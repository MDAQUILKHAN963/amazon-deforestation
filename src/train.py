"""
train.py — training loop for the deforestation U-Net.

Features required by the plan:
  * BCEWithLogitsLoss, AdamW, cosine LR schedule
  * mixed precision (torch.cuda.amp autocast + GradScaler)
  * optional gradient accumulation (config.GRAD_ACCUM)
  * logs train/val loss + metrics each epoch; saves best checkpoint by val IoU
  * clear CUDA-OOM message suggesting smaller batch size
  * --smoke runs 1 epoch on a tiny subset to validate the whole pipeline end-to-end

Usage (on Colab, with DEFOR_DATA pointing at /content/processed):
    python train.py --smoke         # CHECKPOINT 4: tiny end-to-end test
    python train.py                 # full run
    python train.py --epochs 30 --batch-size 4 --lr 1e-4   # sweep without editing config
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

import config as C
from dataset import build_datasets
from model import build_model
from eval import evaluate, confusion_counts, metrics_from_counts, results_table
from utils import set_seed, seed_worker, save_history_csv, save_json, utc_timestamp


def make_optimizer(model, lr=None):
    return torch.optim.AdamW(model.parameters(), lr=lr or C.LR, weight_decay=1e-4)


def train(smoke=False, seed=C.SEED, epochs=None, batch_size=None, lr=None):
    """Train the U-Net. Any of epochs/batch_size/lr left as None falls back to config."""
    set_seed(seed)
    device = C.get_device()
    print(f"device: {device} | seed: {seed}")

    train_ds, val_ds = build_datasets()
    epochs = epochs or C.EPOCHS
    batch = batch_size or C.BATCH_SIZE
    lr = lr or C.LR
    print(f"epochs: {epochs} | batch: {batch} | lr: {lr:g}")

    if smoke:                                   # CHECKPOINT 4: tiny, fast sanity run
        epochs = 1
        train_ds = Subset(train_ds, range(min(20, len(train_ds))))
        val_ds = Subset(val_ds, range(min(20, len(val_ds))))
        print(f"SMOKE TEST: 1 epoch, {len(train_ds)} train / {len(val_ds)} val samples")

    train_loader = DataLoader(train_ds, batch_size=batch, shuffle=True,
                              num_workers=C.NUM_WORKERS, drop_last=True, pin_memory=True,
                              worker_init_fn=seed_worker)
    val_loader = DataLoader(val_ds, batch_size=batch, shuffle=False,
                            num_workers=C.NUM_WORKERS, pin_memory=True)

    model = build_model().to(device)
    opt = make_optimizer(model, lr=lr)
    steps = max(1, len(train_loader) // C.GRAD_ACCUM) * epochs
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    amp_enabled = C.USE_AMP and device.split(":")[0] == "cuda"   # AMP is CUDA-only (not MPS)
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    C.OUTPUTS.mkdir(parents=True, exist_ok=True)
    best_iou = -1.0
    history = []

    try:
        for epoch in range(epochs):
            model.train()
            t0 = time.time()
            run_loss = 0.0
            counts = np.zeros(4)
            opt.zero_grad(set_to_none=True)
            for it, (x, y) in enumerate(train_loader):
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                with torch.autocast(device_type=device.split(":")[0], enabled=amp_enabled):
                    logits = model(x)
                    loss = loss_fn(logits, y) / C.GRAD_ACCUM
                scaler.scale(loss).backward()
                if (it + 1) % C.GRAD_ACCUM == 0:
                    scaler.step(opt); scaler.update()
                    opt.zero_grad(set_to_none=True)
                    sched.step()
                run_loss += loss.item() * C.GRAD_ACCUM
                counts += confusion_counts(logits.detach().float(), y)

            tr = metrics_from_counts(counts)
            val = evaluate(model, val_loader, device)
            dt = time.time() - t0
            print(f"epoch {epoch+1:02d}/{epochs} | {dt:5.1f}s | "
                  f"train loss {run_loss/max(1,len(train_loader)):.4f} "
                  f"IoU {tr['iou']:.3f} | val IoU {val['iou']:.3f} "
                  f"F1 {val['f1']:.3f} acc {val['pixel_acc']*100:.1f}% | lr {sched.get_last_lr()[0]:.2e}")
            history.append({"epoch": epoch + 1, "train_loss": run_loss/max(1,len(train_loader)),
                            "train_iou": tr["iou"], "lr": sched.get_last_lr()[0],
                            "epoch_seconds": round(dt, 2),
                            **{f"val_{k}": v for k, v in val.items()}})
            # rewrite every epoch, not just at the end, so an OOM or a dropped
            # Colab session still leaves the epochs that did finish on disk
            save_history_csv(history, C.OUTPUTS / "history.csv")

            if val["iou"] > best_iou:
                best_iou = val["iou"]
                torch.save({"model": model.state_dict(), "epoch": epoch + 1,
                            "val": val, "config": {k: getattr(C, k) for k in
                            ["ENCODER", "IN_CHANNELS", "VAL_FOLD", "BATCH_SIZE", "LR"]}},
                           C.OUTPUTS / "best.pt")
                if not smoke:
                    print(f"   ✓ new best (val IoU {best_iou:.3f}) -> {C.OUTPUTS/'best.pt'}")

    except torch.cuda.OutOfMemoryError:
        print("\n❌ CUDA OUT OF MEMORY.\n"
              "   Smallest fix first: lower BATCH_SIZE in config.py (8 -> 4 -> 2),\n"
              "   or set GRAD_ACCUM=2 to keep the effective batch size.\n"
              "   Still OOM? reduce IN_CHANNELS or IMG_SIZE.")
        raise

    run_meta = {
        "finished_utc": utc_timestamp(),
        "device": device,
        "seed": seed,
        "epochs": epochs,
        "batch_size": batch,
        "lr": lr,
        "smoke": smoke,
        "encoder": C.ENCODER,
        "in_channels": C.IN_CHANNELS,
        "val_fold": C.VAL_FOLD,
        "grad_accum": C.GRAD_ACCUM,
        "amp": amp_enabled,
        "n_train": len(train_ds),
        "n_val": len(val_ds),
        "epochs_completed": len(history),
        "best_val_iou": best_iou,
        "best_epoch": max(history, key=lambda h: h["val_iou"])["epoch"] if history else None,
    }
    save_json(run_meta, C.OUTPUTS / "run.json")

    print(f"\nbest val IoU: {best_iou:.3f}")
    if not smoke:
        print("\n" + results_table(evaluate(model, val_loader, device)))
    return history


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="1 epoch on 20 samples (pipeline test)")
    ap.add_argument("--seed", type=int, default=C.SEED, help=f"random seed (default {C.SEED})")
    ap.add_argument("--epochs", type=int, help=f"override config.EPOCHS (default {C.EPOCHS})")
    ap.add_argument("--batch-size", type=int, help=f"override config.BATCH_SIZE (default {C.BATCH_SIZE})")
    ap.add_argument("--lr", type=float, help=f"override config.LR (default {C.LR:g})")
    args = ap.parse_args()
    train(smoke=args.smoke, seed=args.seed, epochs=args.epochs,
          batch_size=args.batch_size, lr=args.lr)
