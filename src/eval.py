"""
eval.py — metrics (pixel accuracy, F1, IoU) and qualitative prediction images.

Run standalone to evaluate a checkpoint on the validation fold and dump
side-by-side prediction-vs-truth images for the report.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

import config as C


@torch.no_grad()
def confusion_counts(logits, target, thresh=C.THRESHOLD):
    """Accumulate TP, FP, FN, TN for binary masks given raw logits."""
    pred = (torch.sigmoid(logits) > thresh).float()
    t = target.float()
    tp = (pred * t).sum().item()
    fp = (pred * (1 - t)).sum().item()
    fn = ((1 - pred) * t).sum().item()
    tn = ((1 - pred) * (1 - t)).sum().item()
    return np.array([tp, fp, fn, tn], dtype=np.float64)


def metrics_from_counts(c):
    """c = [TP, FP, FN, TN] -> dict of pixel_acc, f1, iou, precision, recall."""
    tp, fp, fn, tn = c
    eps = 1e-7
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    iou = tp / (tp + fp + fn + eps)
    pixel_acc = (tp + tn) / (tp + fp + fn + tn + eps)
    return {"pixel_acc": pixel_acc, "f1": f1, "iou": iou,
            "precision": precision, "recall": recall}


@torch.no_grad()
def evaluate(model, loader, device, thresh=C.THRESHOLD):
    """Full-pass evaluation; returns the metrics dict."""
    model.eval()
    total = np.zeros(4)
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        with torch.autocast(device_type=device.split(":")[0], enabled=C.USE_AMP):
            logits = model(x)
        total += confusion_counts(logits.float(), y, thresh)
    return metrics_from_counts(total)


def results_table(m):
    """Pretty table comparing our metrics to the paper's single-model numbers."""
    lines = [
        "| Metric         | Ours   | Paper  |",
        "|----------------|--------|--------|",
        f"| Pixel accuracy | {m['pixel_acc']*100:5.1f}% | {C.PAPER_PIXEL_ACC*100:5.1f}% |",
        f"| F1             | {m['f1']:.3f}  | {C.PAPER_F1:.3f}  |",
        f"| IoU            | {m['iou']:.3f}  | {C.PAPER_IOU:.3f}  |",
        f"| Precision      | {m['precision']:.3f}  |   -    |",
        f"| Recall         | {m['recall']:.3f}  |   -    |",
    ]
    return "\n".join(lines)


@torch.no_grad()
def save_predictions(model, dataset, device, out_path, n=8, thresh=C.THRESHOLD):
    """Save a grid of RGB / ground-truth / prediction for n validation samples."""
    import matplotlib.pyplot as plt
    model.eval()
    n = min(n, len(dataset))
    fig, ax = plt.subplots(3, n, figsize=(2.3 * n, 7))
    if n == 1:
        ax = ax.reshape(3, 1)
    mean = dataset.mean; std = dataset.std
    for j in range(n):
        x, y = dataset[j]
        logits = model(x.unsqueeze(0).to(device))
        pred = (torch.sigmoid(logits)[0, 0].cpu().numpy() > thresh).astype(float)
        img = x.numpy() * std + mean                      # un-standardize
        rgb = np.clip(np.stack([img[2], img[1], img[0]], -1) * 3.5, 0, 1)  # B4,B3,B2
        ax[0, j].imshow(rgb)
        ax[1, j].imshow(y[0].numpy(), cmap="Reds", vmin=0, vmax=1)
        ax[2, j].imshow(pred, cmap="Reds", vmin=0, vmax=1)
        for r in range(3):
            ax[r, j].axis("off")
    for r, lab in enumerate(["RGB", "truth", "pred"]):
        ax[r, 0].text(-0.15, 0.5, lab, rotation=90, va="center", ha="right",
                      transform=ax[r, 0].transAxes)
    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=90)
    plt.close()
    print("saved predictions ->", out_path)


if __name__ == "__main__":
    from torch.utils.data import DataLoader
    from dataset import build_datasets
    from model import build_model

    device = "cuda" if torch.cuda.is_available() else "cpu"
    _, val_ds = build_datasets()
    val_loader = DataLoader(val_ds, batch_size=C.BATCH_SIZE, num_workers=C.NUM_WORKERS)

    model = build_model().to(device)
    ckpt = C.OUTPUTS / "best.pt"
    if ckpt.exists():
        model.load_state_dict(torch.load(ckpt, map_location=device)["model"])
        print("loaded", ckpt)
    else:
        print("WARNING: no checkpoint at", ckpt, "- evaluating random weights")

    m = evaluate(model, val_loader, device)
    print("\n" + results_table(m))
    save_predictions(model, val_ds, device, C.OUTPUTS / "predictions.png")
