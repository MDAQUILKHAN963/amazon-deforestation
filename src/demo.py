"""
demo.py — simple interactive prediction.

Given a latitude/longitude inside the study region (and optionally a year), find the
matching satellite tile, run the trained model, and report a deforestation verdict
plus a picture (RGB / true mask / predicted mask).

NOTE: the model needs the 6-channel satellite tiles produced by extract.py — it does
NOT work on ordinary photos. It also only covers the trained Amazon region for
2019-2021. So we look up a real tile by location rather than accepting any image.

Usage (from src/, with DEFOR_DATA / DEFOR_OUT set):
    python demo.py --lat -4.05 --lon -54.90 --year 2021
    python demo.py --random            # pick a random tile
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import config as C
from model import build_model
from dataset import load_manifest

VERDICT_PCT = 1.0   # if >this % of the tile is predicted deforested -> "detected"


def _load(proc_dir=C.DATA_PROC, ckpt=None):
    proc_dir = Path(proc_dir)
    X = np.load(proc_dir / "X.npy", mmap_mode="r")
    Y = np.load(proc_dir / "Y.npy", mmap_mode="r")
    rows = load_manifest(proc_dir)
    norm = json.load(open(proc_dir / "norm_stats.json"))
    mean = np.array(norm["mean"], np.float32).reshape(-1, 1, 1)
    std = np.array(norm["std"], np.float32).reshape(-1, 1, 1)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model().to(device)
    ckpt = Path(ckpt) if ckpt else C.OUTPUTS / "best.pt"
    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=False)["model"])
    model.eval()
    return X, Y, rows, mean, std, model, device


def find_tile(rows, lat, lon, year=None):
    """Return the index of the tile nearest to (lat, lon), optionally matching a year."""
    cand = range(len(rows))
    if year is not None:
        cand = [i for i in cand if rows[i]["mask_date"].startswith(str(year))]
        if not cand:
            print(f"(no tile for year {year}; ignoring year)")
            cand = range(len(rows))
    d2 = [( (rows[i]["lat"]-lat)**2 + (rows[i]["lon"]-lon)**2, i) for i in cand]
    dist, i = min(d2)
    return i, dist ** 0.5


@torch.no_grad()
def predict(idx, X, mean, std, model, device, thresh=C.THRESHOLD):
    img = np.asarray(X[idx], dtype=np.float32)
    x = torch.from_numpy((img - mean) / std).unsqueeze(0).to(device)
    prob = torch.sigmoid(model(x))[0, 0].cpu().numpy()
    return (prob > thresh).astype(np.uint8)


def run(lat=None, lon=None, year=None, random=False, save="/content/outputs/demo_result.png"):
    X, Y, rows, mean, std, model, device = _load()
    if random:
        idx = int(np.random.randint(len(rows)))
    else:
        idx, km = find_tile(rows, lat, lon, year)
        print(f"nearest tile to ({lat}, {lon}): id {idx} at "
              f"({rows[idx]['lat']}, {rows[idx]['lon']}), {rows[idx]['mask_date']} "
              f"(~{km*111:.1f} km away)")

    pred = predict(idx, X, mean, std, model, device)
    truth = np.asarray(Y[idx])
    pct = pred.mean() * 100
    verdict = "🟥 DEFORESTATION DETECTED" if pct > VERDICT_PCT else "🟩 No significant deforestation"
    print(f"\n{verdict}  —  model predicts {pct:.1f}% of this area deforested "
          f"(actual: {truth.mean()*100:.1f}%)")

    # picture
    import matplotlib.pyplot as plt
    img = np.asarray(X[idx], dtype=np.float32)
    rgb = np.clip(np.stack([img[2], img[1], img[0]], -1) * 3.5, 0, 1)   # B4,B3,B2
    fig, ax = plt.subplots(1, 3, figsize=(10, 3.6))
    ax[0].imshow(rgb);                         ax[0].set_title("satellite (RGB)")
    ax[1].imshow(truth, cmap="Reds", vmin=0, vmax=1); ax[1].set_title("actual deforestation")
    ax[2].imshow(pred, cmap="Reds", vmin=0, vmax=1);  ax[2].set_title(f"model prediction ({pct:.0f}%)")
    for a in ax: a.axis("off")
    plt.tight_layout()
    Path(save).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save, dpi=90); plt.close()
    print("saved picture ->", save)
    return idx, pct


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--lat", type=float)
    ap.add_argument("--lon", type=float)
    ap.add_argument("--year", type=int, default=None)
    ap.add_argument("--random", action="store_true")
    a = ap.parse_args()
    if not a.random and (a.lat is None or a.lon is None):
        ap.error("give --lat and --lon, or use --random")
    run(lat=a.lat, lon=a.lon, year=a.year, random=a.random)
