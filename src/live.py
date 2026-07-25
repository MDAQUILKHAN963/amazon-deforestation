"""
live.py — REAL-LIFE deforestation check on ANY location using live Google Earth
Engine (GEE) imagery.

It fetches Sentinel-2 (B2,B3,B4,B8) + Sentinel-1 (VV,VH) for a chosen location and
date, preprocesses them EXACTLY as in training (same bands, same scaling, same
normalization), and runs the trained model. No ground-truth mask exists for a live
location, so we show the satellite image next to the model's prediction.

NOTE on generalization: the model was trained on one Amazon region, so it works best
on similar tropical rainforest. Very different places (cities, deserts, other forests)
would need retraining on local data.

Setup (Colab):
    import ee
    ee.Authenticate()                 # one-time, opens a sign-in link
    import live
    live.init_ee("YOUR_GEE_PROJECT_ID")
    live.predict_live(-4.05, -54.90, "2021-08-01")
"""
from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
import requests
import torch

import config as C
from model import build_model

S2_BANDS = ["B2", "B3", "B4", "B8"]
S1_BANDS = ["VV", "VH"]


def init_ee(project_id):
    """Initialize Earth Engine. Run ee.Authenticate() once before this."""
    import ee
    try:
        ee.Initialize(project=project_id)
    except Exception:
        ee.Authenticate()
        ee.Initialize(project=project_id)
    print("Earth Engine ready (project:", project_id, ")")


def fetch_tile(lat, lon, date, window_days=60, size=256, scale=10):
    """Download a size×size, 6-band tile centered on (lat, lon) for the given date."""
    import ee
    point = ee.Geometry.Point(lon, lat)
    region = point.buffer(size * scale / 2).bounds()      # ~2.56 km square
    d = ee.Date(date)
    start, end = d.advance(-window_days, "day"), d.advance(window_days, "day")

    s2 = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
          .filterBounds(point).filterDate(start, end)
          .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 40))
          .select(S2_BANDS).median())                     # least-cloudy composite

    s1 = (ee.ImageCollection("COPERNICUS/S1_GRD")
          .filterBounds(point).filterDate(start, end)
          .filter(ee.Filter.eq("instrumentMode", "IW"))
          .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
          .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
          .select(S1_BANDS).median())

    img = s2.addBands(s1)
    url = img.getDownloadURL({"region": region, "dimensions": f"{size}x{size}",
                              "format": "NPY"})
    resp = requests.get(url)
    if resp.status_code != 200 or not resp.content:
        raise RuntimeError("Earth Engine returned no imagery for this location/date "
                           "(try a different date or a spot with less cloud).")
    arr = np.load(io.BytesIO(resp.content))               # structured array, fields = band names
    return arr


def preprocess(arr):
    """Same scaling as extract.py: S2 /10000, S1 (dB+30)/35 -> 6-channel float array."""
    opt = [np.clip(arr[b].astype(np.float32) / 10000.0, 0, 1) for b in S2_BANDS]
    sar = [np.clip((arr[b].astype(np.float32) + 30.0) / 35.0, 0, 1) for b in S1_BANDS]
    return np.stack(opt + sar, axis=0)                    # (6, H, W)


def load_model(proc_dir=C.DATA_PROC, ckpt=None):
    norm = json.load(open(Path(proc_dir) / "norm_stats.json"))
    mean = np.array(norm["mean"], np.float32).reshape(-1, 1, 1)
    std = np.array(norm["std"], np.float32).reshape(-1, 1, 1)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model().to(device)
    ckpt = Path(ckpt) if ckpt else C.OUTPUTS / "best.pt"
    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=False)["model"])
    model.eval()
    return model, mean, std, device


@torch.no_grad()
def predict_live(lat, lon, date, bundle=None, save="/content/outputs/live_result.png"):
    """Fetch live imagery for (lat, lon, date), run the model, show + report result."""
    if bundle is None:
        bundle = load_model()
    model, mean, std, device = bundle

    arr = fetch_tile(lat, lon, date)
    img = preprocess(arr)
    x = torch.from_numpy((img - mean) / std).unsqueeze(0).float().to(device)
    prob = torch.sigmoid(model(x))[0, 0].cpu().numpy()
    pred = (prob > C.THRESHOLD).astype(np.uint8)
    pct = pred.mean() * 100
    verdict = "🟥 DEFORESTATION DETECTED" if pct > 1.0 else "🟩 No significant deforestation"

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rgb = np.clip(np.stack([img[2], img[1], img[0]], -1) * 3.5, 0, 1)   # B4,B3,B2
    fig, ax = plt.subplots(1, 2, figsize=(8.5, 4.3))
    ax[0].imshow(rgb);                                ax[0].set_title("Live satellite (RGB)")
    ax[1].imshow(pred, cmap="Reds", vmin=0, vmax=1);  ax[1].set_title(f"Model prediction ({pct:.0f}% deforested)")
    for a in ax:
        a.axis("off")
    fig.suptitle(f"{verdict}   |   ({lat}, {lon})  {date}", fontsize=12)
    plt.tight_layout()
    Path(save).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save, dpi=90); plt.close()
    print(f"{verdict}  —  model predicts {pct:.1f}% of this area deforested")
    print("saved picture ->", save)
    return pct


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", required=True, help="your Google Earth Engine project ID")
    ap.add_argument("--lat", type=float, required=True)
    ap.add_argument("--lon", type=float, required=True)
    ap.add_argument("--date", required=True, help="YYYY-MM-DD")
    a = ap.parse_args()
    init_ee(a.project)
    predict_live(a.lat, a.lon, a.date, save="live_result.png")
