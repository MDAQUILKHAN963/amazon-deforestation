"""
gee_india.py — extend the project to a new region (Hasdeo, India) using LIVE
Google Earth Engine imagery, then fine-tune the Amazon model on it.

Pipeline:
  1. Fetch Sentinel-2 (B2,B3,B4,B8) + Sentinel-1 (VV,VH) tiles for the Hasdeo region
     from Earth Engine, in the same 6-channel format as the Amazon data.
  2. Use the Hansen Global Forest Change "forest loss" layer as ground-truth masks.
  3. Sample tiles CENTERED on deforestation (stratified sampling) + some forest tiles,
     so the dataset is not almost-all forest.
  4. Fine-tune the Amazon-trained model (best.pt) on these tiles with Dice+BCE loss.

This demonstrates transfer learning / domain adaptation: an Amazon-only model fails
on India (predicts ~100% everywhere), and fine-tuning on ~140 live tiles fixes it.

Setup (Colab): authenticate Earth Engine once, then run build_dataset() + finetune().
    import ee; ee.Authenticate(auth_mode="notebook"); ee.Initialize(project="YOUR_PROJECT")
    import gee_india as g
    X, Y = g.build_dataset(out_dir="/content")
    g.finetune(x_path="/content/X_india.npy", y_path="/content/Y_india.npy",
               amazon_ckpt=".../best.pt", out_ckpt="/content/best_india.pt")
"""
from __future__ import annotations

import io
import time

import numpy as np
import requests

BANDS = ["B2", "B3", "B4", "B8", "VV", "VH"]
HANSEN_ASSET = "UMD/hansen/global_forest_change_2023_v1_11"
# Hasdeo forest region, Chhattisgarh, India
HASDEO_BBOX = [82.40, 22.30, 82.90, 23.00]   # [lon_min, lat_min, lon_max, lat_max]


# --------------------------------------------------------------------- fetching
def fetch_tile(lon, lat, year=2023):
    """Fetch one 256x256, 6-channel tile + Hansen loss mask, scaled like training."""
    import ee
    tile = ee.Geometry.Point(lon, lat).buffer(1280).bounds()          # ~2.56 km -> 256 px @10m
    s2 = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED").filterBounds(tile)
          .filterDate(f"{year}-01-01", f"{year}-12-31")
          .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 40)).median()
          .select(["B2", "B3", "B4", "B8"]))
    s1 = (ee.ImageCollection("COPERNICUS/S1_GRD").filterBounds(tile)
          .filterDate(f"{year}-01-01", f"{year}-12-31")
          .filter(ee.Filter.eq("instrumentMode", "IW"))
          .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
          .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
          .select(["VV", "VH"]).median())
    loss = ee.Image(HANSEN_ASSET).select("lossyear").gt(0).rename("loss")
    url = (s2.addBands(s1).addBands(loss)).getDownloadURL(
        {"region": tile, "scale": 10, "format": "NPY"})
    arr = np.load(io.BytesIO(requests.get(url).content))
    st = np.stack([arr[b].astype(np.float32) for b in BANDS], 0)[:, :256, :256]
    if st.shape != (6, 256, 256):
        return None
    st[:4] = np.clip(st[:4] / 10000.0, 0, 1)          # Sentinel-2 optical
    st[4:] = np.clip((st[4:] + 30.0) / 35.0, 0, 1)    # Sentinel-1 radar (dB)
    return st, arr["loss"].astype(np.uint8)[:256, :256]


def build_dataset(out_dir=".", n_defor=100, n_forest=40, year=2023, seed=7):
    """Build the India dataset: tiles centered on deforestation + random forest tiles."""
    import ee
    region = ee.Geometry.Rectangle(HASDEO_BBOX)
    lossbin = ee.Image(HANSEN_ASSET).select("lossyear").gt(0).rename("cls")
    # stratifiedSample pulls points FROM the deforestation class (sparse otherwise)
    centers = lossbin.stratifiedSample(
        numPoints=n_defor, classBand="cls", region=region, scale=30,
        classValues=[1], classPoints=[n_defor], geometries=True, seed=seed
    ).geometry().coordinates().getInfo()
    print("deforestation centers:", len(centers))

    X, Y = [], []
    t0 = time.time()
    for k, (lo, la) in enumerate(centers):
        try:
            r = fetch_tile(lo, la, year)
            if r is not None:
                X.append(r[0]); Y.append(r[1])
        except Exception:
            pass
        if (k + 1) % 20 == 0:
            print(f"  {k+1}/{len(centers)} deforestation tiles | {int(time.time()-t0)}s")

    rng = np.random.default_rng(0)
    lon0, lat0, lon1, lat1 = HASDEO_BBOX
    for _ in range(n_forest):
        try:
            r = fetch_tile(float(rng.uniform(lon0 + .02, lon1 - .02)),
                           float(rng.uniform(lat0 + .02, lat1 - .02)), year)
            if r is not None:
                X.append(r[0]); Y.append(r[1])
        except Exception:
            pass

    X = np.array(X, np.float16); Y = np.array(Y, np.uint8)
    print("dataset:", X.shape, "| deforested fraction:", round(float(Y.mean()), 4))
    np.save(f"{out_dir}/X_india.npy", X); np.save(f"{out_dir}/Y_india.npy", Y)
    return X, Y


# ------------------------------------------------------------------- fine-tuning
def finetune(x_path, y_path, amazon_ckpt, out_ckpt, epochs=25, lr=1e-4, batch=8):
    """Fine-tune the Amazon model on the India tiles (Dice+BCE for class imbalance)."""
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    import segmentation_models_pytorch as smp
    from segmentation_models_pytorch.losses import DiceLoss

    X = np.load(x_path); Y = np.load(y_path); N = len(X)
    Xf = X.astype(np.float32)
    mean = Xf.reshape(N, 6, -1).mean(axis=(0, 2)).reshape(6, 1, 1)
    std = Xf.reshape(N, 6, -1).std(axis=(0, 2)).reshape(6, 1, 1) + 1e-6

    rng = np.random.default_rng(0); perm = rng.permutation(N)
    nval = max(10, N // 5); val_idx, tr_idx = perm[:nval], perm[nval:]

    class DS(Dataset):
        def __init__(self, idx, train): self.idx = idx; self.train = train
        def __len__(self): return len(self.idx)
        def __getitem__(self, i):
            j = self.idx[i]; img = X[j].astype(np.float32).copy(); m = Y[j].astype(np.float32).copy()
            if self.train:
                if np.random.rand() < .5: img = img[:, ::-1, :].copy(); m = m[::-1, :].copy()
                if np.random.rand() < .5: img = img[:, :, ::-1].copy(); m = m[:, ::-1].copy()
                k = np.random.randint(4); img = np.rot90(img, k, (1, 2)).copy(); m = np.rot90(m, k).copy()
            return torch.from_numpy((img - mean) / std), torch.from_numpy(m).unsqueeze(0)

    tr = DataLoader(DS(tr_idx, True), batch_size=batch, shuffle=True, drop_last=True)
    va = DataLoader(DS(val_idx, False), batch_size=batch)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = smp.Unet("efficientnet-b0", encoder_weights=None, in_channels=6, classes=1).to(device)
    model.load_state_dict(torch.load(amazon_ckpt, map_location=device, weights_only=False)["model"])
    print("loaded Amazon weights as starting point | device:", device)

    dice = DiceLoss(mode="binary"); bce = nn.BCEWithLogitsLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    def metrics():
        model.eval(); tp = fp = fn = 0
        with torch.no_grad():
            for x, y in va:
                x, y = x.to(device), y.to(device); p = (torch.sigmoid(model(x)) > .5).float()
                tp += (p * y).sum().item(); fp += (p * (1 - y)).sum().item(); fn += ((1 - p) * y).sum().item()
        return tp / (tp + fp + fn + 1e-7), 2 * tp / (2 * tp + fp + fn + 1e-7)

    best = -1
    for ep in range(epochs):
        model.train()
        for x, y in tr:
            x, y = x.to(device), y.to(device); opt.zero_grad()
            out = model(x); (dice(out, y) + bce(out, y)).backward(); opt.step()
        sched.step(); iou, f1 = metrics()
        print(f"epoch {ep+1:02d} | val IoU {iou:.3f} F1 {f1:.3f}")
        if iou > best:
            best = iou; torch.save({"model": model.state_dict()}, out_ckpt)
    np.save(out_ckpt.replace(".pt", "_mean.npy"), mean)
    np.save(out_ckpt.replace(".pt", "_std.npy"), std)
    print("best val IoU:", round(best, 3), "->", out_ckpt)
    return best
