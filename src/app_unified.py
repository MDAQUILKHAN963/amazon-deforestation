"""
app_unified.py — single Gradio web app covering BOTH regions.

  * Amazon  -> uses the native MultiEarth tiles (the model's training data).
  * Hasdeo (India) -> fetches LIVE Sentinel imagery from Google Earth Engine and
                      runs the fine-tuned India model, compared to Hansen forest loss.

Each region uses the model AND data source it performs well on (mixing them causes
the domain-shift failures we observed). Requires Earth Engine to be initialized.

Run (Colab): authenticate EE, then run this file / paste as a cell.
    import ee; ee.Authenticate(auth_mode="notebook"); ee.Initialize(project="YOUR_PROJECT")
"""
from __future__ import annotations

import csv
import io
import json

import numpy as np
import requests
import torch
import segmentation_models_pytorch as smp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import gradio as gr

# Root holding models + data. On Colab this is your Drive project folder.
D = "/content/drive/MyDrive/amazon-deforestation"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BANDS = ["B2", "B3", "B4", "B8", "VV", "VH"]
HANSEN_ASSET = "UMD/hansen/global_forest_change_2023_v1_11"


def _load_model(path):
    m = smp.Unet("efficientnet-b0", encoder_weights=None, in_channels=6, classes=1).to(DEVICE)
    m.load_state_dict(torch.load(path, map_location=DEVICE, weights_only=False)["model"])
    m.eval()
    return m


AMAZON = _load_model(f"{D}/outputs/best.pt")
INDIA = _load_model(f"{D}/outputs/best_india.pt")

# Amazon native tiles + normalization
AX = np.load(f"{D}/data/processed/X.npy", mmap_mode="r")
AY = np.load(f"{D}/data/processed/Y.npy", mmap_mode="r")
Arows = list(csv.DictReader(open(f"{D}/data/processed/manifest.csv")))
for r in Arows:
    r["lat"] = float(r["lat"]); r["lon"] = float(r["lon"]); r["cloud_frac"] = float(r["cloud_frac"])
_amz = json.load(open(f"{D}/data/processed/norm_stats.json"))
AMEAN = np.array(_amz["mean"], np.float32).reshape(6, 1, 1)
ASTD = np.array(_amz["std"], np.float32).reshape(6, 1, 1)
# India normalization
IMEAN = np.load(f"{D}/outputs/india_mean.npy")
ISTD = np.load(f"{D}/outputs/india_std.npy")


def _predict(model, st, mean, std):
    x = torch.from_numpy(((st - mean) / std)[None]).float().to(DEVICE)
    with torch.no_grad():
        return (torch.sigmoid(model(x))[0, 0].cpu().numpy() > 0.5).astype(np.uint8)


def _show(rgb, truth, pred):
    pct = pred.mean() * 100; actual = truth.mean() * 100
    det = pct > 1; color = "#c62828" if det else "#2e7d32"
    info = (f"<div style='padding:14px;border-radius:12px;background:{color}12;border:1px solid {color}55'>"
            f"<b style='font-size:20px;color:{color}'>"
            f"{'🟥 Deforestation detected' if det else '🟩 No significant deforestation'}</b><br>"
            f"Model prediction: <b>{pct:.1f}%</b> &nbsp;·&nbsp; Actual: <b>{actual:.1f}%</b></div>")
    fig, ax = plt.subplots(1, 3, figsize=(11, 4)); fig.patch.set_facecolor("white")
    ax[0].imshow(rgb); ax[0].set_title("Satellite (RGB)")
    ax[1].imshow(truth, cmap="Reds", vmin=0, vmax=1); ax[1].set_title(f"Actual ({actual:.0f}%)")
    ax[2].imshow(pred, cmap="Reds", vmin=0, vmax=1); ax[2].set_title(f"Model prediction ({pct:.0f}%)")
    for a in ax:
        a.axis("off")
    plt.tight_layout()
    return info, fig


def _amazon(lat, lon, year):
    cand = [i for i, r in enumerate(Arows) if r["cloud_frac"] <= 0.35 and r["mask_date"].startswith(str(year))] \
        or [i for i, r in enumerate(Arows) if r["cloud_frac"] <= 0.35] or list(range(len(Arows)))
    idx = min(cand, key=lambda i: (Arows[i]["lat"] - lat) ** 2 + (Arows[i]["lon"] - lon) ** 2)
    st = np.asarray(AX[idx], dtype=np.float32); truth = np.asarray(AY[idx])
    pred = _predict(AMAZON, st, AMEAN, ASTD)
    return _show(np.clip(np.stack([st[2], st[1], st[0]], -1) * 3.5, 0, 1), truth, pred)


def _india(lat, lon, year):
    import ee
    tile = ee.Geometry.Point(lon, lat).buffer(1280).bounds()
    s2 = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED").filterBounds(tile)
          .filterDate(f"{year}-01-01", f"{year}-12-31").filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 40))
          .median().select(["B2", "B3", "B4", "B8"]))
    s1 = (ee.ImageCollection("COPERNICUS/S1_GRD").filterBounds(tile)
          .filterDate(f"{year}-01-01", f"{year}-12-31").filter(ee.Filter.eq("instrumentMode", "IW"))
          .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
          .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
          .select(["VV", "VH"]).median())
    loss = ee.Image(HANSEN_ASSET).select("lossyear").gt(0).rename("loss")
    arr = np.load(io.BytesIO(requests.get((s2.addBands(s1).addBands(loss))
          .getDownloadURL({"region": tile, "scale": 10, "format": "NPY"})).content))
    st = np.stack([arr[b].astype(np.float32) for b in BANDS], 0)[:, :256, :256]
    if st.shape != (6, 256, 256):
        return "Couldn't fetch a full tile here. Try another spot/year.", None
    st[:4] = np.clip(st[:4] / 10000.0, 0, 1); st[4:] = np.clip((st[4:] + 30.0) / 35.0, 0, 1)
    truth = arr["loss"].astype(np.uint8)[:256, :256]
    pred = _predict(INDIA, st, IMEAN, ISTD)
    return _show(np.clip(np.stack([st[2], st[1], st[0]], -1) * 3.5, 0, 1), truth, pred)


def analyze(region, lat, lon, year):
    lat, lon = float(lat), float(lon)
    if region == "Amazon (Brazil)":
        return _amazon(lat, lon, min([2019, 2020, 2021], key=lambda a: abs(a - int(year))))
    return _india(lat, lon, year)


def build_app():
    with gr.Blocks(theme=gr.themes.Soft(primary_hue="green"), title="Deforestation Detector") as app:
        gr.HTML("<div style='background:linear-gradient(135deg,#1b5e20,#43a047);color:white;"
                "padding:22px;border-radius:14px'><h1>🌍 Deforestation Detector — Amazon & India</h1>"
                "<p>Pick a region and location; the model for that region highlights deforestation "
                "and compares to ground truth.</p></div>")
        with gr.Row():
            with gr.Column(scale=2):
                region = gr.Dropdown(["Hasdeo (India)", "Amazon (Brazil)"], value="Hasdeo (India)", label="Region")
                lat = gr.Number(value=22.341, label="Latitude"); lon = gr.Number(value=82.607, label="Longitude")
                year = gr.Dropdown(["2019", "2020", "2021", "2022", "2023"], value="2023", label="Year")
                go = gr.Button("🔍 Analyze", variant="primary")
                gr.Markdown("**Examples:**")
                gr.Examples([["Hasdeo (India)", 22.341, 82.607, "2023"],
                             ["Hasdeo (India)", 22.688, 82.579, "2023"],
                             ["Amazon (Brazil)", -4.05, -54.90, "2021"]], [region, lat, lon, year])
            with gr.Column(scale=3):
                verdict = gr.HTML(); pic = gr.Plot(show_label=False)
        go.click(analyze, [region, lat, lon, year], [verdict, pic])
    return app


if __name__ == "__main__":
    build_app().launch(share=True)
