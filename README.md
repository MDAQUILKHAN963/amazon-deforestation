# 🌳 Amazon Deforestation Detection (+ India transfer)

Deep-learning **image segmentation** that detects deforested areas in satellite
imagery. A U-Net + EfficientNet model is trained on the Amazon, then **fine-tuned via
transfer learning** to work on a new region (Hasdeo, India) using live Google Earth
Engine data — with a single web app covering both.

Based on the methodology of *"Rapid Deforestation and Burned Area Detection using Deep
Multimodal Learning on Satellite Imagery"* (arXiv:2307.04916), scaled to a single free GPU.

## What it does
- Input: 6-channel satellite tiles — Sentinel-2 optical (B2, B3, B4, B8) + Sentinel-1
  radar (VV, VH).
- Output: a per-pixel mask of deforested vs forested land.
- A Gradio web app: pick a region + location → see satellite image, ground truth, and
  the model's prediction side by side.

## Results (held-out validation)

| Region | Pixel accuracy | F1 | IoU |
|--------|----------------|------|------|
| Amazon (trained from scratch) | 96% | 0.80 | 0.66 |
| Hasdeo, India (fine-tuned)    | —   | 0.53 | 0.37 |

The Amazon model alone predicts ~100% deforestation on India (domain shift);
fine-tuning on ~140 live India tiles fixes it. See [RESULTS.md](RESULTS.md).

## Project structure
```
src/
  config.py        all hyperparameters & paths
  download.py      data acquisition (MultiEarth, Azure)
  extract.py       remote lazy extraction -> 6-channel tiles + masks (X.npy / Y.npy)
  dataset.py       PyTorch Dataset + spatial cross-validation split
  model.py         U-Net + EfficientNet
  train.py         training loop (AMP, AdamW, cosine, BCE)
  eval.py          metrics (pixel acc, F1, IoU) + prediction images
  demo.py          location-based prediction on saved tiles
  live.py          live single-location prediction via Google Earth Engine
  gee_india.py     build India dataset (GEE + Hansen) and fine-tune the model
  app_unified.py   Gradio web app covering Amazon + India
```

## How it works
1. **Amazon:** extract 6-channel tiles + hand-labeled masks from the MultiEarth dataset;
   train a U-Net (EfficientNet-B0) with a spatial train/val split (no leakage).
2. **India:** fetch live Sentinel-1/2 imagery from Google Earth Engine; use the **Hansen
   Global Forest Change** forest-loss layer as ground truth; fine-tune the Amazon model.
3. **App:** each region uses the model + data source it performs well on.

## Run
```bash
pip install -r requirements.txt
# Amazon: train / evaluate (data prepared by extract.py)
python src/train.py
python src/eval.py
# India + web app run on Colab with Earth Engine (see gee_india.py / app_unified.py)
```

## Data & credits
- **MultiEarth 2023** — Amazon imagery + deforestation labels.
- **Sentinel-1 / Sentinel-2** (ESA Copernicus) via **Google Earth Engine**.
- **Hansen Global Forest Change** (University of Maryland) — India ground-truth labels.

*Note: models are region-specific — they perform well on the forest types they were
trained on. Extending to arbitrary global regions is future work.*
