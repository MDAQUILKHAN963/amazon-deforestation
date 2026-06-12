# Results — Amazon Deforestation Segmentation (low-GPU reproduction)

Reproduction of the deforestation-segmentation task from *"Rapid Deforestation and
Burned Area Detection using Deep Multimodal Learning on Satellite Imagery"*
(arXiv:2307.04916, MultiEarth 2023), scaled to a single free Colab T4 GPU. The
fire/burned-area task was intentionally skipped (see *Future work*).

## Data path used: **A — official MultiEarth 2023 data**

The official dataset is still live on public Azure Blob storage (contrary to the
expectation that the 2023 link had expired). The full training imagery is ~300 GB
(`sent2_b1-b4_train.nc` 80 GB, `sent2_b5-b8_train.nc` 85 GB, `sent1_train.nc` 143 GB),
which is impossible to download on Colab. Instead we **read the source NetCDFs
remotely** (fsspec + h5netcdf over HTTP range requests) and extracted only the tiles
we need, in sorted order for near-contiguous reads. Total data transferred: ~1–2 GB
instead of 300 GB.

- **Labels:** `deforestation_train.nc` — 17,215 hand-labeled 256×256 binary masks,
  11 dates (2016–2021), over lat −4.39…−3.33, lon −55.2…−54.48.
- **Imagery:** Sentinel-2 (B2,B3,B4,B8) + Sentinel-1 (VV,VH), matched to each mask
  tile within a ±60-day window. Sentinel-2 coverage begins Dec-2018, so we used the
  three dry-season **August** label dates (2019, 2020, 2021) — lowest cloud.
- **Extracted dataset:** **4,695 tiles**, 6 channels, `X.npy` (4695, 6, 256, 256)
  float16; deforested-pixel fraction 0.20; mean cloud 0.27.

## Method

- **Model:** U-Net + EfficientNet-B0 encoder (`segmentation-models-pytorch`),
  6-channel input, trained from scratch (`encoder_weights=None`, since ImageNet
  weights assume 3 channels).
- **Spatial 5-fold split** (not random): tiles grouped into a 0.15° lat/lon grid,
  whole grid cells assigned to folds, so validation is spatially separated from
  training (avoids leakage from spatial correlation). Reported on `val_fold=0`.
- **Cloud filtering:** tiles with Sentinel-2 cloud fraction > 0.5 dropped.
- **Forest downsampling:** only 50% of "all-forest" (no-deforestation) tiles kept in
  training, so the model can't inflate its score by predicting "all forest".
- **Augmentation:** horizontal/vertical flip, transpose, 90° rotation (no scaling).
- **Satellite input dropout:** SAR (and occasionally NIR/B8) randomly zeroed in
  training to mimic the heavy missingness of the real test set.
- **Training:** BCEWithLogitsLoss, AdamW (lr 3e-4, wd 1e-4), cosine schedule, mixed
  precision (AMP), batch size 8, 10 epochs. Best checkpoint kept by **val IoU**.
- Split sizes (val_fold=0, after filtering): **2,024 train / 1,013 val**.

## Results (best checkpoint, epoch 6, held-out spatial fold)

| Metric         | Ours   | Paper (single model) |
|----------------|--------|----------------------|
| Pixel accuracy | 96.2%  | 90.4%                |
| F1             | 0.799  | 0.871                |
| IoU            | 0.665  | 0.792                |
| Precision      | 0.737  | —                    |
| Recall         | 0.873  | —                    |

Training was ~30 s/epoch on a T4 (~5 min total). Checkpoint: `outputs/best.pt`.
Qualitative predictions: `outputs/predictions.png`.

### Interpretation & caveats
- **Not a head-to-head comparison.** The paper evaluates on MultiEarth's *hidden
  official test set*; we evaluate on a held-out *spatial fold of the training data*.
  Our higher pixel accuracy is therefore not a "win" over the paper — but the
  evaluation is methodologically sound (proper spatial CV, no leakage).
- The result is in the intended "solid reduced-settings" range: 96% pixel accuracy,
  F1 ≈ 0.80. The main gap vs the paper is **IoU**; recall (0.87) exceeds precision
  (0.74), i.e. the model slightly over-predicts deforestation.

## Future work
- **Fire / burned-area segmentation** (the skipped second task; needs ~100 GB data).
- **Dice (or Dice+BCE) loss** — optimizes overlap directly; the most likely single
  change to raise IoU toward the paper.
- **Cloud-aware S2 selection** — pick the least-cloudy image in the window instead of
  the nearest date (we used nearest for extraction speed).
- **Larger encoder (B0 → B3)**, more channels (B11, B12), more label dates.
- **Light ensembling** — average raw predictions from 2–3 model variants.

## Reproduce
1. `extract.py` — remote lazy extraction → `X.npy`/`Y.npy`/`manifest.csv` (run once; data persisted to Drive).
2. `train.py` — training (`DEFOR_DATA=…/processed DEFOR_OUT=…/outputs python train.py`).
3. `eval.py` — metrics table + `predictions.png`.
All hyperparameters live in `config.py`.
