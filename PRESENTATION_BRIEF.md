# Project Brief — Deforestation Detection (Amazon + India)
### Complete reference document for building a presentation

> Hand this file to Claude and ask it to build a slide deck. It contains the full
> story, methods, numbers, and a suggested slide structure.

---

## 1. One-line summary
A deep-learning system that detects deforested areas in satellite imagery. A model is
trained on the Amazon, shown to fail on a new region (domain shift), then **fine-tuned
via transfer learning** to work on Hasdeo, India — all wrapped in a single web app that
uses live satellite data.

## 2. Problem & motivation
- Deforestation (especially in the Amazon and Indian coalfields) is a major
  environmental problem that must be monitored.
- Manually inspecting satellite images across huge areas is slow and impractical.
- **Goal:** automatically detect deforested areas from satellite imagery, pixel by pixel
  (a task called *image segmentation*).

## 3. Objectives
1. Build a deforestation-segmentation model for the Amazon (reproduce a published method
   at small scale, on a single free GPU).
2. Test whether it generalizes to a completely different region (India).
3. Fix the generalization gap using transfer learning + live global satellite data.
4. Deliver an interactive web app anyone can use.

## 4. Data

### Amazon (primary dataset)
- Source: **MultiEarth 2023** challenge dataset (Brazilian Amazon, Para state).
- Region: latitude −4.39° to −3.33°, longitude −55.20° to −54.48°.
- **Sentinel-2** optical bands (B2, B3, B4, B8) + **Sentinel-1** radar bands (VV, VH)
  → 6 channels per tile.
- Labels: hand-drawn deforestation masks (2016–2021).
- Extracted **~4,700 tiles** of 256×256 pixels; ~20% deforested pixels.
- Key trick: the raw imagery is ~300 GB, so we read it **remotely** from Azure and pulled
  only the tiles we needed (~1–2 GB transferred instead of 300 GB).

### India (new region)
- Source: **live Google Earth Engine** — Sentinel-2 + Sentinel-1 for Hasdeo,
  Chhattisgarh (latitude 22.30°–23.00°, longitude 82.40°–82.90°; the Korba coalfield).
- Labels: **Hansen Global Forest Change** (University of Maryland) — a free, worldwide
  forest-loss dataset used as ground truth.
- Built **~140 tiles** by sampling locations centered on deforestation (stratified
  sampling) plus some forest tiles as negatives.

## 5. Method

- **Preprocessing:** stack 6 bands, scale to a standard range (Sentinel-2 reflectance
  ÷10000; Sentinel-1 radar in decibels), normalize per band; remove cloudy images.
- **Model:** **U-Net** with an **EfficientNet-B0** encoder
  (via `segmentation-models-pytorch`). Input 6 channels, output 1 mask.
  - *Encoder* compresses the image to understand it; *decoder* rebuilds a full-size
    deforestation map.
- **Spatial cross-validation split:** tiles grouped into a geographic grid; whole grid
  cells assigned to train/validation — so validation areas are physically separate from
  training (a random split would leak and inflate scores).
- **Training:** BCEWithLogits loss, AdamW optimizer, cosine learning-rate schedule,
  mixed precision (AMP), 10 epochs; best checkpoint kept by validation IoU.
- **Data augmentation:** horizontal/vertical flips, transpose, 90° rotations.

## 6. Results

### Amazon (trained from scratch, held-out spatial fold)
| Metric | Ours | Reference paper |
|--------|------|-----------------|
| Pixel accuracy | 96% | 90.4% |
| F1 score | 0.80 | 0.871 |
| IoU | 0.66 | 0.792 |

*(Not a head-to-head comparison — we test on our own validation split, the paper on a
hidden test set — but the method is sound: proper spatial CV, no leakage.)*

### The generalization problem (key finding)
- Running the **Amazon model directly on India** predicts **~100% deforestation
  everywhere** — a total failure. This is **domain shift**: the model only learned what
  Amazon rainforest looks like.

### India (after fine-tuning)
| Metric | Value |
|--------|-------|
| IoU | 0.37 |
| F1 score | 0.53 |
- Fine-tuning the Amazon model on ~140 live India tiles (Dice + BCE loss, 25 epochs)
  makes it **correctly localize the coal-mine deforestation** instead of predicting
  everything as deforested.
- Honest caveat: with only ~140 small tiles it's a **proof-of-concept** — it reliably
  flags heavy deforestation but fine detail is approximate.

## 7. The web app
- Built with **Gradio**; a single interface for both regions.
- User picks a region + latitude/longitude + year → sees three panels:
  **Satellite image | Actual deforestation | Model prediction** (red = deforested).
- **Amazon** uses its native MultiEarth tiles; **India** fetches **live** imagery from
  Earth Engine and compares to Hansen forest loss.
- Each region uses the model + data source it performs well on (mixing them re-triggers
  the domain-shift failure — an important design lesson).

## 8. Deployment (in progress)
- Temporary sharing: Gradio gives a public `gradio.live` link while the notebook runs.
- Permanent hosting plan: **Hugging Face Spaces** (free), which needs a Google Earth
  Engine **service account** so the server can access satellite data without a login.

## 9. Tech stack
- **Python, PyTorch**, `segmentation-models-pytorch` (U-Net + EfficientNet)
- **Google Earth Engine** (live Sentinel-1/2 imagery)
- **Hansen Global Forest Change** (India labels)
- **Gradio** (web app), **Google Colab** (free T4 GPU), **Google Drive** (storage)

## 10. Challenges & how they were solved
| Challenge | Solution |
|-----------|----------|
| Training imagery was ~300 GB | Remote lazy reading — pulled only needed tiles |
| Severe class imbalance (few deforested pixels) | Stratified sampling + Dice loss + forest downsampling |
| Clouds ruin optical images | Cloud filtering (QA60 band) + used dry-season imagery + added radar |
| Model failed on India | Transfer learning: fine-tuned on live India data |
| No labels for India | Used the Hansen global forest-loss dataset as ground truth |

## 11. Key concepts (one-line each — for a glossary slide)
- **Image segmentation** — labeling every pixel (here: forest vs deforested).
- **U-Net** — an encoder–decoder neural network for segmentation.
- **EfficientNet** — the backbone that extracts features from the image.
- **Sentinel-2 / Sentinel-1** — free ESA optical / radar satellites.
- **Hansen Global Forest Change** — free worldwide forest-loss dataset (the "truth").
- **Google Earth Engine** — Google platform hosting global satellite data.
- **Domain shift** — a model performing poorly on data unlike its training data.
- **Transfer learning / fine-tuning** — adapting a trained model to a new region with a
  little new data.
- **IoU / F1** — overlap-based accuracy metrics for segmentation.

## 12. Future work
- Support any location worldwide (train on more diverse regions).
- Add fire / burned-area detection as a second task.
- Improve the India model with more tiles and longer training.
- Deploy permanently on Hugging Face Spaces.

---

## 13. Suggested slide structure (≈12 slides)
1. **Title** — Deforestation Detection using Deep Learning (Amazon + India)
2. **Problem & motivation** — why automated deforestation monitoring matters
3. **Objective** — the 4 objectives (Section 3)
4. **Data** — Sentinel-1/2 + labels; Amazon vs India table
5. **Method** — the pipeline diagram (data → model → training → app)
6. **The model** — U-Net + EfficientNet, 6-channel input
7. **Amazon results** — metrics table + a prediction image
8. **The generalization problem** — Amazon model on India = 100% (before)
9. **Transfer learning** — fine-tuning on live India data → works (after)
10. **The web app** — screenshot of the Amazon & India detector
11. **Challenges & solutions** — the table from Section 10
12. **Conclusion & future work**

Tone: keep it simple and honest. The strongest story is the **before/after**: a model
that failed on a new region, fixed with transfer learning and live global data.
