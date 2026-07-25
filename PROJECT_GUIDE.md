# Project Guide — Amazon Deforestation Detection (for explaining the project)

A plain-English walkthrough of how the project was built and what the main
deep-learning code does. Study this to explain the project confidently.

## Part 1 — How it was built (the pipeline)
Raw satellite data → cleaned dataset → model → training → evaluation → demo.

1. **Task:** image *segmentation* — label every pixel as forest or deforested.
2. **Data:** MultiEarth 2023 Amazon dataset — Sentinel-2 (optical) + Sentinel-1 (radar)
   imagery, plus hand-labeled deforestation masks.
3. **Match images to labels:** for each label (location + date), take the matching
   satellite image within ±2 months. Stack 6 channels: optical B2,B3,B4,B8 + radar VV,VH.
4. **Clean & save:** drop cloudy images, normalize values, save as `X.npy` (images) and
   `Y.npy` (masks). ~4,700 tiles, each 256×256.
5. **Split:** training vs validation, using a *location-based* split (no leakage).
6. **Model:** U-Net with an EfficientNet backbone.
7. **Train:** 10 epochs — predict, measure error, correct; keep the best model.
8. **Evaluate & demo:** accuracy / F1 / IoU, prediction images, and a web app.

## Part 2 — Main code explained

### Model (model.py)
```python
model = smp.Unet(encoder_name="efficientnet-b0", encoder_weights=None,
                 in_channels=6, classes=1)
```
U-Net = encoder-decoder. Encoder (EfficientNet) understands the image; decoder rebuilds
a pixel-by-pixel deforestation map. 6 inputs (optical+radar), 1 output (yes/no per pixel).

### Dataset (dataset.py)
```python
def __getitem__(self, i):
    img, mask = self.X[i], self.Y[i]
    if self.augment: img, mask = flip_rotate(img, mask)   # more variety
    img = (img - self.mean) / self.std                    # normalization
    return img, mask
```
Feeds one image + label at a time; augments (flips/rotations) and normalizes.

### Training loop (train.py) — the core
```python
for epoch in range(10):
    for images, masks in train_loader:
        predictions = model(images)            # 1. forward: guess
        loss = loss_fn(predictions, masks)     # 2. loss: how wrong
        loss.backward()                        # 3. backward: compute corrections
        optimizer.step()                       # 4. update: adjust model
        optimizer.zero_grad()
```
1. Forward = predict. 2. Loss = error number. 3. Backward = backpropagation (find what
caused the error). 4. Update = optimizer (AdamW) reduces the error. Repeat → learns.
- Loss = BCEWithLogitsLoss (yes/no per pixel). Optimizer = AdamW, lr 3e-4.
- Cosine scheduler lowers lr over time; AMP = faster low-memory training.
- Save the epoch with the best validation IoU.

### Evaluation (eval.py)
```python
pred = (sigmoid(logits) > 0.5)
iou = TP / (TP + FP + FN)
f1  = 2*TP / (2*TP + FP + FN)
```
Threshold at 0.5 → yes/no per pixel, then compare to truth. IoU = overlap of predicted
and real deforested regions.

## Part 3 — Concept cheat-sheet
- Segmentation = labeling every pixel.
- U-Net = encoder-decoder for segmentation. Encoder/EfficientNet = feature extractor.
- Epoch = one full pass over training data. Batch = small group processed together (8).
- Loss = how wrong the prediction is. Backpropagation = how the model learns from errors.
- Optimizer (AdamW) = updates model to reduce error. Learning rate = step size.
- Normalization = scale values for stable training. Augmentation = flip/rotate for variety.
- IoU = overlap between predicted and actual region. Overfitting = memorizing training data.

## Part 4 — Likely mentor questions
- Why U-Net? Standard for segmentation; rebuilds full-resolution maps, keeps detail.
- Why optical + radar? Radar sees through clouds; optical shows vegetation — more robust.
- Why location-based split? Random split leaks info (nearby pixels are similar).
- Results good? 96% acc / 0.80 F1 / 0.66 IoU — solid for small-scale; F1/IoU matter since
  deforested area is the minority class.
- Improvements? Dice loss for IoU, bigger model, live global satellite data.

## Results
| Metric | Ours | Paper |
|---|---|---|
| Pixel accuracy | 96% | 90.4% |
| F1 | 0.80 | 0.871 |
| IoU | 0.66 | 0.792 |
(Tested on our own validation split, not the paper's hidden test set — not head-to-head.)
