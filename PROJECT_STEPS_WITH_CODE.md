# Project Steps — Explained With Code

A detailed, step-by-step walkthrough of how the Amazon Deforestation Detection
project was built, with the main code for each step explained in plain English.
Study this to explain the project (and the code) confidently.

---

## Step 1 — Understand the task (set up the configuration)

Encoded in `config.py` — the file that holds all settings. The two lines that
*define the task*:

```python
IN_CHANNELS = 6      # input: 4 optical + 2 radar bands
CLASSES     = 1      # output: 1 mask (forest vs deforested)
```
**Meaning:** each input image has **6 channels** (layers of data) and the model
outputs **1** result per pixel — a single yes/no (deforested or not). Keeping all
settings in one file lets us change anything (batch size, learning rate) without
hunting through code.

---

## Step 2 — Get the data (download)

The MultiEarth data lives on a public Microsoft Azure server. The masks (labels)
are small, so we downloaded them; the imagery is huge (~300 GB), so we **read it
directly over the internet** without downloading.

**Download the small mask file:**
```python
BASE = "https://rainforestchallenge.blob.core.windows.net/multiearth2023-dataset-final"
wget -c "{BASE}/deforestation_train.nc"      # the deforestation labels (masks)
```

**Open the huge imagery files remotely** (from `extract.py`):
```python
def open_remote(filename):
    url = f"{BASE}/{filename}"
    fobj = fsspec.open(url, mode="rb", block_size=8 << 20).open()  # stream over HTTP
    return xr.open_dataset(fobj, engine="h5netcdf")               # read like a local file
```
**Meaning:** `fsspec` lets us open a file on a remote server and read only the small
pieces we need, instead of downloading all 300 GB. This was the key trick that made
the project possible on a free account.

---

## Step 3 — Match images to labels

Each label is at a (location, date). We found the satellite image at the **same
location** within **±2 months**. (from `extract.py`)

**Turn a location into a lookup key:**
```python
def loc_key(latlon):
    return tuple(np.rint(np.asarray(latlon, float) * 100).astype(int))  # round to 0.01°
```

**Build a lookup table: location → list of (date, image-number):**
```python
def build_loc_dates(ds):
    out = defaultdict(list)
    for i in range(len(ll)):
        out[loc_key(ll[i])].append((dt[i], i))   # group all images by location
    return out
```

**Find the closest image within ±60 days:**
```python
def nearest(cands, target, window=np.timedelta64(60, "D")):
    best = None
    for d, i in cands:
        delta = abs(d - target)
        if delta <= window and (best is None or delta < best[0]):
            best = (delta, d, i)                  # keep the closest date
    return None if best is None else (best[1], best[2])
```
**Meaning:** for every label, we grab the optical image (B2,B3,B4,B8) and radar image
(VV,VH) taken nearest to that label's date, at the exact same spot. That's how a label
gets paired with its 6-channel image.

---

## Step 4 — Clean & save

We removed cloudy images, scaled pixel values to a standard range, and saved
everything. (from `extract.py`)

**Detect clouds** (Sentinel-2 has a special "QA60" cloud layer):
```python
def cloud_fraction(qa60):
    qa = qa60.astype(np.uint16)
    return float((((qa & 1024) | (qa & 2048)) > 0).mean())   # % of pixels that are cloud
```

**Scale each band to roughly 0–1, then stack into 6 channels:**
```python
opt = [np.clip(x / 10000.0, 0, 1) for x in (b2, b3, b4, b8)]      # optical reflectance
sar = [np.clip((x + 30.0) / 35.0, 0, 1) for x in (vv, vh)]        # radar (decibels)
stack = np.stack(opt + sar, axis=0).astype(np.float16)           # shape (6, 256, 256)
```

**Make the binary mask and save:**
```python
mk = (mask_ds.images[midx, 0].data > 0).astype(np.uint8)   # 1 = deforested, 0 = forest
np.save("X.npy", ...)   # all images
np.save("Y.npy", Y)     # all masks
```
**Meaning:** raw satellite numbers are huge and on different scales (optical vs radar),
so we *normalize* them so the model trains stably. We save two big files — `X.npy`
(images) and `Y.npy` (masks) — so we never re-do this slow step.

---

## Step 5 — Split the data (location-based)

We split into training and validation by **geographic grid cells**, so validation
areas are physically separate from training. (from `dataset.py`)

**Assign each tile to a grid cell → a fold:**
```python
def assign_folds(rows, n_folds=5, grid_deg=0.15):
    for i, r in enumerate(rows):
        gr = int((r["lat"] - lat0) // grid_deg)     # grid row
        gc = int((r["lon"] - lon0) // grid_deg)     # grid column
        folds[i] = (gr * ncols + gc) % n_folds      # assign a fold number
    return folds
```

**Pick training vs validation tiles (and filter):**
```python
def select_indices(rows, folds, val_fold, train):
    for i, r in enumerate(rows):
        is_val = (folds[i] == val_fold)
        if is_val == train:            continue     # train uses other folds, val uses this fold
        if r["cloud_frac"] > 0.5:      continue     # drop cloudy tiles
        if train and r["pos_frac"] == 0.0 and rng.random() > 0.5:
            continue                                # drop half the "all-forest" easy tiles
        idx.append(i)
```
**Meaning:** a random split would let the model peek at areas right next to its test
areas (they look almost identical) — that's cheating. Splitting by location blocks
that. We also drop cloudy tiles and reduce boring "all forest" tiles so the model
actually learns deforestation.

---

## Step 6 — Build the model

The whole model is essentially these lines (from `model.py`):
```python
import segmentation_models_pytorch as smp

model = smp.Unet(
    encoder_name="efficientnet-b0",   # backbone that extracts image features
    encoder_weights=None,             # train from scratch (our input has 6 channels, not 3)
    in_channels=6,                    # 4 optical + 2 radar
    classes=1,                        # one output mask
)
```
**Meaning:** a **U-Net** has two halves — the *encoder* (EfficientNet) compresses the
image to understand what's in it, and the *decoder* expands it back to a full-size map
marking deforestation. We use a library so we get this proven architecture in a few
lines.

---

## Step 7 — Train

The training loop — the core of deep learning (from `train.py`):
```python
for epoch in range(10):                         # 10 full passes over the data
    for x, y in train_loader:                   # a batch of 8 images at a time
        x, y = x.to(device), y.to(device)       # move to GPU
        with torch.autocast(device_type="cuda", enabled=amp_enabled):
            logits = model(x)                   # 1. FORWARD: model predicts
            loss = loss_fn(logits, y)           # 2. LOSS: how wrong is it?
        scaler.scale(loss).backward()           # 3. BACKWARD: compute corrections
        scaler.step(opt); scaler.update()       # 4. UPDATE: adjust the model
        opt.zero_grad()                         # reset for next batch
        sched.step()                            # gently lower the learning rate
```

**The settings around it:**
```python
loss_fn = torch.nn.BCEWithLogitsLoss()                   # error for yes/no per pixel
opt     = torch.optim.AdamW(model.parameters(), lr=3e-4) # the optimizer
sched   = CosineAnnealingLR(opt, T_max=steps)            # lowers LR over time
```

**Save the best model:**
```python
if val["iou"] > best_iou:                       # if this epoch is the best so far
    best_iou = val["iou"]
    torch.save({"model": model.state_dict()}, "best.pt")   # save it
```
**Meaning:** the model guesses, we measure how wrong it is (loss), it figures out what
to fix (backward), and the optimizer nudges it to be better (update). Repeat thousands
of times and it learns. We keep only the best-performing version.

---

## Step 8 — Evaluate & demo

**Measure quality** (from `eval.py`):
```python
pred = (torch.sigmoid(logits) > 0.5)            # convert output to yes/no per pixel
tp = (pred * t).sum()                           # correctly found deforestation
fp = (pred * (1 - t)).sum()                     # false alarms
fn = ((1 - pred) * t).sum()                     # missed deforestation
iou = tp / (tp + fp + fn)                       # overlap with truth
f1  = 2*tp / (2*tp + fp + fn)                   # balance of precision & recall
```

**The demo / web app** (from `app.py`):
```python
def analyze(lat, lon, year):
    idx, km = D.find_tile(rows, lat, lon, int(year))   # find the tile for that location
    pred = D.predict(idx, X, MEAN, STD, MODEL, DEVICE) # run the model
    pct = pred.mean() * 100                             # % of area deforested
    verdict = "Deforestation detected" if pct > 1 else "No significant deforestation"
    # then show satellite / actual / prediction images side by side
```
**Meaning:** to test the model fairly we compare its predictions to the true masks and
count correct/incorrect pixels to get IoU and F1. The web app wraps all this so anyone
can pick a location and see the model work.

---

## Quick concept reminders
- **Segmentation** = labeling every pixel.
- **U-Net** = encoder-decoder network for segmentation.
- **Encoder / EfficientNet** = extracts features from the image.
- **Epoch** = one full pass over the training data. **Batch** = small group (8 images).
- **Loss** = how wrong the prediction is. **Backpropagation** = how it learns from errors.
- **Optimizer (AdamW)** = updates the model to reduce error. **Learning rate** = step size.
- **Normalization** = scaling values for stable training. **Augmentation** = flips/rotations for variety.
- **IoU** = overlap between predicted and actual region.
