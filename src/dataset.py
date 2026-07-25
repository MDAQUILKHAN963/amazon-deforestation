"""
dataset.py — PyTorch Dataset + spatial cross-validation split for the
deforestation tiles produced by extract.py (X.npy / Y.npy / manifest.csv).

Design choices follow the plan:
  * SPATIAL 5-fold split (not random): tiles are grouped into a lat/lon grid and
    whole grid cells are assigned to folds, so validation tiles are spatially
    separated from training tiles (random splits leak through spatial correlation).
  * Cloud filtering: tiles whose Sentinel-2 cloud fraction exceeds CLOUD_MAX are
    dropped (nearest-date selection sometimes lands on cloud).
  * Forest downsampling (train only): "easy" all-forest tiles (no deforested pixels)
    are subsampled so the model can't inflate its score by predicting "all forest".
  * Augmentations (train only): flips / transpose / 90-deg rotations — no scaling or
    heavy rotation (the paper found those hurt).
  * Satellite input dropout (train only): randomly zero the SAR (and occasionally an
    optical band) to mirror the heavy missingness of the real test set.

X is opened with mmap so the 3.5 GB array never fully loads into RAM.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

try:
    import albumentations as A
    _HAS_ALBU = True
except ImportError:  # pragma: no cover
    _HAS_ALBU = False

import config as C


# ----------------------------------------------------------------- manifest/split
def load_manifest(proc_dir: Path):
    rows = list(csv.DictReader(open(Path(proc_dir) / "manifest.csv")))
    for r in rows:
        r["id"] = int(r["id"]); r["lat"] = float(r["lat"]); r["lon"] = float(r["lon"])
        r["cloud_frac"] = float(r["cloud_frac"]); r["pos_frac"] = float(r["pos_frac"])
        r["has_s1"] = int(r["has_s1"])
    return rows


def assign_folds(rows, n_folds=C.N_FOLDS, grid_deg=C.GRID_DEG):
    """Assign each sample a spatial fold by its lat/lon grid cell."""
    lats = np.array([r["lat"] for r in rows]); lons = np.array([r["lon"] for r in rows])
    lat0, lon0 = lats.min(), lons.min()
    ncols = int(np.floor((lons.max() - lon0) / grid_deg)) + 1
    folds = np.empty(len(rows), dtype=int)
    for i, r in enumerate(rows):
        gr = int((r["lat"] - lat0) // grid_deg)
        gc = int((r["lon"] - lon0) // grid_deg)
        folds[i] = (gr * ncols + gc) % n_folds
    return folds


def select_indices(rows, folds, val_fold, train: bool, *,
                   cloud_max=C.CLOUD_MAX, forest_keep=C.FOREST_DOWNSAMPLE, seed=C.SEED):
    """Return the sample indices for the train or val split after filtering."""
    rng = np.random.default_rng(seed)
    want_val = not train
    idx = []
    for i, r in enumerate(rows):
        is_val = (folds[i] == val_fold)
        if is_val != want_val:
            continue
        if r["cloud_frac"] > cloud_max:          # drop cloudy tiles from both splits
            continue
        if train and r["pos_frac"] == 0.0:        # forest downsampling (train only)
            if rng.random() > forest_keep:
                continue
        idx.append(i)
    return np.array(idx, dtype=int)


# ----------------------------------------------------------------------- dataset
class DeforestationDataset(Dataset):
    def __init__(self, proc_dir, indices, *, train: bool,
                 norm_stats=None, input_dropout=C.INPUT_DROPOUT, augment=True):
        proc_dir = Path(proc_dir)
        self.X = np.load(proc_dir / "X.npy", mmap_mode="r")   # (N,6,256,256) float16
        self.Y = np.load(proc_dir / "Y.npy", mmap_mode="r")   # (N,256,256) uint8
        self.idx = np.asarray(indices, dtype=int)
        self.train = train
        self.input_dropout = input_dropout if train else 0.0
        self.augment = augment and train

        if norm_stats is None:
            norm_stats = json.load(open(proc_dir / "norm_stats.json"))
        self.mean = np.array(norm_stats["mean"], np.float32).reshape(-1, 1, 1)
        self.std = np.array(norm_stats["std"], np.float32).reshape(-1, 1, 1)

        self.aug = None
        if self.augment and _HAS_ALBU:
            self.aug = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.Transpose(p=0.5),
                A.RandomRotate90(p=0.5),
            ])
        self.groups = C.CHANNEL_GROUPS

    def __len__(self):
        return len(self.idx)

    def _drop_inputs(self, img):
        """Satellite input dropout: zero SAR (and occasionally B8) to mimic missingness."""
        if self.input_dropout <= 0:
            return img
        rng = np.random
        if rng.random() < self.input_dropout:               # drop the whole SAR modality
            for c in self.groups["sar"]:
                img[c] = 0.0
        if rng.random() < self.input_dropout * 0.5:          # occasionally drop NIR (B8)
            img[3] = 0.0
        return img

    def __getitem__(self, i):
        j = self.idx[i]
        img = np.asarray(self.X[j], dtype=np.float32).copy()   # (6,256,256)
        mask = np.asarray(self.Y[j], dtype=np.float32).copy()  # (256,256)

        if self.aug is not None:
            hwc = np.transpose(img, (1, 2, 0))                 # albumentations wants HWC
            out = self.aug(image=hwc, mask=mask)
            img = np.transpose(out["image"], (2, 0, 1)).copy()
            mask = out["mask"]

        img = self._drop_inputs(img)
        img = (img - self.mean) / self.std                     # standardize per band

        return (torch.from_numpy(img),
                torch.from_numpy(mask).unsqueeze(0))           # (6,H,W), (1,H,W)


# ------------------------------------------------------------------- convenience
def build_datasets(proc_dir=C.DATA_PROC, val_fold=C.VAL_FOLD, *, verbose=True):
    """Return (train_ds, val_ds) with the spatial split + filtering applied."""
    proc_dir = Path(proc_dir)
    rows = load_manifest(proc_dir)
    folds = assign_folds(rows)
    norm = json.load(open(proc_dir / "norm_stats.json"))

    tr_idx = select_indices(rows, folds, val_fold, train=True)
    va_idx = select_indices(rows, folds, val_fold, train=False)

    train_ds = DeforestationDataset(proc_dir, tr_idx, train=True, norm_stats=norm)
    val_ds = DeforestationDataset(proc_dir, va_idx, train=False, norm_stats=norm)

    if verbose:
        import collections
        fold_counts = collections.Counter(folds.tolist())
        print(f"total tiles: {len(rows)} | fold sizes: {dict(sorted(fold_counts.items()))}")
        print(f"val_fold={val_fold} -> train: {len(tr_idx)}  val: {len(va_idx)}")
        print(f"  (cloud_max={C.CLOUD_MAX}, forest_keep={C.FOREST_DOWNSAMPLE})")
    return train_ds, val_ds


if __name__ == "__main__":
    tr, va = build_datasets()
    x, y = tr[0]
    print("sample image:", x.shape, x.dtype, "mask:", y.shape, y.dtype,
          "| img mean/std:", round(float(x.mean()), 3), round(float(x.std()), 3),
          "| mask pos frac:", round(float(y.mean()), 3))
