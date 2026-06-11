"""
config.py — all hyperparameters and paths in one place.

Edit values here; the rest of the pipeline imports from this module so you
never have to hunt through training code to change a knob.
"""
import os
from pathlib import Path

# --- paths ---------------------------------------------------------------
# Defaults work for local dev; on Colab set env vars (or just rely on the
# /content defaults) so we read the fast local copy, not Drive:
#   DEFOR_DATA = /content/processed     (extracted X.npy/Y.npy/manifest live here)
#   DEFOR_OUT  = /content/internship/outputs
PROJECT_ROOT  = Path(__file__).resolve().parent.parent
DATA_RAW      = PROJECT_ROOT / "data" / "raw"
DATA_PROC     = Path(os.environ.get("DEFOR_DATA", str(PROJECT_ROOT / "data" / "processed")))
OUTPUTS       = Path(os.environ.get("DEFOR_OUT",  str(PROJECT_ROOT / "outputs")))

# --- channels ------------------------------------------------------------
# Phase-1 channel set: Sentinel-2 [B2,B3,B4,B8] + Sentinel-1 [VV,VH] = 6
IN_CHANNELS   = 6

# --- model ---------------------------------------------------------------
ENCODER       = "efficientnet-b0"   # bump to b3 later if VRAM allows
ENCODER_WEIGHTS = None              # 6 input channels != 3-ch ImageNet; train from scratch
CLASSES       = 1                   # binary mask -> single logit channel

# --- training ------------------------------------------------------------
BATCH_SIZE    = 8          # drop to 4 if CUDA OOM
EPOCHS        = 10
LR            = 3e-4
OPTIMIZER     = "AdamW"
SCHEDULER     = "cosine"
LOSS          = "BCEWithLogits"   # paper used BCE
USE_AMP       = True              # mixed precision — big memory saver on T4
GRAD_ACCUM    = 1                 # raise to simulate a larger batch if memory-tight
INPUT_DROPOUT = 0.5               # max fraction of dates/bands randomly zeroed in training
NUM_WORKERS   = 2

# --- data / split --------------------------------------------------------
IMG_SIZE      = 256
N_FOLDS       = 5          # spatial grid folds
VAL_FOLD      = 0          # which fold is held out for validation
GRID_DEG      = 0.15       # spatial grid cell size (deg) for fold assignment
FOREST_DOWNSAMPLE = 0.5    # keep this fraction of "all-forest" easy tiles in train
CLOUD_MAX     = 0.5        # drop tiles whose Sentinel-2 cloud fraction exceeds this
SEED          = 42

# channel groups (indices into CHANNEL_ORDER [B2,B3,B4,B8,VV,VH]) for input dropout
CHANNEL_GROUPS = {"optical": [0, 1, 2, 3], "sar": [4, 5]}

# --- eval ----------------------------------------------------------------
THRESHOLD     = 0.5        # logit->mask threshold for metrics

# paper reference numbers (single model) for the comparison table
PAPER_PIXEL_ACC = 0.904
PAPER_F1        = 0.871
PAPER_IOU       = 0.792
