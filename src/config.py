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

# --- env override helpers -------------------------------------------------
# Colab/CI runs are configured with env vars, not by editing this file. Each
# knob below reads DEFOR_<NAME> if set, else keeps the literal default.
def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(f"DEFOR_{name}")
    return default if raw is None else int(raw)


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(f"DEFOR_{name}")
    return default if raw is None else float(raw)


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(f"DEFOR_{name}")
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


# --- channels ------------------------------------------------------------
# Phase-1 channel set: Sentinel-2 [B2,B3,B4,B8] + Sentinel-1 [VV,VH] = 6
IN_CHANNELS   = 6

# --- model ---------------------------------------------------------------
ENCODER       = "efficientnet-b0"   # bump to b3 later if VRAM allows
ENCODER_WEIGHTS = None              # 6 input channels != 3-ch ImageNet; train from scratch
CLASSES       = 1                   # binary mask -> single logit channel

# --- training ------------------------------------------------------------
BATCH_SIZE    = _env_int("BATCH_SIZE", 8)     # drop to 4 if CUDA OOM
EPOCHS        = _env_int("EPOCHS", 10)
LR            = _env_float("LR", 3e-4)
OPTIMIZER     = "AdamW"
SCHEDULER     = "cosine"
LOSS          = "BCEWithLogits"   # paper used BCE
USE_AMP       = _env_bool("USE_AMP", True)    # mixed precision — big memory saver on T4
GRAD_ACCUM    = _env_int("GRAD_ACCUM", 1)     # raise to simulate a larger batch if memory-tight
INPUT_DROPOUT = 0.5               # max fraction of dates/bands randomly zeroed in training
NUM_WORKERS   = _env_int("NUM_WORKERS", 2)
PATIENCE      = _env_int("PATIENCE", 0)       # epochs without val-IoU gain before stopping; 0 = never
MIN_DELTA     = _env_float("MIN_DELTA", 1e-4) # IoU gain below this does not count as improvement

# --- data / split --------------------------------------------------------
IMG_SIZE      = 256
N_FOLDS       = 5          # spatial grid folds
VAL_FOLD      = _env_int("VAL_FOLD", 0)       # which fold is held out for validation
GRID_DEG      = 0.15       # spatial grid cell size (deg) for fold assignment
FOREST_DOWNSAMPLE = 0.5    # keep this fraction of "all-forest" easy tiles in train
CLOUD_MAX     = 0.5        # drop tiles whose Sentinel-2 cloud fraction exceeds this
SEED          = _env_int("SEED", 42)

# channel groups (indices into CHANNEL_ORDER [B2,B3,B4,B8,VV,VH]) for input dropout
CHANNEL_GROUPS = {"optical": [0, 1, 2, 3], "sar": [4, 5]}

# --- device --------------------------------------------------------------
# Set DEFOR_DEVICE to force a backend (e.g. "cpu" to debug a CUDA-only crash);
# otherwise pick the best available: CUDA (Colab) > MPS (Apple) > CPU.
DEVICE_OVERRIDE = os.environ.get("DEFOR_DEVICE")


def get_device() -> str:
    """Resolve the training/eval device once, honouring DEFOR_DEVICE."""
    if DEVICE_OVERRIDE:
        return DEVICE_OVERRIDE
    try:
        import torch
    except ImportError:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# --- eval ----------------------------------------------------------------
THRESHOLD     = _env_float("THRESHOLD", 0.5)  # logit->mask threshold for metrics

# paper reference numbers (single model) for the comparison table
PAPER_PIXEL_ACC = 0.904
PAPER_F1        = 0.871
PAPER_IOU       = 0.792
