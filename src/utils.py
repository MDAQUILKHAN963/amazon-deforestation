"""
utils.py — small cross-cutting helpers (seeding, run bookkeeping).

Kept deliberately dependency-light: everything here must import without torch
being present, so the metric/IO helpers stay usable from plain scripts.
"""
from __future__ import annotations

import os
import random

import numpy as np


def set_seed(seed: int, deterministic: bool = False) -> int:
    """Seed python, numpy and (if available) torch. Returns the seed used.

    `deterministic=True` also pins cuDNN to deterministic kernels. That costs
    real throughput on conv-heavy models like ours, so it is off by default and
    reserved for runs whose numbers need to be exactly reproducible.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch
    except ImportError:                      # torch-free contexts (docs, CI lint)
        return seed

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True   # autotune convs for fixed 256x256 input
    return seed


def seed_worker(worker_id: int) -> None:
    """DataLoader worker_init_fn: give each worker a distinct, reproducible seed.

    Without this, every worker inherits the parent's numpy state and the random
    augmentations/input-dropout repeat identically across workers.
    """
    worker_seed = (int(os.environ.get("PYTHONHASHSEED", 0)) + worker_id) % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
