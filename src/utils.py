"""
utils.py — small cross-cutting helpers (seeding, run bookkeeping).

Kept deliberately dependency-light: everything here must import without torch
being present, so the metric/IO helpers stay usable from plain scripts.
"""
from __future__ import annotations

import csv
import json
import os
import random
from datetime import datetime, timezone
from pathlib import Path

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


def save_history_csv(history, path) -> Path:
    """Write a list of per-epoch metric dicts to CSV.

    Columns are the union of every row's keys (a resumed run may start logging
    a field mid-way), ordered by first appearance so `epoch` stays leftmost.
    Missing values are written blank rather than dropping the row.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not history:
        path.write_text("", encoding="utf-8")
        return path

    fields = []
    for row in history:
        for k in row:
            if k not in fields:
                fields.append(k)

    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, restval="")
        w.writeheader()
        w.writerows(history)
    return path


def save_json(obj, path) -> Path:
    """Dump a dict to pretty JSON, coercing anything non-serializable to str."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str, sort_keys=True)
    return path


def utc_timestamp() -> str:
    """ISO-8601 UTC stamp for run metadata (timezone-aware, no local drift)."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")
