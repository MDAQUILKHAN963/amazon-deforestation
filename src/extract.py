"""
extract.py — build the training set by REMOTE lazy reads of the MultiEarth
training imagery (never downloads the ~300 GB of source files).

KEY PERFORMANCE IDEA:
  Random remote reads cost ~2 s; contiguous/sorted reads ~0.15 s (14x). So we do
  NOT interleave the three source files per-sample. Instead we:
    1. Plan every sample's (S2, B8, S1) indices from metadata only (cheap).
    2. Read each file in ONE pass, in sorted index order (near-contiguous, fast).
    3. Assemble into consolidated arrays.

Channels per sample (6): [B2, B3, B4, B8]  (Sentinel-2)  + [VV, VH]  (Sentinel-1)
We anchor on the nearest Sentinel-2 acquisition within +/-60 days of the mask date,
take B8 from that same acquisition, and the nearest Sentinel-1 within the window.
Missing B8/S1 are zero-filled. Cloud fraction (from QA60) is recorded per sample so
cloudy tiles can be filtered later, but we do NOT do extra per-sample cloud reads.

Outputs (to --out, default /content/processed):
    X.npy           float16 (N, 6, 256, 256)   scaled to ~[0,1] per band
    Y.npy           uint8   (N, 256, 256)        binary deforestation masks
    manifest.csv    one row per sample (id, lat, lon, dates, has_b8, has_s1, cloud_frac, pos_frac)
    norm_stats.json per-band mean/std over X (used by dataset.py)
    meta.json       channel order + scaling info

Usage:
    python extract.py --out /content/processed                 # full run
    python extract.py --out /content/processed --limit 20      # smoke test
    python extract.py --out /content/processed --max-samples 3000
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from numpy.lib.format import open_memmap

try:
    import fsspec
    import xarray as xr
except ImportError as e:  # pragma: no cover
    raise SystemExit("Need fsspec, h5netcdf, xarray: pip install h5netcdf fsspec aiohttp xarray") from e

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    def tqdm(x, **k):
        return x

BASE = "https://rainforestchallenge.blob.core.windows.net/multiearth2023-dataset-final"

F_S2_LO  = "sent2_b1-b4_train.nc"          # holds B1,B2,B3,B4,QA60
F_S2_HI  = "sent2_b5-b8_train.nc"          # holds B5,B6,B7,B8,B8A,QA60 — we want B8
F_S1     = "sent1_train.nc"                # holds VH,VV

WINDOW          = np.timedelta64(60, "D")  # +/- 2 months
CHANNEL_ORDER   = ["B2", "B3", "B4", "B8", "VV", "VH"]


def open_remote(filename):
    url = f"{BASE}/{filename}"
    fobj = fsspec.open(url, mode="rb", block_size=8 << 20).open()
    return xr.open_dataset(fobj, engine="h5netcdf")


def loc_key(latlon):
    return tuple(np.rint(np.asarray(latlon, dtype=float) * 100).astype(int))


def band_span(ds):
    names = [str(b) for b in ds.data_band.data]
    return {n: i for i, n in enumerate(names)}


def cloud_fraction(qa60):
    qa = qa60.astype(np.uint16)
    return float((((qa & 1024) | (qa & 2048)) > 0).mean())


def build_loc_dates(ds):
    """loc_key -> list of (date, sample_index)."""
    ll = ds.center_lat_lons.data.astype(float)
    dt = ds.collection_dates.data.astype("datetime64[D]")
    out = defaultdict(list)
    for i in range(len(ll)):
        out[loc_key(ll[i])].append((dt[i], i))
    return out


def build_locdate_index(ds):
    """(loc_key, date) -> sample_index."""
    ll = ds.center_lat_lons.data.astype(float)
    dt = ds.collection_dates.data.astype("datetime64[D]")
    return {(loc_key(ll[i]), dt[i]): i for i in range(len(ll))}


def nearest(cands, target, window=WINDOW):
    """Return (date, idx) of the candidate nearest `target` within window, else None."""
    best = None
    for d, i in cands:
        delta = abs(d - target)
        if delta <= window and (best is None or delta < best[0]):
            best = (delta, d, i)
    return None if best is None else (best[1], best[2])


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/content/processed")
    ap.add_argument("--mask-file", default="/content/raw/deforestation_train.nc")
    ap.add_argument("--limit", type=int, default=None, help="cap usable masks scanned (smoke test)")
    ap.add_argument("--max-samples", type=int, default=None, help="cap planned samples")
    ap.add_argument("--dates", default=None,
                    help="comma-separated mask dates (YYYY-MM-DD) to keep, e.g. "
                         "2019-08-01,2020-08-01,2021-08-01 (default: all usable)")
    args = ap.parse_args(argv)
    keep_dates = None
    if args.dates:
        keep_dates = {np.datetime64(d) for d in args.dates.split(",")}

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    print("Opening sources (mask local, imagery remote)...")
    mask_ds = xr.open_dataset(args.mask_file, cache=False)
    s2lo, s2hi, s1 = open_remote(F_S2_LO), open_remote(F_S2_HI), open_remote(F_S1)

    s2lo_b, s2hi_b, s1_b = band_span(s2lo), band_span(s2hi), band_span(s1)
    print("S2-lo:", list(s2lo_b), "| S2-hi:", list(s2hi_b), "| S1:", list(s1_b))
    for need, have, label in (({"B2", "B3", "B4", "QA60"}, s2lo_b, "S2-lo"),
                              ({"B8"}, s2hi_b, "S2-hi"), ({"VV", "VH"}, s1_b, "S1")):
        if need - set(have):
            raise SystemExit(f"{label} missing band(s) {need - set(have)}; have {list(have)}")
    i_b2, i_b3, i_b4, i_qa = s2lo_b["B2"], s2lo_b["B3"], s2lo_b["B4"], s2lo_b["QA60"]
    i_b8 = s2hi_b["B8"]
    i_vv, i_vh = s1_b["VV"], s1_b["VH"]
    lo_lo = min(i_b2, i_b3, i_b4, i_qa)            # contiguous slice start for S2-lo
    s1_lo, s1_hi = min(i_vv, i_vh), max(i_vv, i_vh)

    print("Indexing imagery metadata (small reads)...")
    s2lo_map = build_loc_dates(s2lo)
    s2hi_idx = build_locdate_index(s2hi)
    s1_map   = build_loc_dates(s1)
    s2_min = min(d for v in s2lo_map.values() for d, _ in v)
    print("S2 earliest date:", s2_min)

    # ---- Pass 1: PLAN every sample from metadata only -----------------------
    m_ll = mask_ds.center_lat_lons.data.astype(float)
    m_dt = mask_ds.collection_dates.data.astype("datetime64[D]")
    usable = [(loc_key(m_ll[i]), m_dt[i], i, float(m_ll[i][0]), float(m_ll[i][1]))
              for i in range(len(m_ll))
              if m_dt[i] >= s2_min - WINDOW and (keep_dates is None or m_dt[i] in keep_dates)]
    usable.sort(key=lambda s: (s[0], s[1]))
    if args.limit:
        usable = usable[: args.limit]

    plan = []   # (lat, lon, mdate, midx, s2_date, s2_idx, b8_idx|None, s1_idx|None)
    for key, mdate, midx, lat, lon in usable:
        s2c = nearest(s2lo_map.get(key, []), mdate)
        if s2c is None:
            continue                       # optical-anchored: no S2 -> skip
        s2_date, s2_idx = s2c
        b8_idx = s2hi_idx.get((key, s2_date))
        s1c = nearest(s1_map.get(key, []), mdate)
        plan.append((lat, lon, mdate, midx, s2_date, s2_idx, b8_idx,
                     None if s1c is None else s1c[1]))
        if args.max_samples and len(plan) >= args.max_samples:
            break

    N = len(plan)
    print(f"planned samples: {N}")
    if N == 0:
        raise SystemExit("No samples planned — nothing to extract.")

    X = open_memmap(out / "X.npy", mode="w+", dtype=np.float16, shape=(N, 6, 256, 256))
    Y = np.zeros((N, 256, 256), np.uint8)
    cloud = np.zeros(N, np.float32)
    has_b8 = np.zeros(N, np.int8); has_s1 = np.zeros(N, np.int8)

    # ---- Pass 2: S2-lo (B2,B3,B4 + QA60), read in sorted index order --------
    for s in tqdm(sorted(range(N), key=lambda s: plan[s][5]), desc="S2 RGB"):
        idx = plan[s][5]
        arr = s2lo.images[idx, lo_lo:i_qa + 1, 0].data        # contiguous bands
        X[s, 0] = np.clip(arr[i_b2 - lo_lo].astype(np.float32) / 10000.0, 0, 1)
        X[s, 1] = np.clip(arr[i_b3 - lo_lo].astype(np.float32) / 10000.0, 0, 1)
        X[s, 2] = np.clip(arr[i_b4 - lo_lo].astype(np.float32) / 10000.0, 0, 1)
        cloud[s] = cloud_fraction(arr[i_qa - lo_lo])

    # ---- Pass 3: S2-hi B8 ---------------------------------------------------
    b8_jobs = [s for s in range(N) if plan[s][6] is not None]
    for s in tqdm(sorted(b8_jobs, key=lambda s: plan[s][6]), desc="S2 B8"):
        b8 = s2hi.images[plan[s][6], i_b8, 0].data
        X[s, 3] = np.clip(b8.astype(np.float32) / 10000.0, 0, 1)
        has_b8[s] = 1

    # ---- Pass 4: S1 VV,VH ---------------------------------------------------
    s1_jobs = [s for s in range(N) if plan[s][7] is not None]
    for s in tqdm(sorted(s1_jobs, key=lambda s: plan[s][7]), desc="S1 VV/VH"):
        blk = s1.images[plan[s][7], s1_lo:s1_hi + 1, 0].data
        X[s, 4] = np.clip((blk[i_vv - s1_lo].astype(np.float32) + 30.0) / 35.0, 0, 1)
        X[s, 5] = np.clip((blk[i_vh - s1_lo].astype(np.float32) + 30.0) / 35.0, 0, 1)
        has_s1[s] = 1

    # ---- masks (local, fast) ------------------------------------------------
    for s in tqdm(range(N), desc="masks"):
        Y[s] = (mask_ds.images[plan[s][3], 0].data > 0).astype(np.uint8)

    X.flush()
    np.save(out / "Y.npy", Y)

    # ---- manifest + stats + meta -------------------------------------------
    with open(out / "manifest.csv", "w", newline="") as fp:
        w = csv.writer(fp)
        w.writerow(["id", "lat", "lon", "mask_date", "s2_date", "has_b8", "has_s1", "cloud_frac", "pos_frac"])
        for s in range(N):
            lat, lon, mdate, midx, s2_date = plan[s][0], plan[s][1], plan[s][2], plan[s][3], plan[s][4]
            w.writerow([s, lat, lon, str(mdate), str(s2_date), int(has_b8[s]), int(has_s1[s]),
                        round(float(cloud[s]), 3), round(float(Y[s].mean()), 4)])

    mean = np.zeros(6); std = np.zeros(6)
    for c in range(6):
        ch = np.asarray(X[:, c], dtype=np.float32)
        mean[c], std[c] = ch.mean(), ch.std()
    json.dump({"channel_order": CHANNEL_ORDER, "mean": mean.tolist(), "std": std.tolist()},
              open(out / "norm_stats.json", "w"), indent=2)
    json.dump({"channel_order": CHANNEL_ORDER,
               "scaling": "S2=uint16/10000 clip[0,1]; S1=(dB+30)/35 clip[0,1]",
               "window_days": 60, "n_samples": N},
              open(out / "meta.json", "w"), indent=2)

    print(f"\nsaved X.npy {X.shape} + Y.npy to {out}")
    print(f"   with B8: {int(has_b8.sum())} | with S1: {int(has_s1.sum())} | "
          f"mean deforested fraction: {Y.mean():.3f} | mean cloud: {cloud.mean():.3f}")
    print("   per-band mean:", np.round(mean, 3), "std:", np.round(std, 3))


if __name__ == "__main__":
    sys.exit(main())
