"""
download.py — Path A data acquisition for the MultiEarth 2023 deforestation
segmentation task.

The official MultiEarth dataset is hosted on public Azure Blob Storage. As of
this writing the direct blob URLs are live (verified HTTP 200), so we download
them directly with resumable HTTP requests — no azcopy or auth needed.

IMPORTANT (memory): the Sentinel image files are large (~13 GB and ~9.5 GB). On
Colab, download these to the EPHEMERAL disk (e.g. /content/raw), NOT to Google
Drive (free Drive is only 15 GB). We later extract small .npy tiles from them and
save ONLY those to Drive.

Usage:
    # default: download the 6-channel deforestation set (S2 + S1 + masks + targets)
    python -m src.download --dest /content/raw

    # optical-only (skip the 9.5 GB SAR file)
    python -m src.download --dest /content/raw --skip-sar
"""
from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

BASE_URL = "https://rainforestchallenge.blob.core.windows.net/multiearth2023-dataset-final"

# (filename, approx size bytes, required?) — sizes used only for a sanity check
FILES = {
    "masks":   ("deforestation_train.nc",                    13_122_354,    True),
    "targets": ("deforestation_segmentation_targets.nc",     94_135,        True),
    "sent2":   ("sent2_deforestation_segmentation.nc",       12_967_173_670, True),
    "sent1":   ("sent1_deforestation_segmentation.nc",       9_454_058_403, False),  # SAR; optional for 4-ch v1
    # landsat8 left out of the default 6-channel set; uncomment to add later:
    # "landsat8": ("landsat8_deforestation_segmentation.nc", None,          False),
}


def _human(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.1f} {unit}"
        n /= 1024


def download_one(filename: str, dest_dir: Path, expected: int | None = None) -> Path:
    """Resumable download of a single blob into dest_dir. Skips if already complete."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    out = dest_dir / filename
    url = f"{BASE_URL}/{filename}"

    # already present and right size? skip.
    if out.exists() and expected is not None and out.stat().st_size == expected:
        print(f"  ✓ already have {filename} ({_human(expected)}) — skipping")
        return out

    # resume from where a partial download left off
    existing = out.stat().st_size if out.exists() else 0
    req = urllib.request.Request(url)
    if existing:
        req.add_header("Range", f"bytes={existing}-")
        print(f"  ↻ resuming {filename} from {_human(existing)}")
    else:
        print(f"  ↓ downloading {filename} ...")

    with urllib.request.urlopen(req) as resp:
        total = existing + int(resp.headers.get("Content-Length", 0))
        mode = "ab" if existing else "wb"
        done = existing
        chunk = 1 << 20  # 1 MB
        with open(out, mode) as f:
            while True:
                buf = resp.read(chunk)
                if not buf:
                    break
                f.write(buf)
                done += len(buf)
                pct = (done / total * 100) if total else 0
                print(f"\r    {filename}: {_human(done)} / {_human(total)} ({pct:4.1f}%)",
                      end="", flush=True)
        print()
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description="Download MultiEarth deforestation data (Path A).")
    ap.add_argument("--dest", required=True,
                    help="Destination dir for raw .nc files (use Colab ephemeral disk, e.g. /content/raw)")
    ap.add_argument("--skip-sar", action="store_true",
                    help="Skip the ~9.5 GB Sentinel-1 SAR file (optical-only 4-channel v1)")
    args = ap.parse_args(argv)

    dest = Path(args.dest)
    print(f"Destination: {dest.resolve()}")

    plan = list(FILES.items())
    for key, (fname, size, required) in plan:
        if key == "sent1" and args.skip_sar:
            print(f"  ⤬ skipping {fname} (--skip-sar)")
            continue
        download_one(fname, dest, expected=size)

    print("\n✅ Download complete. Files in", dest.resolve())
    for p in sorted(dest.glob("*.nc")):
        print(f"   {p.name:50s} {_human(p.stat().st_size)}")


if __name__ == "__main__":
    sys.exit(main())
