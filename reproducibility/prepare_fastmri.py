#!/usr/bin/env python3
"""
prepare_fastmri.py – Consolidate fastMRI .h5 files into a single training
archive for the GPU-aware h5py ML benchmark.

fastMRI source format
---------------------
Each per-patient file contains:
    reconstruction_rss  float32  (num_slices, height, width)

We extract every slice, resize it to a square *target* resolution, and write
all slices into one output dataset with chunk shape (1, target, target).
This maps the paper's three benchmark chunk sizes directly:

    --size 256  → chunks=(1, 256, 256)  ≈ 0.25 MB / chunk
    --size 512  → chunks=(1, 512, 512)  ≈ 1.00 MB / chunk
    --size 1024 → chunks=(1,1024,1024)  ≈ 4.00 MB / chunk

One slice == one chunk, so every DataLoader batch read lands on an integer
number of HDF5 chunks — no cross-chunk scatter reads.

Output HDF5 layout
------------------
    /images          float32  (N, size, size)  chunked (1, size, size)
    /file_index      int32    (N,)              which source file
    /slice_index     int32    (N,)              which slice within that file
    attrs:
        source_dir, size, chunk_mb, n_files, n_slices, fastmri_split

Usage
-----
    # Single chunk variant
    python prepare_fastmri.py /data/fastmri/knee/train knee_512.h5 --size 512

    # All three variants (run from a shell loop)
    for S in 256 512 1024; do
        python prepare_fastmri.py /data/fastmri/knee/train knee_${S}.h5 --size $S
    done

    # Limit to a subset of files while testing
    python prepare_fastmri.py /data/fastmri/knee/train knee_512.h5 \\
        --size 512 --max-files 50

Dependencies
------------
    h5py, numpy, Pillow
    Install with: pip install h5py numpy Pillow
"""

import argparse
import sys
import time
from pathlib import Path

import h5py
import numpy as np

try:
    from PIL import Image as _PILImage
    _PIL_AVAILABLE = True
except ImportError:
    _PIL_AVAILABLE = False


# ---------------------------------------------------------------------------
# Resize helpers
# ---------------------------------------------------------------------------

def _resize_slice(arr: np.ndarray, target: int) -> np.ndarray:
    """Resize a 2-D float32 slice to (target, target) and return float32."""
    h, w = arr.shape
    if h == target and w == target:
        return arr

    if _PIL_AVAILABLE:
        # Normalise to uint16 for Pillow (preserves more dynamic range than uint8)
        lo, hi = arr.min(), arr.max()
        span = hi - lo
        if span > 0:
            scaled = ((arr - lo) / span * 65535).astype(np.uint16)
        else:
            scaled = np.zeros_like(arr, dtype=np.uint16)

        img = _PILImage.fromarray(scaled.astype(np.int32))
        resample = _PILImage.LANCZOS if hasattr(_PILImage, "LANCZOS") else _PILImage.ANTIALIAS
        img = img.resize((target, target), resample=resample)
        out = np.array(img, dtype=np.float32)

        # Rescale back to original range
        out = out / 65535.0 * span + lo
        return out
    else:
        # Fallback: centre-crop / pad (no Pillow)
        out = np.zeros((target, target), dtype=np.float32)
        ch = min(h, target)
        cw = min(w, target)
        r0 = (h - ch) // 2
        c0 = (w - cw) // 2
        dr0 = (target - ch) // 2
        dc0 = (target - cw) // 2
        out[dr0:dr0 + ch, dc0:dc0 + cw] = arr[r0:r0 + ch, c0:c0 + cw]
        return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _count_slices(files: list[Path]) -> int:
    total = 0
    for p in files:
        with h5py.File(p, "r") as f:
            total += f["reconstruction_rss"].shape[0]
    return total


def prepare(
    src_dir: Path,
    out_path: Path,
    size: int,
    max_files: int | None,
    split: str,
) -> None:
    src_files = sorted(src_dir.glob("*.h5"))
    if not src_files:
        sys.exit(f"No .h5 files found in {src_dir}")

    if max_files is not None:
        src_files = src_files[:max_files]

    n_files = len(src_files)
    chunk_bytes = size * size * 4
    chunk_mb = chunk_bytes / 1024 / 1024

    print(f"Source        : {src_dir}  ({n_files} files)")
    print(f"Output        : {out_path}")
    print(f"Target size   : {size}x{size}  ->  chunk = {chunk_mb:.2f} MB")

    # First pass: count total slices so we can pre-allocate
    print("Counting slices …", end=" ", flush=True)
    n_slices = _count_slices(src_files)
    print(f"{n_slices:,}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(out_path, "w") as dst:
        ds_img = dst.create_dataset(
            "images",
            shape=(n_slices, size, size),
            dtype=np.float32,
            chunks=(1, size, size),
        )
        ds_fidx = dst.create_dataset("file_index",  shape=(n_slices,), dtype=np.int32)
        ds_sidx = dst.create_dataset("slice_index", shape=(n_slices,), dtype=np.int32)

        dst.attrs["source_dir"]    = str(src_dir)
        dst.attrs["size"]          = size
        dst.attrs["chunk_mb"]      = chunk_mb
        dst.attrs["n_files"]       = n_files
        dst.attrs["n_slices"]      = n_slices
        dst.attrs["fastmri_split"] = split

        row = 0
        t0 = time.perf_counter()

        for file_idx, src_path in enumerate(src_files):
            with h5py.File(src_path, "r") as src:
                rss = src["reconstruction_rss"]  # (n_slices, H, W) float32
                n = rss.shape[0]

                for s in range(n):
                    sl = rss[s].astype(np.float32)
                    ds_img[row]  = _resize_slice(sl, size)
                    ds_fidx[row] = file_idx
                    ds_sidx[row] = s
                    row += 1

            elapsed = time.perf_counter() - t0
            pct = (file_idx + 1) / n_files * 100
            rate = (file_idx + 1) / elapsed
            eta  = (n_files - file_idx - 1) / rate if rate > 0 else 0
            print(
                f"\r  [{file_idx + 1:>{len(str(n_files))}}/{n_files}]"
                f"  {pct:5.1f}%  {elapsed:6.1f}s elapsed  ETA {eta:5.1f}s"
                f"  rows written: {row:,}",
                end="",
                flush=True,
            )

        print()  # newline after progress

    elapsed = time.perf_counter() - t0
    size_mb = out_path.stat().st_size / 1024 / 1024
    print(f"\nDone in {elapsed:.1f}s  —  {out_path}  ({size_mb:.0f} MB)")
    print(f"Dataset shape : images {(n_slices, size, size)}  "
          f"chunks (1, {size}, {size})  =  {chunk_mb:.2f} MB/chunk")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Consolidate fastMRI .h5 files into one training archive.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "src_dir",
        type=Path,
        help="Directory containing per-patient fastMRI .h5 files",
    )
    parser.add_argument(
        "out_path",
        type=Path,
        help="Output HDF5 file path (e.g. knee_512.h5)",
    )
    parser.add_argument(
        "--size",
        type=int,
        choices=[256, 512, 1024],
        default=512,
        help="Target image size (sets both spatial resolution and chunk size). "
             "256 → 0.25 MB/chunk, 512 → 1 MB/chunk, 1024 → 4 MB/chunk. "
             "Default: 512",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        metavar="N",
        help="Process only the first N source files (useful for quick tests)",
    )
    parser.add_argument(
        "--split",
        default="train",
        help="fastMRI split label stored as metadata (default: train)",
    )

    args = parser.parse_args()

    if not args.src_dir.is_dir():
        sys.exit(f"src_dir does not exist or is not a directory: {args.src_dir}")
    if args.size != 256 and not _PIL_AVAILABLE:
        print(
            "Warning: Pillow not installed — resizing will use centre-crop/pad "
            "fallback instead of LANCZOS. Install Pillow for better quality.",
            file=sys.stderr,
        )

    prepare(args.src_dir, args.out_path, args.size, args.max_files, args.split)


if __name__ == "__main__":
    main()
