#!/usr/bin/env python3
"""
benchmark_batch_overlap.py – Compare the two 3-D whole-slice-batch read
strategies discussed for GPUDataset.read_selection_chunked's fast path
(chunks=(1, H, W), full H x W selected per index along axis 0).

Two ways to read a sequence of batches:

    sync              One call per batch (``gpu_ds[start:stop]``): a single
                       pinned buffer, one HDF5 read + one H2D transfer,
                       synchronous return.  No overlap across batches.

    async_pipelined   ``read_batch_async(start, stop, out=..., buf_idx=...)``
                       called back-to-back with alternating buf_idx (0/1) and
                       alternating GPU output buffers, synchronizing one
                       batch behind -- the double-buffering pattern from the
                       method's own docstring.  The CPU read of batch i+1
                       overlaps the H2D transfer of batch i.

This isolates exactly the trade-off described in the paper
(Section "Chunk-aware reads" / Table 1, case #2): collapsing a batch into
one call removes per-chunk overhead, but forgoes overlap *within* the call;
overlap *across* batches is available if the caller opts into it via the
async variant. This benchmark measures whether that opt-in is worth it for
your storage/transfer balance.

Because overlap only helps once the pipeline has more than one batch, and
only shows up in the *sequence* total (not per-batch call time), this script
times the whole batch sequence, not individual calls.

The input file must be produced by prepare_fastmri.py (or match its layout):
    images  float32  (N, H, W)  chunked (1, H, W)

Usage
-----
    python benchmark_batch_overlap.py knee_512.h5 --batch-size 16 --batches 100

Dependencies
------------
    h5py, cupy, numpy
"""

import argparse
import csv
import sys
import time
from pathlib import Path

import h5py
import numpy as np

try:
    import cupy as cp
    _CUPY_AVAILABLE = True
except ImportError:
    _CUPY_AVAILABLE = False


# ---------------------------------------------------------------------------
# Helpers (batch list matches benchmark_ml_loading.py's convention: every
# batch has the same size, so buffers can be preallocated once)
# ---------------------------------------------------------------------------

def _make_batches(n_slices: int, batch_size: int, seed: int = 0) -> list[tuple[int, int]]:
    rng = np.random.default_rng(seed)
    order = rng.permutation(n_slices)
    batches = []
    for i in range(0, n_slices - batch_size + 1, batch_size):
        idx = np.sort(order[i : i + batch_size])
        batches.append((int(idx[0]), int(idx[0]) + batch_size))
    return batches


def _batch_bytes(n_batches: int, batch_size: int, h: int, w: int) -> float:
    return float(n_batches * batch_size * h * w * 4)


def _stats(times: list[float]) -> dict:
    a = np.array(times)
    return {"mean": a.mean(), "std": a.std(), "min": a.min(), "max": a.max()}


# ---------------------------------------------------------------------------
# Method 1: sync -- one call per batch, no cross-batch overlap
# ---------------------------------------------------------------------------

def _run_sync(gpu_ds, batches: list[tuple[int, int]]) -> float:
    """Return total wall time for the whole batch sequence."""
    t0 = time.perf_counter()
    for start, stop in batches:
        arr = gpu_ds[start:stop]      # read_selection_chunked fast path
    cp.cuda.Device().synchronize()
    return time.perf_counter() - t0


# ---------------------------------------------------------------------------
# Method 2: async_pipelined -- alternating buf_idx, overlap read_i+1 with
# transfer_i (the fixed double-buffering pattern from read_batch_async)
# ---------------------------------------------------------------------------

def _run_async_pipelined(gpu_ds, batches: list[tuple[int, int]],
                         batch_size: int, H: int, W: int, dtype) -> float:
    """Return total wall time for the whole batch sequence."""
    if not batches:
        return 0.0

    t0 = time.perf_counter()

    buf_a = cp.empty((batch_size, H, W), dtype=dtype)
    buf_b = cp.empty((batch_size, H, W), dtype=dtype)

    start0, stop0 = batches[0]
    arr_a, ev_a = gpu_ds.read_batch_async(start0, stop0, out=buf_a, buf_idx=0)

    for i in range(1, len(batches)):
        start_i, stop_i = batches[i]
        buf_nxt = buf_b if i % 2 == 1 else buf_a
        arr_b, ev_b = gpu_ds.read_batch_async(
            start_i, stop_i, out=buf_nxt, buf_idx=i % 2)
        ev_a.synchronize()             # batch i-1 is now safe to have used
        arr_a, ev_a = arr_b, ev_b

    ev_a.synchronize()                 # final batch
    return time.perf_counter() - t0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare sync vs. double-buffered async 3-D whole-slice batch reads.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("h5_file", type=Path,
                        help="Consolidated HDF5 file from prepare_fastmri.py")
    parser.add_argument("--batch-size", type=int, default=16, metavar="B",
                        help="Number of slices per batch (default: 16)")
    parser.add_argument("--batches",    type=int, default=50,  metavar="N",
                        help="Number of batches per repeat (default: 50)")
    parser.add_argument("--repeats",    type=int, default=5,   metavar="R",
                        help="Measurement repeats (default: 5)")
    parser.add_argument("--warmup",     type=int, default=2,   metavar="W",
                        help="Warm-up repeats discarded before measurement (default: 2)")
    parser.add_argument("--seed",       type=int, default=42)
    parser.add_argument("--csv",        type=Path, default=None, metavar="PATH")

    args = parser.parse_args()

    if not _CUPY_AVAILABLE:
        sys.exit("CuPy is required (GPUDataset uses it internally). "
                 "Install with: pip install cupy-cuda12x")
    if not args.h5_file.exists():
        sys.exit(f"File not found: {args.h5_file}")

    with h5py.File(args.h5_file, "r") as meta_f:
        n_slices, H, W = meta_f["images"].shape
        chunk_shape    = meta_f["images"].chunks
        dtype          = meta_f["images"].dtype

    if chunk_shape is None or chunk_shape[0] != 1:
        sys.exit(f"Expected chunks=(1, H, W), got {chunk_shape}. "
                 "This benchmark targets the whole-slice-batch fast path.")

    all_batches = _make_batches(n_slices, args.batch_size, seed=args.seed)
    batches: list[tuple[int, int]] = []
    while len(batches) < args.batches:
        batches.extend(all_batches)
    batches = batches[: args.batches]
    n_batches = len(batches)

    total_bytes = _batch_bytes(n_batches, args.batch_size, H, W)

    print(f"\n{'='*65}")
    print(f"  File        : {args.h5_file}")
    print(f"  Dataset     : images {(n_slices, H, W)}  {dtype}")
    print(f"  Chunk shape : {chunk_shape}")
    print(f"  Batch size  : {args.batch_size} slices "
          f"({args.batch_size * H * W * 4 / 1e6:.1f} MB/batch)")
    print(f"  Batches     : {n_batches}   Repeats: {args.repeats}   "
          f"Warmup: {args.warmup}")
    print(f"{'='*65}\n")

    f_h5 = h5py.File(args.h5_file, "r")
    from h5py.gpu import GPUDataset
    gpu_ds = GPUDataset(f_h5["images"])

    methods = ["sync", "async_pipelined"]
    all_totals: dict[str, list[float]] = {m: [] for m in methods}

    total_repeats = args.warmup + args.repeats
    for rep in range(total_repeats):
        tag = "warmup" if rep < args.warmup else f"rep {rep - args.warmup + 1}/{args.repeats}"
        print(f"  Running {tag} ...", end=" ", flush=True)

        t_sync  = _run_sync(gpu_ds, batches)
        t_async = _run_async_pipelined(gpu_ds, batches, args.batch_size, H, W, dtype)

        if rep < args.warmup:
            print("discarded")
            continue

        all_totals["sync"].append(t_sync)
        all_totals["async_pipelined"].append(t_async)
        print(f"done  (sync {t_sync*1e3:.1f} ms, async {t_async*1e3:.1f} ms total)")

    f_h5.close()

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------
    print(f"\n{'-'*65}")
    print(f"  {'Method':<18s}  {'ms/batch':>12s}  {'BW':>9s}  {'Throughput':>10s}  {'Speedup':>8s}")
    print(f"{'-'*65}")

    stats = {m: _stats(all_totals[m]) for m in methods}
    base_mean = stats["sync"]["mean"]

    for m in methods:
        ms_per_batch = stats[m]["mean"] / n_batches * 1e3
        std_per_batch = stats[m]["std"] / n_batches * 1e3
        bw_gbs = (total_bytes / n_batches) / (stats[m]["mean"] / n_batches) / 1e9
        img_s  = n_batches * args.batch_size / stats[m]["mean"]
        speedup = f"{base_mean / stats[m]['mean']:.2f}x" if m != "sync" else "--"
        print(
            f"  {m:<18s}"
            f"  {ms_per_batch:8.2f} +/- {std_per_batch:5.2f} ms"
            f"  {bw_gbs:6.2f} GB/s"
            f"  {img_s:8.0f} img/s"
            f"  {speedup:>8s}"
        )
    print(f"{'-'*65}\n")

    if args.csv:
        rows = []
        for m in methods:
            for i, t in enumerate(all_totals[m]):
                rows.append({
                    "method": m, "repeat": i, "total_time_s": t,
                    "n_batches": n_batches, "batch_size": args.batch_size,
                    "H": H, "W": W,
                })
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with open(args.csv, "w", newline="") as fout:
            writer = csv.DictWriter(fout, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        print(f"  Raw results written to: {args.csv}\n")


if __name__ == "__main__":
    main()
