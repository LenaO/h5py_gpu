#!/usr/bin/env python3
"""
benchmark_use_cases.py – Empirically compare the four access patterns of
the paper's "Access Patterns and Their Memory Footprints" section:

    1   full dataset          read_chunks_to_gpu() / read_double_buffered()
    2   index-addressed part  GPUDataset[sel]
    3   reduction, discard    reduce_chunks() / reduce_double_buffered()
    4a  reduction, keep data  read_double_buffered() then cp.sum() (non-fused)
    4b  reduction, keep data  read_double_buffered(reduce_fn=...)  (fused)

For each case this script reports:

    time      wall-clock time (mean +/- std over repeats)
    resident  GPU bytes resident *after* the call returns (measured via
              CuPy's default memory pool, reset before each case) -- the
              direct empirical counterpart of the table's third column

It also cross-checks that all reduction variants (#3, #4a, #4b) agree with
a plain NumPy sum on the same data, since a fused reduction that returns the
wrong number is worse than no fused reduction at all.

Usage
-----
    python benchmark_use_cases.py knee_512.h5 --dataset images
    python benchmark_use_cases.py knee_512.h5 --dataset images --repeats 10

Dependencies
------------
    h5py, cupy, numpy
"""

import argparse
import gc
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


def _mempool_used() -> int:
    return cp.get_default_memory_pool().used_bytes()


def _reset_mempool() -> None:
    gc.collect()
    cp.get_default_memory_pool().free_all_blocks()


def _time_and_measure(fn, warmup: int, repeats: int):
    """Run *fn* (no args, returns (result, resident_bytes)); return
    (times: list[float], last_result, last_resident_bytes)."""
    result = None
    resident = 0
    for _ in range(warmup):
        result, resident = fn()
    times = []
    for _ in range(repeats):
        _reset_mempool()
        t0 = time.perf_counter()
        result, resident = fn()
        cp.cuda.Device().synchronize()
        times.append(time.perf_counter() - t0)
    return times, result, resident


def _stats(times: list[float]) -> dict:
    a = np.array(times)
    return {"mean": a.mean(), "std": a.std()}


def _to_scalar(x) -> float:
    return float(cp.asnumpy(x)) if isinstance(x, cp.ndarray) else float(x)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare the four access patterns of the paper's "
                    "\"Access Patterns and Their Memory Footprints\" section.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("h5_file", type=Path, help="HDF5 file to read from")
    parser.add_argument("--dataset", type=str, default="images", metavar="NAME",
                        help="Dataset path inside the file (default: images)")
    parser.add_argument("--repeats", type=int, default=5, metavar="R")
    parser.add_argument("--warmup",  type=int, default=1, metavar="W")

    args = parser.parse_args()

    if not _CUPY_AVAILABLE:
        sys.exit("CuPy is required (GPUDataset uses it internally). "
                 "Install with: pip install cupy-cuda12x")
    if not args.h5_file.exists():
        sys.exit(f"File not found: {args.h5_file}")

    from h5py.gpu import GPUDataset

    f_h5 = h5py.File(args.h5_file, "r")
    ds = f_h5[args.dataset]
    shape, dtype, chunks = ds.shape, ds.dtype, ds.chunks
    nbytes = int(np.prod(shape)) * dtype.itemsize

    print(f"\n{'='*70}")
    print(f"  File    : {args.h5_file}")
    print(f"  Dataset : {args.dataset}  shape={shape}  dtype={dtype}  "
          f"chunks={chunks}")
    print(f"  Size    : {nbytes / 1e6:.1f} MB")
    print(f"  Repeats : {args.repeats}   Warmup: {args.warmup}")
    print(f"{'='*70}\n")

    # Ground truth (CPU), for cross-checking every reduction variant
    print("  Computing ground-truth sum on CPU ...", end=" ", flush=True)
    truth = float(np.asarray(ds).astype(np.float64).sum())
    print(f"done  (sum = {truth:.6g})\n")

    gpu_ds  = GPUDataset(ds)
    chunked = chunks is not None and ds.ndim in (2, 3)

    is_correct = {}

    # ------------------------------------------------------------------
    # Case 1: full dataset
    # ------------------------------------------------------------------
    def _case1():
        arr = (gpu_ds.read_chunks_to_gpu() if chunked
               else gpu_ds.read_double_buffered())
        return arr, _mempool_used()

    times1, arr1, resident1 = _time_and_measure(_case1, args.warmup, args.repeats)

    # ------------------------------------------------------------------
    # Case 2: index-addressed part (middle ~25% along axis 0)
    # ------------------------------------------------------------------
    n0 = shape[0]
    sel_lo, sel_hi = n0 // 2 - n0 // 8, n0 // 2 + n0 // 8
    sel_bytes = (sel_hi - sel_lo) * int(np.prod(shape[1:])) * dtype.itemsize

    def _case2():
        arr = gpu_ds[sel_lo:sel_hi]
        return arr, _mempool_used()

    times2, arr2, resident2 = _time_and_measure(_case2, args.warmup, args.repeats)

    # ------------------------------------------------------------------
    # Case 3: reduction, discard data
    # ------------------------------------------------------------------
    def _case3():
        total = (gpu_ds.reduce_chunks(cp.sum) if chunked
                 else gpu_ds.reduce_double_buffered(cp.sum))
        return total, _mempool_used()

    times3, total3, resident3 = _time_and_measure(_case3, args.warmup, args.repeats)
    is_correct["3  (discard)"] = np.isclose(_to_scalar(total3), truth, rtol=1e-3)

    # ------------------------------------------------------------------
    # Case 4a: reduction, keep data (non-fused: read, then a separate
    # GPU-only reduce() pass)
    # ------------------------------------------------------------------
    def _case4a():
        arr = gpu_ds.read_double_buffered()
        total = cp.sum(arr)
        resident = _mempool_used()
        del arr
        return total, resident

    times4a, total4a, resident4a = _time_and_measure(_case4a, args.warmup, args.repeats)
    is_correct["4a (two-pass)"] = np.isclose(_to_scalar(total4a), truth, rtol=1e-3)

    # ------------------------------------------------------------------
    # Case 4b: reduction, keep data (fused: reduce_fn rides the same
    # stream as the load)
    # ------------------------------------------------------------------
    def _case4b():
        arr, total = gpu_ds.read_double_buffered(reduce_fn=cp.sum)
        resident = _mempool_used()
        del arr
        return total, resident

    times4b, total4b, resident4b = _time_and_measure(_case4b, args.warmup, args.repeats)
    is_correct["4b (fused)"] = np.isclose(_to_scalar(total4b), truth, rtol=1e-3)

    f_h5.close()

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------
    rows = [
        ("1  full dataset",        times1,  resident1),
        ("2  index-addressed part", times2, resident2),
        ("3  reduce, discard",     times3,  resident3),
        ("4a reduce+keep (2-pass)", times4a, resident4a),
        ("4b reduce+keep (fused)",  times4b, resident4b),
    ]

    print(f"{'-'*70}")
    print(f"  {'Case':<26s}  {'time (ms)':>16s}  {'resident (MB)':>14s}")
    print(f"{'-'*70}")
    for label, times, resident in rows:
        st = _stats(times)
        print(f"  {label:<26s}  {st['mean']*1e3:8.2f} +/- {st['std']*1e3:5.2f}"
              f"  {resident/1e6:14.2f}")
    print(f"{'-'*70}")

    st4a = _stats(times4a)
    st4b = _stats(times4b)
    delta = (st4b["mean"] - st4a["mean"]) / st4a["mean"] * 100
    print(f"\n  4b vs 4a (fused vs two-pass):  {delta:+.1f}%  "
          f"({'fused is faster or equal, as expected' if delta <= 5 else 'unexpected: fused is slower'})")

    print(f"\n  Reduction correctness (vs. NumPy ground truth):")
    for label, ok in is_correct.items():
        print(f"    case {label:<15s}  {'OK' if ok else 'MISMATCH'}")
    print()


if __name__ == "__main__":
    main()
