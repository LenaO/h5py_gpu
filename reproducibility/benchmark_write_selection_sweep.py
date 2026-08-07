#!/usr/bin/env python3
"""
benchmark_write_selection_sweep.py - Selection-size sweep for writes: the
write-side mirror of benchmark_selection_sweep.py.

Three methods are compared at each coverage fraction:

    naive_full      Read the entire dataset to a host array, patch the
                    selected sub-region in place, write the ENTIRE array
                    back. The "I don't know about partial writes" pattern
                    -- touches the whole dataset regardless of how much of
                    it actually changed. Should cost roughly the SAME
                    regardless of coverage.

    naive_partial   h5_ds[sel] = cp.asnumpy(src) -- h5py's own hyperslab
                    write, so storage I/O already scales with the
                    selection, but the source is a plain pageable host
                    array (no pinning, no overlap).

    ours            gpu_ds[sel] = src -- plain assignment syntax.
                    GPUDataset.__setitem__ dispatches to
                    write_selection_chunked (double-buffered D2H) for
                    HDF5-chunked 2-D/3-D datasets. For CONTIGUOUS datasets
                    there is no partial-selection double-buffered write in
                    this module -- write_double_buffered() only writes the
                    whole dataset -- so __setitem__ falls back to the same
                    plain h5py hyperslab write as naive_partial. This is a
                    real, current limitation, not a benchmark artifact:
                    expect "ours" to equal "naive_partial" (roughly 1.0x)
                    on the two contiguous layouts, and to behave like the
                    read-side sweep only on the two chunked layouts.

Covers all four combinations of {2-D, 3-D} x {contiguous, HDF5-chunked}.

Usage
-----
    python benchmark_write_selection_sweep.py bench_2d_chunked.h5 --dataset data

    python benchmark_write_selection_sweep.py --all-layouts --csv write_sweep.csv

Dependencies
------------
    h5py, cupy, numpy
"""

import argparse
import csv
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import h5py
import numpy as np

try:
    import cupy as cp
    _CUPY_AVAILABLE = True
except ImportError:
    _CUPY_AVAILABLE = False

REPEATS = 5
WARMUP = 2
COVERAGES = [10, 25, 50, 75, 100]


# ---------------------------------------------------------------------------
# Timing harness (same convention as benchmark_selection_sweep.py)
# ---------------------------------------------------------------------------

def _bench(fn, warmup=WARMUP, repeats=REPEATS):
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    a = np.array(times)
    return {"mean": a.mean(), "std": a.std(), "times": times}


def _bw_gbs(nbytes, seconds):
    return nbytes / seconds / 1e9


# ---------------------------------------------------------------------------
# Selection helpers (identical to benchmark_selection_sweep.py, so coverage
# fractions land on the same regions and results are directly comparable)
# ---------------------------------------------------------------------------

def _select_2d(rows, cols, coverage_pct):
    if coverage_pct >= 100:
        return 0, rows, 0, cols
    frac = coverage_pct / 100.0
    side_r = max(1, int(rows * frac ** 0.5))
    side_c = max(1, int(cols * frac ** 0.5))
    r0 = min(rows // 8, rows - side_r)
    c0 = min(cols // 8, cols - side_c)
    return r0, r0 + side_r, c0, c0 + side_c


def _select_3d(n, coverage_pct):
    if coverage_pct >= 100:
        return 0, n
    n_sel = max(1, int(n * coverage_pct / 100.0))
    s0 = (n - n_sel) // 2
    return s0, s0 + n_sel


# ---------------------------------------------------------------------------
# The three write methods
# ---------------------------------------------------------------------------

def _write_naive_full(h5_ds, sel, src_gpu):
    """Read-patch-write-back the whole dataset -- deliberately wasteful,
    the write-side analogue of naive-full's 'just touch everything'.
    Source is GPU-resident, like the other two methods: the D2H transfer
    is part of what's timed here, not hidden outside the benchmarked call."""
    src_np = cp.asnumpy(src_gpu)
    whole = h5_ds[:]
    whole[sel] = src_np
    h5_ds[:] = whole


def _write_naive_partial(h5_ds, sel, src_gpu):
    """h5py's own hyperslab write. Source is GPU-resident; the D2H transfer
    (plain, pageable, no pinning) is timed as part of this call, mirroring
    how the read-side sweep's naive-partial times its own H2D transfer."""
    src_np = cp.asnumpy(src_gpu)
    h5_ds[sel] = src_np


def _write_ours(gpu_ds, sel, src_gpu):
    """GPUDataset.__setitem__ -- double-buffered for chunked datasets,
    falls back to the naive path for contiguous ones (see module docstring)."""
    gpu_ds[sel] = src_gpu
    cp.cuda.Device().synchronize()


# ---------------------------------------------------------------------------
# Correctness check
# ---------------------------------------------------------------------------

def _check_correctness(h5_ds, gpu_ds, sel, src_gpu):
    src_np = cp.asnumpy(src_gpu)
    _write_naive_partial(h5_ds, sel, src_gpu)
    ok_partial = np.array_equal(np.asarray(h5_ds[sel]), src_np)
    _write_ours(gpu_ds, sel, src_gpu)
    ok_ours = np.array_equal(np.asarray(h5_ds[sel]), src_np)
    return ok_partial, ok_ours


# ---------------------------------------------------------------------------
# Sweep for one dataset
# ---------------------------------------------------------------------------

def run_sweep(h5_path, dataset_name, layout_label=None,
             coverages=COVERAGES, repeats=REPEATS, warmup=WARMUP,
             csv_rows=None):
    from h5py.gpu import GPUDataset

    f = h5py.File(h5_path, "r+")
    ds = f[dataset_name]
    shape, dtype, chunks = ds.shape, ds.dtype, ds.chunks
    ndim = ds.ndim
    chunked = chunks is not None
    label = layout_label or f"{ndim}d_{'chunked' if chunked else 'contiguous'}"

    itemsize = dtype.itemsize
    total_bytes = int(np.prod(shape)) * itemsize

    print(f"\n{'='*78}")
    print(f"  Layout      : {label}")
    print(f"  Dataset     : {dataset_name}  shape={shape}  dtype={dtype}  "
          f"chunks={chunks}")
    print(f"  Total size  : {total_bytes/1e6:.1f} MB")
    print(f"{'='*78}")

    gpu_ds = GPUDataset(ds)

    print(f"\n  {'cov%':>5s}  {'method':<14s}  {'time (ms)':>14s}  "
          f"{'BW (GB/s)':>10s}  {'speedup':>8s}")
    print(f"  {'-'*70}")

    rng = np.random.default_rng(0)

    for cov in coverages:
        if ndim == 2:
            r0, r1, c0, c1 = _select_2d(shape[0], shape[1], cov)
            sel = (slice(r0, r1), slice(c0, c1))
        elif ndim == 3:
            s0, s1 = _select_3d(shape[0], cov)
            sel = slice(s0, s1)
        else:
            f.close()
            sys.exit(f"Unsupported ndim={ndim}; only 2-D and 3-D are covered")

        sel_shape = tuple(
            sl.stop - sl.start
            for sl in (sel if isinstance(sel, tuple) else (sel,))
        ) + shape[len(sel if isinstance(sel, tuple) else (sel,)):]
        sel_elems = int(np.prod(sel_shape))
        sel_bytes = sel_elems * itemsize

        src_np = rng.standard_normal(sel_shape).astype(dtype)
        src_gpu = cp.asarray(src_np)

        ok_partial, ok_ours = _check_correctness(ds, gpu_ds, sel, src_gpu)
        if not (ok_partial and ok_ours):
            print(f"  {cov:>5d}  CORRECTNESS FAILURE: naive_partial_ok="
                  f"{ok_partial} ours_ok={ok_ours} -- skipping timing")
            continue

        results = {}
        results["naive_full"] = _bench(
            lambda: _write_naive_full(ds, sel, src_gpu), warmup, repeats)
        results["naive_partial"] = _bench(
            lambda: _write_naive_partial(ds, sel, src_gpu), warmup, repeats)
        results["ours"] = _bench(
            lambda: _write_ours(gpu_ds, sel, src_gpu), warmup, repeats)

        base_mean = results["naive_full"]["mean"]
        for method in ("naive_full", "naive_partial", "ours"):
            st = results[method]
            bw = _bw_gbs(sel_bytes, st["mean"])
            speedup = base_mean / st["mean"]
            print(f"  {cov:>5d}  {method:<14s}  "
                  f"{st['mean']*1e3:8.2f} +/-{st['std']*1e3:5.2f}  "
                  f"{bw:10.3f}  {speedup:7.2f}x")

            if csv_rows is not None:
                for i, t in enumerate(st["times"]):
                    csv_rows.append({
                        "layout": label, "coverage_pct": cov, "method": method,
                        "repeat": i, "time_s": t,
                        "bw_gbs": _bw_gbs(sel_bytes, t),
                        "selection_bytes": sel_bytes,
                        "total_bytes": total_bytes,
                        "ndim": ndim, "chunked": chunked,
                    })
        print()

    f.close()


# ---------------------------------------------------------------------------
# --all-layouts convenience mode
# ---------------------------------------------------------------------------

def _generate_all_layouts(tmpdir: Path, size_2d: int, chunk_2d: int,
                          n_3d: int, size_3d: int) -> list[tuple[Path, str, str]]:
    repo_root = Path(__file__).resolve().parent
    gen_script = repo_root / "make_benchmark_data.py"

    combos = [
        ("2d_chunked",    "data",   tmpdir / "wsweep_2d_chunked.h5",
         ["--kind", "2d_chunked", "--rows", str(size_2d), "--cols", str(size_2d),
          "--chunk-rows", str(chunk_2d), "--chunk-cols", str(chunk_2d)]),
        ("2d_contiguous", "data",   tmpdir / "wsweep_2d_contiguous.h5",
         ["--kind", "2d_contiguous", "--rows", str(size_2d), "--cols", str(size_2d)]),
        ("3d_chunked",    "images", tmpdir / "wsweep_3d_chunked.h5",
         ["--kind", "3d_chunked", "--n", str(n_3d), "--size", str(size_3d)]),
        ("3d_contiguous", "images", tmpdir / "wsweep_3d_contiguous.h5",
         ["--kind", "3d_contiguous", "--n", str(n_3d), "--size", str(size_3d)]),
    ]

    out = []
    for label, dset, path, extra_args in combos:
        subprocess.run(
            [sys.executable, str(gen_script), str(path), "--force"] + extra_args,
            check=True,
        )
        out.append((path, dset, label))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Write-side selection-size sweep: naive-full vs. "
                    "naive-partial vs. GPU-aware h5py, across coverage "
                    "fractions.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("h5_file", type=Path, nargs="?", default=None)
    parser.add_argument("--dataset", type=str, default=None, metavar="NAME")
    parser.add_argument("--coverages", type=str, default="10,25,50,75,100",
                        metavar="P1,P2,...")
    parser.add_argument("--repeats", type=int, default=REPEATS, metavar="R")
    parser.add_argument("--warmup", type=int, default=WARMUP, metavar="W")
    parser.add_argument("--csv", type=Path, default=None, metavar="PATH")

    parser.add_argument("--all-layouts", action="store_true")
    parser.add_argument("--size-2d", type=int, default=8192, metavar="N")
    parser.add_argument("--chunk-2d", type=int, default=256, metavar="C")
    parser.add_argument("--n-3d", type=int, default=512, metavar="N")
    parser.add_argument("--size-3d", type=int, default=512, metavar="S")
    parser.add_argument("--keep-data", action="store_true")

    args = parser.parse_args()

    if not _CUPY_AVAILABLE:
        sys.exit("CuPy is required (GPUDataset uses it internally). "
                 "Install with: pip install cupy-cuda12x")

    coverages = [int(x) for x in args.coverages.split(",")]
    csv_rows = [] if args.csv else None

    if args.all_layouts:
        tmpdir = Path(tempfile.mkdtemp(prefix="h5py_write_sweep_"))
        try:
            combos = _generate_all_layouts(
                tmpdir, args.size_2d, args.chunk_2d, args.n_3d, args.size_3d)
            for path, dset, label in combos:
                run_sweep(path, dset, layout_label=label,
                         coverages=coverages, repeats=args.repeats,
                         warmup=args.warmup, csv_rows=csv_rows)
        finally:
            if args.keep_data:
                print(f"\nGenerated data kept in: {tmpdir}")
            else:
                shutil.rmtree(tmpdir, ignore_errors=True)
    else:
        if args.h5_file is None or args.dataset is None:
            sys.exit("Provide h5_file and --dataset, or use --all-layouts")
        run_sweep(args.h5_file, args.dataset,
                 coverages=coverages, repeats=args.repeats,
                 warmup=args.warmup, csv_rows=csv_rows)

    if args.csv and csv_rows:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with open(args.csv, "w", newline="") as fout:
            writer = csv.DictWriter(fout, fieldnames=csv_rows[0].keys())
            writer.writeheader()
            writer.writerows(csv_rows)
        print(f"\nRaw results written to: {args.csv}")


if __name__ == "__main__":
    main()
