#!/usr/bin/env python3
"""
benchmark_selection_sweep.py - Selection-size sweep: does reading part of a
dataset cost proportionally less, all the way from disk to GPU?

This is the direct, measurable payoff of index-based access (Section
"Index-Based, Multi-Dimensional Access" / Design Rationale in the paper):
since h5py, NumPy, and CuPy share the same slicing model, a selection that
touches only part of a dataset should only *cost* for that part -- not for
the whole file.

Three methods are compared at each coverage fraction:

    naive_full      h5py_ds[:] -> numpy -> cp.asarray() -> slice on GPU.
                    The "just load it all" pattern. Should cost the SAME
                    regardless of coverage -- it always reads and transfers
                    the entire file, then throws most of it away.

    naive_partial   h5py_ds[sel] -> cp.asarray().
                    Uses h5py's own hyperslab selection, so storage I/O
                    already scales with the selection -- but the transfer
                    to GPU is a single pageable copy (no pinned memory, no
                    overlap). Isolates how much of the benefit comes from
                    HDF5-level selection alone, without the rest of the
                    GPU-transfer machinery.

    ours            GPUDataset[sel] -- plain indexing syntax, for both
                    layouts. __getitem__ dispatches to read_selection_chunked
                    for chunked datasets and read_double_buffered for
                    contiguous ones (see gpu.py's __getitem__ docstring):
                    the same slice on the caller's side always gets the
                    double-buffered treatment appropriate to the layout
                    underneath, without the caller needing to know or care
                    which one it is.

Covers all four combinations of {2-D, 3-D} x {contiguous, HDF5-chunked}.

The key visual signature: naive_full's time (and hence its "useful
bandwidth" = selection bytes / time) should be roughly FLAT across coverage
fractions, while naive_partial and ours should both DROP with coverage --
and ours should track naive_partial closely or beat it, since it adds
pinned buffers and double-buffered overlap on top of the same
storage-level selection scoping.

Usage
-----
    # Single dataset (auto-detects 2-D/3-D and chunked/contiguous from the
    # file itself)
    python benchmark_selection_sweep.py bench_2d_chunked.h5 --dataset data

    # Generate all four layout combinations and sweep each in one run
    python benchmark_selection_sweep.py --all-layouts --csv sweep_results.csv

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
# Timing harness (same convention as bench_sel_opt.py / benchmark_use_cases.py)
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
# Selection helpers
# ---------------------------------------------------------------------------

def _select_2d(rows, cols, coverage_pct):
    """Return (r0, r1, c0, c1) covering roughly coverage_pct% of the array,
    offset away from the origin so the selection is non-trivial (exercises
    edge/interior chunk handling and column alignment, not just the trivial
    from-(0,0) case)."""
    if coverage_pct >= 100:
        return 0, rows, 0, cols
    frac = coverage_pct / 100.0
    side_r = max(1, int(rows * frac ** 0.5))
    side_c = max(1, int(cols * frac ** 0.5))
    r0 = min(rows // 8, rows - side_r)
    c0 = min(cols // 8, cols - side_c)
    return r0, r0 + side_r, c0, c0 + side_c


def _select_3d(n, coverage_pct):
    """Return (s0, s1) along axis 0 covering roughly coverage_pct% of the
    slices -- the realistic 3-D access pattern (a sub-range of a stack of
    images/frames/slices), matching chunks=(1, H, W) layouts and how the
    other benchmarks in this repo already use 3-D data."""
    if coverage_pct >= 100:
        return 0, n
    n_sel = max(1, int(n * coverage_pct / 100.0))
    s0 = (n - n_sel) // 2
    return s0, s0 + n_sel


# ---------------------------------------------------------------------------
# The three read methods
# ---------------------------------------------------------------------------

def _read_naive_full(h5_ds, sel):
    """'Just load it all' -- the pattern this benchmark exists to discourage
    for partial access. Deliberately uses plain cp.asarray(), not any of
    h5py.gpu's pinned-memory machinery: this is what an uninstrumented user
    would actually write."""
    arr_np = h5_ds[:]
    arr_gpu = cp.asarray(arr_np)
    result = arr_gpu[sel]
    cp.cuda.Device().synchronize()
    return result


def _read_naive_partial(h5_ds, sel):
    """h5py's own hyperslab selection (storage I/O already scoped to sel),
    but a plain pageable-memory transfer to the GPU -- isolates the
    HDF5-level selection benefit from the GPU-transfer-layer benefit."""
    arr_np = h5_ds[sel]
    arr_gpu = cp.asarray(arr_np)
    cp.cuda.Device().synchronize()
    return arr_gpu


def _read_ours(gpu_ds, sel):
    """The optimized layer, via plain indexing syntax: GPUDataset.__getitem__
    dispatches to the double-buffered path appropriate for the dataset's
    layout (read_selection_chunked for chunked, read_double_buffered for
    contiguous) regardless of which one it is -- see the module docstring."""
    result = gpu_ds[sel]
    cp.cuda.Device().synchronize()
    return result


# ---------------------------------------------------------------------------
# Correctness check
# ---------------------------------------------------------------------------

def _check_correctness(h5_ds, gpu_ds, sel):
    truth = np.asarray(h5_ds[sel])
    got_partial = cp.asnumpy(_read_naive_partial(h5_ds, sel))
    got_ours = cp.asnumpy(_read_ours(gpu_ds, sel))
    ok_partial = np.array_equal(truth, got_partial)
    ok_ours = np.array_equal(truth, got_ours)
    return ok_partial, ok_ours


# ---------------------------------------------------------------------------
# Sweep for one dataset
# ---------------------------------------------------------------------------

def run_sweep(h5_path, dataset_name, layout_label=None,
             coverages=COVERAGES, repeats=REPEATS, warmup=WARMUP,
             csv_rows=None):
    from h5py.gpu import GPUDataset

    f = h5py.File(h5_path, "r")
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

        sel_elems = int(np.prod(
            [ (sl.stop - sl.start) for sl in (sel if isinstance(sel, tuple) else (sel,)) ]
            + list(shape[len(sel if isinstance(sel, tuple) else (sel,)):])
        ))
        sel_bytes = sel_elems * itemsize

        # Correctness check once per coverage point (cheap relative to the
        # timing loop, and this is exactly the kind of thing that has had
        # subtle bugs before in this codebase).
        ok_partial, ok_ours = _check_correctness(ds, gpu_ds, sel)
        if not (ok_partial and ok_ours):
            print(f"  {cov:>5d}  CORRECTNESS FAILURE: naive_partial_ok="
                  f"{ok_partial} ours_ok={ok_ours} -- skipping timing")
            continue

        results = {}
        results["naive_full"] = _bench(
            lambda: _read_naive_full(ds, sel), warmup, repeats)
        results["naive_partial"] = _bench(
            lambda: _read_naive_partial(ds, sel), warmup, repeats)
        results["ours"] = _bench(
            lambda: _read_ours(gpu_ds, sel), warmup, repeats)

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
    """Returns [(path, dataset_name, layout_label), ...] for all four
    combinations, generated via make_benchmark_data.py."""
    repo_root = Path(__file__).resolve().parent
    gen_script = repo_root / "make_benchmark_data.py"

    combos = [
        ("2d_chunked",    "data",   tmpdir / "sweep_2d_chunked.h5",
         ["--kind", "2d_chunked", "--rows", str(size_2d), "--cols", str(size_2d),
          "--chunk-rows", str(chunk_2d), "--chunk-cols", str(chunk_2d)]),
        ("2d_contiguous", "data",   tmpdir / "sweep_2d_contiguous.h5",
         ["--kind", "2d_contiguous", "--rows", str(size_2d), "--cols", str(size_2d)]),
        ("3d_chunked",    "images", tmpdir / "sweep_3d_chunked.h5",
         ["--kind", "3d_chunked", "--n", str(n_3d), "--size", str(size_3d)]),
        ("3d_contiguous", "images", tmpdir / "sweep_3d_contiguous.h5",
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
        description="Selection-size sweep: naive-full vs. naive-partial vs. "
                    "GPU-aware h5py, across coverage fractions.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("h5_file", type=Path, nargs="?", default=None,
                        help="HDF5 file to sweep (omit when using --all-layouts)")
    parser.add_argument("--dataset", type=str, default=None, metavar="NAME",
                        help="Dataset path inside the file (required unless "
                             "--all-layouts)")
    parser.add_argument("--coverages", type=str, default="10,25,50,75,100",
                        metavar="P1,P2,...",
                        help="Comma-separated coverage percentages "
                             "(default: 10,25,50,75,100)")
    parser.add_argument("--repeats", type=int, default=REPEATS, metavar="R")
    parser.add_argument("--warmup", type=int, default=WARMUP, metavar="W")
    parser.add_argument("--csv", type=Path, default=None, metavar="PATH",
                        help="Write per-repeat raw results to a CSV file")

    parser.add_argument("--all-layouts", action="store_true",
                        help="Generate all four layout combinations "
                             "(2-D/3-D x contiguous/chunked) in a temp "
                             "directory and sweep each in turn")
    parser.add_argument("--size-2d", type=int, default=8192, metavar="N",
                        help="--all-layouts: 2-D dataset is N x N (default: 8192, ~256MB)")
    parser.add_argument("--chunk-2d", type=int, default=256, metavar="C",
                        help="--all-layouts: 2-D chunk size is C x C (default: 256)")
    parser.add_argument("--n-3d", type=int, default=512, metavar="N",
                        help="--all-layouts: 3-D dataset has N slices (default: 512)")
    parser.add_argument("--size-3d", type=int, default=512, metavar="S",
                        help="--all-layouts: 3-D slices are S x S (default: 512, ~512MB total)")
    parser.add_argument("--keep-data", action="store_true",
                        help="--all-layouts: don't delete the generated .h5 "
                             "files afterwards")

    args = parser.parse_args()

    if not _CUPY_AVAILABLE:
        sys.exit("CuPy is required (GPUDataset uses it internally). "
                 "Install with: pip install cupy-cuda12x")

    coverages = [int(x) for x in args.coverages.split(",")]
    csv_rows = [] if args.csv else None

    if args.all_layouts:
        tmpdir = Path(tempfile.mkdtemp(prefix="h5py_sel_sweep_"))
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
