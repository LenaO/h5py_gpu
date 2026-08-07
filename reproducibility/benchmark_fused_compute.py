#!/usr/bin/env python3
"""
benchmark_fused_compute.py - Does fusing a transform/reduction into the I/O
pipeline actually hide its cost, compared to the "old" three-phase approach
(read data, copy to GPU, execute kernel)?

This tests the paper's central overlap claim (Section "Overlapping I/O,
Transfer, and Compute" / "Streaming Reductions"): a transform or reduction
enqueued on the same CUDA stream as each piece's H2D transfer should add no
wall-clock cost as long as its GPU time does not itself exceed the I/O time
for the next piece -- the double-buffering loop hides T_h2d + T_compute
behind T_io, not just T_h2d alone.

Two methods, compared across a sweep of per-piece COMPUTE COST:

    naive   The old approach, three separate serial phases:
              1. read the whole dataset:  h5py_ds[:] -> numpy
              2. copy the whole thing to the GPU:  cp.asarray()
              3. THEN run the transform/reduction as one big kernel call
                 over the entire resident array.
            Total time = T_io + T_h2d + T_compute, always -- compute is
            pure overhead added on top of a fully serial pipeline.

    fused   read_double_buffered(transform=fn) / reduce_double_buffered(fn),
            for BOTH chunked and contiguous layouts -- deliberately the
            same row-band method either way, not a different one per
            layout. The same per-row-band computation runs on the same
            stream as that band's H2D transfer, immediately afterward,
            while the CPU is already reading the next band. Total time
            should stay close to max(T_io, T_h2d + T_compute) per band --
            i.e. roughly FLAT as compute cost grows, until the per-band
            compute time exceeds the per-band I/O time, at which point
            fused starts growing too (compute stops being free once it
            becomes the bottleneck instead of storage I/O).

    Why row-band uniformly, not read_chunks_to_gpu/reduce_chunks for the
    chunked case: this benchmark's job is to isolate whether fusion hides
    compute, not how many chunks get batched into one piece -- that
    question already has its own dedicated comparison
    (benchmark_chunk_vs_rowband.py). Using one method for both layouts
    here keeps the two questions separate instead of conflating them. For
    a dataset chunked as (1, H, W), row-band height defaults to
    dataset.chunks[0] == 1, so a row-band IS one chunk there -- the
    per-piece granularity, and hence the result, is unchanged from the
    read_chunks_to_gpu/reduce_chunks version of this benchmark; only the
    method name is now uniform across layouts.

The compute cost itself is a synthetic, controllable knob -- N repeated
applications of a cheap elementwise op (sqrt(|x|+1)) per piece -- not a
specific real workload. This lets the sweep isolate exactly where the
crossover from "fully hidden" to "starts costing wall-clock time" happens,
which is the useful number to report, not any single point measurement.

Both a TRANSFORM case (keeps the data, applies the op in place) and a
REDUCE case (discards each piece after reducing it) are covered, matching
the two fused code paths the paper describes.

Usage
-----
    python benchmark_fused_compute.py --layout 3d_chunked \\
        --n-ops 0,1,2,4,8,16,32,64,128,256

    python benchmark_fused_compute.py --layout 2d_contiguous --csv fused.csv

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
DEFAULT_N_OPS = [0, 1, 2, 4, 8, 16, 32, 64, 128, 256]


# ---------------------------------------------------------------------------
# Synthetic, controllable per-element compute cost
# ---------------------------------------------------------------------------

def _make_transform_fn(n_ops: int):
    """N repeated applications of a cheap elementwise op, done as ONE kernel
    launch with an internal loop (via cp.ElementwiseKernel), not N separate
    launches. This matters: N separate cp.sqrt/cp.abs calls per chunk would
    multiply fixed per-launch overhead by the chunk count for the fused
    path specifically (256 chunks x N launches vs. naive's N launches over
    the whole array in one shot) -- a real effect, but an artifact of how a
    real expensive transform would be written (as one kernel doing more
    work, not hundreds of chained trivial launches), not something this
    benchmark is meant to measure. A single kernel with an internal loop
    isolates the actual question: does overlap hide compute *time*,
    independent of how many kernels launch it.

    Purely a compute-cost knob -- chunking never changes the result, since
    the op is local to each element (no cross-element or cross-chunk
    dependency)."""
    def _fn(x):
        return _COMPUTE_COST_KERNEL(x, n_ops)
    return _fn


if _CUPY_AVAILABLE:
    _COMPUTE_COST_KERNEL = cp.ElementwiseKernel(
        "T x, int32 n_ops", "T out",
        """
        T v = x;
        for (int i = 0; i < n_ops; i++) {
            v = sqrt(abs(v) + (T)1.0);
        }
        out = v;
        """,
        "h5py_gpu_bench_compute_cost_kernel",
    )


def _make_reduce_fn(n_ops: int):
    """Same elementwise cost knob, followed by a sum. Must be paired with
    combine_fn=cp.sum (NOT left to default to this function itself) --
    otherwise the module would try to re-apply the whole compute chain to
    the array of partial sums, which is wrong. See reduce_chunks'
    combine_fn docs for the same sum-then-combine pattern."""
    transform = _make_transform_fn(n_ops)
    def _fn(x):
        return cp.sum(transform(x))
    return _fn


# ---------------------------------------------------------------------------
# Timing harness (same convention as the other benchmark scripts here)
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
    return {"mean": a.mean(), "std": a.std()}


# ---------------------------------------------------------------------------
# The two methods, transform case
# ---------------------------------------------------------------------------

def _naive_transform(h5_ds, n_ops):
    arr_np = h5_ds[:]
    arr_gpu = cp.asarray(arr_np)
    fn = _make_transform_fn(n_ops)
    result = fn(arr_gpu)
    cp.cuda.Device().synchronize()
    return result


def _fused_transform(gpu_ds, chunked, n_ops):
    # Deliberately the same row-band method for both layouts -- see the
    # module docstring for why this benchmark no longer branches on
    # `chunked` the way it used to.
    fn = _make_transform_fn(n_ops)
    result = gpu_ds.read_double_buffered(transform=fn)
    cp.cuda.Device().synchronize()
    return result


# ---------------------------------------------------------------------------
# The two methods, reduce case
# ---------------------------------------------------------------------------

def _naive_reduce(h5_ds, n_ops):
    arr_np = h5_ds[:]
    arr_gpu = cp.asarray(arr_np)
    fn = _make_reduce_fn(n_ops)
    result = fn(arr_gpu)
    cp.cuda.Device().synchronize()
    return result


def _fused_reduce(gpu_ds, chunked, n_ops):
    # Same reasoning as _fused_transform: one method for both layouts.
    fn = _make_reduce_fn(n_ops)
    result = gpu_ds.reduce_double_buffered(fn, combine_fn=cp.sum)
    cp.cuda.Device().synchronize()
    return result


# ---------------------------------------------------------------------------
# Correctness checks
# ---------------------------------------------------------------------------

def _check_transform_correctness(h5_ds, gpu_ds, chunked, n_ops):
    truth = _make_transform_fn(n_ops)(cp.asarray(np.asarray(h5_ds[:])))
    got = _fused_transform(gpu_ds, chunked, n_ops)
    return cp.allclose(truth, got, rtol=1e-4).item()


def _check_reduce_correctness(h5_ds, gpu_ds, chunked, n_ops):
    truth = _make_reduce_fn(n_ops)(cp.asarray(np.asarray(h5_ds[:])))
    got = _fused_reduce(gpu_ds, chunked, n_ops)
    return cp.allclose(truth, got, rtol=1e-3).item()  # summation order differs -> looser tol


# ---------------------------------------------------------------------------
# Sweep for one dataset
# ---------------------------------------------------------------------------

def run_sweep(h5_path: Path, dataset_name: str, n_ops_list: list[int],
             repeats: int, warmup: int, csv_rows: list | None,
             layout_label: str | None = None) -> None:
    from h5py.gpu import GPUDataset

    f = h5py.File(h5_path, "r")
    ds = f[dataset_name]
    chunked = ds.chunks is not None
    label = layout_label or f"{ds.ndim}d_{'chunked' if chunked else 'contiguous'}"
    total_mb = int(np.prod(ds.shape)) * ds.dtype.itemsize / 1e6

    print(f"\n{'='*84}")
    print(f"  Layout      : {label}")
    print(f"  Dataset     : {dataset_name}  shape={ds.shape}  dtype={ds.dtype}  "
          f"chunks={ds.chunks}")
    print(f"  Total size  : {total_mb:.1f} MB")
    print(f"{'='*84}")

    gpu_ds = GPUDataset(ds)

    for case, naive_fn, fused_fn, check_fn in (
        ("transform", _naive_transform, _fused_transform, _check_transform_correctness),
        ("reduce",    _naive_reduce,    _fused_reduce,    _check_reduce_correctness),
    ):
        print(f"\n  -- {case} --")
        print(f"  {'n_ops':>6s}  {'naive (ms)':>14s}  {'fused (ms)':>14s}  "
              f"{'speedup':>8s}  {'correct':>8s}")
        print(f"  {'-'*60}")

        for n_ops in n_ops_list:
            ok = check_fn(ds, gpu_ds, chunked, n_ops)
            if not ok:
                print(f"  {n_ops:>6d}  CORRECTNESS FAILURE -- skipping timing")
                continue

            st_naive = _bench(lambda: naive_fn(ds, n_ops), warmup, repeats)
            st_fused = _bench(lambda: fused_fn(gpu_ds, chunked, n_ops), warmup, repeats)
            speedup = st_naive["mean"] / st_fused["mean"]

            print(f"  {n_ops:>6d}  "
                  f"{st_naive['mean']*1e3:8.2f} +/-{st_naive['std']*1e3:5.2f}  "
                  f"{st_fused['mean']*1e3:8.2f} +/-{st_fused['std']*1e3:5.2f}  "
                  f"{speedup:7.2f}x  {'yes':>8s}")

            if csv_rows is not None:
                csv_rows.append({
                    "layout": label, "case": case, "n_ops": n_ops,
                    "naive_s": st_naive["mean"], "naive_std_s": st_naive["std"],
                    "fused_s": st_fused["mean"], "fused_std_s": st_fused["std"],
                    "speedup": speedup,
                })

    f.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fused transform/reduce vs. the old read-copy-execute "
                    "approach, swept across per-chunk compute cost.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("h5_file", type=Path, nargs="?", default=None,
                        help="Existing HDF5 file (omit to auto-generate one "
                             "via --layout)")
    parser.add_argument("--dataset", type=str, default=None, metavar="NAME",
                        help="Dataset path inside h5_file (required if "
                             "h5_file is given)")
    parser.add_argument("--layout", choices=["2d_chunked", "2d_contiguous",
                                             "3d_chunked", "3d_contiguous"],
                        default="3d_chunked",
                        help="Auto-generate a dataset of this layout if no "
                             "h5_file is given (default: 3d_chunked)")
    parser.add_argument("--n-ops", type=str,
                        default=",".join(str(n) for n in DEFAULT_N_OPS),
                        metavar="N1,N2,...",
                        help="Comma-separated per-chunk compute-cost knob "
                             "values (default: " +
                             ",".join(str(n) for n in DEFAULT_N_OPS) + ")")
    parser.add_argument("--repeats", type=int, default=REPEATS, metavar="R")
    parser.add_argument("--warmup", type=int, default=WARMUP, metavar="W")
    parser.add_argument("--csv", type=Path, default=None, metavar="PATH")

    # Auto-generation sizing (only used when h5_file is omitted)
    parser.add_argument("--size-2d", type=int, default=8192, metavar="N")
    parser.add_argument("--chunk-2d", type=int, default=256, metavar="C")
    parser.add_argument("--n-3d", type=int, default=256, metavar="N")
    parser.add_argument("--size-3d", type=int, default=512, metavar="S")
    parser.add_argument("--keep-data", action="store_true")

    args = parser.parse_args()

    if not _CUPY_AVAILABLE:
        sys.exit("CuPy is required (GPUDataset uses it internally). "
                 "Install with: pip install cupy-cuda12x")

    n_ops_list = [int(x) for x in args.n_ops.split(",")]
    csv_rows = [] if args.csv else None

    if args.h5_file is not None:
        if args.dataset is None:
            sys.exit("Provide --dataset when passing an explicit h5_file")
        run_sweep(args.h5_file, args.dataset, n_ops_list, args.repeats,
                 args.warmup, csv_rows)
    else:
        tmpdir = Path(tempfile.mkdtemp(prefix="h5py_fused_compute_"))
        gen_script = Path(__file__).resolve().parent / "make_benchmark_data.py"
        dataset_name = "images" if args.layout.startswith("3d") else "data"
        path = tmpdir / f"fused_{args.layout}.h5"

        if args.layout in ("2d_chunked", "2d_contiguous"):
            gen_args = ["--kind", args.layout,
                       "--rows", str(args.size_2d), "--cols", str(args.size_2d)]
            if args.layout == "2d_chunked":
                gen_args += ["--chunk-rows", str(args.chunk_2d),
                            "--chunk-cols", str(args.chunk_2d)]
        else:
            gen_args = ["--kind", args.layout,
                       "--n", str(args.n_3d), "--size", str(args.size_3d)]

        try:
            subprocess.run(
                [sys.executable, str(gen_script), str(path), "--force"] + gen_args,
                check=True,
            )
            run_sweep(path, dataset_name, n_ops_list, args.repeats,
                     args.warmup, csv_rows, layout_label=args.layout)
        finally:
            if args.keep_data:
                print(f"\nGenerated data kept in: {tmpdir}")
            else:
                shutil.rmtree(tmpdir, ignore_errors=True)

    if args.csv and csv_rows:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with open(args.csv, "w", newline="") as fout:
            writer = csv.DictWriter(fout, fieldnames=csv_rows[0].keys())
            writer.writeheader()
            writer.writerows(csv_rows)
        print(f"\nRaw results written to: {args.csv}")


if __name__ == "__main__":
    main()
