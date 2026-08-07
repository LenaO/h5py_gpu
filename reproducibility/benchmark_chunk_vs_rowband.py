#!/usr/bin/env python3
"""
benchmark_chunk_vs_rowband.py - Isolates the overhead difference between
processing a chunked dataset one HDF5 chunk at a time vs. batching whole
row-bands (every chunk in a row, one call) -- the rationale behind
choosing row-bands over chunk-by-chunk throughout this module, for both
reads and writes.

Both methods in each pair move exactly the same data, in the same order,
through the same double-buffering pattern; the only difference is how many
Python-level calls (and HDF5 read_direct/write_direct calls) that data is
split across:

    read:  read_chunks_to_gpu()      chunk-by-chunk (one call per chunk)
           read_double_buffered()    row-band (one call per row of chunks)

    write: write_chunks_from_gpu()   chunk-by-chunk
           write_double_buffered()   row-band

Dataset size is held fixed; chunk size is swept, so the number of chunks
batched into one row-band varies (dataset_width / chunk_width chunks per
band). The expected signature: at large chunk sizes (few chunks per row)
the two methods should be close; as chunk size shrinks (more chunks per
row), row-band's advantage should grow, since chunk-by-chunk pays its
fixed per-call overhead once per chunk while row-band pays it once per row
regardless of how many chunks that row contains.

Usage
-----
    python benchmark_chunk_vs_rowband.py --size 8192 \\
        --chunk-sizes 128,256,512,1024,2048 --csv chunk_vs_rowband.csv
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


def _run_read_pair(path, dataset_name, repeats, warmup):
    from h5py.gpu import GPUDataset

    with h5py.File(path, "r") as f:
        ds = f[dataset_name]
        gpu_ds = GPUDataset(ds)

        truth = ds[:]
        chunkwise = cp.asnumpy(gpu_ds.read_chunks_to_gpu())
        rowband = cp.asnumpy(gpu_ds.read_double_buffered())
        if not (np.array_equal(truth, chunkwise) and np.array_equal(truth, rowband)):
            raise RuntimeError("read correctness check failed")

        st_chunk = _bench(lambda: gpu_ds.read_chunks_to_gpu(), warmup, repeats)
        st_band  = _bench(lambda: gpu_ds.read_double_buffered(), warmup, repeats)
    return st_chunk, st_band


def _run_write_pair(path, dataset_name, repeats, warmup):
    from h5py.gpu import GPUDataset

    with h5py.File(path, "r+") as f:
        ds = f[dataset_name]
        gpu_ds = GPUDataset(ds)
        shape, dtype = ds.shape, ds.dtype

        rng = np.random.default_rng(0)
        src_np = rng.standard_normal(shape).astype(dtype)
        src_gpu = cp.asarray(src_np)

        gpu_ds.write_chunks_from_gpu(src_gpu)
        ok_chunk = np.array_equal(np.asarray(ds[:]), src_np)
        gpu_ds.write_double_buffered(src_gpu)
        ok_band = np.array_equal(np.asarray(ds[:]), src_np)
        if not (ok_chunk and ok_band):
            raise RuntimeError("write correctness check failed")

        st_chunk = _bench(lambda: gpu_ds.write_chunks_from_gpu(src_gpu), warmup, repeats)
        st_band  = _bench(lambda: gpu_ds.write_double_buffered(src_gpu), warmup, repeats)
    return st_chunk, st_band


def run_sweep(size, chunk_sizes, repeats, warmup, csv_rows, keep_data):
    repo_dir = Path(__file__).resolve().parent
    gen_script = repo_dir / "make_benchmark_data.py"
    tmpdir = Path(tempfile.mkdtemp(prefix="h5py_chunk_vs_rowband_"))

    total_bytes = size * size * 4
    print(f"\n{'='*86}")
    print(f"  Dataset: {size}x{size} float32 ({total_bytes/1e6:.1f} MB), "
          f"sweeping chunk size")
    print(f"{'='*86}")
    print(f"  {'chunk':>6s}  {'chunks/row':>10s}  {'op':<6s}  "
          f"{'chunk-wise (ms)':>16s}  {'row-band (ms)':>14s}  {'speedup':>8s}")
    print(f"  {'-'*80}")

    try:
        for chunk in chunk_sizes:
            path = tmpdir / f"sweep_{chunk}.h5"
            subprocess.run(
                [sys.executable, str(gen_script), str(path),
                 "--kind", "2d_chunked", "--rows", str(size), "--cols", str(size),
                 "--chunk-rows", str(chunk), "--chunk-cols", str(chunk), "--force"],
                check=True, capture_output=True,
            )
            chunks_per_row = -(-size // chunk)  # ceil

            st_rc, st_rb = _run_read_pair(path, "data", repeats, warmup)
            speedup_r = st_rc["mean"] / st_rb["mean"]
            print(f"  {chunk:>6d}  {chunks_per_row:>10d}  {'read':<6s}  "
                  f"{st_rc['mean']*1e3:13.2f}  {st_rb['mean']*1e3:11.2f}  "
                  f"{speedup_r:7.2f}x")

            st_wc, st_wb = _run_write_pair(path, "data", repeats, warmup)
            speedup_w = st_wc["mean"] / st_wb["mean"]
            print(f"  {chunk:>6d}  {chunks_per_row:>10d}  {'write':<6s}  "
                  f"{st_wc['mean']*1e3:13.2f}  {st_wb['mean']*1e3:11.2f}  "
                  f"{speedup_w:7.2f}x")

            if csv_rows is not None:
                for op, st_chunk, st_band in [("read", st_rc, st_rb),
                                              ("write", st_wc, st_wb)]:
                    for i in range(len(st_chunk["times"])):
                        csv_rows.append({
                            "size": size, "chunk": chunk,
                            "chunks_per_row": chunks_per_row, "op": op,
                            "repeat": i,
                            "chunkwise_s": st_chunk["times"][i],
                            "rowband_s": st_band["times"][i],
                        })
            if not keep_data:
                path.unlink(missing_ok=True)
    finally:
        if keep_data:
            print(f"\nGenerated data kept in: {tmpdir}")
        else:
            shutil.rmtree(tmpdir, ignore_errors=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Chunk-by-chunk vs. row-band: isolating the Python/HDF5 "
                    "per-call overhead this module's row-band design avoids.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--size", type=int, default=8192, metavar="N",
                        help="Dataset is N x N float32 (default: 8192, ~256MB)")
    parser.add_argument("--chunk-sizes", type=str, default="128,256,512,1024,2048",
                        metavar="C1,C2,...")
    parser.add_argument("--repeats", type=int, default=REPEATS, metavar="R")
    parser.add_argument("--warmup", type=int, default=WARMUP, metavar="W")
    parser.add_argument("--csv", type=Path, default=None, metavar="PATH")
    parser.add_argument("--keep-data", action="store_true")
    args = parser.parse_args()

    if not _CUPY_AVAILABLE:
        sys.exit("CuPy is required (GPUDataset uses it internally). "
                 "Install with: pip install cupy-cuda12x")

    chunk_sizes = [int(x) for x in args.chunk_sizes.split(",")]
    csv_rows = [] if args.csv else None

    run_sweep(args.size, chunk_sizes, args.repeats, args.warmup, csv_rows,
             args.keep_data)

    if args.csv and csv_rows:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with open(args.csv, "w", newline="") as fout:
            writer = csv.DictWriter(fout, fieldnames=csv_rows[0].keys())
            writer.writeheader()
            writer.writerows(csv_rows)
        print(f"\nRaw results written to: {args.csv}")


if __name__ == "__main__":
    main()
