#!/usr/bin/env python3
"""
make_benchmark_data.py – Generate synthetic HDF5 files for the GPU-aware
h5py benchmarks, without needing the real fastMRI dataset.

Four layouts, matching what the benchmark scripts in this repo expect:

    3d_chunked      /images  float32  (N, size, size)  chunked (1, size, size)
                    Same layout as prepare_fastmri.py's output -- drop-in for
                    benchmark_ml_loading.py and benchmark_batch_overlap.py.

    3d_contiguous   /images  float32  (N, size, size)  no chunking
                    3-D counterpart of 2d_contiguous, for the contiguous
                    read_double_buffered path on 3-D data.

    2d_chunked      /data    float32  (rows, cols)  chunked (chunk_rows, chunk_cols)
                    For benchmark_use_cases.py and bench_sel_opt.py-style
                    chunk-aware benchmarks.

    2d_contiguous   /data    float32  (rows, cols)  no chunking
                    For the contiguous (read_double_buffered /
                    reduce_double_buffered) code paths in benchmark_use_cases.py.

Data is synthetic (reproducible via --seed) -- structureless random floats
are fine for I/O and transfer benchmarks, which only care about size and
layout, not content. Written in row-band batches so generation itself
doesn't need the whole array resident in host memory.

Usage
-----
    # 3-D, chunked (1, H, W) -- e.g. for benchmark_ml_loading.py
    python make_benchmark_data.py bench_3d_512.h5 --kind 3d_chunked \\
        --n 1024 --size 512

    # 2-D chunked
    python make_benchmark_data.py bench_2d_chunked.h5 --kind 2d_chunked \\
        --rows 8192 --cols 8192 --chunk-rows 256 --chunk-cols 256

    # 2-D contiguous
    python make_benchmark_data.py bench_2d_cont.h5 --kind 2d_contiguous \\
        --rows 8192 --cols 8192

Dependencies
------------
    h5py, numpy
"""

import argparse
import sys
import time
from pathlib import Path

import h5py
import numpy as np


def _fill_rows(dset, n_rows, row_shape, batch_rows, rng, dtype):
    """Write pseudo-random data row-band by row-band (bounded host memory)."""
    total = 0
    while total < n_rows:
        n = min(batch_rows, n_rows - total)
        block = rng.standard_normal((n,) + row_shape).astype(dtype)
        dset[total : total + n] = block
        total += n
    return total


def make_3d_chunked(path: Path, n: int, size: int, batch_rows: int,
                    seed: int, split: str) -> None:
    dtype = np.float32
    rng = np.random.default_rng(seed)

    with h5py.File(path, "w") as f:
        images = f.create_dataset(
            "images", shape=(n, size, size), dtype=dtype,
            chunks=(1, size, size),
        )
        t0 = time.perf_counter()
        _fill_rows(images, n, (size, size), max(1, batch_rows), rng, dtype)
        elapsed = time.perf_counter() - t0

        f.create_dataset("file_index", data=np.zeros(n, dtype=np.int32))
        f.create_dataset("slice_index", data=np.arange(n, dtype=np.int32))
        f.attrs["source_dir"]     = "synthetic"
        f.attrs["size"]           = size
        f.attrs["chunk_mb"]       = size * size * 4 / 1e6
        f.attrs["n_files"]        = 1
        f.attrs["n_slices"]       = n
        f.attrs["fastmri_split"]  = split

    nbytes = n * size * size * 4
    print(f"  Wrote {path}  images=({n},{size},{size}) float32  "
          f"chunks=(1,{size},{size})  {nbytes/1e6:.1f} MB  "
          f"in {elapsed:.1f}s ({nbytes/elapsed/1e6:.0f} MB/s)")


def make_3d_contiguous(path: Path, n: int, size: int, batch_rows: int,
                       seed: int) -> None:
    dtype = np.float32
    rng = np.random.default_rng(seed)

    with h5py.File(path, "w") as f:
        images = f.create_dataset("images", shape=(n, size, size), dtype=dtype)
        t0 = time.perf_counter()
        _fill_rows(images, n, (size, size), max(1, batch_rows), rng, dtype)
        elapsed = time.perf_counter() - t0
        f.attrs["source"] = "synthetic"

    nbytes = n * size * size * 4
    print(f"  Wrote {path}  images=({n},{size},{size}) float32  contiguous  "
          f"{nbytes/1e6:.1f} MB  in {elapsed:.1f}s "
          f"({nbytes/elapsed/1e6:.0f} MB/s)")


def make_2d_chunked(path: Path, rows: int, cols: int,
                    chunk_rows: int, chunk_cols: int, batch_rows: int,
                    seed: int) -> None:
    dtype = np.float32
    rng = np.random.default_rng(seed)

    with h5py.File(path, "w") as f:
        data = f.create_dataset(
            "data", shape=(rows, cols), dtype=dtype,
            chunks=(chunk_rows, chunk_cols),
        )
        t0 = time.perf_counter()
        _fill_rows(data, rows, (cols,), max(chunk_rows, batch_rows), rng, dtype)
        elapsed = time.perf_counter() - t0
        f.attrs["source"] = "synthetic"

    nbytes = rows * cols * 4
    print(f"  Wrote {path}  data=({rows},{cols}) float32  "
          f"chunks=({chunk_rows},{chunk_cols})  {nbytes/1e6:.1f} MB  "
          f"in {elapsed:.1f}s ({nbytes/elapsed/1e6:.0f} MB/s)")


def make_2d_contiguous(path: Path, rows: int, cols: int, batch_rows: int,
                       seed: int) -> None:
    dtype = np.float32
    rng = np.random.default_rng(seed)

    with h5py.File(path, "w") as f:
        data = f.create_dataset("data", shape=(rows, cols), dtype=dtype)
        t0 = time.perf_counter()
        _fill_rows(data, rows, (cols,), max(1, batch_rows), rng, dtype)
        elapsed = time.perf_counter() - t0
        f.attrs["source"] = "synthetic"

    nbytes = rows * cols * 4
    print(f"  Wrote {path}  data=({rows},{cols}) float32  contiguous  "
          f"{nbytes/1e6:.1f} MB  in {elapsed:.1f}s "
          f"({nbytes/elapsed/1e6:.0f} MB/s)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate synthetic HDF5 files for the GPU-aware h5py benchmarks.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("out_file", type=Path, help="Output .h5 path")
    parser.add_argument("--kind", choices=["3d_chunked", "3d_contiguous",
                                           "2d_chunked", "2d_contiguous"],
                        required=True)

    # 3d_chunked
    parser.add_argument("--n",    type=int, default=1024, metavar="N",
                        help="3d_chunked: number of slices (default: 1024)")
    parser.add_argument("--size", type=int, default=512,  metavar="S",
                        help="3d_chunked: H=W=size per slice (default: 512)")
    parser.add_argument("--split", type=str, default="synthetic", metavar="NAME",
                        help="3d_chunked: fastmri_split attribute value")

    # 2d_*
    parser.add_argument("--rows", type=int, default=8192, metavar="R",
                        help="2d_*: number of rows (default: 8192)")
    parser.add_argument("--cols", type=int, default=8192, metavar="C",
                        help="2d_*: number of columns (default: 8192)")
    parser.add_argument("--chunk-rows", type=int, default=256, metavar="CR",
                        help="2d_chunked: chunk height (default: 256)")
    parser.add_argument("--chunk-cols", type=int, default=256, metavar="CC",
                        help="2d_chunked: chunk width (default: 256)")

    parser.add_argument("--batch-rows", type=int, default=64, metavar="B",
                        help="Row-band size used while generating (default: 64); "
                             "bounds host memory during creation, not part of "
                             "the dataset's own chunking")
    parser.add_argument("--seed", type=int, default=0, metavar="SEED")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite out_file if it already exists")

    args = parser.parse_args()

    if args.out_file.exists() and not args.force:
        sys.exit(f"{args.out_file} already exists (use --force to overwrite)")

    if args.kind == "3d_chunked":
        make_3d_chunked(args.out_file, args.n, args.size, args.batch_rows,
                        args.seed, args.split)
    elif args.kind == "3d_contiguous":
        make_3d_contiguous(args.out_file, args.n, args.size, args.batch_rows,
                           args.seed)
    elif args.kind == "2d_chunked":
        make_2d_chunked(args.out_file, args.rows, args.cols,
                        args.chunk_rows, args.chunk_cols, args.batch_rows,
                        args.seed)
    else:
        make_2d_contiguous(args.out_file, args.rows, args.cols,
                           args.batch_rows, args.seed)


if __name__ == "__main__":
    main()
