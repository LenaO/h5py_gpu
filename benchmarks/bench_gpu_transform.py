"""
GPU transform pipeline benchmark.

Measures how well element-wise GPU compute is hidden behind HDF5 I/O when
using the ``transform`` parameter and multi-stream parallelism.

Usage
-----
    python benchmarks/bench_gpu_transform.py [--rows N] [--cols N]
                                             [--dtype DTYPE]
                                             [--hdf5-chunk-rows N]
                                             [--hdf5-chunk-cols N]
                                             [--repeats N] [--warmup N]

Three benchmark sections:

  Section 1 -- 1-D dataset  (read_double_buffered + transform sweep)
    Transforms: none, scale (x*2), sqrt, exp, exp(sqrt)
    Metric: overlap% = how much of the compute time is hidden behind I/O

  Section 2 -- 2-D dataset, single stream  (read_chunks_to_gpu + transform sweep)
    Same transforms as Section 1

  Section 3 -- 2-D dataset, multi-stream  (read_chunks_parallel, exp transform)
    Vary n_streams: 1, 2, 4, 8
    Metric: speedup vs n_streams=1

All sections report:
    TIME(s)   : wall time of the pipelined call
    BW(GB/s)  : useful read throughput  (dataset bytes / time)
    COMP(s)   : pure GPU compute time (transform applied to already-loaded array)
    OVERLAP%  : fraction of compute time hidden behind I/O
                100% -> compute is completely free; 0% -> fully sequential
"""

import argparse
import os
import sys
import tempfile
import time

import numpy as np

import h5py
from h5py.gpu import GPUDataset

try:
    import cupy as cp
except ImportError:
    sys.exit("CuPy is required to run this benchmark.")


# ---------------------------------------------------------------------------
# Transforms (ordered roughly by compute intensity)
# ---------------------------------------------------------------------------

def _make_transforms():
    """Return list of (label, callable_or_None)."""
    return [
        ("none",        None),
        ("x * 2",       lambda x: x * 2.0),
        ("sqrt",        cp.sqrt),
        ("exp",         cp.exp),
        ("exp(sqrt(x))", lambda x: cp.exp(cp.sqrt(x))),
    ]


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _flush():
    cp.cuda.runtime.deviceSynchronize()


def _time_fn(fn, repeats, warmup):
    for _ in range(warmup):
        fn(); _flush()
    times = []
    for _ in range(repeats):
        _flush()
        t0 = time.perf_counter()
        fn(); _flush()
        times.append(time.perf_counter() - t0)
    return float(np.mean(times))


def _gb(n):
    return n / 1024**3


def _bar(v, vmax, width=24):
    f = int(round(v / vmax * width)) if vmax > 0 else 0
    return "#" * f + "." * (width - f)


def _compute_only_time(arr, transform, repeats, warmup):
    """Time applying transform to the full already-loaded GPU array."""
    if transform is None:
        return 0.0
    return _time_fn(lambda: transform(arr), repeats, warmup)


def _overlap_pct(t_pipe, t_io, t_comp):
    """Percentage of compute time hidden behind I/O.

    overlap = (t_io + t_comp - t_pipe) / t_comp  (clamped to [0, 100])
    """
    if t_comp <= 0:
        return 100.0
    hidden = t_io + t_comp - t_pipe
    return max(0.0, min(100.0, hidden / t_comp * 100.0))


def _print_header(extra_col=False):
    if extra_col:
        print(f"\n  {'TRANSFORM':<16} {'TIME(s)':>8}  {'BW(GB/s)':>9}  "
              f"{'COMP(s)':>8}  {'OVERLAP':>8}  {'SPEEDUP':>8}")
        print(f"  {'-'*16}  {'-'*8}  {'-'*9}  {'-'*8}  {'-'*8}  {'-'*8}")
    else:
        print(f"\n  {'TRANSFORM':<16} {'TIME(s)':>8}  {'BW(GB/s)':>9}  "
              f"{'COMP(s)':>8}  {'OVERLAP':>8}")
        print(f"  {'-'*16}  {'-'*8}  {'-'*9}  {'-'*8}  {'-'*8}")


def _print_row(label, t_pipe, bw, t_comp, t_io, speedup_vs=None):
    overlap = _overlap_pct(t_pipe, t_io, t_comp)
    overlap_s = f"{overlap:6.1f}%" if t_comp > 0 else "   N/A  "
    row = (f"  {label:<16}  {t_pipe:8.4f}  {bw:9.3f}  "
           f"{t_comp:8.4f}  {overlap_s}")
    if speedup_vs is not None:
        row += f"  {speedup_vs/t_pipe:>7.2f}x"
    print(row)


# ---------------------------------------------------------------------------
# Section 1: 1-D — read_double_buffered + transform sweep
# ---------------------------------------------------------------------------

def bench_1d(path, n_elems, dtype, chunk_size, repeats, warmup):
    transforms = _make_transforms()
    total_bytes = n_elems * dtype.itemsize

    with h5py.File(path, "r") as f:
        gpu_ds = GPUDataset(f["ds_1d"])
        # Measure pure I/O time (no transform)
        t_io = _time_fn(lambda: gpu_ds.read_double_buffered(chunk_size=chunk_size),
                        repeats, warmup)

        # Load full array once for compute-only timing
        arr_gpu = gpu_ds.read_double_buffered(chunk_size=chunk_size)

        print(f"\n  chunk_size={chunk_size}  ({chunk_size * dtype.itemsize / 1024:.1f} KB/band)")
        _print_header()

        for label, tfm in transforms:
            t_comp = _compute_only_time(arr_gpu, tfm, repeats, warmup)
            t_pipe = _time_fn(
                lambda t=tfm: gpu_ds.read_double_buffered(chunk_size=chunk_size,
                                                          transform=t),
                repeats, warmup)
            bw = _gb(total_bytes) / t_pipe
            _print_row(label, t_pipe, bw, t_comp, t_io)

        # Bar chart
        results = []
        for label, tfm in transforms:
            t_comp = _compute_only_time(arr_gpu, tfm, repeats, warmup)
            t_pipe = _time_fn(
                lambda t=tfm: gpu_ds.read_double_buffered(chunk_size=chunk_size,
                                                          transform=t),
                repeats, warmup)
            results.append((label, _gb(total_bytes) / t_pipe))

        max_bw = max(bw for _, bw in results)
        print(f"\n  Bandwidth  (each # ~= {max_bw/24:.3f} GB/s)\n")
        for label, bw in results:
            print(f"  {label:<16}  {_bar(bw, max_bw)}  {bw:.3f} GB/s")


# ---------------------------------------------------------------------------
# Section 2: 2-D single stream — read_chunks_to_gpu + transform sweep
# ---------------------------------------------------------------------------

def bench_2d_single(path, shape, dtype, chunks, repeats, warmup):
    transforms = _make_transforms()
    total_bytes = int(np.prod(shape)) * dtype.itemsize

    with h5py.File(path, "r") as f:
        gpu_ds = GPUDataset(f["ds_2d"])
        t_io = _time_fn(lambda: gpu_ds.read_chunks_to_gpu(), repeats, warmup)
        arr_gpu = gpu_ds.read_chunks_to_gpu()

        print(f"\n  chunks={chunks}  ({int(np.prod(chunks)) * dtype.itemsize / 1024:.1f} KB/chunk)")
        _print_header()

        for label, tfm in transforms:
            t_comp = _compute_only_time(arr_gpu, tfm, repeats, warmup)
            t_pipe = _time_fn(
                lambda t=tfm: gpu_ds.read_chunks_to_gpu(transform=t),
                repeats, warmup)
            bw = _gb(total_bytes) / t_pipe
            _print_row(label, t_pipe, bw, t_comp, t_io)

        results = []
        for label, tfm in transforms:
            t_pipe = _time_fn(
                lambda t=tfm: gpu_ds.read_chunks_to_gpu(transform=t),
                repeats, warmup)
            results.append((label, _gb(total_bytes) / t_pipe))

        max_bw = max(bw for _, bw in results)
        print(f"\n  Bandwidth  (each # ~= {max_bw/24:.3f} GB/s)\n")
        for label, bw in results:
            print(f"  {label:<16}  {_bar(bw, max_bw)}  {bw:.3f} GB/s")


# ---------------------------------------------------------------------------
# Section 3: 2-D multi-stream — n_streams sweep, heavy transform (exp)
# ---------------------------------------------------------------------------

def bench_2d_parallel(path, shape, dtype, chunks, repeats, warmup):
    total_bytes = int(np.prod(shape)) * dtype.itemsize
    stream_counts = [1, 2, 4, 8]
    transforms_to_test = [
        ("none",  None),
        ("exp",   cp.exp),
        ("exp(sqrt(x))", lambda x: cp.exp(cp.sqrt(x))),
    ]

    with h5py.File(path, "r") as f:
        gpu_ds = GPUDataset(f["ds_2d"])

        for tfm_label, tfm in transforms_to_test:
            print(f"\n  transform: {tfm_label}  chunks={chunks}")
            print(f"\n  {'N_STREAMS':>10}  {'TIME(s)':>8}  {'BW(GB/s)':>9}  {'SPEEDUP':>8}")
            print(f"  {'-'*10}  {'-'*8}  {'-'*9}  {'-'*8}")

            t_base = None
            results = []
            for n in stream_counts:
                t = _time_fn(
                    lambda ns=n, t=tfm: gpu_ds.read_chunks_parallel(
                        n_streams=ns, transform=t),
                    repeats, warmup)
                bw = _gb(total_bytes) / t
                if t_base is None:
                    t_base = t
                speedup = t_base / t
                results.append((n, t, bw, speedup))
                print(f"  {n:>10}  {t:8.4f}  {bw:9.3f}  {speedup:>7.2f}x")

            max_bw = max(bw for _, _, bw, _ in results)
            print(f"\n  Bandwidth  (each # ~= {max_bw/24:.3f} GB/s)\n")
            for n, _, bw, _ in results:
                print(f"  {'n='+str(n):<16}  {_bar(bw, max_bw)}  {bw:.3f} GB/s")


# ---------------------------------------------------------------------------
# Top-level runner
# ---------------------------------------------------------------------------

def run(rows, cols, dtype, hdf5_chunk_rows, hdf5_chunk_cols, repeats, warmup):
    dtype     = np.dtype(dtype)
    n_elems   = rows * cols          # 1-D dataset is same total size
    shape_2d  = (rows, cols)
    chunks_2d = (hdf5_chunk_rows, hdf5_chunk_cols)

    # Use positive values so sqrt/log don't produce NaN
    data_1d = np.random.rand(n_elems).astype(dtype) + 0.01
    data_2d = np.random.rand(*shape_2d).astype(dtype) + 0.01

    chunk_size_1d = max(1, n_elems // 16)   # ~16 bands for 1-D

    print(f"\n{'='*72}")
    print(f"  h5py GPU transform benchmark")
    print(f"  1-D size   : {n_elems:,} elements  "
          f"({_gb(data_1d.nbytes):.3f} GB)  dtype={dtype}")
    print(f"  2-D shape  : {shape_2d}  chunks={chunks_2d}  "
          f"({_gb(data_2d.nbytes):.3f} GB)")
    print(f"  repeats    : {repeats}   warmup : {warmup}")
    print(f"{'='*72}")

    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "bench.h5")
        with h5py.File(path, "w") as f:
            f.create_dataset("ds_1d", data=data_1d)
            f.create_dataset("ds_2d", data=data_2d, chunks=chunks_2d)

        # ---- Section 1 -------------------------------------------------------
        print(f"\n{'='*72}")
        print(f"  SECTION 1: 1-D  read_double_buffered  + transform sweep")
        print(f"  Columns:")
        print(f"    COMP(s)   : pure GPU compute time (no I/O)")
        print(f"    OVERLAP % : fraction of compute time hidden behind I/O")
        bench_1d(path, n_elems, dtype, chunk_size_1d, repeats, warmup)

        # ---- Section 2 -------------------------------------------------------
        print(f"\n{'='*72}")
        print(f"  SECTION 2: 2-D  read_chunks_to_gpu  (single stream) + transform sweep")
        bench_2d_single(path, shape_2d, dtype, chunks_2d, repeats, warmup)

        # ---- Section 3 -------------------------------------------------------
        print(f"\n{'='*72}")
        print(f"  SECTION 3: 2-D  read_chunks_parallel  (n_streams sweep)")
        print(f"  SPEEDUP is relative to n_streams=1 for the same transform.")
        bench_2d_parallel(path, shape_2d, dtype, chunks_2d, repeats, warmup)

    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--rows",            type=int, default=4096)
    p.add_argument("--cols",            type=int, default=4096)
    p.add_argument("--dtype",           type=str, default="float32")
    p.add_argument("--hdf5-chunk-rows", type=int, default=None,
                   help="HDF5 chunk rows (default: rows//16)")
    p.add_argument("--hdf5-chunk-cols", type=int, default=None,
                   help="HDF5 chunk cols (default: cols//16)")
    p.add_argument("--repeats",         type=int, default=5)
    p.add_argument("--warmup",          type=int, default=2)
    args = p.parse_args()

    hdf5_chunk_rows = args.hdf5_chunk_rows or max(1, args.rows // 16)
    hdf5_chunk_cols = args.hdf5_chunk_cols or max(1, args.cols // 16)

    run(args.rows, args.cols, args.dtype,
        hdf5_chunk_rows, hdf5_chunk_cols,
        args.repeats, args.warmup)
