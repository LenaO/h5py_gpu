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
                                             [--length N] [--hdf5-chunk-1d N]
                                             [--repeats N] [--warmup N]

Five benchmark sections:

  Section 1 -- 1-D contiguous dataset  (read_double_buffered + transform sweep)
    Transforms: none, scale (x*2), sqrt, exp, exp(sqrt)
    Metric: overlap% = how much of the compute time is hidden behind I/O
    Auto chunk_size = length // 16

  Section 2 -- 2-D chunked dataset, single stream  (read_chunks_to_gpu + transform sweep)
    Same transforms as Section 1

  Section 3 -- 2-D chunked dataset, multi-stream  (read_chunks_parallel, exp transform)
    Vary n_streams: 1, 2, 4, 8
    Metric: speedup vs n_streams=1

  Section 4 -- 1-D chunked dataset  (read_double_buffered + transform sweep)
    Same as Section 1 but the HDF5 dataset is stored with HDF5 chunks.
    Auto chunk_size aligns to the HDF5 chunk boundary.

  Section 5 -- 2-D contiguous (non-chunked) dataset  (read_double_buffered + transform sweep)
    read_double_buffered processes row bands; chunk_size = rows // 8.

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
        ("none",          None),
        ("x * 2",         lambda x: x * 2.0),
        ("sqrt",          cp.sqrt),
        ("exp",           cp.exp),
        ("exp(sqrt(x))",  lambda x: cp.exp(cp.sqrt(x))),
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
# Section 1 / 4: 1-D dataset — read_double_buffered + transform sweep
# ---------------------------------------------------------------------------

def bench_1d(path, ds_name, n_elems, dtype, chunk_size, repeats, warmup):
    """Transform sweep on a 1-D dataset read via read_double_buffered.

    Parameters
    ----------
    ds_name    : str  — HDF5 dataset name (e.g. ``"ds_1d"`` or ``"ds_1d_chunked"``)
    chunk_size : int  — row-band size passed to read_double_buffered
    """
    transforms  = _make_transforms()
    total_bytes = n_elems * dtype.itemsize

    with h5py.File(path, "r") as f:
        gpu_ds = GPUDataset(f[ds_name])
        t_io   = _time_fn(lambda: gpu_ds.read_double_buffered(chunk_size=chunk_size),
                          repeats, warmup)
        arr_gpu = gpu_ds.read_double_buffered(chunk_size=chunk_size)

        print(f"\n  chunk_size={chunk_size}  "
              f"({chunk_size * dtype.itemsize / 1024:.1f} KB/band)")
        _print_header()

        for label, tfm in transforms:
            t_comp = _compute_only_time(arr_gpu, tfm, repeats, warmup)
            t_pipe = _time_fn(
                lambda t=tfm: gpu_ds.read_double_buffered(chunk_size=chunk_size,
                                                          transform=t),
                repeats, warmup)
            bw = _gb(total_bytes) / t_pipe
            _print_row(label, t_pipe, bw, t_comp, t_io)

        # Bar chart (re-time for clean numbers)
        results = []
        for label, tfm in transforms:
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
# Section 2: 2-D chunked, single stream — read_chunks_to_gpu + transform sweep
# ---------------------------------------------------------------------------

def bench_2d_single(path, shape, dtype, chunks, repeats, warmup):
    transforms  = _make_transforms()
    total_bytes = int(np.prod(shape)) * dtype.itemsize

    with h5py.File(path, "r") as f:
        gpu_ds = GPUDataset(f["ds_2d"])
        t_io   = _time_fn(lambda: gpu_ds.read_chunks_to_gpu(), repeats, warmup)
        arr_gpu = gpu_ds.read_chunks_to_gpu()

        print(f"\n  chunks={chunks}  "
              f"({int(np.prod(chunks)) * dtype.itemsize / 1024:.1f} KB/chunk)")
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
# Section 3: 2-D chunked, multi-stream — n_streams sweep, heavy transform
# ---------------------------------------------------------------------------

def bench_2d_parallel(path, shape, dtype, chunks, repeats, warmup):
    total_bytes = int(np.prod(shape)) * dtype.itemsize
    stream_counts = [1, 2, 4, 8]
    transforms_to_test = [
        ("none",          None),
        ("exp",           cp.exp),
        ("exp(sqrt(x))",  lambda x: cp.exp(cp.sqrt(x))),
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
# Section 5: 2-D contiguous — read_double_buffered + transform sweep
# ---------------------------------------------------------------------------

def bench_2d_contiguous(path, shape, dtype, chunk_size, repeats, warmup):
    """Transform sweep on a 2-D contiguous dataset read via read_double_buffered.

    Parameters
    ----------
    chunk_size : int  — number of rows per band passed to read_double_buffered
    """
    transforms  = _make_transforms()
    total_bytes = int(np.prod(shape)) * dtype.itemsize
    rows, cols  = shape

    with h5py.File(path, "r") as f:
        gpu_ds  = GPUDataset(f["ds_2d_contig"])
        t_io    = _time_fn(lambda: gpu_ds.read_double_buffered(chunk_size=chunk_size),
                           repeats, warmup)
        arr_gpu = gpu_ds.read_double_buffered(chunk_size=chunk_size)

        band_mb = chunk_size * cols * dtype.itemsize / 1024**2
        print(f"\n  chunk_size={chunk_size} rows  ({band_mb:.2f} MB/band)")
        _print_header()

        for label, tfm in transforms:
            t_comp = _compute_only_time(arr_gpu, tfm, repeats, warmup)
            t_pipe = _time_fn(
                lambda t=tfm: gpu_ds.read_double_buffered(chunk_size=chunk_size,
                                                          transform=t),
                repeats, warmup)
            bw = _gb(total_bytes) / t_pipe
            _print_row(label, t_pipe, bw, t_comp, t_io)

        results = []
        for label, tfm in transforms:
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
# Top-level runner
# ---------------------------------------------------------------------------

def run(rows, cols, dtype, hdf5_chunk_rows, hdf5_chunk_cols,
        length, hdf5_chunk_1d, repeats, warmup):
    dtype       = np.dtype(dtype)
    shape_2d    = (rows, cols)
    chunks_2d   = (hdf5_chunk_rows, hdf5_chunk_cols)

    # Use positive values so sqrt/log don't produce NaN
    data_1d = np.random.rand(length).astype(dtype)  + 0.01
    data_2d = np.random.rand(*shape_2d).astype(dtype) + 0.01

    chunk_size_1d_contig  = max(1, length // 16)   # ~16 bands, contiguous
    chunk_size_2d_contig  = max(1, rows   // 8)    # ~8 bands, contiguous 2-D

    print(f"\n{'='*72}")
    print(f"  h5py GPU transform benchmark")
    print(f"  1-D size   : {length:,} elements  "
          f"({_gb(data_1d.nbytes):.3f} GB)  dtype={dtype}")
    print(f"  1-D chunk  : {hdf5_chunk_1d} elements  "
          f"({hdf5_chunk_1d * dtype.itemsize / 1024**2:.2f} MB/chunk)")
    print(f"  2-D shape  : {shape_2d}  chunks={chunks_2d}  "
          f"({_gb(data_2d.nbytes):.3f} GB)")
    print(f"  repeats    : {repeats}   warmup : {warmup}")
    print(f"{'='*72}")

    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "bench.h5")
        with h5py.File(path, "w") as f:
            f.create_dataset("ds_1d",        data=data_1d)
            f.create_dataset("ds_1d_chunked", data=data_1d,
                             chunks=(hdf5_chunk_1d,))
            f.create_dataset("ds_2d",        data=data_2d, chunks=chunks_2d)
            f.create_dataset("ds_2d_contig", data=data_2d)

        # ── Section 1: 1-D contiguous ─────────────────────────────────────
        print(f"\n{'='*72}")
        print(f"  SECTION 1: 1-D contiguous  —  read_double_buffered + transform")
        print(f"  shape=({length},)  (contiguous)  chunk_size={chunk_size_1d_contig}")
        print(f"  Columns:")
        print(f"    COMP(s)   : pure GPU compute time (no I/O)")
        print(f"    OVERLAP % : fraction of compute time hidden behind I/O")
        bench_1d(path, "ds_1d", length, dtype,
                 chunk_size_1d_contig, repeats, warmup)

        # ── Section 2: 2-D chunked, single stream ─────────────────────────
        print(f"\n{'='*72}")
        print(f"  SECTION 2: 2-D chunked  —  read_chunks_to_gpu + transform "
              f"(single stream)")
        print(f"  shape={shape_2d}  chunks={chunks_2d}")
        bench_2d_single(path, shape_2d, dtype, chunks_2d, repeats, warmup)

        # ── Section 3: 2-D chunked, multi-stream ──────────────────────────
        print(f"\n{'='*72}")
        print(f"  SECTION 3: 2-D chunked  —  read_chunks_parallel (n_streams sweep)")
        print(f"  shape={shape_2d}  chunks={chunks_2d}")
        print(f"  SPEEDUP is relative to n_streams=1 for the same transform.")
        bench_2d_parallel(path, shape_2d, dtype, chunks_2d, repeats, warmup)

        # ── Section 4: 1-D chunked ────────────────────────────────────────
        print(f"\n{'='*72}")
        print(f"  SECTION 4: 1-D chunked  —  read_double_buffered + transform")
        print(f"  shape=({length},)  chunks=({hdf5_chunk_1d},)  "
              f"chunk_size={hdf5_chunk_1d} (HDF5-aligned)")
        bench_1d(path, "ds_1d_chunked", length, dtype,
                 hdf5_chunk_1d, repeats, warmup)

        # ── Section 5: 2-D contiguous ─────────────────────────────────────
        print(f"\n{'='*72}")
        print(f"  SECTION 5: 2-D contiguous  —  read_double_buffered + transform")
        print(f"  shape={shape_2d}  (contiguous)  chunk_size={chunk_size_2d_contig} rows")
        bench_2d_contiguous(path, shape_2d, dtype,
                            chunk_size_2d_contig, repeats, warmup)

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
                   help="HDF5 chunk rows for 2-D dataset (default: rows//16)")
    p.add_argument("--hdf5-chunk-cols", type=int, default=None,
                   help="HDF5 chunk cols for 2-D dataset (default: cols//16)")
    p.add_argument("--length",          type=int, default=None,
                   help="Number of elements in the 1-D dataset "
                        "(default: rows * cols)")
    p.add_argument("--hdf5-chunk-1d",   type=int, default=None,
                   help="HDF5 chunk size for the 1-D chunked dataset "
                        "(default: length // 16)")
    p.add_argument("--repeats",         type=int, default=5)
    p.add_argument("--warmup",          type=int, default=2)
    args = p.parse_args()

    hdf5_chunk_rows = args.hdf5_chunk_rows or max(1, args.rows // 16)
    hdf5_chunk_cols = args.hdf5_chunk_cols or max(1, args.cols // 16)
    length          = args.length          or args.rows * args.cols
    hdf5_chunk_1d   = args.hdf5_chunk_1d   or max(1, length // 16)

    run(args.rows, args.cols, args.dtype,
        hdf5_chunk_rows, hdf5_chunk_cols,
        length, hdf5_chunk_1d,
        args.repeats, args.warmup)
