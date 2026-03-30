"""
GPU write benchmark: double-buffered full-dataset write and selection write.

Usage
-----
    python benchmarks/bench_gpu_write.py [--rows N] [--cols N] [--dtype DTYPE]
                                         [--hdf5-chunk-rows N] [--hdf5-chunk-cols N]
                                         [--repeats N] [--warmup N]

Two benchmark sections are run:

  Section 1 -- Full-dataset write, varying the row-band chunk size
    baseline : h5py dataset[:] = numpy_array  (CPU -> HDF5 directly)
    auto     : GPUDataset.write_double_buffered() -- HDF5-chunk-aligned default
    double   : GPUDataset.write_double_buffered(chunk_size=K)  (log sweep)
    tiles    : GPUDataset.write_chunks_from_gpu()

  Section 2 -- Selection write on a 2-D chunked dataset
    baseline : h5py dataset[sel] = numpy_patch  (CPU -> HDF5)
    chunked  : GPUDataset.write_selection_chunked(gpu_patch, sel)

    Sub-sections:
      (a) Coverage sweep  -- selection covers 10 / 25 / 50 / 75 / 100 % of the dataset,
                             selection starts at a non-chunk-aligned offset
      (b) Alignment sweep -- fixed 50 % coverage, aligned vs misaligned selections

    Reported metrics (per selection):
      sel MB    : bytes of data actually written (useful payload)
      rw MB     : bytes read+written by HDF5 (partial-chunk read-modify-write overhead)
      waste %   : extra I/O fraction from partial-chunk overhead
      BW (GB/s) : useful throughput  (sel_bytes / wall_time)
      speedup   : baseline_time / chunked_time
"""

import argparse
import os
import sys
import tempfile
import time

import numpy as np

import h5py
from h5py.gpu import GPUDataset, _normalize_sel, _iter_touched_chunks

try:
    import cupy as cp
except ImportError:
    sys.exit("CuPy is required to run this benchmark.")


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _flush_gpu():
    cp.cuda.runtime.deviceSynchronize()


def _time_fn(fn, repeats, warmup):
    """Return (mean_s, min_s, [times]) for *fn*."""
    for _ in range(warmup):
        fn()
        _flush_gpu()
    times = []
    for _ in range(repeats):
        _flush_gpu()
        t0 = time.perf_counter()
        fn()
        _flush_gpu()
        times.append(time.perf_counter() - t0)
    return float(np.mean(times)), float(np.min(times)), times


def _gb(nbytes):
    return nbytes / 1024**3


def _mb(nbytes):
    return nbytes / 1024**2


def _bar(value, max_value, width=28):
    filled = int(round(value / max_value * width))
    return "#" * filled + "." * (width - filled)


def _chunk_sizes(n_rows, hdf5_chunk_rows):
    sizes = set()
    k = 1
    while k <= n_rows:
        sizes.add(k)
        k *= 2
    sizes.add(hdf5_chunk_rows)
    return sorted(sizes)


def _sel_row(rows, frac, align, chunk_rows):
    size = max(1, int(rows * frac))
    if align:
        r0 = (rows // 3 // chunk_rows) * chunk_rows
    else:
        r0 = rows // 3 + chunk_rows // 2
    r0 = min(r0, rows - size)
    return r0, r0 + size


def _sel_col(cols, frac, align, chunk_cols):
    size = max(1, int(cols * frac))
    if align:
        c0 = (cols // 4 // chunk_cols) * chunk_cols
    else:
        c0 = cols // 4 + chunk_cols // 3
    c0 = min(c0, cols - size)
    return c0, c0 + size


# ---------------------------------------------------------------------------
# Section 1: full-dataset write sweep
# ---------------------------------------------------------------------------

def bench_full_write(f, gpu_ds, data, dtype, hdf5_chunks, repeats, warmup):
    rows, cols = data.shape
    hdf5_chunk_rows, hdf5_chunk_cols = hdf5_chunks
    total_bytes = data.nbytes
    gpu_data = cp.asarray(data)
    ds = f["ds"]

    # baseline: numpy -> HDF5 via h5py (no GPU transfer)
    mean_b, _, _ = _time_fn(lambda: ds.__setitem__(np.s_[:], data), repeats, warmup)
    bw_b = _gb(total_bytes) / mean_b

    print(f"\n{'-'*72}")
    print(f"  {'METHOD':<34} {'CHUNK':>7}  {'TIME (s)':>8}  "
          f"{'BW (GB/s)':>9}  {'SPEEDUP':>7}")
    print(f"{'-'*72}")
    print(f"  {'baseline (h5py numpy write)':<34} {'N/A':>7}  "
          f"{mean_b:8.4f}  {bw_b:9.3f}  {'1.00x':>7}")

    # write_double_buffered auto (HDF5-chunk-aligned default)
    mean_a, _, _ = _time_fn(
        lambda: gpu_ds.write_double_buffered(gpu_data), repeats, warmup)
    bw_a = _gb(total_bytes) / mean_a
    sp_a = mean_b / mean_a
    auto_mb = hdf5_chunk_rows * hdf5_chunk_cols * dtype.itemsize / 1024**2
    print(f"  {'double  auto (HDF5-aligned)*':<34} {auto_mb:>6.1f}M"
          f"  {mean_a:8.4f}  {bw_a:9.3f}  {sp_a:>6.2f}x")

    results = []
    for cs in _chunk_sizes(rows, hdf5_chunk_rows):
        chunk_mb = cs * cols * dtype.itemsize / 1024**2
        marker = "*" if cs == hdf5_chunk_rows else " "
        mean_c, _, _ = _time_fn(
            lambda c=cs: gpu_ds.write_double_buffered(gpu_data, chunk_size=c),
            repeats, warmup)
        bw_c = _gb(total_bytes) / mean_c
        sp_c = mean_b / mean_c
        results.append((cs, chunk_mb, mean_c, bw_c, sp_c, marker))
        label = f"double  chunk={cs:>5} rows{marker}"
        print(f"  {label:<34} {chunk_mb:>6.1f}M  "
              f"{mean_c:8.4f}  {bw_c:9.3f}  {sp_c:>6.2f}x")

    # write_chunks_from_gpu (tile-by-tile)
    mean_wc, _, _ = _time_fn(
        lambda: gpu_ds.write_chunks_from_gpu(gpu_data), repeats, warmup)
    bw_wc = _gb(total_bytes) / mean_wc
    sp_wc = mean_b / mean_wc
    tile_mb = hdf5_chunk_rows * hdf5_chunk_cols * dtype.itemsize / 1024**2
    print(f"  {'write_chunks_from_gpu (tiles)*':<34} {tile_mb:>6.1f}M  "
          f"{mean_wc:8.4f}  {bw_wc:9.3f}  {sp_wc:>6.2f}x")

    print(f"{'-'*72}")
    print(f"  * = aligned to HDF5 chunk shape ({hdf5_chunks})")

    all_bws = [bw_b, bw_a, bw_wc] + [r[3] for r in results]
    max_bw = max(all_bws)
    print(f"\n  Bandwidth  (each # ~= {max_bw/28:.2f} GB/s)\n")
    print(f"  {'baseline':<26}  {_bar(bw_b, max_bw)}  {bw_b:.3f} GB/s")
    print(f"  {'auto (HDF5-aligned)*':<26}  {_bar(bw_a, max_bw)}  {bw_a:.3f} GB/s")
    for cs, _, _, bw_c, _, marker in results:
        print(f"  {f'chunk={cs}{marker}':<26}  {_bar(bw_c, max_bw)}  {bw_c:.3f} GB/s")
    print(f"  {'write_chunks_from_gpu*':<26}  {_bar(bw_wc, max_bw)}  {bw_wc:.3f} GB/s")

    hdf5_r = next(r for r in results if r[0] == hdf5_chunk_rows)
    print(f"\n  HDF5-aligned double (chunk={hdf5_chunk_rows}): "
          f"{hdf5_r[3]:.3f} GB/s  ({hdf5_r[4]:.2f}x)")
    print(f"  write_chunks_from_gpu:             "
          f"{bw_wc:.3f} GB/s  ({sp_wc:.2f}x)")


# ---------------------------------------------------------------------------
# Section 2: selection write benchmark helpers
# ---------------------------------------------------------------------------

def _bench_one_sel_write(f_ds, gpu_ds, sel, dtype, shape, chunks, repeats, warmup):
    r0, r1 = sel[0].start, sel[0].stop
    c0, c1 = sel[1].start, sel[1].stop
    sel_shape = (r1 - r0, c1 - c0)
    sel_bytes = sel_shape[0] * sel_shape[1] * dtype.itemsize

    norm, _ = _normalize_sel((sel[0], sel[1]), shape)
    touched = list(_iter_touched_chunks(shape, chunks, norm))
    n_chunks = len(touched)
    rw_bytes = sum(int(np.prod(s)) for _, s, _, _ in touched) * dtype.itemsize
    waste_pct = 100.0 * (rw_bytes - sel_bytes) / rw_bytes if rw_bytes else 0.0

    cpu_patch = np.random.rand(*sel_shape).astype(dtype)
    gpu_patch = cp.asarray(cpu_patch)

    # baseline: numpy patch -> h5py selection write
    mean_b, _, _ = _time_fn(
        lambda: f_ds.__setitem__((sel[0], sel[1]), cpu_patch), repeats, warmup)
    bw_b = _gb(sel_bytes) / mean_b

    # write_selection_chunked: GPU -> pinned -> HDF5 per touched chunk
    mean_c, _, _ = _time_fn(
        lambda: gpu_ds.write_selection_chunked(gpu_patch, (sel[0], sel[1])),
        repeats, warmup)
    bw_c = _gb(sel_bytes) / mean_c

    return dict(
        sel_mb=_mb(sel_bytes),
        rw_mb=_mb(rw_bytes),
        n_chunks=n_chunks,
        waste_pct=waste_pct,
        mean_b=mean_b, bw_b=bw_b,
        mean_c=mean_c, bw_c=bw_c,
        speedup=mean_b / mean_c,
    )


def _print_sel_header():
    print(f"\n  {'SELECTION':<22} {'SEL MB':>6}  {'RW MB':>6}  "
          f"{'CHUNKS':>6}  {'WASTE':>6}  "
          f"{'BASE BW':>7}  {'SEL BW':>7}  {'SPEEDUP':>7}")
    print(f"  {'-'*22}  {'-'*6}  {'-'*6}  {'-'*6}  {'-'*6}  "
          f"{'-'*7}  {'-'*7}  {'-'*7}")


def _print_sel_row(label, s):
    print(f"  {label:<22}  {s['sel_mb']:>6.1f}  {s['rw_mb']:>6.1f}  "
          f"{s['n_chunks']:>6}  {s['waste_pct']:>5.1f}%  "
          f"{s['bw_b']:>7.3f}  {s['bw_c']:>7.3f}  {s['speedup']:>6.2f}x")


def bench_coverage_sweep(f, gpu_ds, data, dtype, chunks, repeats, warmup):
    rows, cols = data.shape
    cr, cc = chunks
    coverages = [0.10, 0.25, 0.50, 0.75, 1.00]
    f_ds = f["ds"]

    print(f"\n  Coverage sweep (misaligned selections, chunks={chunks})")
    _print_sel_header()

    for frac in coverages:
        r0, r1 = _sel_row(rows, frac, align=False, chunk_rows=cr)
        c0, c1 = _sel_col(cols, frac, align=False, chunk_cols=cc)
        sel = (slice(r0, r1), slice(c0, c1))
        s = _bench_one_sel_write(
            f_ds, gpu_ds, sel, dtype, data.shape, chunks, repeats, warmup)
        label = f"{int(frac*100):3d}%  [{r0}:{r1}, {c0}:{c1}]"
        _print_sel_row(label, s)


def bench_alignment_sweep(f, gpu_ds, data, dtype, chunks, repeats, warmup):
    rows, cols = data.shape
    cr, cc = chunks
    frac = 0.50
    f_ds = f["ds"]

    print(f"\n  Alignment sweep (~50% coverage, chunks={chunks})")
    _print_sel_header()

    for aligned, label_prefix in [(False, "misaligned"), (True, "aligned  ")]:
        r0, r1 = _sel_row(rows, frac, align=aligned, chunk_rows=cr)
        c0, c1 = _sel_col(cols, frac, align=aligned, chunk_cols=cc)
        sel = (slice(r0, r1), slice(c0, c1))
        s = _bench_one_sel_write(
            f_ds, gpu_ds, sel, dtype, data.shape, chunks, repeats, warmup)
        label = f"{label_prefix} [{r0}:{r1}, {c0}:{c1}]"
        _print_sel_row(label, s)


# ---------------------------------------------------------------------------
# Top-level runner
# ---------------------------------------------------------------------------

def run(rows, cols, dtype, hdf5_chunk_rows, hdf5_chunk_cols, repeats, warmup):
    dtype = np.dtype(dtype)
    data  = np.random.rand(rows, cols).astype(dtype)
    hdf5_chunks = (hdf5_chunk_rows, hdf5_chunk_cols)

    print(f"\n{'='*72}")
    print(f"  h5py GPU write benchmark")
    print(f"  dataset    : ({rows}, {cols})  dtype={dtype}  "
          f"size={_gb(data.nbytes):.3f} GB")
    print(f"  HDF5 chunks: {hdf5_chunks}  "
          f"({hdf5_chunk_rows * hdf5_chunk_cols * dtype.itemsize / 1024**2:.2f} MB/chunk)")
    print(f"  repeats    : {repeats}   warmup : {warmup}")
    print(f"{'='*72}")

    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "bench.h5")
        with h5py.File(path, "w") as f:
            f.create_dataset("ds", shape=data.shape, dtype=dtype, chunks=hdf5_chunks)

        with h5py.File(path, "r+") as f:
            gpu_ds = GPUDataset(f["ds"])

            print(f"\n{'='*72}")
            print(f"  SECTION 1: full-dataset write  (write_double_buffered sweep)")
            bench_full_write(f, gpu_ds, data, dtype, hdf5_chunks, repeats, warmup)

            print(f"\n{'='*72}")
            print(f"  SECTION 2: selection write  (write_selection_chunked)")
            print(f"\n  Columns:")
            print(f"    SEL MB  : bytes of data actually written (useful payload)")
            print(f"    RW MB   : bytes read+written by HDF5 "
                  f"(partial-chunk read-modify-write)")
            print(f"    WASTE % : extra I/O fraction from partial-chunk overhead")
            print(f"    BASE BW : baseline GB/s  (h5py native numpy write)")
            print(f"    SEL BW  : write_selection_chunked GB/s "
                  f"(GPU -> pinned -> HDF5)")

            bench_coverage_sweep(f, gpu_ds, data, dtype, hdf5_chunks, repeats, warmup)
            bench_alignment_sweep(f, gpu_ds, data, dtype, hdf5_chunks, repeats, warmup)

    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--rows",             type=int, default=4096)
    p.add_argument("--cols",             type=int, default=4096)
    p.add_argument("--dtype",            type=str, default="float32")
    p.add_argument("--hdf5-chunk-rows",  type=int, default=None,
                   help="HDF5 chunk size along axis 0 (default: rows//16)")
    p.add_argument("--hdf5-chunk-cols",  type=int, default=None,
                   help="HDF5 chunk size along axis 1 (default: cols//16)")
    p.add_argument("--repeats",          type=int, default=5)
    p.add_argument("--warmup",           type=int, default=2)
    args = p.parse_args()

    hdf5_chunk_rows = args.hdf5_chunk_rows or max(1, args.rows // 16)
    hdf5_chunk_cols = args.hdf5_chunk_cols or max(1, args.cols // 16)

    run(args.rows, args.cols, args.dtype,
        hdf5_chunk_rows, hdf5_chunk_cols,
        args.repeats, args.warmup)
