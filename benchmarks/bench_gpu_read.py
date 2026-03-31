"""
GPU read benchmark: double-buffered full-read and chunk-based selection read.

Usage
-----
    python benchmarks/bench_gpu_read.py [--rows N] [--cols N] [--dtype DTYPE]
                                        [--hdf5-chunk-rows N] [--hdf5-chunk-cols N]
                                        [--length N] [--hdf5-chunk-1d N]
                                        [--repeats N] [--warmup N]

Five benchmark sections are run:

  Section 1 -- Full 2-D chunked dataset read, varying the row-band chunk size
    baseline : GPUDataset[:] -- simple pinned-memory read (no pipelining)
    auto     : GPUDataset.read_double_buffered() -- HDF5-chunk-aligned default
    double   : GPUDataset.read_double_buffered(chunk_size=K)  (log sweep)

  Section 2 -- Selection read on a 2-D chunked dataset
    baseline : h5py native partial read + simple H2D (no double-buffering)
    chunked  : GPUDataset.read_selection_chunked() -- full-chunk pipeline

    Sub-sections:
      (a) Coverage sweep  -- selection covers 10 / 25 / 50 / 75 / 100 % of the dataset,
                             selection starts at a non-chunk-aligned offset
      (b) Alignment sweep -- fixed 50 % coverage, aligned vs misaligned selections

    Reported metrics (per selection):
      sel MB    : bytes of data actually requested (useful payload)
      touched   : number of HDF5 chunks read from storage
      waste %   : fraction of read bytes that are discarded after cropping
      BW (GB/s) : useful throughput  (sel_bytes / wall_time)
      speedup   : baseline_time / chunked_time

  Section 3 -- Full 2-D **contiguous** (non-chunked) dataset read
    Same row-band chunk sweep as Section 1 but no HDF5 chunks on disk.
    Auto default is rows // 8.

  Section 4 -- 1-D **contiguous** dataset read
    baseline : GPUDataset[:]
    double   : GPUDataset.read_double_buffered(chunk_size=K)  (log sweep)
    Auto default is length // 8.

  Section 5 -- 1-D **chunked** dataset read
    Same as Section 4 but the HDF5 dataset is stored with chunks=(hdf5_chunk_1d,).
    Auto default aligns to the HDF5 chunk boundary.
"""

import argparse
import os
import sys
import tempfile
import time

import numpy as np

import h5py
from h5py.gpu import GPUDataset, _numpy_to_gpu, _normalize_sel, _iter_touched_chunks

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


def _chunk_sizes(n_rows, ref_chunk):
    """Log-2 sweep of row-band sizes plus ref_chunk."""
    sizes = set()
    k = 1
    while k <= n_rows:
        sizes.add(k)
        k *= 2
    sizes.add(ref_chunk)
    return sorted(sizes)


def _chunk_sizes_1d(n, ref_chunk=None):
    """Log-2 sweep of element counts plus optional ref_chunk."""
    sizes = set()
    k = 1
    while k <= n:
        sizes.add(k)
        k *= 2
    if ref_chunk is not None:
        sizes.add(ref_chunk)
    return sorted(sizes)


def _sel_row(rows, frac, align, chunk_rows):
    """Build a row range covering *frac* of *rows*, aligned or not."""
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


def _print_dbl_header():
    print(f"\n  {'METHOD':<34} {'CHUNK':>7}  {'TIME (s)':>8}  "
          f"{'BW (GB/s)':>9}  {'SPEEDUP':>7}")
    print(f"  {'-'*34}  {'-'*7}  {'-'*8}  {'-'*9}  {'-'*7}")


def _print_dbl_barchart(bw_b, bw_a, results, auto_label):
    all_bws = [bw_b, bw_a] + [r[3] for r in results]
    max_bw = max(all_bws)
    print(f"\n  Bandwidth  (each # ~= {max_bw/28:.2f} GB/s)\n")
    print(f"  {'baseline':<26}  {_bar(bw_b, max_bw)}  {bw_b:.3f} GB/s")
    print(f"  {auto_label:<26}  {_bar(bw_a, max_bw)}  {bw_a:.3f} GB/s")
    for cs, _, _, bw_c, _, marker in results:
        print(f"  {f'chunk={cs}{marker}':<26}  {_bar(bw_c, max_bw)}  {bw_c:.3f} GB/s")


# ---------------------------------------------------------------------------
# Section 1: full 2-D chunked read
# ---------------------------------------------------------------------------

def bench_full_read(f, gpu_ds, data, dtype, hdf5_chunks, repeats, warmup):
    rows, cols = data.shape
    hdf5_chunk_rows, hdf5_chunk_cols = hdf5_chunks
    total_bytes = data.nbytes

    mean_b, _, _ = _time_fn(lambda: gpu_ds[:], repeats, warmup)
    bw_b = _gb(total_bytes) / mean_b

    print(f"\n{'-'*72}")
    _print_dbl_header()
    print(f"  {'baseline (simple read)':<34} {'N/A':>7}  "
          f"{mean_b:8.4f}  {bw_b:9.3f}  {'1.00x':>7}")

    mean_a, _, _ = _time_fn(lambda: gpu_ds.read_double_buffered(), repeats, warmup)
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
            lambda c=cs: gpu_ds.read_double_buffered(chunk_size=c), repeats, warmup)
        bw_c = _gb(total_bytes) / mean_c
        sp_c = mean_b / mean_c
        results.append((cs, chunk_mb, mean_c, bw_c, sp_c, marker))
        label = f"double  chunk={cs:>5} rows{marker}"
        print(f"  {label:<34} {chunk_mb:>6.1f}M  "
              f"{mean_c:8.4f}  {bw_c:9.3f}  {sp_c:>6.2f}x")

    print(f"{'-'*72}")
    print(f"  * = aligned to HDF5 chunk rows ({hdf5_chunk_rows})")

    _print_dbl_barchart(bw_b, bw_a, results, "auto (HDF5-aligned)*")

    best = max(results, key=lambda r: r[3])
    hdf5_r = next(r for r in results if r[0] == hdf5_chunk_rows)
    print(f"\n  HDF5-aligned (chunk={hdf5_chunk_rows}): "
          f"{hdf5_r[3]:.3f} GB/s  ({hdf5_r[4]:.2f}x)")
    print(f"  Best         (chunk={best[0]}): "
          f"{best[3]:.3f} GB/s  ({best[4]:.2f}x)")


# ---------------------------------------------------------------------------
# Section 2: selection benchmark helpers
# ---------------------------------------------------------------------------

def _bench_one_sel(f_ds, gpu_ds, sel, dtype, shape, chunks, repeats, warmup):
    r0, r1 = sel[0].start, sel[0].stop
    c0, c1 = sel[1].start, sel[1].stop
    sel_bytes = (r1 - r0) * (c1 - c0) * dtype.itemsize

    norm, _ = _normalize_sel((sel[0], sel[1]), shape)
    touched = list(_iter_touched_chunks(shape, chunks, norm))
    n_chunks = len(touched)
    read_bytes = sum(int(np.prod(s)) for _, s, _, _ in touched) * dtype.itemsize
    waste_pct = 100.0 * (read_bytes - sel_bytes) / read_bytes if read_bytes else 0.0

    mean_b, _, _ = _time_fn(
        lambda: _numpy_to_gpu(f_ds[sel[0], sel[1]]), repeats, warmup)
    bw_b = _gb(sel_bytes) / mean_b

    mean_c, _, _ = _time_fn(
        lambda: gpu_ds.read_selection_chunked((sel[0], sel[1])), repeats, warmup)
    bw_c = _gb(sel_bytes) / mean_c

    return dict(
        sel_mb=_mb(sel_bytes),
        read_mb=_mb(read_bytes),
        n_chunks=n_chunks,
        waste_pct=waste_pct,
        mean_b=mean_b, bw_b=bw_b,
        mean_c=mean_c, bw_c=bw_c,
        speedup=mean_b / mean_c,
    )


def _print_sel_header():
    print(f"\n  {'SELECTION':<22} {'SEL MB':>6}  {'READ MB':>7}  "
          f"{'CHUNKS':>6}  {'WASTE':>6}  "
          f"{'BASE BW':>7}  {'CHUNK BW':>8}  {'SPEEDUP':>7}")
    print(f"  {'-'*22}  {'-'*6}  {'-'*7}  {'-'*6}  {'-'*6}  "
          f"{'-'*7}  {'-'*8}  {'-'*7}")


def _print_sel_row(label, s):
    print(f"  {label:<22}  {s['sel_mb']:>6.1f}  {s['read_mb']:>7.1f}  "
          f"{s['n_chunks']:>6}  {s['waste_pct']:>5.1f}%  "
          f"{s['bw_b']:>7.3f}  {s['bw_c']:>8.3f}  {s['speedup']:>6.2f}x")


def bench_coverage_sweep(f, gpu_ds, data, dtype, chunks, repeats, warmup):
    rows, cols = data.shape
    cr, cc = chunks
    coverages = [0.10, 0.25, 0.50, 0.75, 1.00]

    print(f"\n  Coverage sweep (misaligned selections, chunks={chunks})")
    _print_sel_header()

    for frac in coverages:
        r0, r1 = _sel_row(rows, frac, align=False, chunk_rows=cr)
        c0, c1 = _sel_col(cols, frac, align=False, chunk_cols=cc)
        sel = (slice(r0, r1), slice(c0, c1))
        s = _bench_one_sel(f["ds"], gpu_ds, sel, dtype, data.shape, chunks, repeats, warmup)
        label = f"{int(frac*100):3d}%  [{r0}:{r1}, {c0}:{c1}]"
        _print_sel_row(label, s)


def bench_alignment_sweep(f, gpu_ds, data, dtype, chunks, repeats, warmup):
    rows, cols = data.shape
    cr, cc = chunks
    frac = 0.50

    print(f"\n  Alignment sweep (~50% coverage, chunks={chunks})")
    _print_sel_header()

    for aligned, label_prefix in [(False, "misaligned"), (True, "aligned  ")]:
        r0, r1 = _sel_row(rows, frac, align=aligned, chunk_rows=cr)
        c0, c1 = _sel_col(cols, frac, align=aligned, chunk_cols=cc)
        sel = (slice(r0, r1), slice(c0, c1))
        s = _bench_one_sel(f["ds"], gpu_ds, sel, dtype, data.shape, chunks, repeats, warmup)
        label = f"{label_prefix} [{r0}:{r1}, {c0}:{c1}]"
        _print_sel_row(label, s)


# ---------------------------------------------------------------------------
# Section 3: full 2-D contiguous (non-chunked) read
# ---------------------------------------------------------------------------

def bench_contiguous_2d_read(f, gpu_ds, data, dtype, repeats, warmup):
    rows, cols = data.shape
    total_bytes = data.nbytes
    auto_rows = max(1, rows // 8)

    mean_b, _, _ = _time_fn(lambda: gpu_ds[:], repeats, warmup)
    bw_b = _gb(total_bytes) / mean_b

    print(f"\n{'-'*72}")
    _print_dbl_header()
    print(f"  {'baseline (simple read)':<34} {'N/A':>7}  "
          f"{mean_b:8.4f}  {bw_b:9.3f}  {'1.00x':>7}")

    mean_a, _, _ = _time_fn(lambda: gpu_ds.read_double_buffered(), repeats, warmup)
    bw_a = _gb(total_bytes) / mean_a
    sp_a = mean_b / mean_a
    auto_mb = auto_rows * cols * dtype.itemsize / 1024**2
    print(f"  {'double  auto (rows//8)*':<34} {auto_mb:>6.1f}M"
          f"  {mean_a:8.4f}  {bw_a:9.3f}  {sp_a:>6.2f}x")

    results = []
    for cs in _chunk_sizes(rows, auto_rows):
        chunk_mb = cs * cols * dtype.itemsize / 1024**2
        marker = "*" if cs == auto_rows else " "
        mean_c, _, _ = _time_fn(
            lambda c=cs: gpu_ds.read_double_buffered(chunk_size=c), repeats, warmup)
        bw_c = _gb(total_bytes) / mean_c
        sp_c = mean_b / mean_c
        results.append((cs, chunk_mb, mean_c, bw_c, sp_c, marker))
        label = f"double  chunk={cs:>5} rows{marker}"
        print(f"  {label:<34} {chunk_mb:>6.1f}M  "
              f"{mean_c:8.4f}  {bw_c:9.3f}  {sp_c:>6.2f}x")

    print(f"{'-'*72}")
    print(f"  * = auto default (rows // 8 = {auto_rows})")

    _print_dbl_barchart(bw_b, bw_a, results, "auto (rows//8)*")

    best = max(results, key=lambda r: r[3])
    auto_r = next(r for r in results if r[0] == auto_rows)
    print(f"\n  Auto default (chunk={auto_rows}): "
          f"{auto_r[3]:.3f} GB/s  ({auto_r[4]:.2f}x)")
    print(f"  Best         (chunk={best[0]}): "
          f"{best[3]:.3f} GB/s  ({best[4]:.2f}x)")


# ---------------------------------------------------------------------------
# Section 4 / 5: 1-D dataset read (contiguous or HDF5-chunked)
# ---------------------------------------------------------------------------

def bench_1d_read(f, gpu_ds, data, dtype, hdf5_chunk, repeats, warmup):
    """Benchmark a 1-D dataset.

    Parameters
    ----------
    hdf5_chunk : int or None
        HDF5 chunk size (elements).  ``None`` means the dataset is contiguous.
    """
    n = data.size
    total_bytes = data.nbytes

    if hdf5_chunk is not None:
        auto_n = hdf5_chunk
        auto_label = f"auto (HDF5-aligned)*"
        auto_note  = f"* = aligned to HDF5 chunk ({hdf5_chunk} elements)"
    else:
        auto_n = max(1, n // 8)
        auto_label = f"auto (length//8)*"
        auto_note  = f"* = auto default (length // 8 = {auto_n})"

    mean_b, _, _ = _time_fn(lambda: gpu_ds[:], repeats, warmup)
    bw_b = _gb(total_bytes) / mean_b

    print(f"\n{'-'*72}")
    _print_dbl_header()
    print(f"  {'baseline (simple read)':<34} {'N/A':>7}  "
          f"{mean_b:8.4f}  {bw_b:9.3f}  {'1.00x':>7}")

    mean_a, _, _ = _time_fn(lambda: gpu_ds.read_double_buffered(), repeats, warmup)
    bw_a = _gb(total_bytes) / mean_a
    sp_a = mean_b / mean_a
    auto_mb = auto_n * dtype.itemsize / 1024**2
    print(f"  {f'double  {auto_label}':<34} {auto_mb:>6.2f}M"
          f"  {mean_a:8.4f}  {bw_a:9.3f}  {sp_a:>6.2f}x")

    results = []
    for cs in _chunk_sizes_1d(n, auto_n):
        chunk_mb = cs * dtype.itemsize / 1024**2
        marker = "*" if cs == auto_n else " "
        mean_c, _, _ = _time_fn(
            lambda c=cs: gpu_ds.read_double_buffered(chunk_size=c), repeats, warmup)
        bw_c = _gb(total_bytes) / mean_c
        sp_c = mean_b / mean_c
        results.append((cs, chunk_mb, mean_c, bw_c, sp_c, marker))
        label = f"double  chunk={cs:>7} elems{marker}"
        print(f"  {label:<34} {chunk_mb:>6.2f}M  "
              f"{mean_c:8.4f}  {bw_c:9.3f}  {sp_c:>6.2f}x")

    print(f"{'-'*72}")
    print(f"  {auto_note}")

    _print_dbl_barchart(bw_b, bw_a, results, auto_label)

    best = max(results, key=lambda r: r[3])
    auto_r = next(r for r in results if r[0] == auto_n)
    print(f"\n  Auto default (chunk={auto_n}): "
          f"{auto_r[3]:.3f} GB/s  ({auto_r[4]:.2f}x)")
    print(f"  Best         (chunk={best[0]}): "
          f"{best[3]:.3f} GB/s  ({best[4]:.2f}x)")


# ---------------------------------------------------------------------------
# Top-level runner
# ---------------------------------------------------------------------------

def run(rows, cols, dtype, hdf5_chunk_rows, hdf5_chunk_cols,
        length, hdf5_chunk_1d, repeats, warmup):
    dtype = np.dtype(dtype)
    data_2d   = np.random.rand(rows, cols).astype(dtype)
    data_1d   = np.random.rand(length).astype(dtype)
    hdf5_chunks = (hdf5_chunk_rows, hdf5_chunk_cols)

    print(f"\n{'='*72}")
    print(f"  h5py GPU read benchmark")
    print(f"  2-D dataset : ({rows}, {cols})  dtype={dtype}  "
          f"size={_gb(data_2d.nbytes):.3f} GB")
    print(f"  HDF5 chunks : {hdf5_chunks}  "
          f"({hdf5_chunk_rows * hdf5_chunk_cols * dtype.itemsize / 1024**2:.2f} MB/chunk)")
    print(f"  1-D dataset : ({length},)  dtype={dtype}  "
          f"size={_gb(data_1d.nbytes):.3f} GB")
    print(f"  HDF5 chunk  : {hdf5_chunk_1d} elements  "
          f"({hdf5_chunk_1d * dtype.itemsize / 1024**2:.2f} MB/chunk)")
    print(f"  repeats     : {repeats}   warmup : {warmup}")
    print(f"{'='*72}")

    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "bench.h5")
        with h5py.File(path, "w") as f:
            f.create_dataset("ds",            data=data_2d, chunks=hdf5_chunks)
            f.create_dataset("ds_contig_2d",  data=data_2d)
            f.create_dataset("ds_contig_1d",  data=data_1d)
            f.create_dataset("ds_chunked_1d", data=data_1d,
                             chunks=(hdf5_chunk_1d,))

        with h5py.File(path, "r") as f:

            # ── Section 1: 2-D chunked, full read ─────────────────────────
            print(f"\n{'='*72}")
            print(f"  SECTION 1: 2-D chunked dataset — full read "
                  f"(read_double_buffered sweep)")
            print(f"  shape={data_2d.shape}  chunks={hdf5_chunks}")
            bench_full_read(f, GPUDataset(f["ds"]), data_2d, dtype,
                            hdf5_chunks, repeats, warmup)

            # ── Section 2: 2-D chunked, selection read ─────────────────────
            print(f"\n{'='*72}")
            print(f"  SECTION 2: 2-D chunked dataset — selection read "
                  f"(read_selection_chunked)")
            print(f"\n  Columns:")
            print(f"    SEL MB  : bytes of data actually requested (useful payload)")
            print(f"    READ MB : bytes read from storage (whole chunks, may include waste)")
            print(f"    WASTE % : fraction of read bytes discarded after crop")
            print(f"    BASE BW : baseline GB/s  (h5py native + simple H2D)")
            print(f"    CHUNK BW: read_selection_chunked GB/s (full-chunk pipeline)")
            bench_coverage_sweep(f, GPUDataset(f["ds"]), data_2d, dtype,
                                 hdf5_chunks, repeats, warmup)
            bench_alignment_sweep(f, GPUDataset(f["ds"]), data_2d, dtype,
                                  hdf5_chunks, repeats, warmup)

            # ── Section 3: 2-D contiguous, full read ──────────────────────
            print(f"\n{'='*72}")
            print(f"  SECTION 3: 2-D contiguous (non-chunked) dataset — full read")
            print(f"  shape={data_2d.shape}  (contiguous)")
            bench_contiguous_2d_read(f, GPUDataset(f["ds_contig_2d"]),
                                     data_2d, dtype, repeats, warmup)

            # ── Section 4: 1-D contiguous, full read ──────────────────────
            print(f"\n{'='*72}")
            print(f"  SECTION 4: 1-D contiguous dataset — full read")
            print(f"  shape=({length},)  (contiguous)")
            bench_1d_read(f, GPUDataset(f["ds_contig_1d"]), data_1d, dtype,
                          hdf5_chunk=None, repeats=repeats, warmup=warmup)

            # ── Section 5: 1-D chunked, full read ─────────────────────────
            print(f"\n{'='*72}")
            print(f"  SECTION 5: 1-D chunked dataset — full read")
            print(f"  shape=({length},)  chunks=({hdf5_chunk_1d},)")
            bench_1d_read(f, GPUDataset(f["ds_chunked_1d"]), data_1d, dtype,
                          hdf5_chunk=hdf5_chunk_1d, repeats=repeats, warmup=warmup)

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
                   help="HDF5 chunk size along axis 0 (default: rows//16)")
    p.add_argument("--hdf5-chunk-cols", type=int, default=None,
                   help="HDF5 chunk size along axis 1 (default: cols//16)")
    p.add_argument("--length",          type=int, default=None,
                   help="Number of elements in the 1-D dataset "
                        "(default: rows * cols)")
    p.add_argument("--hdf5-chunk-1d",   type=int, default=None,
                   help="HDF5 chunk size for the 1-D dataset "
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
