"""
GPU compressed-read benchmark: read_chunks_compressed pipeline vs. h5py baseline.

Measures how much faster ``GPUDataset.read_chunks_compressed()`` is compared to
the standard h5py path (CPU decompression into pageable memory, then H2D copy)
across different compression codecs, levels, shuffle settings, and chunk sizes.

Usage
-----
    python benchmarks/bench_gpu_compressed.py [--rows N] [--cols N] [--dtype DTYPE]
                                              [--hdf5-chunk-rows N]
                                              [--hdf5-chunk-cols N]
                                              [--repeats N] [--warmup N]

Three benchmark sections:

  Section 1 -- gzip / deflate  (always available; always CPU decomp via zlib)
    (a) Level sweep  : clevel = 1, 4, 6, 9  (no shuffle)
    (b) Shuffle sweep: clevel = 4, without vs. with byte-shuffle
    Reported metrics per row: COMP_MB, RATIO, BASE_BW, RCC_BW, SPEEDUP

  Section 2 -- LZ4 and Zstd  (requires ``hdf5plugin``; GPU path via nvCOMP if installed)
    (a) LZ4 without shuffle and with shuffle
    (b) Zstd level sweep: clevel = 1, 3, 9, 19 (no shuffle)
    (c) Zstd shuffle sweep: clevel = 3, without vs. with byte-shuffle
    The DECOMP column shows whether CPU or GPU (nvCOMP) decompression was used.

  Section 3 -- HDF5 chunk-size sweep  (gzip clevel=4, no shuffle)
    Vary HDF5 chunk rows over a log-2 sweep from 1 to rows.
    Shows how chunk granularity affects compressed-pipeline throughput.

Column definitions
------------------
  COMP MB  : compressed bytes stored on disk (from HDF5 storage_size)
  RATIO    : uncompressed / compressed  (>1 means the file is smaller on disk)
  BASE BW  : GB/s with the h5py baseline (f["ds"][:] + _numpy_to_gpu)
  RCC BW   : GB/s with read_chunks_compressed (pinned-memory pipeline)
  SPEEDUP  : baseline_time / rcc_time  (>1 means rcc is faster)
  DECOMP   : decompression site: CPU (zlib/lz4/zstd lib) or GPU (nvCOMP)
"""

import argparse
import os
import sys
import tempfile
import time

import numpy as np

import h5py
from h5py.gpu import GPUDataset, _numpy_to_gpu

try:
    import cupy as cp
except ImportError:
    sys.exit("CuPy is required to run this benchmark.")

# ---------------------------------------------------------------------------
# Optional compression libraries
# ---------------------------------------------------------------------------

try:
    import hdf5plugin as _hdf5plugin
    _HDF5PLUGIN = True
except ImportError:
    _HDF5PLUGIN = False

try:
    import lz4.block  # noqa: F401
    _LZ4_CPU = True
except ImportError:
    _LZ4_CPU = False

try:
    import zstd  # noqa: F401
    _ZSTD_CPU = True
except ImportError:
    _ZSTD_CPU = False

try:
    from nvidia import nvcomp  # noqa: F401
    _NVCOMP = True
except ImportError:
    _NVCOMP = False

_LZ4_AVAIL  = _HDF5PLUGIN and (_LZ4_CPU or _NVCOMP)
_ZSTD_AVAIL = _HDF5PLUGIN and (_ZSTD_CPU or _NVCOMP)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _flush_gpu():
    cp.cuda.runtime.deviceSynchronize()


def _time_fn(fn, repeats, warmup):
    """Return (mean_s, min_s, times_list) for *fn*."""
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


def _storage_mb(dataset):
    """Compressed bytes on disk in MB, via HDF5 storage_size API."""
    try:
        return _mb(dataset.id.get_storage_size())
    except AttributeError:
        return float("nan")


def _bar(value, max_value, width=24):
    filled = int(round(value / max_value * width)) if max_value > 0 else 0
    return "#" * filled + "." * (width - filled)


def _infer_decomp_path(filter_name):
    """Return 'GPU' when nvCOMP will handle decompression, else 'CPU'."""
    if filter_name in ("lz4", "zstd") and _NVCOMP:
        return "GPU"
    return "CPU"


def _chunk_sizes(n_rows, ref_chunk):
    """Log-2 sweep of row counts, always including ref_chunk."""
    sizes = set()
    k = 1
    while k <= n_rows:
        sizes.add(k)
        k *= 2
    sizes.add(ref_chunk)
    return sorted(sizes)


# ---------------------------------------------------------------------------
# Core measurement
# ---------------------------------------------------------------------------

def _bench_one(dataset, gpu_ds, uncompressed_bytes, repeats, warmup):
    """Time baseline and read_chunks_compressed; return a metrics dict."""
    comp_mb = _storage_mb(dataset)
    if comp_mb > 0 and not np.isnan(comp_mb):
        ratio = uncompressed_bytes / (comp_mb * 1024**2)
    else:
        ratio = float("nan")

    mean_b, _, _ = _time_fn(lambda: _numpy_to_gpu(dataset[:]), repeats, warmup)
    bw_b = _gb(uncompressed_bytes) / mean_b

    mean_c, _, _ = _time_fn(lambda: gpu_ds.read_chunks_compressed(), repeats, warmup)
    bw_c = _gb(uncompressed_bytes) / mean_c

    return dict(
        comp_mb=comp_mb,
        ratio=ratio,
        bw_b=bw_b,
        bw_c=bw_c,
        mean_b=mean_b,
        mean_c=mean_c,
        speedup=mean_b / mean_c,
    )


# ---------------------------------------------------------------------------
# Print helpers
# ---------------------------------------------------------------------------

def _print_header():
    print(f"\n  {'CODEC':<17} {'SHUF':>4}  {'COMP MB':>7}  {'RATIO':>5}  "
          f"{'BASE BW':>7}  {'RCC BW':>7}  {'SPEEDUP':>7}  {'DECOMP':>7}")
    print(f"  {'-'*17}  {'-'*4}  {'-'*7}  {'-'*5}  "
          f"{'-'*7}  {'-'*7}  {'-'*7}  {'-'*7}")


def _print_row(label, shuffle, m, decomp_path):
    shuf_str = "yes" if shuffle else "no"
    ratio_str = f"{m['ratio']:.2f}" if not np.isnan(m["ratio"]) else "  n/a"
    print(f"  {label:<17}  {shuf_str:>4}  {m['comp_mb']:>7.1f}  {ratio_str:>5}  "
          f"{m['bw_b']:>7.3f}  {m['bw_c']:>7.3f}  {m['speedup']:>6.2f}x  "
          f"{decomp_path:>7}")


def _print_barchart(rows, title="RCC BW (GB/s)"):
    """rows: list of (label, bw) tuples."""
    max_bw = max(bw for _, bw in rows) if rows else 1.0
    print(f"\n  {title}  (each # ~= {max_bw/24:.2f} GB/s)\n")
    for label, bw in rows:
        print(f"  {label:<32}  {_bar(bw, max_bw)}  {bw:.3f} GB/s")


# ---------------------------------------------------------------------------
# Dataset creation helpers
# ---------------------------------------------------------------------------

def _create_gzip_datasets(wf, data, hdf5_chunks):
    """Write all Section 1 datasets into an open writable HDF5 file."""
    for lvl in (1, 4, 6, 9):
        wf.create_dataset(f"gzip_lvl{lvl}", data=data, chunks=hdf5_chunks,
                          compression="gzip", compression_opts=lvl)
    wf.create_dataset("gzip4_noshuffle", data=data, chunks=hdf5_chunks,
                      compression="gzip", compression_opts=4, shuffle=False)
    wf.create_dataset("gzip4_shuffle", data=data, chunks=hdf5_chunks,
                      compression="gzip", compression_opts=4, shuffle=True)


def _create_lz4_datasets(wf, data, hdf5_chunks):
    """Write Section 2a datasets (LZ4).  Requires hdf5plugin."""
    wf.create_dataset("lz4_noshuffle", data=data, chunks=hdf5_chunks,
                      **_hdf5plugin.LZ4())
    wf.create_dataset("lz4_shuffle", data=data, chunks=hdf5_chunks,
                      shuffle=True, **_hdf5plugin.LZ4())


def _create_zstd_datasets(wf, data, hdf5_chunks):
    """Write Section 2b/2c datasets (Zstd).  Requires hdf5plugin."""
    for lvl in (1, 3, 9, 19):
        wf.create_dataset(f"zstd_lvl{lvl}", data=data, chunks=hdf5_chunks,
                          **_hdf5plugin.Zstd(clevel=lvl))
    wf.create_dataset("zstd3_noshuffle", data=data, chunks=hdf5_chunks,
                      **_hdf5plugin.Zstd(clevel=3))
    wf.create_dataset("zstd3_shuffle", data=data, chunks=hdf5_chunks,
                      shuffle=True, **_hdf5plugin.Zstd(clevel=3))


def _create_sweep_datasets(wf, data, cols, hdf5_chunk_rows):
    """Write Section 3 datasets (gzip-4, varying chunk rows)."""
    rows = data.shape[0]
    for cs in _chunk_sizes(rows, hdf5_chunk_rows):
        wf.create_dataset(f"sweep_chunk{cs}", data=data, chunks=(cs, cols),
                          compression="gzip", compression_opts=4)


# ---------------------------------------------------------------------------
# Section 1: gzip / deflate
# ---------------------------------------------------------------------------

def bench_gzip(rf, data, dtype, hdf5_chunks, repeats, warmup):
    uncompressed_bytes = data.nbytes

    print(f"\n  shape={data.shape}  dtype={dtype}  chunks={hdf5_chunks}  "
          f"uncompressed={_mb(uncompressed_bytes):.1f} MB")

    # ── 1a: level sweep ──────────────────────────────────────────────────────
    print(f"\n  (a) Compression-level sweep (no shuffle)")
    _print_header()
    barchart = []
    for lvl in (1, 4, 6, 9):
        m = _bench_one(rf[f"gzip_lvl{lvl}"], GPUDataset(rf[f"gzip_lvl{lvl}"]),
                       uncompressed_bytes, repeats, warmup)
        label = f"gzip-{lvl}"
        _print_row(label, shuffle=False, m=m, decomp_path="CPU")
        barchart.append((label, m["bw_c"]))

    # ── 1b: shuffle sweep ────────────────────────────────────────────────────
    print(f"\n  (b) Shuffle sweep (gzip level=4)")
    _print_header()
    for ds_key, shuffle, label in [
        ("gzip4_noshuffle", False, "gzip-4"),
        ("gzip4_shuffle",   True,  "gzip-4+shuffle"),
    ]:
        m = _bench_one(rf[ds_key], GPUDataset(rf[ds_key]),
                       uncompressed_bytes, repeats, warmup)
        _print_row(label, shuffle=shuffle, m=m, decomp_path="CPU")
        barchart.append((label + (" +shuf" if shuffle else "      "), m["bw_c"]))

    _print_barchart(barchart, title="Section 1 — RCC BW (GB/s)")


# ---------------------------------------------------------------------------
# Section 2: LZ4 and Zstd
# ---------------------------------------------------------------------------

def bench_lz4_zstd(rf, data, dtype, hdf5_chunks, repeats, warmup):
    uncompressed_bytes = data.nbytes
    barchart = []

    # ── 2a: LZ4 ─────────────────────────────────────────────────────────────
    if _LZ4_AVAIL:
        print(f"\n  (a) LZ4 (without and with byte-shuffle)")
        _print_header()
        decomp = _infer_decomp_path("lz4")
        for ds_key, shuffle, label in [
            ("lz4_noshuffle", False, "lz4"),
            ("lz4_shuffle",   True,  "lz4+shuffle"),
        ]:
            m = _bench_one(rf[ds_key], GPUDataset(rf[ds_key]),
                           uncompressed_bytes, repeats, warmup)
            _print_row(label, shuffle=shuffle, m=m, decomp_path=decomp)
            barchart.append((label + (" +shuf" if shuffle else "      "), m["bw_c"]))
    else:
        print(f"\n  (a) LZ4  [SKIPPED — install lz4 or nvidia-nvcomp to enable]")

    # ── 2b: Zstd level sweep ─────────────────────────────────────────────────
    if _ZSTD_AVAIL:
        print(f"\n  (b) Zstd level sweep (no shuffle)")
        _print_header()
        decomp = _infer_decomp_path("zstd")
        for lvl in (1, 3, 9, 19):
            m = _bench_one(rf[f"zstd_lvl{lvl}"], GPUDataset(rf[f"zstd_lvl{lvl}"]),
                           uncompressed_bytes, repeats, warmup)
            label = f"zstd-{lvl}"
            _print_row(label, shuffle=False, m=m, decomp_path=decomp)
            barchart.append((label, m["bw_c"]))

        # ── 2c: Zstd shuffle sweep ────────────────────────────────────────────
        print(f"\n  (c) Zstd shuffle sweep (level=3)")
        _print_header()
        for ds_key, shuffle, label in [
            ("zstd3_noshuffle", False, "zstd-3"),
            ("zstd3_shuffle",   True,  "zstd-3+shuffle"),
        ]:
            m = _bench_one(rf[ds_key], GPUDataset(rf[ds_key]),
                           uncompressed_bytes, repeats, warmup)
            _print_row(label, shuffle=shuffle, m=m, decomp_path=decomp)
            barchart.append((label + (" +shuf" if shuffle else "      "), m["bw_c"]))
    else:
        print(f"\n  (b)(c) Zstd  [SKIPPED — install zstd or nvidia-nvcomp to enable]")

    if barchart:
        _print_barchart(barchart, title="Section 2 — RCC BW (GB/s)")


# ---------------------------------------------------------------------------
# Section 3: HDF5 chunk-size sweep (gzip-4)
# ---------------------------------------------------------------------------

def bench_chunk_sweep(rf, data, dtype, hdf5_chunk_rows, repeats, warmup):
    rows, cols = data.shape
    uncompressed_bytes = data.nbytes
    sizes = _chunk_sizes(rows, hdf5_chunk_rows)

    print(f"\n  shape={data.shape}  dtype={dtype}  codec=gzip-4  no shuffle")
    _print_header()

    barchart_b = []
    barchart_c = []
    for cs in sizes:
        ds_key = f"sweep_chunk{cs}"
        m = _bench_one(rf[ds_key], GPUDataset(rf[ds_key]),
                       uncompressed_bytes, repeats, warmup)
        marker = "*" if cs == hdf5_chunk_rows else " "
        label = f"chunk={cs}{marker}"
        _print_row(label, shuffle=False, m=m, decomp_path="CPU")
        barchart_b.append((f"base  chunk={cs}{marker}", m["bw_b"]))
        barchart_c.append((f"rcc   chunk={cs}{marker}", m["bw_c"]))

    print(f"\n  * = default chunk rows ({hdf5_chunk_rows})")

    _print_barchart(barchart_b + barchart_c,
                    title="Section 3 — BASE and RCC BW (GB/s)")


# ---------------------------------------------------------------------------
# Top-level runner
# ---------------------------------------------------------------------------

def run(rows, cols, dtype, hdf5_chunk_rows, hdf5_chunk_cols, repeats, warmup):
    dtype = np.dtype(dtype)
    rng = np.random.default_rng(42)
    data = rng.random((rows, cols)).astype(dtype)
    hdf5_chunks = (hdf5_chunk_rows, hdf5_chunk_cols)
    cols_ = cols  # alias for inner helpers

    # ── Environment banner ───────────────────────────────────────────────────
    print(f"\n{'='*74}")
    print(f"  h5py GPU compressed-read benchmark")
    print(f"  dataset     : ({rows}, {cols})  dtype={dtype}  "
          f"uncompressed={_mb(data.nbytes):.1f} MB")
    print(f"  HDF5 chunks : {hdf5_chunks}  "
          f"({hdf5_chunk_rows * hdf5_chunk_cols * dtype.itemsize / 1024**2:.2f} MB/chunk)")
    print(f"  repeats     : {repeats}   warmup : {warmup}")
    print(f"  hdf5plugin  : {'yes' if _HDF5PLUGIN else 'NO  (pip install hdf5plugin)'}")
    print(f"  lz4 (CPU)   : {'yes' if _LZ4_CPU   else 'NO  (pip install lz4)'}")
    print(f"  zstd (CPU)  : {'yes' if _ZSTD_CPU  else 'NO  (pip install zstd)'}")
    nvcomp_note = ("yes — LZ4/Zstd will use GPU decompression"
                   if _NVCOMP else "NO  (pip install nvidia-nvcomp-cu12 "
                                   "--extra-index-url https://pypi.nvidia.com)")
    print(f"  nvCOMP(GPU) : {nvcomp_note}")
    print(f"{'='*74}")

    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "bench_compressed.h5")

        # ── Write all datasets (one pass) ────────────────────────────────────
        print(f"\n  Writing datasets ...", end="", flush=True)
        with h5py.File(path, "w") as wf:
            _create_gzip_datasets(wf, data, hdf5_chunks)
            if _LZ4_AVAIL:
                _create_lz4_datasets(wf, data, hdf5_chunks)
            if _ZSTD_AVAIL:
                _create_zstd_datasets(wf, data, hdf5_chunks)
            _create_sweep_datasets(wf, data, cols_, hdf5_chunk_rows)
        print(f" done.")

        # ── Section 1: gzip ─────────────────────────────────────────────────
        print(f"\n{'='*74}")
        print(f"  SECTION 1: gzip / deflate  (CPU decompression via zlib)")
        with h5py.File(path, "r") as rf:
            bench_gzip(rf, data, dtype, hdf5_chunks, repeats, warmup)

        # ── Section 2: LZ4 + Zstd ───────────────────────────────────────────
        print(f"\n{'='*74}")
        if _LZ4_AVAIL or _ZSTD_AVAIL:
            decomp_note = "GPU (nvCOMP)" if _NVCOMP else "CPU"
            print(f"  SECTION 2: LZ4 and Zstd  (decompression path: {decomp_note})")
            with h5py.File(path, "r") as rf:
                bench_lz4_zstd(rf, data, dtype, hdf5_chunks, repeats, warmup)
        else:
            print(f"  SECTION 2: LZ4 and Zstd  [SKIPPED]")
            print(f"    Requires hdf5plugin plus at least one of: lz4, zstd, nvidia-nvcomp.")
            print(f"    pip install hdf5plugin lz4 zstd")

        # ── Section 3: chunk-size sweep ──────────────────────────────────────
        print(f"\n{'='*74}")
        print(f"  SECTION 3: HDF5 chunk-size sweep  (gzip level=4, no shuffle)")
        with h5py.File(path, "r") as rf:
            bench_chunk_sweep(rf, data, dtype, hdf5_chunk_rows, repeats, warmup)

    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--rows",            type=int, default=4096)
    p.add_argument("--cols",            type=int, default=4096)
    p.add_argument("--dtype",           type=str, default="float32")
    p.add_argument("--hdf5-chunk-rows", type=int, default=None,
                   help="HDF5 chunk size along axis 0 (default: rows // 16)")
    p.add_argument("--hdf5-chunk-cols", type=int, default=None,
                   help="HDF5 chunk size along axis 1 (default: cols // 16)")
    p.add_argument("--repeats",         type=int, default=5)
    p.add_argument("--warmup",          type=int, default=2)
    args = p.parse_args()

    hdf5_chunk_rows = args.hdf5_chunk_rows or max(1, args.rows // 16)
    hdf5_chunk_cols = args.hdf5_chunk_cols or max(1, args.cols // 16)

    run(args.rows, args.cols, args.dtype,
        hdf5_chunk_rows, hdf5_chunk_cols,
        args.repeats, args.warmup)
