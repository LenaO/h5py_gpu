"""
GPU compressed-read benchmark: read_chunks_compressed pipeline vs. h5py baseline.

Measures how much faster ``GPUDataset.read_chunks_compressed()`` is compared to
the standard h5py path (CPU decompression into pageable memory, then H2D copy)
across different compression codecs, levels, shuffle settings, and chunk sizes.

Five synthetic int32 datasets are used to cover the full compressibility spectrum:

  constant    all elements = 42                       (trivially compressible)
  small_range uniform int32 in [0, 15]               (good  — 4 bits effective)
  sparse      5 % nonzero, nonzero values in [1,100]  (good  — long zero-runs)
  random_u8   uniform int32 in [0, 255]               (medium)
  random_i32  full int32 range                        (poor  — high entropy)

Using int32 throughout makes the compression ratio differences obvious and
exercises byte-shuffle (which reorders bytes within each element).

Usage
-----
    python benchmarks/bench_gpu_compressed.py [--rows N] [--cols N]
                                              [--hdf5-chunk-rows N]
                                              [--hdf5-chunk-cols N]
                                              [--repeats N] [--warmup N]

Three benchmark sections:

  Section 1 -- gzip / deflate  (always available; always CPU decomp via zlib)
    For each of the 5 data types:
    (a) Level sweep  : clevel = 1, 4, 6, 9  (no shuffle)
    (b) Shuffle sweep: clevel = 4, without vs. with byte-shuffle
    Reported metrics per row: COMP_MB, RATIO, BASE_BW, RCC_BW, SPEEDUP

  Section 2 -- LZ4 and Zstd  (requires ``hdf5plugin``; GPU path via nvCOMP)
    For each of the 5 data types:
    (a) LZ4 without shuffle and with shuffle
    (b) Zstd level sweep: clevel = 1, 3, 9, 19 (no shuffle)
    (c) Zstd shuffle sweep: clevel = 3, without vs. with byte-shuffle

  Section 3 -- HDF5 chunk-size sweep  (gzip clevel=4, no shuffle)
    Run for two representative datasets: small_range (good compression)
    and random_i32 (poor compression).  Shows how chunk granularity affects
    throughput differently depending on compressibility.

Column definitions
------------------
  COMP MB  : compressed bytes stored on disk (from HDF5 storage_size)
  RATIO    : uncompressed / compressed  (>1 means the file is smaller on disk)
  BASE BW  : GB/s with the h5py baseline (f["ds"][:] + H2D copy)
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
# Synthetic dataset generators
# ---------------------------------------------------------------------------

def _make_datasets(rows, cols, rng):
    """Return list of (name, description, int32 array) covering compressibility spectrum."""
    shape = (rows, cols)

    # sparse: 5 % nonzero
    sparse = np.zeros(shape, dtype=np.int32)
    mask   = rng.random(shape) < 0.05
    sparse[mask] = rng.integers(1, 101, int(mask.sum()), dtype=np.int32)

    return [
        ("constant",    "all value 42",
         np.full(shape, 42, dtype=np.int32)),

        ("small_range", "uniform [0, 15]",
         rng.integers(0, 16, shape, dtype=np.int32)),

        ("sparse",      "5% nonzero, values [1, 100]",
         sparse),

        ("random_u8",   "uniform [0, 255]",
         rng.integers(0, 256, shape, dtype=np.int32)),

        ("random_i32",  "full int32 range",
         rng.integers(np.iinfo(np.int32).min, np.iinfo(np.int32).max,
                      shape, dtype=np.int32)),
    ]


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _flush_gpu():
    cp.cuda.runtime.deviceSynchronize()


def _time_fn(fn, repeats, warmup):
    for _ in range(warmup):
        fn(); _flush_gpu()
    times = []
    for _ in range(repeats):
        _flush_gpu()
        t0 = time.perf_counter()
        fn(); _flush_gpu()
        times.append(time.perf_counter() - t0)
    return float(np.mean(times)), float(np.min(times)), times


def _gb(nbytes):
    return nbytes / 1024**3


def _mb(nbytes):
    return nbytes / 1024**2


def _storage_mb(dataset):
    try:
        return _mb(dataset.id.get_storage_size())
    except AttributeError:
        return float("nan")


def _bar(value, max_value, width=24):
    filled = int(round(value / max_value * width)) if max_value > 0 else 0
    return "#" * filled + "." * (width - filled)


def _infer_decomp_path(filter_name):
    if filter_name in ("lz4", "zstd") and _NVCOMP:
        return "GPU"
    return "CPU"


_SWEEP_NAMES = {"small_range", "random_i32"}


def _chunk_sizes(n_rows, ref_chunk):
    """Log-2 sweep starting from n_rows//32 to avoid tiny-chunk write overhead."""
    sizes = set()
    k = max(1, n_rows >> 5)
    while k <= n_rows:
        sizes.add(k)
        k *= 2
    sizes.add(ref_chunk)
    return sorted(sizes)


# ---------------------------------------------------------------------------
# Core measurement
# ---------------------------------------------------------------------------

def _bench_one(dataset, gpu_ds, uncompressed_bytes, repeats, warmup):
    comp_mb = _storage_mb(dataset)
    ratio   = (uncompressed_bytes / (comp_mb * 1024**2)
               if comp_mb > 0 and not np.isnan(comp_mb) else float("nan"))

    mean_b, _, _ = _time_fn(lambda: _numpy_to_gpu(dataset[:]), repeats, warmup)
    bw_b = _gb(uncompressed_bytes) / mean_b

    mean_c, _, _ = _time_fn(lambda: gpu_ds.read_chunks_compressed(), repeats, warmup)
    bw_c = _gb(uncompressed_bytes) / mean_c

    return dict(comp_mb=comp_mb, ratio=ratio,
                bw_b=bw_b, bw_c=bw_c,
                mean_b=mean_b, mean_c=mean_c,
                speedup=mean_b / mean_c)


# ---------------------------------------------------------------------------
# Print helpers
# ---------------------------------------------------------------------------

def _print_header():
    print(f"\n  {'CODEC':<17} {'SHUF':>4}  {'COMP MB':>7}  {'RATIO':>5}  "
          f"{'BASE BW':>7}  {'RCC BW':>7}  {'SPEEDUP':>7}  {'DECOMP':>7}")
    print(f"  {'-'*17}  {'-'*4}  {'-'*7}  {'-'*5}  "
          f"{'-'*7}  {'-'*7}  {'-'*7}  {'-'*7}")


def _print_row(label, shuffle, m, decomp_path):
    shuf_str  = "yes" if shuffle else "no"
    ratio_str = f"{m['ratio']:.2f}" if not np.isnan(m["ratio"]) else "  n/a"
    print(f"  {label:<17}  {shuf_str:>4}  {m['comp_mb']:>7.1f}  {ratio_str:>5}  "
          f"{m['bw_b']:>7.3f}  {m['bw_c']:>7.3f}  {m['speedup']:>6.2f}x  "
          f"{decomp_path:>7}")


def _ds_header(name, desc, data, hdf5_chunks):
    unc_mb = _mb(data.nbytes)
    print(f"\n  ── {name}  ({desc})  "
          f"uncompressed={unc_mb:.0f} MB  chunks={hdf5_chunks} ──")


def _print_barchart(rows, title="RCC BW (GB/s)"):
    max_bw = max(bw for _, bw in rows) if rows else 1.0
    print(f"\n  {title}  (each # ~= {max_bw/24:.2f} GB/s)\n")
    for label, bw in rows:
        print(f"  {label:<36}  {_bar(bw, max_bw)}  {bw:.3f} GB/s")


# ---------------------------------------------------------------------------
# Dataset creation helpers  (prefix isolates each data type in the HDF5 file)
# ---------------------------------------------------------------------------

def _create_gzip_datasets(wf, data, hdf5_chunks, prefix):
    for lvl in (1, 4, 6, 9):
        wf.create_dataset(f"{prefix}/gzip_lvl{lvl}", data=data, chunks=hdf5_chunks,
                          compression="gzip", compression_opts=lvl)
    wf.create_dataset(f"{prefix}/gzip4_noshuffle", data=data, chunks=hdf5_chunks,
                      compression="gzip", compression_opts=4, shuffle=False)
    wf.create_dataset(f"{prefix}/gzip4_shuffle",   data=data, chunks=hdf5_chunks,
                      compression="gzip", compression_opts=4, shuffle=True)


def _create_lz4_datasets(wf, data, hdf5_chunks, prefix):
    wf.create_dataset(f"{prefix}/lz4_noshuffle", data=data, chunks=hdf5_chunks,
                      **_hdf5plugin.LZ4())
    wf.create_dataset(f"{prefix}/lz4_shuffle",   data=data, chunks=hdf5_chunks,
                      shuffle=True, **_hdf5plugin.LZ4())


def _create_zstd_datasets(wf, data, hdf5_chunks, prefix):
    for lvl in (1, 3, 9, 19):
        wf.create_dataset(f"{prefix}/zstd_lvl{lvl}", data=data, chunks=hdf5_chunks,
                          **_hdf5plugin.Zstd(clevel=lvl))
    wf.create_dataset(f"{prefix}/zstd3_noshuffle", data=data, chunks=hdf5_chunks,
                      **_hdf5plugin.Zstd(clevel=3))
    wf.create_dataset(f"{prefix}/zstd3_shuffle",   data=data, chunks=hdf5_chunks,
                      shuffle=True, **_hdf5plugin.Zstd(clevel=3))


def _create_sweep_datasets(wf, data, hdf5_chunk_rows, prefix):
    rows, cols = data.shape
    for cs in _chunk_sizes(rows, hdf5_chunk_rows):
        wf.create_dataset(f"{prefix}/sweep_chunk{cs}", data=data,
                          chunks=(cs, cols), compression="gzip", compression_opts=4)


# ---------------------------------------------------------------------------
# Section 1: gzip / deflate
# ---------------------------------------------------------------------------

def bench_gzip(rf, datasets, hdf5_chunks, repeats, warmup):
    all_barchart = []

    for name, desc, data in datasets:
        uncompressed_bytes = data.nbytes
        _ds_header(name, desc, data, hdf5_chunks)

        # (a) level sweep
        print(f"\n  (a) Level sweep  (no shuffle)")
        _print_header()
        for lvl in (1, 4, 6, 9):
            key = f"{name}/gzip_lvl{lvl}"
            m   = _bench_one(rf[key], GPUDataset(rf[key]),
                             uncompressed_bytes, repeats, warmup)
            label = f"gzip-{lvl}"
            _print_row(label, shuffle=False, m=m, decomp_path="CPU")
            all_barchart.append((f"{name:<12} gzip-{lvl}", m["bw_c"]))

        # (b) shuffle sweep
        print(f"\n  (b) Shuffle sweep  (gzip level=4)")
        _print_header()
        for key_sfx, shuffle, label in [
            ("gzip4_noshuffle", False, "gzip-4"),
            ("gzip4_shuffle",   True,  "gzip-4+shuffle"),
        ]:
            key = f"{name}/{key_sfx}"
            m   = _bench_one(rf[key], GPUDataset(rf[key]),
                             uncompressed_bytes, repeats, warmup)
            _print_row(label, shuffle=shuffle, m=m, decomp_path="CPU")
            suffix = "+shuf" if shuffle else "     "
            all_barchart.append((f"{name:<12} gzip-4{suffix}", m["bw_c"]))

    _print_barchart(all_barchart, title="Section 1 — RCC BW (GB/s)")


# ---------------------------------------------------------------------------
# Section 2: LZ4 and Zstd
# ---------------------------------------------------------------------------

def bench_lz4_zstd(rf, datasets, repeats, warmup):
    all_barchart = []

    for name, desc, data in datasets:
        uncompressed_bytes = data.nbytes
        _ds_header(name, desc, data, hdf5_chunks=None)

        # (a) LZ4
        if _LZ4_AVAIL:
            print(f"\n  (a) LZ4")
            _print_header()
            decomp = _infer_decomp_path("lz4")
            for key_sfx, shuffle, label in [
                ("lz4_noshuffle", False, "lz4"),
                ("lz4_shuffle",   True,  "lz4+shuffle"),
            ]:
                key = f"{name}/{key_sfx}"
                m   = _bench_one(rf[key], GPUDataset(rf[key]),
                                 uncompressed_bytes, repeats, warmup)
                _print_row(label, shuffle=shuffle, m=m, decomp_path=decomp)
                suffix = "+shuf" if shuffle else "     "
                all_barchart.append((f"{name:<12} lz4{suffix}", m["bw_c"]))
        else:
            print(f"\n  (a) LZ4  [SKIPPED]")

        # (b) Zstd level sweep
        if _ZSTD_AVAIL:
            print(f"\n  (b) Zstd level sweep  (no shuffle)")
            _print_header()
            decomp = _infer_decomp_path("zstd")
            for lvl in (1, 3, 9, 19):
                key = f"{name}/zstd_lvl{lvl}"
                m   = _bench_one(rf[key], GPUDataset(rf[key]),
                                 uncompressed_bytes, repeats, warmup)
                label = f"zstd-{lvl}"
                _print_row(label, shuffle=False, m=m, decomp_path=decomp)
                all_barchart.append((f"{name:<12} zstd-{lvl}", m["bw_c"]))

            # (c) Zstd shuffle sweep
            print(f"\n  (c) Zstd shuffle sweep  (level=3)")
            _print_header()
            for key_sfx, shuffle, label in [
                ("zstd3_noshuffle", False, "zstd-3"),
                ("zstd3_shuffle",   True,  "zstd-3+shuffle"),
            ]:
                key = f"{name}/{key_sfx}"
                m   = _bench_one(rf[key], GPUDataset(rf[key]),
                                 uncompressed_bytes, repeats, warmup)
                _print_row(label, shuffle=shuffle, m=m, decomp_path=decomp)
                suffix = "+shuf" if shuffle else "     "
                all_barchart.append((f"{name:<12} zstd-3{suffix}", m["bw_c"]))
        else:
            print(f"\n  (b)(c) Zstd  [SKIPPED]")

    if all_barchart:
        _print_barchart(all_barchart, title="Section 2 — RCC BW (GB/s)")


# ---------------------------------------------------------------------------
# Section 3: chunk-size sweep (gzip-4, two representative datasets)
# ---------------------------------------------------------------------------

def bench_chunk_sweep(rf, datasets, hdf5_chunk_rows, repeats, warmup):
    # Run only for best and worst compressibility to keep output manageable
    sweep_names = {"small_range", "random_i32"}
    sweep_data  = [(n, d, a) for n, d, a in datasets if n in sweep_names]

    for name, desc, data in sweep_data:
        rows, cols         = data.shape
        uncompressed_bytes = data.nbytes
        sizes              = _chunk_sizes(rows, hdf5_chunk_rows)

        print(f"\n  ── {name}  ({desc})  codec=gzip-4  no shuffle ──")
        _print_header()

        barchart_b = []
        barchart_c = []
        for cs in sizes:
            key = f"{name}/sweep_chunk{cs}"
            m   = _bench_one(rf[key], GPUDataset(rf[key]),
                             uncompressed_bytes, repeats, warmup)
            marker = "*" if cs == hdf5_chunk_rows else " "
            label  = f"chunk={cs}{marker}"
            _print_row(label, shuffle=False, m=m, decomp_path="CPU")
            barchart_b.append((f"{name}  base  chunk={cs}{marker}", m["bw_b"]))
            barchart_c.append((f"{name}  rcc   chunk={cs}{marker}", m["bw_c"]))

        print(f"  * = default chunk rows ({hdf5_chunk_rows})")
        _print_barchart(barchart_b + barchart_c,
                        title=f"Section 3 / {name} — BASE and RCC BW (GB/s)")


# ---------------------------------------------------------------------------
# Top-level runner
# ---------------------------------------------------------------------------

def run(rows, cols, hdf5_chunk_rows, hdf5_chunk_cols, repeats, warmup, tmp_dir=None):
    rng         = np.random.default_rng(42)
    datasets    = _make_datasets(rows, cols, rng)
    hdf5_chunks = (hdf5_chunk_rows, hdf5_chunk_cols)
    unc_mb      = rows * cols * np.dtype(np.int32).itemsize / 1024**2

    # Banner
    print(f"\n{'='*74}")
    print(f"  h5py GPU compressed-read benchmark")
    print(f"  shape       : ({rows}, {cols})  dtype=int32  "
          f"uncompressed={unc_mb:.1f} MB per dataset")
    print(f"  HDF5 chunks : {hdf5_chunks}  "
          f"({hdf5_chunk_rows * hdf5_chunk_cols * 4 / 1024**2:.2f} MB/chunk)")
    print(f"  repeats     : {repeats}   warmup : {warmup}")
    print(f"  tmp dir     : {tmp_dir or '(system default)' }")
    print(f"  hdf5plugin  : {'yes' if _HDF5PLUGIN else 'NO  (pip install hdf5plugin)'}")
    print(f"  lz4 (CPU)   : {'yes' if _LZ4_CPU   else 'NO  (pip install lz4)'}")
    print(f"  zstd (CPU)  : {'yes' if _ZSTD_CPU  else 'NO  (pip install zstd)'}")
    nvcomp_note = ("yes — LZ4/Zstd will use GPU decompression"
                   if _NVCOMP else "NO  (pip install nvidia-nvcomp-cu12 "
                                   "--extra-index-url https://pypi.nvidia.com)")
    print(f"  nvCOMP(GPU) : {nvcomp_note}")
    print(f"\n  Datasets:")
    for name, desc, arr in datasets:
        print(f"    {name:<14}  {desc}")
    print(f"{'='*74}")

    with tempfile.TemporaryDirectory(dir=tmp_dir) as td:
        path = os.path.join(td, "bench_compressed.h5")

        # Write all datasets (one pass)
        print(f"\n  Writing datasets ...", end="", flush=True)
        with h5py.File(path, "w") as wf:
            for name, _, data in datasets:
                _create_gzip_datasets(wf, data, hdf5_chunks, name)
                if _LZ4_AVAIL:
                    _create_lz4_datasets(wf, data, hdf5_chunks, name)
                if _ZSTD_AVAIL:
                    _create_zstd_datasets(wf, data, hdf5_chunks, name)
                if name in _SWEEP_NAMES:   # only write sweep data where it's used
                    _create_sweep_datasets(wf, data, hdf5_chunk_rows, name)
                print(".", end="", flush=True)
        print(" done.")

        # Section 1: gzip
        print(f"\n{'='*74}")
        print(f"  SECTION 1: gzip / deflate  (CPU decompression via zlib)")
        with h5py.File(path, "r") as rf:
            bench_gzip(rf, datasets, hdf5_chunks, repeats, warmup)

        # Section 2: LZ4 + Zstd
        print(f"\n{'='*74}")
        if _LZ4_AVAIL or _ZSTD_AVAIL:
            decomp_note = "GPU (nvCOMP)" if _NVCOMP else "CPU"
            print(f"  SECTION 2: LZ4 and Zstd  (decompression path: {decomp_note})")
            with h5py.File(path, "r") as rf:
                bench_lz4_zstd(rf, datasets, repeats, warmup)
        else:
            print(f"  SECTION 2: LZ4 and Zstd  [SKIPPED]")
            print(f"    pip install hdf5plugin lz4 zstd")

        # Section 3: chunk-size sweep (small_range + random_i32 only)
        print(f"\n{'='*74}")
        print(f"  SECTION 3: chunk-size sweep  (gzip level=4, no shuffle)")
        print(f"  Datasets: small_range (good compression) and "
              f"random_i32 (poor compression)")
        with h5py.File(path, "r") as rf:
            bench_chunk_sweep(rf, datasets, hdf5_chunk_rows, repeats, warmup)

    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--rows",            type=int, default=2048)
    p.add_argument("--cols",            type=int, default=2048)
    p.add_argument("--hdf5-chunk-rows", type=int, default=None,
                   help="HDF5 chunk rows for 2-D datasets (default: rows // 16)")
    p.add_argument("--hdf5-chunk-cols", type=int, default=None,
                   help="HDF5 chunk cols for 2-D datasets (default: cols // 16)")
    p.add_argument("--tmp-dir",          type=str,  default=None,
                   help="Directory for the temporary HDF5 file "
                        "(default: system temp). "
                        "Use to benchmark network or non-default filesystems.")
    p.add_argument("--repeats",         type=int, default=5)
    p.add_argument("--warmup",          type=int, default=2)
    args = p.parse_args()

    hdf5_chunk_rows = args.hdf5_chunk_rows or max(1, args.rows // 16)
    hdf5_chunk_cols = args.hdf5_chunk_cols or max(1, args.cols // 16)

    run(args.rows, args.cols,
        hdf5_chunk_rows, hdf5_chunk_cols,
        args.repeats, args.warmup,
        tmp_dir=args.tmp_dir)
