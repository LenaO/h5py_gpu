"""
GPU compressed-read benchmark using real floating-point datasets.

Measures ``GPUDataset.read_chunks_compressed()`` vs the h5py baseline
(CPU decompression + H2D copy) on real scientific float32 data downloaded
from the FPsingle corpus:
  https://userweb.cs.txstate.edu/~burtscher/research/datasets/FPsingle/

Because the internal structure of these datasets is unknown, all HDF5
datasets are stored as **1-D chunks**.  The dataset length is fixed by the
file — it cannot be changed from the command line.

Usage
-----
    python benchmarks/bench_gpu_compressed_real.py --dataset num_brain
    python benchmarks/bench_gpu_compressed_real.py --list

    Optional flags:
      --datasets-dir PATH   directory containing the decompressed .sp files
                            (default: datasets/raw)
      --hdf5-chunk N        1-D HDF5 chunk size in number of elements
                            (default: length // 16)
      --repeats N           timing repetitions (default: 5)
      --warmup  N           warmup iterations   (default: 2)

Available datasets (pass the name without the .sp extension):
    obs_error   obs_info   obs_spitzer   obs_temp
    num_brain   num_comet  num_control   num_plasma
    msg_bt      msg_lu     msg_sp        msg_sppm    msg_sweep3d

Three benchmark sections:

  Section 1 -- gzip / deflate  (always available; CPU decompression via zlib)
    (a) Level sweep  : clevel = 1, 4, 6, 9  (no shuffle)
    (b) Shuffle sweep: clevel = 4, without vs. with byte-shuffle

  Section 2 -- LZ4 and Zstd  (requires hdf5plugin; GPU path via nvCOMP)
    (a) LZ4 without shuffle and with shuffle
    (b) Zstd level sweep: clevel = 1, 3, 9, 19
    (c) Zstd shuffle sweep: clevel = 3, without vs. with byte-shuffle

  Section 3 -- HDF5 chunk-size sweep  (gzip clevel=4, no shuffle)
    Log-2 sweep of 1-D chunk sizes.  Shows how chunk granularity affects
    compressed-pipeline throughput for this particular dataset.

Column definitions
------------------
  COMP MB  : compressed bytes on disk (HDF5 storage_size)
  RATIO    : uncompressed / compressed  (>1 → file is smaller on disk)
  BASE BW  : GB/s with h5py baseline (f["ds"][:] + H2D copy)
  RCC BW   : GB/s with read_chunks_compressed
  SPEEDUP  : baseline_time / rcc_time
  DECOMP   : decompression site — CPU or GPU (nvCOMP)
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
# Optional compression libraries (same guards as bench_gpu_compressed.py)
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
# Known datasets
# ---------------------------------------------------------------------------

KNOWN_DATASETS = [
    ("obs_error",    "observational"),
    ("obs_info",     "observational"),
    ("obs_spitzer",  "observational"),
    ("obs_temp",     "observational"),
    ("num_brain",    "numeric-simulation"),
    ("num_comet",    "numeric-simulation"),
    ("num_control",  "numeric-simulation"),
    ("num_plasma",   "numeric-simulation"),
    ("msg_bt",       "parallel-messages"),
    ("msg_lu",       "parallel-messages"),
    ("msg_sp",       "parallel-messages"),
    ("msg_sppm",     "parallel-messages"),
    ("msg_sweep3d",  "parallel-messages"),
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


def _chunk_sizes_1d(n, ref_chunk):
    """Log-2 sweep of 1-D chunk sizes, always including ref_chunk."""
    sizes = set()
    k = max(1, n >> 10)
    while k <= n:
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


def _print_barchart(rows, title="RCC BW (GB/s)"):
    max_bw = max(bw for _, bw in rows) if rows else 1.0
    print(f"\n  {title}  (each # ~= {max_bw/24:.2f} GB/s)\n")
    for label, bw in rows:
        print(f"  {label:<32}  {_bar(bw, max_bw)}  {bw:.3f} GB/s")


# ---------------------------------------------------------------------------
# Dataset creation helpers
# ---------------------------------------------------------------------------

def _create_gzip_datasets(wf, data, chunk):
    for lvl in (1, 4, 6, 9):
        wf.create_dataset(f"gzip_lvl{lvl}", data=data, chunks=(chunk,),
                          compression="gzip", compression_opts=lvl)
    wf.create_dataset("gzip4_noshuffle", data=data, chunks=(chunk,),
                      compression="gzip", compression_opts=4, shuffle=False)
    wf.create_dataset("gzip4_shuffle", data=data, chunks=(chunk,),
                      compression="gzip", compression_opts=4, shuffle=True)


def _create_lz4_datasets(wf, data, chunk):
    wf.create_dataset("lz4_noshuffle", data=data, chunks=(chunk,),
                      **_hdf5plugin.LZ4())
    wf.create_dataset("lz4_shuffle", data=data, chunks=(chunk,),
                      shuffle=True, **_hdf5plugin.LZ4())


def _create_zstd_datasets(wf, data, chunk):
    for lvl in (1, 3, 9, 19):
        wf.create_dataset(f"zstd_lvl{lvl}", data=data, chunks=(chunk,),
                          **_hdf5plugin.Zstd(clevel=lvl))
    wf.create_dataset("zstd3_noshuffle", data=data, chunks=(chunk,),
                      **_hdf5plugin.Zstd(clevel=3))
    wf.create_dataset("zstd3_shuffle", data=data, chunks=(chunk,),
                      shuffle=True, **_hdf5plugin.Zstd(clevel=3))


def _create_sweep_datasets(wf, data, auto_chunk):
    n = data.shape[0]
    for cs in _chunk_sizes_1d(n, auto_chunk):
        wf.create_dataset(f"sweep_chunk{cs}", data=data, chunks=(cs,),
                          compression="gzip", compression_opts=4)


# ---------------------------------------------------------------------------
# Section 1: gzip / deflate
# ---------------------------------------------------------------------------

def bench_gzip(rf, data, chunk, repeats, warmup):
    uncompressed_bytes = data.nbytes

    print(f"\n  n={len(data):,}  dtype={data.dtype}  chunk={chunk:,}  "
          f"({chunk * data.dtype.itemsize / 1024**2:.2f} MB/chunk)  "
          f"uncompressed={_mb(uncompressed_bytes):.1f} MB")

    print(f"\n  (a) Compression-level sweep (no shuffle)")
    _print_header()
    barchart = []
    for lvl in (1, 4, 6, 9):
        m = _bench_one(rf[f"gzip_lvl{lvl}"], GPUDataset(rf[f"gzip_lvl{lvl}"]),
                       uncompressed_bytes, repeats, warmup)
        label = f"gzip-{lvl}"
        _print_row(label, shuffle=False, m=m, decomp_path="CPU")
        barchart.append((label, m["bw_c"]))

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

def bench_lz4_zstd(rf, data, repeats, warmup):
    uncompressed_bytes = data.nbytes
    barchart = []

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
# Section 3: chunk-size sweep (gzip-4, 1-D)
# ---------------------------------------------------------------------------

def bench_chunk_sweep(rf, data, auto_chunk, repeats, warmup):
    n                  = data.shape[0]
    uncompressed_bytes = data.nbytes
    sizes              = _chunk_sizes_1d(n, auto_chunk)

    print(f"\n  n={n:,}  dtype={data.dtype}  codec=gzip-4  no shuffle")
    _print_header()

    barchart_b = []
    barchart_c = []
    for cs in sizes:
        chunk_mb = cs * data.dtype.itemsize / 1024**2
        m = _bench_one(rf[f"sweep_chunk{cs}"], GPUDataset(rf[f"sweep_chunk{cs}"]),
                       uncompressed_bytes, repeats, warmup)
        marker = "*" if cs == auto_chunk else " "
        label  = f"chunk={cs:,}{marker}"
        _print_row(label, shuffle=False, m=m, decomp_path="CPU")
        barchart_b.append((f"base  chunk={cs:,}{marker}", m["bw_b"]))
        barchart_c.append((f"rcc   chunk={cs:,}{marker}", m["bw_c"]))

    print(f"\n  * = default chunk ({auto_chunk:,} elements, "
          f"{auto_chunk * data.dtype.itemsize / 1024**2:.2f} MB)")
    _print_barchart(barchart_b + barchart_c,
                    title="Section 3 — BASE and RCC BW (GB/s)")


# ---------------------------------------------------------------------------
# Top-level runner
# ---------------------------------------------------------------------------

def run(data_path, auto_chunk, repeats, warmup):
    # Load dataset
    data = np.fromfile(data_path, dtype=np.float32)
    if data.size == 0:
        sys.exit(f"ERROR: {data_path} is empty or could not be read.")

    ds_name   = os.path.splitext(os.path.basename(data_path))[0]
    n_elems   = data.shape[0]
    unc_bytes = data.nbytes

    if auto_chunk is None:
        auto_chunk = max(1, n_elems // 16)

    # Banner
    print(f"\n{'='*74}")
    print(f"  h5py GPU compressed-read benchmark — real dataset")
    print(f"  dataset     : {ds_name}  ({data_path})")
    print(f"  length      : {n_elems:,} float32 values  "
          f"({_mb(unc_bytes):.1f} MB uncompressed)")
    print(f"  HDF5 chunk  : {auto_chunk:,} elements  "
          f"({auto_chunk * data.dtype.itemsize / 1024**2:.2f} MB/chunk)")
    print(f"  repeats     : {repeats}   warmup : {warmup}")
    print(f"  hdf5plugin  : {'yes' if _HDF5PLUGIN else 'NO  (pip install hdf5plugin)'}")
    print(f"  lz4 (CPU)   : {'yes' if _LZ4_CPU   else 'NO  (pip install lz4)'}")
    print(f"  zstd (CPU)  : {'yes' if _ZSTD_CPU  else 'NO  (pip install zstd)'}")
    nvcomp_note = ("yes — LZ4/Zstd will use GPU decompression" if _NVCOMP
                   else "NO  (pip install nvidia-nvcomp-cu12 "
                        "--extra-index-url https://pypi.nvidia.com)")
    print(f"  nvCOMP(GPU) : {nvcomp_note}")
    print(f"{'='*74}")

    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "bench_real.h5")

        # Write all datasets in a single pass
        print(f"\n  Writing HDF5 datasets from {ds_name} ...", end="", flush=True)
        with h5py.File(path, "w") as wf:
            _create_gzip_datasets(wf, data, auto_chunk)
            if _LZ4_AVAIL:
                _create_lz4_datasets(wf, data, auto_chunk)
            if _ZSTD_AVAIL:
                _create_zstd_datasets(wf, data, auto_chunk)
            _create_sweep_datasets(wf, data, auto_chunk)
        print(" done.")

        # Section 1: gzip
        print(f"\n{'='*74}")
        print(f"  SECTION 1: gzip / deflate  (CPU decompression via zlib)")
        with h5py.File(path, "r") as rf:
            bench_gzip(rf, data, auto_chunk, repeats, warmup)

        # Section 2: LZ4 + Zstd
        print(f"\n{'='*74}")
        if _LZ4_AVAIL or _ZSTD_AVAIL:
            decomp_note = "GPU (nvCOMP)" if _NVCOMP else "CPU"
            print(f"  SECTION 2: LZ4 and Zstd  (decompression path: {decomp_note})")
            with h5py.File(path, "r") as rf:
                bench_lz4_zstd(rf, data, repeats, warmup)
        else:
            print(f"  SECTION 2: LZ4 and Zstd  [SKIPPED]")
            print(f"    pip install hdf5plugin lz4 zstd")

        # Section 3: chunk-size sweep
        print(f"\n{'='*74}")
        print(f"  SECTION 3: 1-D chunk-size sweep  (gzip level=4, no shuffle)")
        with h5py.File(path, "r") as rf:
            bench_chunk_sweep(rf, data, auto_chunk, repeats, warmup)

    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _resolve_path(name, datasets_dir):
    """Resolve a dataset name or path to an absolute file path."""
    # If it looks like a path, use it directly
    if os.sep in name or "/" in name or name.endswith(".sp"):
        return os.path.abspath(name)
    # Otherwise look in datasets_dir with .sp extension
    candidate = os.path.join(datasets_dir, name + ".sp")
    if os.path.exists(candidate):
        return candidate
    # Also try without adding .sp (in case user typed the full filename)
    candidate2 = os.path.join(datasets_dir, name)
    if os.path.exists(candidate2):
        return candidate2
    return None


def _list_available(datasets_dir):
    print(f"\nAvailable datasets in {os.path.abspath(datasets_dir)}:\n")
    found = False
    for name, category in KNOWN_DATASETS:
        path = os.path.join(datasets_dir, name + ".sp")
        if os.path.exists(path):
            n      = os.path.getsize(path) // 4
            mb     = os.path.getsize(path) / 1024**2
            status = f"{n:>12,} floats  {mb:6.0f} MB"
            found  = True
        else:
            status = "  [not downloaded]"
        print(f"  {name:<20}  ({category:<22})  {status}")
    if not found:
        print("  None found. Run:  python download_datasets.py")
    print()


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--dataset",      type=str, default=None,
                   help="Dataset name (e.g. num_brain) or path to a .sp file")
    p.add_argument("--datasets-dir", type=str, default="datasets/raw",
                   help="Directory containing decompressed .sp files "
                        "(default: datasets/raw)")
    p.add_argument("--hdf5-chunk",   type=int, default=None,
                   help="1-D HDF5 chunk size in elements (default: length // 16)")
    p.add_argument("--repeats",      type=int, default=5)
    p.add_argument("--warmup",       type=int, default=2)
    p.add_argument("--list",         action="store_true",
                   help="List available datasets and exit")
    args = p.parse_args()

    if args.list:
        _list_available(args.datasets_dir)
        sys.exit(0)

    if args.dataset is None:
        p.error("--dataset is required (use --list to see available datasets)")

    data_path = _resolve_path(args.dataset, args.datasets_dir)
    if data_path is None or not os.path.exists(data_path):
        print(f"ERROR: dataset '{args.dataset}' not found in {args.datasets_dir}/",
              file=sys.stderr)
        print(f"       Run: python download_datasets.py --only {args.dataset}",
              file=sys.stderr)
        _list_available(args.datasets_dir)
        sys.exit(1)

    run(data_path, args.hdf5_chunk, args.repeats, args.warmup)
