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

Seven benchmark sections:

  Section 1 -- 2-D chunked dataset  (read_double_buffered + transform)
    (a) Chunk-size sweep with heaviest transform (exp(sqrt(x))):
        shows how OVERLAP% varies with the row-band size.
    (b) Transform sweep at the HDF5-aligned auto chunk size:
        shows all transforms at the operating point used in practice.

  Section 2 -- 2-D chunked dataset, single stream  (read_chunks_to_gpu + transform sweep)
    Same transforms as Section 1b.

  Section 3 -- 2-D chunked dataset, multi-stream  (read_chunks_parallel, exp transform)
    Vary n_streams: 1, 2, 4, 8
    Metric: speedup vs n_streams=1

  Section 4 -- 2-D chunked dataset — full read method comparison with transforms
    Side-by-side comparison of every 2-D full-read method:
      baseline (gpu_ds[:]), read_double_buffered, read_chunks_to_gpu,
      read_chunks_parallel(n=4)
    Two sub-tables: (a) no transform, (b) heavy transform (exp(sqrt(x))).
    For (b), the baseline row uses baseline + sequential compute as reference.

  Section 5 -- 2-D contiguous (non-chunked) dataset  (read_double_buffered + transform)
    Same two-part structure as Section 1.

  Section 6 -- 1-D contiguous dataset  (read_double_buffered + transform)
    Same two-part structure as Section 1.

  Section 7 -- 1-D chunked dataset  (read_double_buffered + transform)
    Same two-part structure as Section 1.

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


_HEAVY_LABEL = "exp(sqrt(x))"
_HEAVY_TFM   = lambda x: cp.exp(cp.sqrt(x))


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
    return float(np.mean(times)), float(np.min(times)), times


def _gb(n):
    return n / 1024**3


def _bar(v, vmax, width=28):
    f = int(round(v / vmax * width)) if vmax > 0 else 0
    return "#" * f + "." * (width - f)


def _compute_only_time(arr, transform, repeats, warmup):
    """Time applying transform to the full already-loaded GPU array."""
    if transform is None:
        return 0.0
    mean, _, _ = _time_fn(lambda: transform(arr), repeats, warmup)
    return mean


def _overlap_pct(t_pipe, t_io, t_comp):
    """Percentage of compute time hidden behind I/O.

    overlap = (t_io + t_comp - t_pipe) / t_comp  (clamped to [0, 100])
    """
    if t_comp <= 0:
        return 100.0
    hidden = t_io + t_comp - t_pipe
    return max(0.0, min(100.0, hidden / t_comp * 100.0))


def _chunk_sizes(n_rows, ref_chunk):
    """Log-2 sweep of row-band sizes, always including ref_chunk."""
    sizes = set()
    k = max(1, n_rows >> 5)
    while k <= n_rows:
        sizes.add(k)
        k *= 2
    sizes.add(ref_chunk)
    return sorted(sizes)


def _chunk_sizes_1d(n, ref_chunk=None):
    """Log-2 sweep of element counts plus optional ref_chunk."""
    sizes = set()
    k = n >> 10
    while k <= n:
        sizes.add(k)
        k *= 2
    if ref_chunk is not None:
        sizes.add(ref_chunk)
    return sorted(sizes)


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


def _print_chunk_sweep_header():
    print(f"\n  {'METHOD':<34} {'CHUNK':>7}  {'TIME(s)':>8}  "
          f"{'BW(GB/s)':>9}  {'OVERLAP':>8}")
    print(f"  {'-'*34}  {'-'*7}  {'-'*8}  {'-'*9}  {'-'*8}")


# ---------------------------------------------------------------------------
# Section 1 / 5: 2-D dataset — read_double_buffered chunk-size sweep + transform sweep
# ---------------------------------------------------------------------------

def bench_2d_dbl(path, ds_name, shape, dtype, chunks, repeats, warmup):
    """2-D chunked read_double_buffered: chunk-size sweep then transform sweep.

    Parameters
    ----------
    ds_name : str  — HDF5 dataset name
    chunks  : (int, int)  — HDF5 chunk shape; auto chunk_size = chunks[0] rows
    """
    rows, cols = shape
    hdf5_chunk_rows = chunks[0]
    total_bytes = int(np.prod(shape)) * dtype.itemsize
    auto_rows = hdf5_chunk_rows

    with h5py.File(path, "r") as f:
        gpu_ds = GPUDataset(f[ds_name])
        arr_gpu = gpu_ds.read_double_buffered()

        # -- (a) Chunk-size sweep with heavy transform -------------------------
        t_comp_heavy = _compute_only_time(arr_gpu, _HEAVY_TFM, repeats, warmup)
        print(f"\n  Chunk-size sweep  "
              f"(transform: {_HEAVY_LABEL},  compute-only: {t_comp_heavy:.4f} s)")
        _print_chunk_sweep_header()

        chunk_results = []
        for cs in _chunk_sizes(rows, hdf5_chunk_rows):
            chunk_mb = cs * cols * dtype.itemsize / 1024**2
            marker = "*" if cs == hdf5_chunk_rows else " "
            t_io, _, _ = _time_fn(
                lambda c=cs: gpu_ds.read_double_buffered(chunk_size=c),
                repeats, warmup)
            t_pipe, _, _ = _time_fn(
                lambda c=cs: gpu_ds.read_double_buffered(chunk_size=c,
                                                         transform=_HEAVY_TFM),
                repeats, warmup)
            bw = _gb(total_bytes) / t_pipe
            overlap = _overlap_pct(t_pipe, t_io, t_comp_heavy)
            label = f"double  chunk={cs:>5} rows{marker}"
            print(f"  {label:<34} {chunk_mb:>6.1f}M  {t_pipe:8.4f}  "
                  f"{bw:9.3f}  {overlap:>6.1f}%")
            chunk_results.append((cs, bw, marker))

        print(f"  * = aligned to HDF5 chunk rows ({hdf5_chunk_rows})")

        max_bw = max(bw for _, bw, _ in chunk_results)
        print(f"\n  Bandwidth  (each # ~= {max_bw/28:.2f} GB/s)\n")
        for cs, bw, marker in chunk_results:
            print(f"  {f'chunk={cs}{marker}':<26}  {_bar(bw, max_bw)}  {bw:.3f} GB/s")

        # -- (b) Transform sweep at HDF5-aligned auto chunk size --------------
        auto_mb = auto_rows * cols * dtype.itemsize / 1024**2
        t_io, _, _ = _time_fn(
            lambda: gpu_ds.read_double_buffered(chunk_size=auto_rows),
            repeats, warmup)

        print(f"\n  Transform sweep  "
              f"(chunk_size={auto_rows} rows*,  {auto_mb:.1f} MB/band,  HDF5-aligned)")
        _print_header()

        results = []
        for label, tfm in _make_transforms():
            t_comp = _compute_only_time(arr_gpu, tfm, repeats, warmup)
            t_pipe, _, _ = _time_fn(
                lambda t=tfm: gpu_ds.read_double_buffered(chunk_size=auto_rows,
                                                          transform=t),
                repeats, warmup)
            bw = _gb(total_bytes) / t_pipe
            _print_row(label, t_pipe, bw, t_comp, t_io)
            results.append((label, bw))

        max_bw = max(bw for _, bw in results)
        print(f"\n  Bandwidth  (each # ~= {max_bw/28:.2f} GB/s)\n")
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
        t_io, _, _ = _time_fn(lambda: gpu_ds.read_chunks_to_gpu(), repeats, warmup)
        arr_gpu = gpu_ds.read_chunks_to_gpu()

        print(f"\n  chunks={chunks}  "
              f"({int(np.prod(chunks)) * dtype.itemsize / 1024:.1f} KB/chunk)")
        _print_header()

        results = []
        for label, tfm in transforms:
            t_comp = _compute_only_time(arr_gpu, tfm, repeats, warmup)
            t_pipe, _, _ = _time_fn(
                lambda t=tfm: gpu_ds.read_chunks_to_gpu(transform=t),
                repeats, warmup)
            bw = _gb(total_bytes) / t_pipe
            _print_row(label, t_pipe, bw, t_comp, t_io)
            results.append((label, bw))

        max_bw = max(bw for _, bw in results)
        print(f"\n  Bandwidth  (each # ~= {max_bw/28:.2f} GB/s)\n")
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
                t, _, _ = _time_fn(
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
            print(f"\n  Bandwidth  (each # ~= {max_bw/28:.2f} GB/s)\n")
            for n, _, bw, _ in results:
                print(f"  {'n='+str(n):<16}  {_bar(bw, max_bw)}  {bw:.3f} GB/s")


# ---------------------------------------------------------------------------
# Section 4: 2-D chunked — full read method comparison with transforms
# ---------------------------------------------------------------------------

def bench_method_comparison(path, shape, dtype, chunks, repeats, warmup):
    """Compare all 2-D full-read methods without and with the heavy transform.

    The baseline (gpu_ds[:]) does not support a transform parameter; for the
    "with transform" sub-table its row shows baseline + sequential compute
    (t_base + t_comp) as the cost of the naive approach.
    """
    total_bytes = int(np.prod(shape)) * dtype.itemsize

    with h5py.File(path, "r") as f:
        gpu_ds = GPUDataset(f["ds_2d"])
        arr_gpu = gpu_ds.read_chunks_to_gpu()
        t_comp = _compute_only_time(arr_gpu, _HEAVY_TFM, repeats, warmup)

        # -- (a) No transform — mirrors read benchmark Section 6 --------------
        print(f"\n  shape={shape}  dtype={dtype}  chunks={chunks}  "
              f"size={_gb(total_bytes):.3f} GB")
        print(f"\n  No transform")
        print(f"\n  {'METHOD':<36} {'TIME(s)':>8}  {'BW(GB/s)':>9}  {'SPEEDUP':>7}")
        print(f"  {'-'*36}  {'-'*8}  {'-'*9}  {'-'*7}")

        methods_no_tfm = [
            ("baseline (gpu_ds[:])",        lambda: gpu_ds[:]),
            ("read_double_buffered(auto)",   lambda: gpu_ds.read_double_buffered()),
            ("read_chunks_to_gpu()",         lambda: gpu_ds.read_chunks_to_gpu()),
            ("read_chunks_parallel(n=4)",    lambda: gpu_ds.read_chunks_parallel(n_streams=4)),
        ]

        t_base_no_tfm = None
        bw_list_no_tfm = []
        for label, fn in methods_no_tfm:
            t, _, _ = _time_fn(fn, repeats, warmup)
            bw = _gb(total_bytes) / t
            if t_base_no_tfm is None:
                t_base_no_tfm = t
                sp = "1.00x"
            else:
                sp = f"{t_base_no_tfm / t:.2f}x"
            print(f"  {label:<36} {t:8.4f}  {bw:9.3f}  {sp:>7}")
            bw_list_no_tfm.append((label, bw))

        max_bw = max(bw for _, bw in bw_list_no_tfm)
        print(f"\n  Bandwidth  (each # ~= {max_bw/28:.2f} GB/s)\n")
        for label, bw in bw_list_no_tfm:
            print(f"  {label:<36}  {_bar(bw, max_bw)}  {bw:.3f} GB/s")

        # -- (b) Heavy transform — shows which methods hide compute -----------
        t_base_io, _, _ = _time_fn(lambda: gpu_ds[:], repeats, warmup)
        t_seq = t_base_io + t_comp  # naive: read then compute sequentially

        print(f"\n  With transform: {_HEAVY_LABEL}  "
              f"[compute-only: {t_comp:.4f} s]")
        print(f"\n  {'METHOD':<36} {'TIME(s)':>8}  {'BW(GB/s)':>9}  "
              f"{'COMP(s)':>8}  {'OVERLAP':>8}  {'SPEEDUP':>7}")
        print(f"  {'-'*36}  {'-'*8}  {'-'*9}  {'-'*8}  {'-'*8}  {'-'*7}")

        # Baseline row: sequential (no pipeline)
        bw_seq = _gb(total_bytes) / t_seq
        print(f"  {'baseline + sequential':<36} {t_seq:8.4f}  {bw_seq:9.3f}  "
              f"{t_comp:8.4f}  {'   0.0%':>8}  {'1.00x':>7}")

        methods_with_tfm = [
            ("read_double_buffered(auto)",   lambda: gpu_ds.read_double_buffered(
                                                 transform=_HEAVY_TFM)),
            ("read_chunks_to_gpu()",         lambda: gpu_ds.read_chunks_to_gpu(
                                                 transform=_HEAVY_TFM)),
            ("read_chunks_parallel(n=4)",    lambda: gpu_ds.read_chunks_parallel(
                                                 n_streams=4, transform=_HEAVY_TFM)),
        ]

        t_io_dbl, _, _ = _time_fn(lambda: gpu_ds.read_double_buffered(),
                                   repeats, warmup)
        t_io_chunks, _, _ = _time_fn(lambda: gpu_ds.read_chunks_to_gpu(),
                                      repeats, warmup)
        t_io_par, _, _ = _time_fn(lambda: gpu_ds.read_chunks_parallel(n_streams=4),
                                   repeats, warmup)
        t_ios = [t_io_dbl, t_io_chunks, t_io_par]

        bw_list_tfm = []
        for (label, fn), t_io in zip(methods_with_tfm, t_ios):
            t, _, _ = _time_fn(fn, repeats, warmup)
            bw = _gb(total_bytes) / t
            overlap = _overlap_pct(t, t_io, t_comp)
            sp = f"{t_seq / t:.2f}x"
            print(f"  {label:<36} {t:8.4f}  {bw:9.3f}  "
                  f"{t_comp:8.4f}  {overlap:>6.1f}%  {sp:>7}")
            bw_list_tfm.append((label, bw))

        all_bws = [bw_seq] + [bw for _, bw in bw_list_tfm]
        max_bw = max(all_bws)
        print(f"\n  Bandwidth  (each # ~= {max_bw/28:.2f} GB/s)\n")
        print(f"  {'baseline + sequential':<36}  {_bar(bw_seq, max_bw)}  {bw_seq:.3f} GB/s")
        for label, bw in bw_list_tfm:
            print(f"  {label:<36}  {_bar(bw, max_bw)}  {bw:.3f} GB/s")


# ---------------------------------------------------------------------------
# Section 5 / 6 / 7: generic dataset — chunk-size sweep + transform sweep
# ---------------------------------------------------------------------------

def bench_generic_dbl(path, ds_name, total_bytes, dtype, auto_chunk,
                       chunk_sizes, auto_label, auto_note,
                       chunk_label_fn, chunk_mb_fn,
                       repeats, warmup):
    """Shared implementation for 2-D contiguous, 1-D contiguous, 1-D chunked.

    Parameters
    ----------
    auto_chunk     : chunk size used for auto default and transform sweep
    chunk_sizes    : iterable of chunk sizes for the sweep
    auto_label     : string shown in the transform-sweep sub-header
    auto_note      : footnote for the auto default
    chunk_label_fn : (cs, marker) -> row label string
    chunk_mb_fn    : cs -> float, MB per chunk/band
    """
    with h5py.File(path, "r") as f:
        gpu_ds = GPUDataset(f[ds_name])
        arr_gpu = gpu_ds.read_double_buffered(chunk_size=auto_chunk)

        # -- (a) Chunk-size sweep with heavy transform -------------------------
        t_comp_heavy = _compute_only_time(arr_gpu, _HEAVY_TFM, repeats, warmup)
        print(f"\n  Chunk-size sweep  "
              f"(transform: {_HEAVY_LABEL},  compute-only: {t_comp_heavy:.4f} s)")
        _print_chunk_sweep_header()

        chunk_results = []
        for cs in chunk_sizes:
            chunk_mb = chunk_mb_fn(cs)
            marker = "*" if cs == auto_chunk else " "
            t_io, _, _ = _time_fn(
                lambda c=cs: gpu_ds.read_double_buffered(chunk_size=c),
                repeats, warmup)
            t_pipe, _, _ = _time_fn(
                lambda c=cs: gpu_ds.read_double_buffered(chunk_size=c,
                                                         transform=_HEAVY_TFM),
                repeats, warmup)
            bw = _gb(total_bytes) / t_pipe
            overlap = _overlap_pct(t_pipe, t_io, t_comp_heavy)
            label = chunk_label_fn(cs, marker)
            print(f"  {label:<34} {chunk_mb:>6.2f}M  {t_pipe:8.4f}  "
                  f"{bw:9.3f}  {overlap:>6.1f}%")
            chunk_results.append((cs, bw, marker))

        print(f"  {auto_note}")

        max_bw = max(bw for _, bw, _ in chunk_results)
        print(f"\n  Bandwidth  (each # ~= {max_bw/28:.2f} GB/s)\n")
        for cs, bw, marker in chunk_results:
            short = chunk_label_fn(cs, marker).strip()
            print(f"  {short:<26}  {_bar(bw, max_bw)}  {bw:.3f} GB/s")

        # -- (b) Transform sweep at auto chunk size ---------------------------
        auto_mb = chunk_mb_fn(auto_chunk)
        t_io, _, _ = _time_fn(
            lambda: gpu_ds.read_double_buffered(chunk_size=auto_chunk),
            repeats, warmup)

        print(f"\n  Transform sweep  ({auto_label},  {auto_mb:.2f} MB/band)")
        _print_header()

        results = []
        for label, tfm in _make_transforms():
            t_comp = _compute_only_time(arr_gpu, tfm, repeats, warmup)
            t_pipe, _, _ = _time_fn(
                lambda t=tfm: gpu_ds.read_double_buffered(chunk_size=auto_chunk,
                                                          transform=t),
                repeats, warmup)
            bw = _gb(total_bytes) / t_pipe
            _print_row(label, t_pipe, bw, t_comp, t_io)
            results.append((label, bw))

        max_bw = max(bw for _, bw in results)
        print(f"\n  Bandwidth  (each # ~= {max_bw/28:.2f} GB/s)\n")
        for label, bw in results:
            print(f"  {label:<16}  {_bar(bw, max_bw)}  {bw:.3f} GB/s")


def bench_2d_contiguous(path, shape, dtype, repeats, warmup):
    rows, cols = shape
    total_bytes = int(np.prod(shape)) * dtype.itemsize
    auto_rows = max(1, rows // 8)
    bench_generic_dbl(
        path, "ds_2d_contig", total_bytes, dtype,
        auto_chunk   = auto_rows,
        chunk_sizes  = _chunk_sizes(rows, auto_rows),
        auto_label   = f"chunk_size={auto_rows} rows* (rows//8)",
        auto_note    = f"* = auto default (rows // 8 = {auto_rows})",
        chunk_label_fn = lambda cs, m: f"double  chunk={cs:>5} rows{m}",
        chunk_mb_fn    = lambda cs: cs * cols * dtype.itemsize / 1024**2,
        repeats=repeats, warmup=warmup,
    )


def bench_1d(path, ds_name, n_elems, dtype, hdf5_chunk, repeats, warmup):
    """1-D dataset (contiguous or HDF5-chunked): chunk-size sweep + transform sweep.

    Parameters
    ----------
    hdf5_chunk : int or None — HDF5 chunk size; None means contiguous dataset.
    """
    total_bytes = n_elems * dtype.itemsize
    if hdf5_chunk is not None:
        auto_chunk = hdf5_chunk
        auto_label = f"chunk_size={hdf5_chunk}* (HDF5-aligned)"
        auto_note  = f"* = aligned to HDF5 chunk ({hdf5_chunk} elements)"
    else:
        auto_chunk = max(1, n_elems // 8)
        auto_label = f"chunk_size={auto_chunk}* (length//8)"
        auto_note  = f"* = auto default (length // 8 = {auto_chunk})"

    bench_generic_dbl(
        path, ds_name, total_bytes, dtype,
        auto_chunk   = auto_chunk,
        chunk_sizes  = _chunk_sizes_1d(n_elems, auto_chunk),
        auto_label   = auto_label,
        auto_note    = auto_note,
        chunk_label_fn = lambda cs, m: f"double  chunk={cs:>7} elems{m}",
        chunk_mb_fn    = lambda cs: cs * dtype.itemsize / 1024**2,
        repeats=repeats, warmup=warmup,
    )


# ---------------------------------------------------------------------------
# Top-level runner
# ---------------------------------------------------------------------------

def run(rows, cols, dtype, hdf5_chunk_rows, hdf5_chunk_cols,
        length, hdf5_chunk_1d, repeats, warmup):
    dtype     = np.dtype(dtype)
    shape_2d  = (rows, cols)
    chunks_2d = (hdf5_chunk_rows, hdf5_chunk_cols)

    # Use positive values so sqrt/exp don't produce NaN
    data_1d = np.random.rand(length).astype(dtype) + 0.01
    data_2d = np.random.rand(*shape_2d).astype(dtype) + 0.01

    print(f"\n{'='*72}")
    print(f"  h5py GPU transform benchmark")
    print(f"  2-D dataset : {shape_2d}  dtype={dtype}  "
          f"size={_gb(data_2d.nbytes):.3f} GB")
    print(f"  HDF5 chunks : {chunks_2d}  "
          f"({chunks_2d[0] * chunks_2d[1] * dtype.itemsize / 1024**2:.2f} MB/chunk)")
    print(f"  1-D dataset : ({length},)  dtype={dtype}  "
          f"size={_gb(data_1d.nbytes):.3f} GB")
    print(f"  HDF5 chunk  : {hdf5_chunk_1d} elements  "
          f"({hdf5_chunk_1d * dtype.itemsize / 1024**2:.2f} MB/chunk)")
    print(f"  repeats     : {repeats}   warmup : {warmup}")
    print(f"{'='*72}")

    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "bench.h5")
        with h5py.File(path, "w") as f:
            f.create_dataset("ds_2d",        data=data_2d, chunks=chunks_2d)
            f.create_dataset("ds_2d_contig", data=data_2d)
            f.create_dataset("ds_1d",        data=data_1d)
            f.create_dataset("ds_1d_chunked", data=data_1d,
                             chunks=(hdf5_chunk_1d,))

        # ── Section 1: 2-D chunked, read_double_buffered ──────────────────
        print(f"\n{'='*72}")
        print(f"  SECTION 1: 2-D chunked dataset — read_double_buffered + transform")
        print(f"  shape={shape_2d}  chunks={chunks_2d}")
        print(f"  Columns:")
        print(f"    COMP(s)   : pure GPU compute time (no I/O)")
        print(f"    OVERLAP % : fraction of compute time hidden behind I/O")
        bench_2d_dbl(path, "ds_2d", shape_2d, dtype, chunks_2d, repeats, warmup)

        # ── Section 2: 2-D chunked, single stream ─────────────────────────
        print(f"\n{'='*72}")
        print(f"  SECTION 2: 2-D chunked — read_chunks_to_gpu + transform "
              f"(single stream)")
        print(f"  shape={shape_2d}  chunks={chunks_2d}")
        bench_2d_single(path, shape_2d, dtype, chunks_2d, repeats, warmup)

        # ── Section 3: 2-D chunked, multi-stream ──────────────────────────
        print(f"\n{'='*72}")
        print(f"  SECTION 3: 2-D chunked — read_chunks_parallel (n_streams sweep)")
        print(f"  shape={shape_2d}  chunks={chunks_2d}")
        print(f"  SPEEDUP is relative to n_streams=1 for the same transform.")
        bench_2d_parallel(path, shape_2d, dtype, chunks_2d, repeats, warmup)

        # ── Section 4: 2-D chunked, method comparison ─────────────────────
        print(f"\n{'='*72}")
        print(f"  SECTION 4: 2-D chunked dataset — full read method comparison")
        print(f"  shape={shape_2d}  chunks={chunks_2d}")
        bench_method_comparison(path, shape_2d, dtype, chunks_2d, repeats, warmup)

        # ── Section 5: 2-D contiguous ─────────────────────────────────────
        print(f"\n{'='*72}")
        print(f"  SECTION 5: 2-D contiguous (non-chunked) dataset — "
              f"read_double_buffered + transform")
        print(f"  shape={shape_2d}  (contiguous)")
        bench_2d_contiguous(path, shape_2d, dtype, repeats, warmup)

        # ── Section 6: 1-D contiguous ─────────────────────────────────────
        print(f"\n{'='*72}")
        print(f"  SECTION 6: 1-D contiguous dataset — read_double_buffered + transform")
        print(f"  shape=({length},)  (contiguous)")
        bench_1d(path, "ds_1d", length, dtype,
                 hdf5_chunk=None, repeats=repeats, warmup=warmup)

        # ── Section 7: 1-D chunked ────────────────────────────────────────
        print(f"\n{'='*72}")
        print(f"  SECTION 7: 1-D chunked dataset — read_double_buffered + transform")
        print(f"  shape=({length},)  chunks=({hdf5_chunk_1d},)")
        bench_1d(path, "ds_1d_chunked", length, dtype,
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
