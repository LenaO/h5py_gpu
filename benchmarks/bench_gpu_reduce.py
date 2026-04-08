"""
GPU reduction benchmark.

Measures throughput and compute-overlap when reducing HDF5 datasets directly
on the GPU without fully loading them into GPU memory.

Usage
-----
    python benchmarks/bench_gpu_reduce.py [--rows N] [--cols N]
                                          [--dtype DTYPE]
                                          [--hdf5-chunk-rows N]
                                          [--hdf5-chunk-cols N]
                                          [--length N] [--hdf5-chunk-1d N]
                                          [--repeats N] [--warmup N]

Seven benchmark sections:

  Section 1 -- 2-D chunked dataset  (reduce_chunks)
    The chunk granularity is fixed by the HDF5 layout, so there is no
    chunk-size sweep.  Two sub-tables:
    (a) Reduce-fn sweep, no transform: sum / max / min / mean / sum-of-squares
    (b) Reduce-fn sweep with heavy transform (exp(sqrt(x))):
        also reports COMP(s) and OVERLAP% to show compute hiding.

  Section 2 -- 2-D chunked dataset  (reduce_double_buffered)
    (a) Chunk-size sweep with heavy reduce (exp(sqrt(x)) + sum):
        shows OVERLAP% vs row-band size.
    (b) Reduce-fn sweep at HDF5-aligned auto chunk size.

  Section 3 -- 2-D contiguous (non-chunked) dataset  (reduce_double_buffered)
    Same two-part structure as Section 2.

  Section 4 -- 1-D contiguous dataset  (reduce_double_buffered)
    Same two-part structure as Section 2.

  Section 5 -- 1-D chunked dataset  (reduce_double_buffered)
    Same two-part structure as Section 2.

  Section 6 -- 2-D chunked dataset — method comparison
    Side-by-side comparison of:
      numpy baseline: h5py f["ds"][:] → cp.asarray → reduce  (standard user path)
      GPUDataset[:] + sequential reduce
      reduce_chunks()
      reduce_double_buffered(auto)
    Two sub-tables: (a) sum (I/O-bound), (b) exp(sqrt(x)) + sum (compute-heavy).
    Speedup is relative to the numpy baseline (1.00x).

All sections report:
    TIME(s)   : wall time of the pipelined call
    BW(GB/s)  : dataset bytes / time  (useful read throughput)
    COMP(s)   : pure GPU compute time (reduce applied to already-loaded array)
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
# Reduce functions
# ---------------------------------------------------------------------------

def _make_reduces(n_total):
    """Return list of (label, reduce_fn, combine_fn, transform).

    n_total : total number of elements in the dataset, used for mean.
    """
    return [
        ("sum",             cp.sum,  None,                             None),
        ("max",             cp.max,  None,                             None),
        ("min",             cp.min,  None,                             None),
        ("mean",            cp.sum,  lambda x: cp.sum(x) / n_total,   None),
        ("sum(x**2)",       lambda x: cp.sum(x ** 2),  cp.sum,        None),
        ("exp(sqrt)+sum",   cp.sum,  None,  lambda x: cp.exp(cp.sqrt(x))),
    ]


_HEAVY_LABEL    = "exp(sqrt(x)) + sum"
_HEAVY_REDUCE   = cp.sum
_HEAVY_TFM      = lambda x: cp.exp(cp.sqrt(x))


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


def _compute_only_time(arr, transform, reduce_fn, repeats, warmup):
    """Time transform (optional) + reduce applied to an already-loaded GPU array."""
    def fn():
        x = transform(arr) if transform is not None else arr
        return reduce_fn(x)
    mean, _, _ = _time_fn(fn, repeats, warmup)
    return mean


def _overlap_pct(t_pipe, t_io, t_comp):
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
    sizes = set()
    k = n >> 10
    while k <= n:
        sizes.add(k)
        k *= 2
    if ref_chunk is not None:
        sizes.add(ref_chunk)
    return sorted(sizes)


def _print_reduce_header(with_overlap=False):
    if with_overlap:
        print(f"\n  {'REDUCE':<20} {'TIME(s)':>8}  {'BW(GB/s)':>9}  "
              f"{'COMP(s)':>8}  {'OVERLAP':>8}")
        print(f"  {'-'*20}  {'-'*8}  {'-'*9}  {'-'*8}  {'-'*8}")
    else:
        print(f"\n  {'REDUCE':<20} {'TIME(s)':>8}  {'BW(GB/s)':>9}")
        print(f"  {'-'*20}  {'-'*8}  {'-'*9}")


def _print_chunk_sweep_header():
    print(f"\n  {'METHOD':<34} {'CHUNK':>7}  {'TIME(s)':>8}  "
          f"{'BW(GB/s)':>9}  {'OVERLAP':>8}")
    print(f"  {'-'*34}  {'-'*7}  {'-'*8}  {'-'*9}  {'-'*8}")


# ---------------------------------------------------------------------------
# Section 1: 2-D chunked — reduce_chunks
# ---------------------------------------------------------------------------

def bench_reduce_chunks(path, shape, dtype, chunks, repeats, warmup):
    """reduce_chunks: reduce-fn sweep without and with heavy transform."""
    total_bytes = int(np.prod(shape)) * dtype.itemsize
    n_total     = int(np.prod(shape))

    with h5py.File(path, "r") as f:
        gpu_ds  = GPUDataset(f["ds_2d"])
        arr_gpu = gpu_ds.read_chunks_to_gpu()

        # I/O-only time (for overlap calculation)
        t_io, _, _ = _time_fn(lambda: gpu_ds.read_chunks_to_gpu(), repeats, warmup)

        # -- (a) Reduce-fn sweep, no transform --------------------------------
        print(f"\n  chunks={chunks}  "
              f"({int(np.prod(chunks)) * dtype.itemsize / 1024**2:.2f} MB/chunk)")
        print(f"\n  No transform")
        _print_reduce_header(with_overlap=False)

        results_no_tfm = []
        for label, rfn, cfn, tfm in _make_reduces(n_total):
            if tfm is not None:
                continue  # skip transform rows in this sub-table
            t, _, _ = _time_fn(
                lambda r=rfn, c=cfn: gpu_ds.reduce_chunks(r, combine_fn=c),
                repeats, warmup)
            bw = _gb(total_bytes) / t
            print(f"  {label:<20}  {t:8.4f}  {bw:9.3f}")
            results_no_tfm.append((label, bw))

        max_bw = max(bw for _, bw in results_no_tfm)
        print(f"\n  Bandwidth  (each # ~= {max_bw/28:.2f} GB/s)\n")
        for label, bw in results_no_tfm:
            print(f"  {label:<20}  {_bar(bw, max_bw)}  {bw:.3f} GB/s")

        # -- (b) Reduce-fn sweep with heavy transform -------------------------
        print(f"\n  With transform: exp(sqrt(x))")
        _print_reduce_header(with_overlap=True)

        results_tfm = []
        for label, rfn, cfn, tfm in _make_reduces(n_total):
            # use the heavy transform for all rows here
            t_comp = _compute_only_time(arr_gpu, _HEAVY_TFM, rfn, repeats, warmup)
            t, _, _ = _time_fn(
                lambda r=rfn, c=cfn: gpu_ds.reduce_chunks(
                    r, combine_fn=c, transform=_HEAVY_TFM),
                repeats, warmup)
            bw = _gb(total_bytes) / t
            overlap = _overlap_pct(t, t_io, t_comp)
            overlap_s = f"{overlap:6.1f}%"
            print(f"  {label:<20}  {t:8.4f}  {bw:9.3f}  {t_comp:8.4f}  {overlap_s}")
            results_tfm.append((label, bw))

        max_bw = max(bw for _, bw in results_tfm)
        print(f"\n  Bandwidth  (each # ~= {max_bw/28:.2f} GB/s)\n")
        for label, bw in results_tfm:
            print(f"  {label:<20}  {_bar(bw, max_bw)}  {bw:.3f} GB/s")


# ---------------------------------------------------------------------------
# Sections 2–5: generic reduce_double_buffered
# ---------------------------------------------------------------------------

def bench_reduce_dbl(path, ds_name, total_bytes, n_total, dtype, auto_chunk,
                     chunk_sizes, auto_label, auto_note,
                     chunk_label_fn, chunk_mb_fn,
                     repeats, warmup):
    """Shared implementation for all reduce_double_buffered sections.

    (a) Chunk-size sweep with heavy reduce (exp(sqrt) + sum)
    (b) Reduce-fn sweep at auto chunk size
    """
    with h5py.File(path, "r") as f:
        gpu_ds  = GPUDataset(f[ds_name])
        arr_gpu = gpu_ds.read_double_buffered(chunk_size=auto_chunk)

        # -- (a) Chunk-size sweep with heavy reduce ---------------------------
        t_comp_heavy = _compute_only_time(arr_gpu, _HEAVY_TFM, _HEAVY_REDUCE,
                                          repeats, warmup)
        print(f"\n  Chunk-size sweep  "
              f"(reduce: {_HEAVY_LABEL},  compute-only: {t_comp_heavy:.4f} s)")
        _print_chunk_sweep_header()

        chunk_results = []
        for cs in chunk_sizes:
            chunk_mb = chunk_mb_fn(cs)
            marker = "*" if cs == auto_chunk else " "
            t_io, _, _ = _time_fn(
                lambda c=cs: gpu_ds.read_double_buffered(chunk_size=c),
                repeats, warmup)
            t_pipe, _, _ = _time_fn(
                lambda c=cs: gpu_ds.reduce_double_buffered(
                    _HEAVY_REDUCE, transform=_HEAVY_TFM, chunk_size=c),
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

        # -- (b) Reduce-fn sweep at auto chunk size ---------------------------
        t_io, _, _ = _time_fn(
            lambda: gpu_ds.read_double_buffered(chunk_size=auto_chunk),
            repeats, warmup)
        auto_mb = chunk_mb_fn(auto_chunk)

        print(f"\n  Reduce-fn sweep  ({auto_label},  {auto_mb:.2f} MB/band)")
        _print_reduce_header(with_overlap=True)

        results = []
        for label, rfn, cfn, tfm in _make_reduces(n_total):
            t_comp = _compute_only_time(arr_gpu, tfm, rfn, repeats, warmup)
            t, _, _ = _time_fn(
                lambda r=rfn, c=cfn, t=tfm: gpu_ds.reduce_double_buffered(
                    r, combine_fn=c, transform=t, chunk_size=auto_chunk),
                repeats, warmup)
            bw = _gb(total_bytes) / t
            overlap = _overlap_pct(t, t_io, t_comp)
            overlap_s = f"{overlap:6.1f}%" if t_comp > 0 else "   N/A  "
            print(f"  {label:<20}  {t:8.4f}  {bw:9.3f}  {t_comp:8.4f}  {overlap_s}")
            results.append((label, bw))

        max_bw = max(bw for _, bw in results)
        print(f"\n  Bandwidth  (each # ~= {max_bw/28:.2f} GB/s)\n")
        for label, bw in results:
            print(f"  {label:<20}  {_bar(bw, max_bw)}  {bw:.3f} GB/s")


def bench_2d_chunked_dbl(path, shape, dtype, chunks, repeats, warmup):
    rows, cols = shape
    total_bytes = int(np.prod(shape)) * dtype.itemsize
    auto_rows = chunks[0]
    bench_reduce_dbl(
        path, "ds_2d", total_bytes, int(np.prod(shape)), dtype,
        auto_chunk     = auto_rows,
        chunk_sizes    = _chunk_sizes(rows, auto_rows),
        auto_label     = f"chunk_size={auto_rows} rows* (HDF5-aligned)",
        auto_note      = f"* = aligned to HDF5 chunk rows ({auto_rows})",
        chunk_label_fn = lambda cs, m: f"double  chunk={cs:>5} rows{m}",
        chunk_mb_fn    = lambda cs: cs * cols * dtype.itemsize / 1024**2,
        repeats=repeats, warmup=warmup,
    )


def bench_2d_contiguous(path, shape, dtype, repeats, warmup):
    rows, cols = shape
    total_bytes = int(np.prod(shape)) * dtype.itemsize
    auto_rows = max(1, rows // 8)
    bench_reduce_dbl(
        path, "ds_2d_contig", total_bytes, int(np.prod(shape)), dtype,
        auto_chunk     = auto_rows,
        chunk_sizes    = _chunk_sizes(rows, auto_rows),
        auto_label     = f"chunk_size={auto_rows} rows* (rows//8)",
        auto_note      = f"* = auto default (rows // 8 = {auto_rows})",
        chunk_label_fn = lambda cs, m: f"double  chunk={cs:>5} rows{m}",
        chunk_mb_fn    = lambda cs: cs * cols * dtype.itemsize / 1024**2,
        repeats=repeats, warmup=warmup,
    )


def bench_1d(path, ds_name, n_elems, dtype, hdf5_chunk, repeats, warmup):
    total_bytes = n_elems * dtype.itemsize
    if hdf5_chunk is not None:
        auto_chunk = hdf5_chunk
        auto_label = f"chunk_size={hdf5_chunk}* (HDF5-aligned)"
        auto_note  = f"* = aligned to HDF5 chunk ({hdf5_chunk} elements)"
    else:
        auto_chunk = max(1, n_elems // 8)
        auto_label = f"chunk_size={auto_chunk}* (length//8)"
        auto_note  = f"* = auto default (length // 8 = {auto_chunk})"
    bench_reduce_dbl(
        path, ds_name, total_bytes, n_elems, dtype,
        auto_chunk     = auto_chunk,
        chunk_sizes    = _chunk_sizes_1d(n_elems, auto_chunk),
        auto_label     = auto_label,
        auto_note      = auto_note,
        chunk_label_fn = lambda cs, m: f"double  chunk={cs:>7} elems{m}",
        chunk_mb_fn    = lambda cs: cs * dtype.itemsize / 1024**2,
        repeats=repeats, warmup=warmup,
    )


# ---------------------------------------------------------------------------
# Section 6: 2-D chunked — method comparison
# ---------------------------------------------------------------------------

def bench_method_comparison(path, shape, dtype, chunks, repeats, warmup):
    """Compare numpy baseline vs reduce_chunks vs reduce_double_buffered.

    Two sub-tables:
    (a) sum — I/O-bound, shows raw throughput difference between methods.
    (b) exp(sqrt(x)) + sum — compute-heavy, shows value of pipelining.

    Three baseline rows:
      numpy h5py[:] + H2D + reduce : standard path (h5py numpy read → cp.asarray → reduce)
      GPUDataset[:] + seq. reduce  : extension read → sequential reduce (no pipeline)
    Speedup is relative to the numpy baseline (first row = 1.00x).
    """
    total_bytes = int(np.prod(shape)) * dtype.itemsize
    n_total     = int(np.prod(shape))

    with h5py.File(path, "r") as f:
        gpu_ds  = GPUDataset(f["ds_2d"])
        h5_ds   = f["ds_2d"]           # raw h5py dataset for numpy baseline
        arr_gpu = gpu_ds.read_chunks_to_gpu()

        print(f"\n  shape={shape}  dtype={dtype}  chunks={chunks}  "
              f"size={_gb(total_bytes):.3f} GB")

        for sub_label, rfn, cfn, tfm in [
            ("sum  (I/O-bound)",          cp.sum,          None,  None),
            ("exp(sqrt(x)) + sum  (compute-heavy)", cp.sum, None, _HEAVY_TFM),
        ]:
            t_comp = _compute_only_time(arr_gpu, tfm, rfn, repeats, warmup)

            # Numpy baseline: standard h5py read → H2D copy → sequential reduce
            def _numpy_baseline(r=rfn, t=tfm):
                arr_np  = h5_ds[:]                # standard h5py read (numpy)
                arr_gpu2 = cp.asarray(arr_np)     # host-to-device copy
                x = t(arr_gpu2) if t is not None else arr_gpu2
                return r(x)

            t_numpy, _, _ = _time_fn(_numpy_baseline, repeats, warmup)
            bw_numpy = _gb(total_bytes) / t_numpy

            # GPUDataset baseline: gpu_ds[:] then reduce sequentially
            t_load, _, _ = _time_fn(lambda: gpu_ds[:], repeats, warmup)

            def _seq_baseline(r=rfn, t=tfm):
                arr = gpu_ds[:]
                x = t(arr) if t is not None else arr
                return r(x)

            t_seq, _, _ = _time_fn(_seq_baseline, repeats, warmup)
            bw_seq = _gb(total_bytes) / t_seq

            print(f"\n  Reduce: {sub_label}")
            if t_comp > 0:
                print(f"  compute-only: {t_comp:.4f} s")

            print(f"\n  {'METHOD':<36} {'TIME(s)':>8}  {'BW(GB/s)':>9}  "
                  f"{'OVERLAP':>8}  {'SPEEDUP':>7}")
            print(f"  {'-'*36}  {'-'*8}  {'-'*9}  {'-'*8}  {'-'*7}")

            # numpy baseline row (reference: 1.00x)
            overlap_np = "   N/A  " if t_comp == 0 else "   0.0%"
            print(f"  {'numpy h5py[:] + H2D + reduce':<36} {t_numpy:8.4f}  "
                  f"{bw_numpy:9.3f}  {overlap_np:>8}  {'1.00x':>7}")

            # GPUDataset sequential baseline row
            overlap_seq = "   N/A  " if t_comp == 0 else "   0.0%"
            print(f"  {'GPUDataset[:] + seq. reduce':<36} {t_seq:8.4f}  "
                  f"{bw_seq:9.3f}  {overlap_seq:>8}  {f'{t_numpy/t_seq:.2f}x':>7}")

            methods = [
                ("reduce_chunks()",
                 lambda r=rfn, c=cfn, t=tfm: gpu_ds.reduce_chunks(
                     r, combine_fn=c, transform=t),
                 lambda: gpu_ds.read_chunks_to_gpu()),
                ("reduce_double_buffered(auto)",
                 lambda r=rfn, c=cfn, t=tfm: gpu_ds.reduce_double_buffered(
                     r, combine_fn=c, transform=t),
                 lambda: gpu_ds.read_double_buffered()),
            ]

            bw_list = [(sub_label, bw_numpy), ("GPUDataset seq.", bw_seq)]
            for label, fn, io_fn in methods:
                t_io_m, _, _ = _time_fn(io_fn, repeats, warmup)
                t, _, _ = _time_fn(fn, repeats, warmup)
                bw = _gb(total_bytes) / t
                overlap = _overlap_pct(t, t_io_m, t_comp)
                overlap_s = f"{overlap:6.1f}%" if t_comp > 0 else "   N/A  "
                sp = f"{t_numpy / t:.2f}x"
                print(f"  {label:<36} {t:8.4f}  {bw:9.3f}  {overlap_s:>8}  {sp:>7}")
                bw_list.append((label, bw))

            max_bw = max(bw for _, bw in bw_list)
            print(f"\n  Bandwidth  (each # ~= {max_bw/28:.2f} GB/s)\n")
            print(f"  {'numpy h5py[:] + H2D + reduce':<36}  "
                  f"{_bar(bw_numpy, max_bw)}  {bw_numpy:.3f} GB/s")
            print(f"  {'GPUDataset[:] + seq. reduce':<36}  "
                  f"{_bar(bw_seq, max_bw)}  {bw_seq:.3f} GB/s")
            for label, bw in bw_list[2:]:
                print(f"  {label:<36}  {_bar(bw, max_bw)}  {bw:.3f} GB/s")


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
    print(f"  h5py GPU reduce benchmark")
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
            f.create_dataset("ds_2d",         data=data_2d, chunks=chunks_2d)
            f.create_dataset("ds_2d_contig",  data=data_2d)
            f.create_dataset("ds_1d",         data=data_1d)
            f.create_dataset("ds_1d_chunked", data=data_1d,
                             chunks=(hdf5_chunk_1d,))

        # ── Section 1: 2-D chunked, reduce_chunks ─────────────────────────
        print(f"\n{'='*72}")
        print(f"  SECTION 1: 2-D chunked dataset — reduce_chunks")
        print(f"  shape={shape_2d}  chunks={chunks_2d}")
        print(f"  Columns:")
        print(f"    COMP(s)   : pure GPU compute time (no I/O)")
        print(f"    OVERLAP % : fraction of compute time hidden behind I/O")
        bench_reduce_chunks(path, shape_2d, dtype, chunks_2d, repeats, warmup)

        # ── Section 2: 2-D chunked, reduce_double_buffered ────────────────
        print(f"\n{'='*72}")
        print(f"  SECTION 2: 2-D chunked dataset — reduce_double_buffered")
        print(f"  shape={shape_2d}  chunks={chunks_2d}")
        bench_2d_chunked_dbl(path, shape_2d, dtype, chunks_2d, repeats, warmup)

        # ── Section 3: 2-D contiguous, reduce_double_buffered ─────────────
        print(f"\n{'='*72}")
        print(f"  SECTION 3: 2-D contiguous (non-chunked) dataset — "
              f"reduce_double_buffered")
        print(f"  shape={shape_2d}  (contiguous)")
        bench_2d_contiguous(path, shape_2d, dtype, repeats, warmup)

        # ── Section 4: 1-D contiguous, reduce_double_buffered ─────────────
        print(f"\n{'='*72}")
        print(f"  SECTION 4: 1-D contiguous dataset — reduce_double_buffered")
        print(f"  shape=({length},)  (contiguous)")
        bench_1d(path, "ds_1d", length, dtype,
                 hdf5_chunk=None, repeats=repeats, warmup=warmup)

        # ── Section 5: 1-D chunked, reduce_double_buffered ────────────────
        print(f"\n{'='*72}")
        print(f"  SECTION 5: 1-D chunked dataset — reduce_double_buffered")
        print(f"  shape=({length},)  chunks=({hdf5_chunk_1d},)")
        bench_1d(path, "ds_1d_chunked", length, dtype,
                 hdf5_chunk=hdf5_chunk_1d, repeats=repeats, warmup=warmup)

        # ── Section 6: 2-D chunked, method comparison ─────────────────────
        print(f"\n{'='*72}")
        print(f"  SECTION 6: 2-D chunked dataset — method comparison")
        print(f"  shape={shape_2d}  chunks={chunks_2d}")
        bench_method_comparison(path, shape_2d, dtype, chunks_2d, repeats, warmup)

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
