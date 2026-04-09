"""
GPUCachedDataset benchmark.

Measures the cost of loading a dataset into GPU memory once and then
applying multiple operations entirely on the GPU — no disk I/O after
the initial preload.

Usage
-----
    python benchmarks/bench_gpu_cached.py [--rows N] [--cols N]
                                          [--dtype DTYPE]
                                          [--hdf5-chunk-rows N]
                                          [--hdf5-chunk-cols N]
                                          [--length N] [--hdf5-chunk-1d N]
                                          [--repeats N] [--warmup N]

Four benchmark sections:

  Section 1 -- Load cost (preload)
    Compares GPUCachedDataset.preload() against the underlying streaming
    read methods it delegates to (read_chunks_to_gpu / read_double_buffered)
    for all four dataset types: 2-D chunked, 2-D contiguous, 1-D contiguous,
    1-D chunked.

  Section 2 -- Amortization: N repeated operations
    For N = 1, 2, 4, 8, 16 operations (cp.sum):
      cache strategy : preload once, then N × cached.reduce(cp.sum)
      stream strategy: N × reduce_double_buffered(cp.sum)  (re-reads each time)
    Reports total time, per-operation time, and speedup.
    The cross-over point shows when caching starts to pay off.
    Tested for both a lightweight (sum) and heavy (exp(sqrt)+sum) operation.

  Section 3 -- Pure GPU operations (no I/O)
    Benchmarks operations applied to the already-loaded cache:
    (a) Reduce sweep: sum / max / min / mean / sum(x²) / exp(sqrt)+sum
        Compared against the equivalent streaming reduce for context.
    (b) Indexing sweep: cached[sel] for 10/25/50/75/100% coverage
        Compared against read_double_buffered(sel=...) and
        read_selection_chunked() to show the I/O-free speedup.
    (c) Transform chain: series of in-place updates to the cache
        (each measured as the incremental GPU cost after a fresh preload).

  Section 4 -- Multi-operation pipeline
    A realistic scenario: apply a fixed set of operations to one dataset.
    Compares two strategies:
      cache   : preload once, then apply all ops on the GPU
      stream  : each op triggers a fresh read (no caching)
    Reports total and per-operation cost for each strategy.
"""

import argparse
import os
import sys
import tempfile
import time

import numpy as np

import h5py
from h5py.gpu import GPUDataset, GPUCachedDataset

try:
    import cupy as cp
except ImportError:
    sys.exit("CuPy is required to run this benchmark.")


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


def _sel_row(rows, frac, chunk_rows):
    size = max(1, int(rows * frac))
    r0   = rows // 3 + chunk_rows // 2
    r0   = min(r0, rows - size)
    return r0, r0 + size


def _sel_col(cols, frac, chunk_cols):
    size = max(1, int(cols * frac))
    c0   = cols // 4 + chunk_cols // 3
    c0   = min(c0, cols - size)
    return c0, c0 + size


_HEAVY_TFM   = lambda x: cp.exp(cp.sqrt(x))
_HEAVY_LABEL = "exp(sqrt(x))"


def _make_reduces(n_total):
    return [
        ("sum",             cp.sum,  None,                             None),
        ("max",             cp.max,  None,                             None),
        ("min",             cp.min,  None,                             None),
        ("mean",            cp.sum,  lambda x: cp.sum(x) / n_total,   None),
        ("sum(x**2)",       lambda x: cp.sum(x ** 2),  cp.sum,        None),
        ("exp(sqrt)+sum",   cp.sum,  None,  _HEAVY_TFM),
    ]


# ---------------------------------------------------------------------------
# Section 1: Load cost
# ---------------------------------------------------------------------------

def bench_load_cost(path, datasets, dtype, repeats, warmup):
    """Compare preload() against the underlying streaming read it delegates to.

    datasets : list of (ds_name, shape, hdf5_chunk, label)
    """
    print(f"\n  {'DATASET':<32} {'METHOD':<30} {'TIME(s)':>8}  {'BW(GB/s)':>9}")
    print(f"  {'-'*32}  {'-'*30}  {'-'*8}  {'-'*9}")

    results = []
    with h5py.File(path, "r") as f:
        for ds_name, shape, hdf5_chunk, label in datasets:
            total_bytes = int(np.prod(shape)) * dtype.itemsize
            gpu_ds = GPUDataset(f[ds_name])

            # preload()
            def _preload(d=gpu_ds):
                c = GPUCachedDataset(d, preload=False)
                c.preload()
                c.free()

            t_pre, _, _ = _time_fn(_preload, repeats, warmup)
            bw_pre = _gb(total_bytes) / t_pre
            print(f"  {label:<32}  {'preload()':<30}  {t_pre:8.4f}  {bw_pre:9.3f}")

            # underlying read method
            if hdf5_chunk is not None and len(shape) == 2:
                method_label = "read_chunks_to_gpu()"
                t_read, _, _ = _time_fn(lambda d=gpu_ds: d.read_chunks_to_gpu(),
                                        repeats, warmup)
            else:
                method_label = "read_double_buffered()"
                t_read, _, _ = _time_fn(lambda d=gpu_ds: d.read_double_buffered(),
                                        repeats, warmup)
            bw_read = _gb(total_bytes) / t_read
            print(f"  {'':32}  {method_label:<30}  {t_read:8.4f}  {bw_read:9.3f}")

            overhead = (t_pre - t_read) / t_read * 100
            print(f"  {'':32}  {'overhead':30}  {overhead:+.1f}%")
            results.append((label, bw_pre, bw_read))

    max_bw = max(max(bw_pre, bw_read) for _, bw_pre, bw_read in results)
    print(f"\n  Bandwidth  (each # ~= {max_bw/28:.2f} GB/s)\n")
    for label, bw_pre, bw_read in results:
        print(f"  {label+' preload()':<38}  {_bar(bw_pre, max_bw)}  {bw_pre:.3f} GB/s")
        print(f"  {label+' read()':<38}  {_bar(bw_read, max_bw)}  {bw_read:.3f} GB/s")


# ---------------------------------------------------------------------------
# Section 2: Amortization — N repeated operations
# ---------------------------------------------------------------------------

def bench_amortization(path, shape, dtype, chunks, repeats, warmup):
    """Cache vs stream for N repeated operations.

    For each (op_label, cache_op, stream_op) pair, sweeps N = 1..16 and
    reports total time, per-op time, and speedup of cache over stream.
    """
    total_bytes = int(np.prod(shape)) * dtype.itemsize
    n_total     = int(np.prod(shape))
    n_ops_list  = [1, 2, 4, 8, 16]

    ops = [
        (
            "sum  (I/O-bound)",
            lambda cached:         cached.reduce(cp.sum),
            lambda gpu_ds:         gpu_ds.reduce_double_buffered(cp.sum),
        ),
        (
            "exp(sqrt)+sum  (compute-heavy)",
            lambda cached:         cached.reduce(cp.sum, transform=_HEAVY_TFM),
            lambda gpu_ds:         gpu_ds.reduce_double_buffered(
                                       cp.sum, transform=_HEAVY_TFM),
        ),
    ]

    with h5py.File(path, "r") as f:
        gpu_ds = GPUDataset(f["ds_2d"])

        for op_label, cache_op, stream_op in ops:
            print(f"\n  Operation: {op_label}")
            print(f"\n  {'N OPS':>6}  {'CACHE total':>12}  {'CACHE /op':>10}  "
                  f"{'STREAM total':>13}  {'STREAM /op':>10}  {'SPEEDUP':>8}")
            print(f"  {'-'*6}  {'-'*12}  {'-'*10}  {'-'*13}  {'-'*10}  {'-'*8}")

            # Warm-up preload time (not counted in per-N timings below)
            cached = GPUCachedDataset(gpu_ds, preload=False)
            t_load, _, _ = _time_fn(
                lambda: cached.reload(), repeats, warmup)

            results = []
            for n in n_ops_list:
                # Cache: preload once, then N ops
                def _cache_strategy(n=n, c=cached):
                    c.reload()
                    for _ in range(n):
                        cache_op(c)

                t_cache, _, _ = _time_fn(_cache_strategy, repeats, warmup)

                # Stream: N separate reads+ops (no caching)
                def _stream_strategy(n=n, d=gpu_ds):
                    for _ in range(n):
                        stream_op(d)

                t_stream, _, _ = _time_fn(_stream_strategy, repeats, warmup)

                speedup = t_stream / t_cache
                results.append((n, t_cache, t_stream, speedup))
                print(f"  {n:>6}  {t_cache:>12.4f}  {t_cache/n:>10.4f}  "
                      f"{t_stream:>13.4f}  {t_stream/n:>10.4f}  {speedup:>7.2f}x")

            cached.free()

            # Find break-even
            breakeven = next((n for n, _, _, sp in results if sp >= 1.0), None)
            if breakeven:
                print(f"\n  Cache pays off at N >= {breakeven} operations"
                      f"  (load cost: {t_load:.4f} s)")
            else:
                print(f"\n  Cache does not pay off within N={n_ops_list[-1]}"
                      f"  (load cost: {t_load:.4f} s)")


# ---------------------------------------------------------------------------
# Section 3: Pure GPU operations
# ---------------------------------------------------------------------------

def bench_pure_gpu_reduce(path, shape, dtype, chunks, repeats, warmup):
    """Reduce sweep on cached data vs equivalent streaming reduce."""
    total_bytes = int(np.prod(shape)) * dtype.itemsize
    n_total     = int(np.prod(shape))

    print(f"\n  {'REDUCE':<20} {'CACHED(s)':>10}  {'STREAM(s)':>10}  "
          f"{'CACHED BW':>10}  {'SPEEDUP':>8}")
    print(f"  {'-'*20}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*8}")

    results = []
    with h5py.File(path, "r") as f:
        gpu_ds = GPUDataset(f["ds_2d"])
        cached = GPUCachedDataset(gpu_ds)

        for label, rfn, cfn, tfm in _make_reduces(n_total):
            # Cached: pure GPU
            t_cached, _, _ = _time_fn(
                lambda r=rfn, t=tfm: cached.reduce(r, transform=t),
                repeats, warmup)

            # Streaming: read + reduce each time
            t_stream, _, _ = _time_fn(
                lambda r=rfn, c=cfn, t=tfm: gpu_ds.reduce_double_buffered(
                    r, combine_fn=c, transform=t),
                repeats, warmup)

            bw_cached = _gb(total_bytes) / t_cached
            speedup   = t_stream / t_cached
            print(f"  {label:<20}  {t_cached:>10.4f}  {t_stream:>10.4f}  "
                  f"{bw_cached:>10.3f}  {speedup:>7.2f}x")
            results.append((label, bw_cached))

        cached.free()

    max_bw = max(bw for _, bw in results)
    print(f"\n  Cached bandwidth  (each # ~= {max_bw/28:.2f} GB/s)\n")
    for label, bw in results:
        print(f"  {label:<20}  {_bar(bw, max_bw)}  {bw:.3f} GB/s")


def bench_pure_gpu_indexing(path, shape, dtype, chunks, repeats, warmup):
    """Selection sweep on cached data vs streaming selection reads."""
    rows, cols  = shape
    cr, cc      = chunks
    total_bytes = int(np.prod(shape)) * dtype.itemsize
    coverages   = [0.10, 0.25, 0.50, 0.75, 1.00]

    print(f"\n  {'SELECTION':<22} {'SEL MB':>6}  {'CACHED(s)':>10}  "
          f"{'DBL(s)':>8}  {'SEL_C(s)':>9}  {'SP(dbl)':>8}  {'SP(sel)':>8}")
    print(f"  {'-'*22}  {'-'*6}  {'-'*10}  {'-'*8}  {'-'*9}  {'-'*8}  {'-'*8}")

    with h5py.File(path, "r") as f:
        gpu_ds = GPUDataset(f["ds_2d"])
        cached = GPUCachedDataset(gpu_ds)

        for frac in coverages:
            r0, r1 = _sel_row(rows, frac, cr)
            c0, c1 = _sel_col(cols, frac, cc)
            sel     = (slice(r0, r1), slice(c0, c1))
            sel_bytes = (r1 - r0) * (c1 - c0) * dtype.itemsize

            t_cached, _, _ = _time_fn(
                lambda s=sel: cached[s], repeats, warmup)
            t_dbl, _, _ = _time_fn(
                lambda s=sel: gpu_ds.read_double_buffered(sel=s),
                repeats, warmup)
            t_sel, _, _ = _time_fn(
                lambda s=sel: gpu_ds.read_selection_chunked(s),
                repeats, warmup)

            sp_dbl = t_dbl / t_cached
            sp_sel = t_sel / t_cached
            label  = f"{int(frac*100):3d}%  [{r0}:{r1}, {c0}:{c1}]"
            print(f"  {label:<22}  {sel_bytes/1024**2:>6.1f}  "
                  f"{t_cached:>10.4f}  {t_dbl:>8.4f}  {t_sel:>9.4f}  "
                  f"{sp_dbl:>7.2f}x  {sp_sel:>7.2f}x")

        cached.free()


def bench_transform_chain(path, shape, dtype, chunks, repeats, warmup):
    """Time each in-place cache transform incrementally (after a fresh preload).

    Each row is: preload() + transform(fn) — so the load cost is included
    to show the total cost of "load and update the cache".
    Also shown: the streaming equivalent (read_double_buffered + transform)
    for reference.
    """
    total_bytes = int(np.prod(shape)) * dtype.itemsize

    transforms = [
        ("x *= 2  (in-place)",    lambda x: x.__imul__(2.0)),
        ("sqrt    (out-of-place)", cp.sqrt),
        ("exp     (out-of-place)", cp.exp),
        ("exp(sqrt(x))",           lambda x: cp.exp(cp.sqrt(x))),
    ]

    print(f"\n  {'TRANSFORM':<26} {'PRELOAD+TFM(s)':>15}  "
          f"{'TFM-only(s)':>12}  {'STREAM(s)':>10}")
    print(f"  {'-'*26}  {'-'*15}  {'-'*12}  {'-'*10}")

    with h5py.File(path, "r") as f:
        gpu_ds = GPUDataset(f["ds_2d"])

        for label, tfm in transforms:
            # preload + transform (total cost to get transformed data into cache)
            def _full(d=gpu_ds, t=tfm):
                c = GPUCachedDataset(d, preload=False)
                c.preload()
                c.transform(t)
                c.free()

            t_full, _, _ = _time_fn(_full, repeats, warmup)

            # transform only (incremental GPU cost, load not counted)
            cached = GPUCachedDataset(gpu_ds)

            def _tfm_only(c=cached, t=tfm):
                c.reload()
                c.transform(t)

            t_tfm, _, _ = _time_fn(_tfm_only, repeats, warmup)
            cached.free()

            # streaming equivalent (read_double_buffered returns transformed array)
            t_stream, _, _ = _time_fn(
                lambda t=tfm: gpu_ds.read_double_buffered(transform=t),
                repeats, warmup)

            print(f"  {label:<26}  {t_full:>15.4f}  {t_tfm:>12.4f}  {t_stream:>10.4f}")


# ---------------------------------------------------------------------------
# Section 4: Multi-operation pipeline
# ---------------------------------------------------------------------------

def bench_pipeline(path, shape, dtype, chunks, repeats, warmup):
    """Realistic pipeline: one load, many operations.

    Pipeline (applied to the 2-D chunked dataset):
      1. reduce(cp.sum)
      2. reduce(cp.max)
      3. reduce(cp.sum, transform=exp(sqrt))
      4. cached[sel_50pct]   (indexing)
      5. transform(cp.sqrt)  (update cache in-place)
      6. reduce(cp.sum)      (reduce on transformed cache)

    Compares:
      cache  : preload once, run all 6 ops on GPU
      stream : each op triggers a fresh read (reduce_double_buffered or
               read_double_buffered), no caching
    """
    rows, cols  = shape
    cr, cc      = chunks
    total_bytes = int(np.prod(shape)) * dtype.itemsize

    r0, r1 = _sel_row(rows, 0.50, cr)
    c0, c1 = _sel_col(cols, 0.50, cc)
    sel_50  = (slice(r0, r1), slice(c0, c1))

    pipeline_labels = [
        "reduce(sum)",
        "reduce(max)",
        "reduce(exp(sqrt)+sum)",
        "cached[50% sel]",
        "transform(sqrt)",
        "reduce(sum) [on sqrt cache]",
    ]

    with h5py.File(path, "r") as f:
        gpu_ds = GPUDataset(f["ds_2d"])

        # ── Cache strategy ─────────────────────────────────────────────────
        cached = GPUCachedDataset(gpu_ds, preload=False)

        def _run_cache(c=cached):
            c.reload()
            c.reduce(cp.sum)
            c.reduce(cp.max)
            c.reduce(cp.sum, transform=_HEAVY_TFM)
            c[sel_50]
            c.transform(cp.sqrt)
            c.reduce(cp.sum)

        t_cache_total, _, _ = _time_fn(_run_cache, repeats, warmup)

        # Time the load separately so we can show amortized ops cost
        t_load, _, _ = _time_fn(lambda: cached.reload(), repeats, warmup)

        # Time each GPU op individually (no I/O)
        cached.reload()
        op_times_cache = []
        for lbl, fn in [
            ("reduce(sum)",                lambda c=cached: c.reduce(cp.sum)),
            ("reduce(max)",                lambda c=cached: c.reduce(cp.max)),
            ("reduce(exp(sqrt)+sum)",      lambda c=cached: c.reduce(
                                               cp.sum, transform=_HEAVY_TFM)),
            ("cached[50% sel]",            lambda c=cached: c[sel_50]),
            ("transform(sqrt)",            lambda c=cached: c.transform(cp.sqrt)),
            ("reduce(sum) [on sqrt cache]",lambda c=cached: c.reduce(cp.sum)),
        ]:
            t, _, _ = _time_fn(fn, repeats, warmup)
            op_times_cache.append((lbl, t))
        cached.free()

        # ── Stream strategy ────────────────────────────────────────────────
        def _run_stream(d=gpu_ds):
            d.reduce_double_buffered(cp.sum)
            d.reduce_double_buffered(cp.max)
            d.reduce_double_buffered(cp.sum, transform=_HEAVY_TFM)
            d.read_double_buffered(sel=sel_50)
            d.read_double_buffered(transform=cp.sqrt)
            d.reduce_double_buffered(cp.sum, transform=cp.sqrt)

        t_stream_total, _, _ = _time_fn(_run_stream, repeats, warmup)

        op_times_stream = []
        for lbl, fn in [
            ("reduce(sum)",                lambda d=gpu_ds: d.reduce_double_buffered(cp.sum)),
            ("reduce(max)",                lambda d=gpu_ds: d.reduce_double_buffered(cp.max)),
            ("reduce(exp(sqrt)+sum)",      lambda d=gpu_ds: d.reduce_double_buffered(
                                               cp.sum, transform=_HEAVY_TFM)),
            ("read[50% sel]",              lambda d=gpu_ds: d.read_double_buffered(
                                               sel=sel_50)),
            ("read+transform(sqrt)",       lambda d=gpu_ds: d.read_double_buffered(
                                               transform=cp.sqrt)),
            ("reduce(sum)+transform(sqrt)",lambda d=gpu_ds: d.reduce_double_buffered(
                                               cp.sum, transform=cp.sqrt)),
        ]:
            t, _, _ = _time_fn(fn, repeats, warmup)
            op_times_stream.append((lbl, t))

    # -- Print results --------------------------------------------------------
    print(f"\n  {'OPERATION':<32} {'CACHE(s)':>10}  {'STREAM(s)':>10}  {'SPEEDUP':>8}")
    print(f"  {'-'*32}  {'-'*10}  {'-'*10}  {'-'*8}")
    print(f"  {'preload / (n/a)':<32}  {t_load:>10.4f}  {'---':>10}")

    for (lbl, tc), (_, ts) in zip(op_times_cache, op_times_stream):
        sp = ts / tc
        print(f"  {lbl:<32}  {tc:>10.4f}  {ts:>10.4f}  {sp:>7.2f}x")

    print(f"  {'-'*32}  {'-'*10}  {'-'*10}  {'-'*8}")
    sp_total = t_stream_total / t_cache_total
    print(f"  {'TOTAL (6 ops)':<32}  {t_cache_total:>10.4f}  "
          f"{t_stream_total:>10.4f}  {sp_total:>7.2f}x")

    t_gpu_ops = sum(t for _, t in op_times_cache)
    print(f"\n  Load:     {t_load:.4f} s")
    print(f"  GPU ops:  {t_gpu_ops:.4f} s  ({t_gpu_ops/t_cache_total*100:.1f}% of total)")
    print(f"  Overhead: {(t_cache_total - t_load - t_gpu_ops):.4f} s")


# ---------------------------------------------------------------------------
# Top-level runner
# ---------------------------------------------------------------------------

def run(rows, cols, dtype, hdf5_chunk_rows, hdf5_chunk_cols,
        length, hdf5_chunk_1d, repeats, warmup, tmp_dir=None):
    dtype     = np.dtype(dtype)
    shape_2d  = (rows, cols)
    chunks_2d = (hdf5_chunk_rows, hdf5_chunk_cols)

    data_1d = np.random.rand(length).astype(dtype) + 0.01
    data_2d = np.random.rand(*shape_2d).astype(dtype) + 0.01

    print(f"\n{'='*72}")
    print(f"  h5py GPU cached-dataset benchmark")
    print(f"  2-D dataset : {shape_2d}  dtype={dtype}  "
          f"size={_gb(data_2d.nbytes):.3f} GB")
    print(f"  HDF5 chunks : {chunks_2d}  "
          f"({chunks_2d[0] * chunks_2d[1] * dtype.itemsize / 1024**2:.2f} MB/chunk)")
    print(f"  1-D dataset : ({length},)  dtype={dtype}  "
          f"size={_gb(data_1d.nbytes):.3f} GB")
    print(f"  HDF5 chunk  : {hdf5_chunk_1d} elements  "
          f"({hdf5_chunk_1d * dtype.itemsize / 1024**2:.2f} MB/chunk)")
    print(f"  repeats     : {repeats}   warmup : {warmup}")
    print(f"  tmp dir     : {tmp_dir or '(system default)' }")
    print(f"{'='*72}")

    with tempfile.TemporaryDirectory(dir=tmp_dir) as td:
        path = os.path.join(td, "bench.h5")
        with h5py.File(path, "w") as f:
            f.create_dataset("ds_2d",         data=data_2d, chunks=chunks_2d)
            f.create_dataset("ds_2d_contig",  data=data_2d)
            f.create_dataset("ds_1d",         data=data_1d)
            f.create_dataset("ds_1d_chunked", data=data_1d,
                             chunks=(hdf5_chunk_1d,))

        ds_list = [
            ("ds_2d",         shape_2d,   chunks_2d,        "2-D chunked"),
            ("ds_2d_contig",  shape_2d,   None,             "2-D contiguous"),
            ("ds_1d",         (length,),  None,             "1-D contiguous"),
            ("ds_1d_chunked", (length,),  (hdf5_chunk_1d,), "1-D chunked"),
        ]

        # ── Section 1: Load cost ───────────────────────────────────────────
        print(f"\n{'='*72}")
        print(f"  SECTION 1: Load cost — preload() vs underlying read method")
        bench_load_cost(path, ds_list, dtype, repeats, warmup)

        # ── Section 2: Amortization ────────────────────────────────────────
        print(f"\n{'='*72}")
        print(f"  SECTION 2: Amortization — N repeated operations")
        print(f"  shape={shape_2d}  chunks={chunks_2d}")
        print(f"  cache  : preload once + N × cached.reduce(op)")
        print(f"  stream : N × reduce_double_buffered(op)  (re-reads each time)")
        bench_amortization(path, shape_2d, dtype, chunks_2d, repeats, warmup)

        # ── Section 3: Pure GPU operations ────────────────────────────────
        print(f"\n{'='*72}")
        print(f"  SECTION 3: Pure GPU operations (no I/O after preload)")
        print(f"  shape={shape_2d}  chunks={chunks_2d}")

        print(f"\n{'─'*72}")
        print(f"  3a. Reduce sweep")
        bench_pure_gpu_reduce(path, shape_2d, dtype, chunks_2d, repeats, warmup)

        print(f"\n{'─'*72}")
        print(f"  3b. Indexing sweep — cached[sel] vs streaming selection reads")
        bench_pure_gpu_indexing(path, shape_2d, dtype, chunks_2d, repeats, warmup)

        print(f"\n{'─'*72}")
        print(f"  3c. Transform chain — incremental cost of cache updates")
        bench_transform_chain(path, shape_2d, dtype, chunks_2d, repeats, warmup)

        # ── Section 4: Multi-operation pipeline ───────────────────────────
        print(f"\n{'='*72}")
        print(f"  SECTION 4: Multi-operation pipeline")
        print(f"  shape={shape_2d}  chunks={chunks_2d}")
        print(f"  6 operations: reduce×3, indexing×1, transform×1, reduce×1")
        bench_pipeline(path, shape_2d, dtype, chunks_2d, repeats, warmup)

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
    p.add_argument("--tmp-dir",          type=str,  default=None,
                   help="Directory for the temporary HDF5 file "
                        "(default: system temp). "
                        "Use to benchmark network or non-default filesystems.")
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
        args.repeats, args.warmup,
        tmp_dir=args.tmp_dir)
