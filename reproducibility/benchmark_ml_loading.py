#!/usr/bin/env python3
"""
benchmark_ml_loading.py – Measure DataLoader-style I/O throughput for the
GPU-aware h5py ML benchmark.

Simulates one training epoch: a shuffled permutation of slice indices is split
into contiguous batches and read sequentially.  Three methods are compared:

    baseline      standard h5py  →  numpy  →  GPU  (cp.array / pinned copy)
    gpu_raw       GPUDataset.__getitem__  →  cupy   (read_selection_chunked)
    gpu_norm      gpu_raw + in-situ per-image z-score normalisation (fused
                  on the same CUDA stream, no extra pass over GPU memory)

The input file must be produced by prepare_fastmri.py:
    images  float32  (N, H, W)  chunked (1, H, W)

Metrics reported
----------------
    GB/s     effective I/O bandwidth  (batch bytes / wall time)
    img/s    images loaded per second
    ms/batch mean batch time
    speedup  vs. baseline

Usage
-----
    # Quick test (single file, 20 batches × 5 repeats)
    python benchmark_ml_loading.py knee_512.h5 --batches 20 --repeats 5

    # Full benchmark matching paper parameters
    python benchmark_ml_loading.py knee_1024.h5 \\
        --batch-size 16 --batches 100 --repeats 5 --csv results_1024.csv

    # Show effect of normalisation overhead
    python benchmark_ml_loading.py knee_512.h5 --with-norm

Dependencies
------------
    h5py, cupy, numpy
    (Optional for CSV output: nothing extra needed)
"""

import argparse
import csv
import sys
import time
from pathlib import Path

import h5py
import numpy as np

try:
    import cupy as cp
    _CUPY_AVAILABLE = True
except ImportError:
    _CUPY_AVAILABLE = False

try:
    from mpi4py import MPI as _MPI
    _MPI_AVAILABLE = True
except ImportError:
    _MPI = None
    _MPI_AVAILABLE = False

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _check_backend(backend: str) -> None:
    if not _CUPY_AVAILABLE:
        sys.exit("CuPy is required for all backends (GPUDataset uses it internally). "
                 "Install with: pip install cupy-cuda12x")
    if backend == "torch" and not _TORCH_AVAILABLE:
        sys.exit("PyTorch backend requested but torch is not installed. "
                 "Install with: pip install torch")


def _make_batches(n_slices: int, batch_size: int, seed: int = 0) -> list[tuple[int, int]]:
    """Return a list of (start, stop) pairs covering a shuffled epoch.

    The epoch permutation is sorted within each batch so that HDF5 reads are
    always contiguous (matching how real DataLoaders work: shuffle at the
    index level, but read storage sequentially within a batch for I/O efficiency).
    """
    rng = np.random.default_rng(seed)
    order = rng.permutation(n_slices)
    batches = []
    for i in range(0, n_slices - batch_size + 1, batch_size):
        idx = np.sort(order[i : i + batch_size])
        # Store as (first, last+1) only when contiguous; otherwise keep array.
        # For chunks=(1, H, W) a contiguous slice touches the fewest chunks.
        # We always pick the min..max range so the read is a single contiguous
        # slice — this is how curated HPC datasets are typically accessed.
        batches.append((int(idx[0]), int(idx[-1]) + 1))
    return batches


def _batch_bytes(batches: list[tuple[int, int]], h: int, w: int) -> float:
    total = sum((stop - start) * h * w * 4 for start, stop in batches)
    return float(total)


# ---------------------------------------------------------------------------
# Benchmark runners
# ---------------------------------------------------------------------------

def _run_baseline(
    dataset: h5py.Dataset,
    batches: list[tuple[int, int]],
    backend: str = "cupy",
) -> list[float]:
    """h5py → numpy → pinned copy → GPU."""
    times = []
    for start, stop in batches:
        t0 = time.perf_counter()
        arr_np = dataset[start:stop]
        if backend == "cupy":
            pinned = cp.cuda.alloc_pinned_memory(arr_np.nbytes)
            buf = np.frombuffer(pinned, dtype=arr_np.dtype).reshape(arr_np.shape)
            np.copyto(buf, arr_np)
            arr_gpu = cp.array(buf)
            cp.cuda.Device().synchronize()
        else:
            arr_gpu = torch.as_tensor(arr_np).pin_memory().cuda(non_blocking=True)
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    return times


def _run_gpu(
    gpu_ds,
    batches: list[tuple[int, int]],
    normalize: bool = False,
    backend: str = "cupy",
) -> list[float]:
    """GPUDataset → CuPy array, or CuPy → DLPack → PyTorch tensor."""
    # CuPy fused transform (only used in cupy path)
    if normalize and backend == "cupy":
        def _norm(x):
            mu = x.mean()
            sd = x.std()
            return (x - mu) / (sd + 1e-6)
        transform = _norm
    else:
        transform = None

    times = []
    for start, stop in batches:
        t0 = time.perf_counter()
        if backend == "torch":
            # GPUDataset(backend="torch") already returns a torch.Tensor
            arr = gpu_ds[start:stop]
            if normalize:
                mu = arr.mean()
                sd = arr.std()
                arr = (arr - mu) / (sd + 1e-6)
            torch.cuda.synchronize()
        else:  # "cupy" or "kvikio"
            if transform is not None:   # fused CuPy normalize (cupy only)
                sel = (slice(start, stop), slice(None), slice(None))
                arr = gpu_ds.read_selection_chunked(sel, transform=transform)
            else:
                arr = gpu_ds[start:stop]
                if normalize:           # post-process normalize (kvikio path)
                    mu = arr.mean()
                    sd = arr.std()
                    arr = (arr - mu) / (sd + 1e-6)
            cp.cuda.Device().synchronize()
        times.append(time.perf_counter() - t0)
    return times


# ---------------------------------------------------------------------------
# Multi-GPU worker
# ---------------------------------------------------------------------------

def _gpu_worker(
    gpu_id: int,
    h5_path: str,
    batches: list[tuple[int, int]],
    normalize: bool,
    n_warmup: int,
    n_repeats: int,
    backend: str = "cupy",
) -> list[float]:
    """Subprocess entry: benchmarks one GPU, returns measurement batch times."""
    cp.cuda.Device(gpu_id).use()
    if backend == "torch":
        torch.cuda.set_device(gpu_id)
    f = h5py.File(h5_path, "r")
    from h5py.gpu import GPUDataset
    gpu_ds = GPUDataset(f["images"], backend=backend)
    all_times: list[float] = []
    for rep in range(n_warmup + n_repeats):
        times = _run_gpu(gpu_ds, batches, normalize=normalize, backend=backend)
        if rep >= n_warmup:
            all_times.extend(times)
    f.close()
    return all_times


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

def _stats(times: list[float]) -> dict:
    a = np.array(times)
    return {
        "mean": a.mean(),
        "std":  a.std(),
        "min":  a.min(),
        "max":  a.max(),
    }


def _report_row(label: str, st: dict, batch_bytes_total: float,
                n_batches: int, baseline_mean: float | None) -> None:
    total_bytes = batch_bytes_total          # bytes for all batches
    bw_gbs  = (total_bytes / n_batches) / st["mean"] / 1e9
    img_s   = (total_bytes / (st["mean"] * 4)) / (
        total_bytes / n_batches / 4           # images per batch
    )  # images / second
    img_s   = 1.0 / st["mean"] * (total_bytes / n_batches / 4)

    speedup = f"{baseline_mean / st['mean']:.2f}×" if baseline_mean else "—"

    print(
        f"  {label:<18s}"
        f"  {st['mean']*1e3:7.1f} ± {st['std']*1e3:5.1f} ms"
        f"  {bw_gbs:6.2f} GB/s"
        f"  {img_s:8.0f} img/s"
        f"  {speedup:>8s}"
    )
    return st["mean"]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark GPU-aware h5py vs. baseline for ML data loading.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("h5_file", type=Path,
                        help="Consolidated HDF5 file from prepare_fastmri.py")
    parser.add_argument("--batch-size", type=int, default=16, metavar="B",
                        help="Number of slices per batch (default: 16)")
    parser.add_argument("--batches",    type=int, default=50,  metavar="N",
                        help="Number of batches to measure per repeat (default: 50)")
    parser.add_argument("--repeats",    type=int, default=5,   metavar="R",
                        help="Number of measurement repeats (default: 5)")
    parser.add_argument("--warmup",     type=int, default=2,   metavar="W",
                        help="Warm-up repeats discarded before measurement (default: 2)")
    parser.add_argument("--with-norm",  action="store_true",
                        help="Also benchmark gpu_norm (in-situ z-score normalisation)")
    parser.add_argument("--csv",        type=Path, default=None, metavar="PATH",
                        help="Write per-repeat raw results to a CSV file")
    parser.add_argument("--seed",       type=int, default=42)
    parser.add_argument("--gpus",         type=int, default=1,   metavar="N",
                        help="GPUs to benchmark in parallel; each gets 1/N of the "
                             "batches (default: 1)")
    parser.add_argument("--gpus-per-node", type=int, default=None, metavar="G",
                        help="GPUs per node for MPI runs (default: auto-detect)")
    parser.add_argument("--backend",      choices=["cupy", "torch"], default="cupy",
                        help="GPU array backend: cupy (default) or torch. "
                             "Both require CuPy; torch adds a zero-copy DLPack "
                             "conversion after the GPUDataset read.")

    args = parser.parse_args()
    _check_backend(args.backend)

    if not args.h5_file.exists():
        sys.exit(f"File not found: {args.h5_file}")

    # ------------------------------------------------------------------
    # Open file and inspect
    # ------------------------------------------------------------------
    with h5py.File(args.h5_file, "r") as meta_f:
        n_slices, H, W = meta_f["images"].shape
        chunk_shape     = meta_f["images"].chunks
        chunk_mb        = float(meta_f.attrs.get("chunk_mb", chunk_shape[1] * chunk_shape[2] * 4 / 1e6))
        split           = meta_f.attrs.get("fastmri_split", "?")

    print(f"\n{'='*65}")
    print(f"  File        : {args.h5_file}")
    print(f"  Dataset     : images {(n_slices, H, W)}  float32")
    print(f"  Chunk shape : {chunk_shape}  ({chunk_mb:.2f} MB/chunk)")
    print(f"  Split       : {split}")
    print(f"  Batch size  : {args.batch_size} slices  "
          f"({args.batch_size * H * W * 4 / 1e6:.1f} MB/batch  "
          f"= {args.batch_size} chunks)")
    print(f"  Batches/rep : {args.batches}   Repeats: {args.repeats}   "
          f"Warmup: {args.warmup}")
    print(f"{'='*65}\n")

    # Pre-generate batch index list (same permutation seed for all repeats)
    all_batches = _make_batches(n_slices, args.batch_size, seed=args.seed)
    # Limit to requested number of batches (wrap around if fewer than available)
    batches: list[tuple[int, int]] = []
    while len(batches) < args.batches:
        batches.extend(all_batches)
    batches = batches[: args.batches]

    total_bytes = _batch_bytes(batches, H, W)
    n_batches   = len(batches)

    # ------------------------------------------------------------------
    # MPI multi-node path
    # ------------------------------------------------------------------
    _comm     = _MPI.COMM_WORLD if _MPI_AVAILABLE else None
    _mpi_rank = _comm.Get_rank() if _comm else 0
    _mpi_size = _comm.Get_size() if _comm else 1

    if _mpi_size > 1:
        gpus_per_node = args.gpus_per_node or cp.cuda.runtime.getDeviceCount()
        cp.cuda.Device(_mpi_rank % gpus_per_node).use()
        if args.backend == "torch":
            torch.cuda.set_device(_mpi_rank % gpus_per_node)

        my_batches = batches[_mpi_rank::_mpi_size]
        my_bytes   = _batch_bytes(my_batches, H, W)
        n_my       = len(my_batches)

        # baseline on rank 0 only (single-GPU reference)
        baseline_times: list[float] = []
        if _mpi_rank == 0:
            _f = h5py.File(args.h5_file, "r")
            for rep in range(args.warmup + args.repeats):
                times = _run_baseline(_f["images"], my_batches, backend=args.backend)
                if rep >= args.warmup:
                    baseline_times.extend(times)
            _f.close()

        # GPU methods: all ranks run in parallel, results gathered to rank 0
        gpu_mpi_results: dict[str, list[list[float]]] = {}
        _f = h5py.File(args.h5_file, "r")
        from h5py.gpu import GPUDataset
        _gpu_ds = GPUDataset(_f["images"], backend=args.backend)

        gpu_methods_mpi = [("gpu_raw", False)]
        if args.with_norm:
            gpu_methods_mpi.append(("gpu_norm", True))

        for method_name, normalize in gpu_methods_mpi:
            if _mpi_rank == 0:
                print(f"  [{method_name}] running across {_mpi_size} MPI ranks ...",
                      end=" ", flush=True)
            my_times: list[float] = []
            for rep in range(args.warmup + args.repeats):
                times = _run_gpu(_gpu_ds, my_batches, normalize=normalize, backend=args.backend)
                if rep >= args.warmup:
                    my_times.extend(times)
            all_rank_times = _comm.gather(my_times, root=0)
            if _mpi_rank == 0:
                gpu_mpi_results[method_name] = all_rank_times
                print("done")

        _f.close()

        if _mpi_rank == 0:
            base_st   = _stats(baseline_times)
            base_mean = base_st["mean"]
            base_bw   = (my_bytes / n_my) / base_mean / 1e9

            print(f"\n{'─'*65}")
            print(f"  MPI benchmark  ({_mpi_size} ranks, {gpus_per_node} GPU/node)")
            print(f"  {'Method':<28s}  {'ms/batch':>14s}  {'BW':>9s}  {'Throughput':>10s}  {'Speedup':>8s}")
            print(f"{'─'*65}")
            _report_row("baseline (rank 0)", base_st, my_bytes, n_my, None)

            for method_name in [m for m in ["gpu_raw", "gpu_norm"] if m in gpu_mpi_results]:
                agg_bw   = 0.0
                agg_imgs = 0.0
                for rank_id, rank_times in enumerate(gpu_mpi_results[method_name]):
                    st       = _stats(rank_times)
                    rb       = batches[rank_id::_mpi_size]
                    rb_bytes = _batch_bytes(rb, H, W)
                    n_rb     = len(rb)
                    bw       = (rb_bytes / n_rb) / st["mean"] / 1e9
                    img_s    = 1.0 / st["mean"] * (rb_bytes / n_rb / 4)
                    agg_bw   += bw
                    agg_imgs += img_s
                    print(
                        f"  {f'{method_name}[rank{rank_id}]':<28s}"
                        f"  {st['mean']*1e3:7.1f} +/- {st['std']*1e3:5.1f} ms"
                        f"  {bw:6.2f} GB/s"
                        f"  {img_s:8.0f} img/s"
                    )
                speedup = f"{agg_bw / base_bw:.2f}x"
                print(
                    f"  {f'{method_name}[{_mpi_size}x aggregate]':<28s}"
                    f"  {'':>14s}  "
                    f"  {agg_bw:6.2f} GB/s"
                    f"  {agg_imgs:8.0f} img/s"
                    f"  {speedup:>8s}"
                )
            print(f"{'─'*65}\n")
        return  # MPI path complete

    # ------------------------------------------------------------------
    # Run measurements
    # ------------------------------------------------------------------
    methods = ["baseline", "gpu_raw"]
    if args.with_norm:
        methods.append("gpu_norm")

    all_results: dict[str, list[float]] = {m: [] for m in methods}

    f_h5   = h5py.File(args.h5_file, "r")
    dataset = f_h5["images"]

    from h5py.gpu import GPUDataset
    gpu_ds = GPUDataset(dataset, backend=args.backend)

    total_repeats = args.warmup + args.repeats
    for rep in range(total_repeats):
        tag = "warmup" if rep < args.warmup else f"rep {rep - args.warmup + 1}/{args.repeats}"
        print(f"  Running {tag} …", end=" ", flush=True)

        t_rep = {}
        t_rep["baseline"] = _run_baseline(dataset, batches, backend=args.backend)
        t_rep["gpu_raw"]  = _run_gpu(gpu_ds, batches, normalize=False, backend=args.backend)
        if args.with_norm:
            t_rep["gpu_norm"] = _run_gpu(gpu_ds, batches, normalize=True, backend=args.backend)

        if rep < args.warmup:
            print("discarded")
            continue

        for m in methods:
            all_results[m].extend(t_rep[m])
        print(f"done  (baseline mean {np.mean(t_rep['baseline'])*1e3:.1f} ms/batch)")

    f_h5.close()

    # ------------------------------------------------------------------
    # Multi-GPU measurement
    # ------------------------------------------------------------------
    multi_gpu_results: dict[str, list[list[float]]] = {}

    if args.gpus > 1:
        from concurrent.futures import ProcessPoolExecutor

        gpu_batches = [batches[i::args.gpus] for i in range(args.gpus)]
        gpu_methods_to_run = [("gpu_raw", False)]
        if args.with_norm:
            gpu_methods_to_run.append(("gpu_norm", True))

        for method_name, normalize in gpu_methods_to_run:
            print(f"\n  [{method_name}] dispatching across {args.gpus} GPUs ...",
                  end=" ", flush=True)
            with ProcessPoolExecutor(max_workers=args.gpus) as pool:
                futures = [
                    pool.submit(
                        _gpu_worker, gpu_id, str(args.h5_file),
                        gpu_batches[gpu_id], normalize, args.warmup, args.repeats,
                        args.backend,
                    )
                    for gpu_id in range(args.gpus)
                ]
                multi_gpu_results[method_name] = [f.result() for f in futures]
            print("done")

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------
    print(f"\n{'─'*65}")
    print(f"  {'Method':<18s}  {'ms/batch':>14s}  {'BW':>9s}  {'Throughput':>10s}  {'Speedup':>8s}")
    print(f"{'─'*65}")

    stats     = {m: _stats(all_results[m]) for m in methods}
    base_mean = stats["baseline"]["mean"]

    for m in methods:
        _report_row(m, stats[m], total_bytes, n_batches, base_mean if m != "baseline" else None)

    print(f"{'─'*65}")
    print(f"\n  Normalisation overhead  (gpu_norm vs gpu_raw):  ", end="")
    if args.with_norm:
        overhead = (stats["gpu_norm"]["mean"] - stats["gpu_raw"]["mean"]) / stats["gpu_raw"]["mean"] * 100
        print(f"{overhead:+.1f}%")
    else:
        print("(run with --with-norm to measure)")

    # ------------------------------------------------------------------
    # Multi-GPU report
    # ------------------------------------------------------------------
    if args.gpus > 1:
        baseline_bw = (total_bytes / n_batches) / stats["baseline"]["mean"] / 1e9

        print(f"\n{'─'*65}")
        print(f"  Multi-GPU ({args.gpus} devices)  --  each GPU processes 1/{args.gpus} of batches")
        print(f"  {'Label':<26s}  {'ms/batch':>14s}  {'BW':>9s}  {'Throughput':>10s}  {'vs baseline':>11s}")
        print(f"{'─'*65}")

        for method_name in [m for m in ["gpu_raw", "gpu_norm"] if m in multi_gpu_results]:
            agg_bw   = 0.0
            agg_imgs = 0.0
            for gpu_id in range(args.gpus):
                times     = multi_gpu_results[method_name][gpu_id]
                st        = _stats(times)
                per_bytes = _batch_bytes(gpu_batches[gpu_id], H, W)
                n_per     = len(gpu_batches[gpu_id])
                bw        = (per_bytes / n_per) / st["mean"] / 1e9
                img_s     = 1.0 / st["mean"] * (per_bytes / n_per / 4)
                agg_bw   += bw
                agg_imgs += img_s
                print(
                    f"  {f'{method_name}[gpu{gpu_id}]':<26s}"
                    f"  {st['mean']*1e3:7.1f} +/- {st['std']*1e3:5.1f} ms"
                    f"  {bw:6.2f} GB/s"
                    f"  {img_s:8.0f} img/s"
                )
            speedup = f"{agg_bw / baseline_bw:.2f}x"
            print(
                f"  {f'{method_name}[{args.gpus}x aggregate]':<26s}"
                f"  {'':>14s}  "
                f"  {agg_bw:6.2f} GB/s"
                f"  {agg_imgs:8.0f} img/s"
                f"  {speedup:>11s}"
            )

        print(f"{'─'*65}")

    # ------------------------------------------------------------------
    # CSV output
    # ------------------------------------------------------------------
    if args.csv:
        rows = []
        for m in methods:
            for i, t in enumerate(all_results[m]):
                bw = (total_bytes / n_batches) / t / 1e9
                rows.append({
                    "method":    m,
                    "repeat":    i // n_batches,
                    "batch":     i  % n_batches,
                    "time_s":    t,
                    "bw_gbs":    bw,
                    "img_s":     args.batch_size / t,
                    "chunk_mb":  chunk_mb,
                    "batch_size": args.batch_size,
                    "H":         H,
                    "W":         W,
                })
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with open(args.csv, "w", newline="") as fout:
            writer = csv.DictWriter(fout, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        print(f"\n  Raw results written to: {args.csv}")

    print()


if __name__ == "__main__":
    main()
