#!/usr/bin/env python3
"""
benchmark_worker_reader_scaling.py - Does the GPU-aware loader need as many
CPU workers as the baseline to keep the GPU fed?

Motivated by an observed pattern in real multi-GPU training runs: baseline
throughput (h5py + PyTorch DataLoader, num_workers CPU processes) scales
noticeably with worker count, while gpu-mode throughput (GPUDataset +
_GPUPrefetchLoader, N reader threads) stays roughly flat regardless of
reader count -- because double buffering and pinned transfers already hide
most of the I/O behind the model's forward/backward pass, so extra CPU
parallelism has little left to overlap. The practical consequence: gpu mode
with very few readers can match a baseline that needs several CPU workers
to reach the same throughput -- fewer CPU cores required for the same GPU
utilization, not just a flat speedup at matched worker/reader count.

This sweeps --workers for --mode baseline and --readers for --mode gpu
(single GPU, single process -- the effect is per-process/per-GPU, so it
does not need a multi-GPU DDP run to reproduce or measure), running
train_fastmri_resnet.py as a subprocess for each point and parsing its
own printed throughput summary. Reports:

    baseline: img/s vs. --workers
    gpu:      img/s vs. --readers

and the approximate CROSSOVER -- the smallest baseline --workers value
whose throughput reaches or exceeds gpu mode's (already-flat) throughput --
since that crossover count is the concrete number worth quoting in the
paper ("gpu mode with 1 reader matches baseline needing N workers").

Usage
-----
    python benchmark_worker_reader_scaling.py knee_512.h5 \\
        --baseline-workers 1,2,4,8,16 --gpu-readers 0,1,2,4,8 \\
        --steps 50 --repeats 3 --csv worker_reader_scaling.csv

    # Auto-generate synthetic data instead of pointing at a real file
    python benchmark_worker_reader_scaling.py --layout 3d_chunked \\
        --baseline-workers 1,2,4,8 --gpu-readers 0,1,2,4

Caveats
-------
Baseline's PyTorch DataLoader workers are spawned lazily on the first
iteration, so their one-time process-startup cost lands inside the first
measured step rather than being excluded by a warmup phase (this script's
--steps run is the entire timed window; train_fastmri_resnet.py itself has
no separate untimed warmup). At high --workers counts with a small --steps,
this can measurably depress baseline's apparent throughput -- if the
baseline numbers look noisier or worse than expected at high worker counts,
increase --steps before concluding anything from a single short run.

Each repeat uses a different --seed, so batch order (a full random shuffle
by default) differs run to run for both modes equally -- intentional, to
average out order-dependent I/O variance rather than replaying one fixed
access pattern.

Dependencies
------------
    h5py, cupy, numpy, torch, torchvision (same as train_fastmri_resnet.py)
"""

import argparse
import csv
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

_TOTAL_TIME_RE = re.compile(r"Total training time\s*:\s*([\d.]+)s")
_MEAN_BW_RE    = re.compile(r"Mean I/O bandwidth\s*:\s*([\d.]+) GB/s")


def _run_once(train_script: Path, h5_file: Path, mode: str, batch_size: int,
             steps: int, extra_args: list[str], seed: int) -> dict:
    """Run train_fastmri_resnet.py once as a subprocess; parse its summary."""
    cmd = [sys.executable, str(train_script), str(h5_file),
          "--mode", mode, "--epochs", "1", "--steps", str(steps),
          "--batch-size", str(batch_size), "--seed", str(seed)] + extra_args
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"train_fastmri_resnet.py failed (mode={mode}, args={extra_args}):\n"
            f"--- stdout ---\n{proc.stdout[-3000:]}\n--- stderr ---\n{proc.stderr[-3000:]}"
        )
    time_match = _TOTAL_TIME_RE.search(proc.stdout)
    bw_match   = _MEAN_BW_RE.search(proc.stdout)
    if time_match is None:
        raise RuntimeError(
            f"Could not parse 'Total training time' from output "
            f"(mode={mode}, args={extra_args}):\n{proc.stdout[-3000:]}"
        )
    total_s = float(time_match.group(1))
    mean_bw = float(bw_match.group(1)) if bw_match else float("nan")
    n_images = steps * batch_size
    img_s = n_images / total_s
    return {"total_s": total_s, "mean_bw_gbs": mean_bw, "img_s": img_s}


def _bench_point(train_script, h5_file, mode, batch_size, steps, extra_args,
                 repeats, seed_base):
    results = [
        _run_once(train_script, h5_file, mode, batch_size, steps, extra_args,
                 seed=seed_base + i)
        for i in range(repeats)
    ]
    img_s_vals = np.array([r["img_s"] for r in results])
    return {"mean_img_s": img_s_vals.mean(), "std_img_s": img_s_vals.std(),
           "raw": results}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sweep --workers (baseline) / --readers (gpu) and "
                    "compare data-loading throughput.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("h5_file", type=Path, nargs="?", default=None,
                        help="Existing consolidated fastMRI-layout HDF5 file "
                             "(omit to auto-generate one via --layout)")
    parser.add_argument("--layout", choices=["3d_chunked"], default="3d_chunked",
                        help="Auto-generated data layout if h5_file is omitted "
                             "(only 3d_chunked matches train_fastmri_resnet.py's "
                             "expected schema)")
    parser.add_argument("--n-3d", type=int, default=2048, metavar="N",
                        help="Auto-generated data: number of slices (default: 2048)")
    parser.add_argument("--size-3d", type=int, default=512, metavar="S",
                        help="Auto-generated data: slice size (default: 512)")

    parser.add_argument("--baseline-workers", type=str, default="1,2,4,8,16",
                        metavar="W1,W2,...",
                        help="Comma-separated --workers values to sweep for "
                             "--mode baseline (default: 1,2,4,8,16)")
    parser.add_argument("--gpu-readers", type=str, default="0,1,2,4,8",
                        metavar="R1,R2,...",
                        help="Comma-separated --readers values to sweep for "
                             "--mode gpu (default: 0,1,2,4,8)")
    parser.add_argument("--gpu-prefetch", type=int, default=2, metavar="P",
                        help="Fixed --prefetch used for every gpu-mode run "
                             "(default: 2)")
    parser.add_argument("--batch-size", type=int, default=16, metavar="B")
    parser.add_argument("--steps", type=int, default=50, metavar="N",
                        help="Steps per run (default: 50) -- kept modest so "
                             "the sweep finishes in reasonable time; must be "
                             "less than one epoch's worth of batches for the "
                             "chosen dataset size")
    parser.add_argument("--repeats", type=int, default=3, metavar="R")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--csv", type=Path, default=None, metavar="PATH")
    parser.add_argument("--keep-data", action="store_true")

    args = parser.parse_args()

    repo_dir = Path(__file__).resolve().parent
    train_script = repo_dir / "train_fastmri_resnet.py"

    baseline_workers = [int(x) for x in args.baseline_workers.split(",")]
    gpu_readers      = [int(x) for x in args.gpu_readers.split(",")]

    cleanup_dir = None
    if args.h5_file is not None:
        h5_file = args.h5_file
    else:
        cleanup_dir = Path(tempfile.mkdtemp(prefix="h5py_worker_reader_"))
        h5_file = cleanup_dir / "worker_reader_scaling.h5"
        gen_script = repo_dir / "make_benchmark_data.py"
        subprocess.run(
            [sys.executable, str(gen_script), str(h5_file),
             "--kind", args.layout, "--n", str(args.n_3d), "--size", str(args.size_3d),
             "--force"],
            check=True,
        )

    csv_rows = []

    try:
        print(f"\n{'='*70}")
        print(f"  File       : {h5_file}")
        print(f"  Batch size : {args.batch_size}   Steps/run: {args.steps}   "
              f"Repeats: {args.repeats}")
        print(f"{'='*70}\n")

        print(f"  {'mode':<10s}  {'workers/readers':>16s}  {'img/s':>10s}  "
              f"{'std':>8s}")
        print(f"  {'-'*55}")

        baseline_points = {}
        for w in baseline_workers:
            st = _bench_point(train_script, h5_file, "baseline", args.batch_size,
                              args.steps, ["--workers", str(w)],
                              args.repeats, args.seed)
            baseline_points[w] = st
            print(f"  {'baseline':<10s}  {w:>16d}  {st['mean_img_s']:>10.1f}  "
                  f"{st['std_img_s']:>8.1f}")
            for r in st["raw"]:
                csv_rows.append({"mode": "baseline", "workers_or_readers": w,
                                "img_s": r["img_s"], "total_s": r["total_s"],
                                "mean_bw_gbs": r["mean_bw_gbs"]})

        gpu_points = {}
        for rdr in gpu_readers:
            st = _bench_point(train_script, h5_file, "gpu", args.batch_size,
                              args.steps,
                              ["--readers", str(rdr), "--prefetch", str(args.gpu_prefetch)],
                              args.repeats, args.seed)
            gpu_points[rdr] = st
            print(f"  {'gpu':<10s}  {rdr:>16d}  {st['mean_img_s']:>10.1f}  "
                  f"{st['std_img_s']:>8.1f}")
            for r in st["raw"]:
                csv_rows.append({"mode": "gpu", "workers_or_readers": rdr,
                                "img_s": r["img_s"], "total_s": r["total_s"],
                                "mean_bw_gbs": r["mean_bw_gbs"]})

        print(f"\n{'-'*70}")

        # gpu mode's flat baseline: use the single lowest-reader-count point
        # (closest to a caller who wants to spend as few CPU threads as
        # possible) as the throughput baseline for the crossover search.
        gpu_ref_readers = min(gpu_points)
        gpu_ref_img_s = gpu_points[gpu_ref_readers]["mean_img_s"]
        print(f"  gpu mode @ readers={gpu_ref_readers}: "
              f"{gpu_ref_img_s:.1f} img/s (reference)")

        crossover_w = None
        for w in sorted(baseline_points):
            if baseline_points[w]["mean_img_s"] >= gpu_ref_img_s:
                crossover_w = w
                break

        if crossover_w is not None:
            print(f"  Crossover: baseline needs --workers={crossover_w} to "
                  f"reach gpu mode's throughput at --readers={gpu_ref_readers}")
        else:
            max_w = max(baseline_points)
            print(f"  No crossover: baseline did not reach gpu mode's "
                  f"throughput even at --workers={max_w} "
                  f"({baseline_points[max_w]['mean_img_s']:.1f} img/s vs. "
                  f"{gpu_ref_img_s:.1f} img/s)")
        print(f"{'-'*70}\n")

    finally:
        if cleanup_dir is not None:
            if args.keep_data:
                print(f"Generated data kept in: {cleanup_dir}")
            else:
                shutil.rmtree(cleanup_dir, ignore_errors=True)

    if args.csv and csv_rows:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with open(args.csv, "w", newline="") as fout:
            writer = csv.DictWriter(fout, fieldnames=csv_rows[0].keys())
            writer.writeheader()
            writer.writerows(csv_rows)
        print(f"Raw results written to: {args.csv}")


if __name__ == "__main__":
    main()
