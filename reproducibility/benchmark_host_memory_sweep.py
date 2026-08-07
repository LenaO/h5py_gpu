#!/usr/bin/env python3
"""
benchmark_host_memory_sweep.py - Host memory footprint vs. dataset size:
does loading a whole dataset onto the GPU require holding it, all at once,
in host RAM first?

This is the direct payoff of the module's reusable pinned-buffer design
(paper Section "Reusable Transfer Resources"): each dataset wrapper streams
data through two small, reused pinned host buffers -- sized to one HDF5
chunk or row-band, never to the dataset -- rather than staging the whole
file through host memory before any of it reaches the GPU.

Two methods, both reading the ENTIRE dataset (the "full dataset" case, #1
in the paper's Table 1), swept across dataset SIZE:

    naive   h5py_ds[:] -> a full numpy array in host RAM -> cp.asarray().
            Host RAM peak scales linearly with dataset size: the whole file
            sits in host memory at once before any of it reaches the GPU.

    ours    read_chunks_to_gpu() (chunked datasets) or
            read_double_buffered() (contiguous datasets) -- streams the
            dataset through two small reused pinned buffers, overlapping
            storage reads with H2D transfers. Host RAM peak should stay
            FLAT regardless of dataset size, even though the resulting GPU
            array is exactly as large as the dataset -- host and device
            memory usage decouple.

GPU memory used is also reported, as a sanity check: it should grow with
dataset size for BOTH methods (the final array is the same size either
way) -- this benchmark is specifically about the HOST side; it does not
claim GPU memory stays flat for a full-dataset load (see
benchmark_use_cases.py / Table 1's case #3 for the reduction case, where
GPU memory stays flat too).

Because peak host memory (Windows: peak working set; POSIX: ru_maxrss) is
a MONOTONIC, whole-process high-water mark that never resets, each (size,
method) measurement is taken in its own freshly spawned subprocess --
otherwise a single large measurement would contaminate every later,
smaller one for the rest of the process's lifetime.

Usage
-----
    python benchmark_host_memory_sweep.py --layout 3d_chunked \\
        --sizes-mb 128,256,512,1024,2048,4096

    python benchmark_host_memory_sweep.py --all-layouts \\
        --sizes-mb 128,512,2048 --csv host_mem_sweep.csv

Choose --sizes-mb with the test machine's available RAM in mind: the naive
method genuinely needs that much host memory, by design -- that is the
point being demonstrated, not a bug. Default sizes are kept modest (up to
4 GB) to stay safe on a typical laptop; on an HPC node with more RAM, much
larger sizes make the divergence even more dramatic.

Dependencies
------------
    h5py, cupy, numpy. No extra dependency for the memory measurement
    itself -- uses ctypes + GetProcessMemoryInfo on Windows, the `resource`
    module on POSIX.
"""

import argparse
import csv
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

_LAYOUTS = ["2d_chunked", "2d_contiguous", "3d_chunked", "3d_contiguous"]
_SLICE_SIZE_3D = 512  # fixed per-slice H=W for 3-D layouts; only n varies


# ---------------------------------------------------------------------------
# Peak host memory measurement (monotonic high-water mark for this process)
# ---------------------------------------------------------------------------

def _peak_host_bytes() -> int:
    """Return this process's peak resident memory (working set) in bytes
    since process start. Monotonic: never decreases, even after freeing
    memory -- see the module docstring for why each measurement needs its
    own fresh subprocess."""
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        class _PROCESS_MEMORY_COUNTERS(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        kernel32 = ctypes.windll.kernel32
        psapi = ctypes.windll.psapi
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        psapi.GetProcessMemoryInfo.argtypes = [
            wintypes.HANDLE, ctypes.POINTER(_PROCESS_MEMORY_COUNTERS),
            wintypes.DWORD,
        ]
        psapi.GetProcessMemoryInfo.restype = wintypes.BOOL

        counters = _PROCESS_MEMORY_COUNTERS()
        counters.cb = ctypes.sizeof(_PROCESS_MEMORY_COUNTERS)
        handle = kernel32.GetCurrentProcess()
        ok = psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb)
        if not ok:
            raise OSError("GetProcessMemoryInfo failed")
        return int(counters.PeakWorkingSetSize)
    else:
        import resource
        ru_maxrss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # Linux reports ru_maxrss in KB; macOS reports it in bytes.
        return ru_maxrss if sys.platform == "darwin" else ru_maxrss * 1024


# ---------------------------------------------------------------------------
# Worker: performs exactly ONE read, then reports its own peak memory
# ---------------------------------------------------------------------------

def _run_worker(file: Path, dataset: str, method: str) -> None:
    import h5py
    import cupy as cp
    from h5py.gpu import GPUDataset

    f = h5py.File(file, "r")
    ds = f[dataset]

    if method == "naive":
        arr_np = ds[:]
        arr_gpu = cp.asarray(arr_np)
    else:
        gpu_ds = GPUDataset(ds)
        if ds.chunks is not None:
            arr_gpu = gpu_ds.read_chunks_to_gpu()
        else:
            arr_gpu = gpu_ds.read_double_buffered()
    cp.cuda.Device().synchronize()

    gpu_bytes = cp.get_default_memory_pool().used_bytes()
    host_bytes = _peak_host_bytes()
    f.close()

    # Single machine-parseable line, printed last: the driver reads the
    # LAST line with this prefix, so any incidental library stdout noise
    # earlier cannot corrupt the result.
    print(f"RESULT host_bytes={host_bytes} gpu_bytes={gpu_bytes}")


def _spawn_worker(h5_path: Path, dataset_name: str, method: str) -> tuple[int, int]:
    proc = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--worker",
         "--file", str(h5_path), "--dataset", dataset_name, "--method", method],
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"Worker failed (method={method}, file={h5_path}):\n"
            f"--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}"
        )
    for line in reversed(proc.stdout.splitlines()):
        if line.startswith("RESULT "):
            parts = dict(kv.split("=") for kv in line[len("RESULT "):].split())
            return int(parts["host_bytes"]), int(parts["gpu_bytes"])
    raise RuntimeError(
        f"Worker produced no RESULT line (method={method}, file={h5_path}):\n"
        f"--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}"
    )


# ---------------------------------------------------------------------------
# Driver: generate data, sweep sizes, print + collect results
# ---------------------------------------------------------------------------

def _gen_args_for_size(layout: str, size_mb: int) -> list[str]:
    target_bytes = size_mb * 1024 * 1024
    if layout in ("2d_chunked", "2d_contiguous"):
        side = max(64, int(round((target_bytes / 4) ** 0.5)))
        args = ["--kind", layout, "--rows", str(side), "--cols", str(side)]
        if layout == "2d_chunked":
            args += ["--chunk-rows", "256", "--chunk-cols", "256"]
        return args
    else:
        n = max(1, int(round(target_bytes / (_SLICE_SIZE_3D ** 2 * 4))))
        return ["--kind", layout, "--n", str(n), "--size", str(_SLICE_SIZE_3D)]


def run_sweep_for_layout(layout: str, sizes_mb: list[int], tmpdir: Path,
                         csv_rows: list | None, keep_data: bool) -> None:
    dataset_name = "images" if layout.startswith("3d") else "data"
    gen_script = Path(__file__).resolve().parent / "make_benchmark_data.py"

    print(f"\n{'='*78}")
    print(f"  Layout: {layout}")
    print(f"{'='*78}")
    print(f"  {'size (MB)':>10s}  {'method':<8s}  {'host peak (MB)':>16s}  "
          f"{'GPU used (MB)':>14s}")
    print(f"  {'-'*60}")

    for size_mb in sizes_mb:
        path = tmpdir / f"sweep_{layout}_{size_mb}mb.h5"
        gen_args = _gen_args_for_size(layout, size_mb)

        gen = subprocess.run(
            [sys.executable, str(gen_script), str(path), "--force"] + gen_args,
            capture_output=True, text=True,
        )
        if gen.returncode != 0:
            raise RuntimeError(
                f"Data generation failed for {layout} @ {size_mb}MB:\n"
                f"{gen.stdout}\n{gen.stderr}"
            )

        for method in ("naive", "ours"):
            host_bytes, gpu_bytes = _spawn_worker(path, dataset_name, method)
            print(f"  {size_mb:>10d}  {method:<8s}  {host_bytes/1e6:16.1f}  "
                  f"{gpu_bytes/1e6:14.1f}")
            if csv_rows is not None:
                csv_rows.append({
                    "layout": layout, "size_mb": size_mb, "method": method,
                    "host_peak_bytes": host_bytes, "gpu_used_bytes": gpu_bytes,
                })

        if not keep_data:
            path.unlink(missing_ok=True)  # bound disk usage during the sweep
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Host memory footprint vs. dataset size: naive "
                    "(load-to-numpy-then-GPU) vs. streaming through "
                    "reused pinned buffers.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    # Internal worker mode -- one clean subprocess per (size, method).
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--file", type=Path, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--dataset", type=str, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--method", choices=["naive", "ours"], default=None,
                        help=argparse.SUPPRESS)

    parser.add_argument("--layout", choices=_LAYOUTS, default=None,
                        help="Single layout to sweep")
    parser.add_argument("--all-layouts", action="store_true",
                        help="Sweep all four layout combinations")
    parser.add_argument("--sizes-mb", type=str,
                        default="128,256,512,1024,2048,4096",
                        metavar="M1,M2,...",
                        help="Comma-separated dataset sizes in MB "
                             "(default: 128,256,512,1024,2048,4096)")
    parser.add_argument("--csv", type=Path, default=None, metavar="PATH",
                        help="Write raw (layout, size, method) results to a CSV file")
    parser.add_argument("--keep-data", action="store_true",
                        help="Don't delete generated .h5 files after each "
                             "size point (for debugging; uses more disk)")

    args = parser.parse_args()

    if args.worker:
        _run_worker(args.file, args.dataset, args.method)
        return

    try:
        import cupy  # noqa: F401
    except ImportError:
        sys.exit("CuPy is required (GPUDataset uses it internally). "
                 "Install with: pip install cupy-cuda12x")

    if not args.layout and not args.all_layouts:
        sys.exit("Provide --layout <name> or --all-layouts")

    sizes_mb = [int(x) for x in args.sizes_mb.split(",")]
    layouts = _LAYOUTS if args.all_layouts else [args.layout]

    tmpdir = Path(tempfile.mkdtemp(prefix="h5py_host_mem_sweep_"))
    csv_rows = [] if args.csv else None
    try:
        for layout in layouts:
            run_sweep_for_layout(layout, sizes_mb, tmpdir, csv_rows,
                                 keep_data=args.keep_data)
    finally:
        if args.keep_data:
            print(f"\nGenerated data kept in: {tmpdir}")
        else:
            shutil.rmtree(tmpdir, ignore_errors=True)

    if args.csv and csv_rows:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with open(args.csv, "w", newline="") as fout:
            writer = csv.DictWriter(fout, fieldnames=csv_rows[0].keys())
            writer.writeheader()
            writer.writerows(csv_rows)
        print(f"\nRaw results written to: {args.csv}")


if __name__ == "__main__":
    main()
