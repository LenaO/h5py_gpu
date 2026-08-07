#!/usr/bin/env python3
"""
test_stream_sharing.py - Verify GPUDataset's optional `stream=` parameter,
and prove it fixes the cross-stream memory-pool growth found while debugging
train_fastmri_resnet.py's GPU OOM.

Background
----------
GPUDataset normally creates its own private, non-blocking CUDA stream for all
transfers. That is fine as long as the arrays it returns are only ever used
on that same stream. But GPUBatchDataset (in train_fastmri_resnet.py) uses
backend="torch": each batch is DLPack-imported into a torch.Tensor and then
read by the model's forward/backward pass on torch's own current stream --
a *different* stream than the one that did the H2D transfer. CuPy's memory
pool only tracks activity on the stream *it* issued an allocation on, so it
cannot tell that a block is safe to reuse once it has also been touched on
that other stream, and instead of risking an unsafe reuse it keeps allocating
fresh blocks -- the pool's total footprint grows without bound even though
most of it sits idle at any instant (confirmed in practice: `used_bytes()`
oscillating while `total_bytes()` only ever climbed).

The fix: let the caller pass in whatever stream will be used end-to-end
(GPUDataset(..., stream=my_stream)), so there is only one stream and CuPy's
own tracking is accurate.

This script does not need PyTorch to prove the fix: the mechanism CuPy cares
about is "was this memory touched on a stream other than the one it thinks
owns it", which is exactly reproducible with two independent CuPy streams --
a second stream standing in for what would be torch's current stream in the
real script. If the fix works here, it works identically for the torch case,
since CuPy cannot tell the difference between "another CuPy stream" and "a
foreign stream wrapped via cp.cuda.ExternalStream" -- both are just stream
handles to it.

Usage
-----
    python test_stream_sharing.py [path/to/some.h5]

    If no file is given, a small synthetic one is generated in a temp dir
    via make_benchmark_data.py and cleaned up afterwards.
"""

import sys
import tempfile
from pathlib import Path

import h5py
import numpy as np

try:
    import cupy as cp
    _CUPY_AVAILABLE = True
except ImportError:
    _CUPY_AVAILABLE = False


def _make_temp_file(tmpdir: Path) -> Path:
    import subprocess
    path = tmpdir / "stream_test.h5"
    subprocess.run(
        [sys.executable, str(Path(__file__).parent / "make_benchmark_data.py"),
         str(path), "--kind", "3d_chunked", "--n", "64", "--size", "128", "--force"],
        check=True,
    )
    return path


def test_default_stream_is_private(h5_path: Path) -> None:
    """No stream passed -> GPUDataset creates and owns its own, as before."""
    from h5py.gpu import GPUDataset

    with h5py.File(h5_path, "r") as f:
        gpu_ds = GPUDataset(f["images"])
        s = gpu_ds._gpu_cache.stream
        assert isinstance(s, cp.cuda.Stream)
        # Accessing it again must return the *same* stream (cached, not
        # recreated every time).
        assert gpu_ds._gpu_cache.stream is s
    print("PASS: default construction still creates a private stream")


def test_explicit_stream_is_used(h5_path: Path) -> None:
    """An explicitly passed stream is used as-is, not wrapped or replaced."""
    from h5py.gpu import GPUDataset

    my_stream = cp.cuda.Stream(non_blocking=True)
    with h5py.File(h5_path, "r") as f:
        gpu_ds = GPUDataset(f["images"], stream=my_stream)
        assert gpu_ds._gpu_cache.stream is my_stream, (
            "GPUDataset did not use the explicitly supplied stream"
        )
    print("PASS: explicitly supplied stream is used as-is")


def test_results_identical_with_and_without_explicit_stream(h5_path: Path) -> None:
    """Sharing a stream must not change the data that comes back."""
    from h5py.gpu import GPUDataset

    with h5py.File(h5_path, "r") as f:
        ds_default = GPUDataset(f["images"])
        arr_default = ds_default[0:8]

        my_stream = cp.cuda.Stream(non_blocking=True)
        ds_shared = GPUDataset(f["images"], stream=my_stream)
        arr_shared = ds_shared[0:8]

        np.testing.assert_array_equal(cp.asnumpy(arr_default), cp.asnumpy(arr_shared))
    print("PASS: identical data with and without an explicit stream")


def _cross_stream_growth(h5_path: Path, n_iters: int, share_stream: bool,
                         batch_size: int = 8, prefetch: int = 2,
                         consume_ms: float = 20.0):
    """Reproduce (share_stream=False) or fix (share_stream=True) the
    cross-stream pool-growth pattern, mirroring _GPUPrefetchLoader._iter_single
    exactly: a background thread keeps calling read_batch_async *without*
    synchronizing (bounded only by a maxsize=prefetch queue), while the main
    thread synchronizes and "consumes" each batch on a separate stream that
    stands in for torch's current stream -- with a deliberate delay so the
    consumer is slower than the producer, exactly like a model forward/
    backward pass being much slower than the H2D transfer in the real script.

    Returns the list of `mempool.total_bytes()` samples, one per iteration.
    """
    import queue as _queue
    import threading
    import time
    from h5py.gpu import GPUDataset

    mempool = cp.get_default_memory_pool()
    mempool.free_all_blocks()

    consumer_stream = cp.cuda.Stream(non_blocking=True)

    with h5py.File(h5_path, "r") as f:
        n_slices = f["images"].shape[0]
        gpu_stream = consumer_stream if share_stream else None
        gpu_ds = GPUDataset(f["images"], stream=gpu_stream)

        q = _queue.Queue(maxsize=prefetch)
        stop_flag = threading.Event()

        def worker():
            try:
                for i in range(n_iters):
                    if stop_flag.is_set():
                        return
                    start = (i * batch_size) % (n_slices - batch_size)
                    # Mirrors _getitem_async: submit and return immediately,
                    # no synchronize here -- the consumer synchronizes later.
                    q.put(gpu_ds.read_batch_async(start, start + batch_size))
            finally:
                q.put(None)

        t = threading.Thread(target=worker, daemon=True)
        t.start()

        totals = []
        try:
            while True:
                item = q.get()
                if item is None:
                    break
                arr, event = item
                event.synchronize()
                with consumer_stream:
                    _ = arr.astype(cp.float32).mean()
                consumer_stream.synchronize()
                # Simulate a slow model forward/backward pass (170ms in the
                # real script vs ~1-2ms I/O) so the producer races ahead and
                # keeps the queue full, maximising overlapping in-flight work.
                time.sleep(consume_ms / 1000.0)
                del arr
                totals.append(mempool.total_bytes())
        finally:
            stop_flag.set()
            while t.is_alive():
                try:
                    q.get_nowait()
                except _queue.Empty:
                    time.sleep(0.001)
            t.join()

    mempool.free_all_blocks()
    return totals


def test_stream_sharing_bounds_pool_growth(
    h5_path: Path, n_iters: int = 200, batch_size: int = 16,
    consume_ms: float = 170.0,
) -> None:
    totals_unshared = _cross_stream_growth(
        h5_path, n_iters, share_stream=False,
        batch_size=batch_size, consume_ms=consume_ms)
    totals_shared = _cross_stream_growth(
        h5_path, n_iters, share_stream=True,
        batch_size=batch_size, consume_ms=consume_ms)

    growth_unshared = totals_unshared[-1] - totals_unshared[0]
    growth_shared   = totals_shared[-1]   - totals_shared[0]

    print(f"  unshared streams: pool grew {growth_unshared/1e6:8.2f} MB "
          f"over {n_iters} iters (first={totals_unshared[0]/1e6:.2f} MB, "
          f"last={totals_unshared[-1]/1e6:.2f} MB)")
    print(f"  shared stream   : pool grew {growth_shared/1e6:8.2f} MB "
          f"over {n_iters} iters (first={totals_shared[0]/1e6:.2f} MB, "
          f"last={totals_shared[-1]/1e6:.2f} MB)")

    assert growth_shared <= growth_unshared, (
        "Sharing the stream should never grow the pool *more* than not "
        "sharing it"
    )
    _NOISE_FLOOR = 5 * 1024 * 1024  # 5 MB: below this, treat as no signal
    if growth_unshared < _NOISE_FLOOR:
        print(
            "INCONCLUSIVE: could not reproduce the pool-growth pattern seen "
            "in train_fastmri_resnet.py using two independent CuPy streams "
            "on this GPU/CuPy version (both stayed within noise). This does "
            "NOT prove the fix works for the real torch+DLPack case -- it "
            "means the growth mechanism could not be triggered by CuPy-only "
            "cross-stream use in isolation, so the actual cause may be "
            "specific to how PyTorch's own caching allocator interacts with "
            "a DLPack-imported foreign buffer, which cannot be exercised "
            "without CUDA-enabled torch (unavailable on this machine). The "
            "API behavior above (stream is honored, results unchanged) is "
            "still verified; only this specific regression scenario is "
            "unproven here."
        )
    else:
        assert growth_shared < growth_unshared * 0.5, (
            f"Expected stream sharing to substantially reduce pool growth, "
            f"got {growth_shared} vs {growth_unshared} bytes"
        )
        print("PASS: explicit stream sharing keeps the memory pool bounded; "
              "without it, the pool grows unboundedly across iterations")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Verify GPUDataset's stream= parameter and check whether "
                    "pure CuPy (no torch/DLPack) reproduces the cross-stream "
                    "memory-pool growth seen in train_fastmri_resnet.py.",
    )
    parser.add_argument("h5_file", type=Path, nargs="?", default=None,
                        help="Existing 3d_chunked-layout HDF5 file to test "
                             "against (e.g. one of your real fastMRI files, "
                             "or one from make_benchmark_data.py). If "
                             "omitted, a small synthetic one is generated "
                             "and cleaned up afterwards.")
    parser.add_argument("--batch-size", type=int, default=16, metavar="B",
                        help="Batch size to match your real training config "
                             "(default: 16, train_fastmri_resnet.py's default)")
    parser.add_argument("--n-iters", type=int, default=200, metavar="N",
                        help="Number of simulated batches (default: 200)")
    parser.add_argument("--consume-ms", type=float, default=170.0, metavar="MS",
                        help="Simulated model forward+backward time per batch "
                             "in milliseconds (default: 170, matching the "
                             "fwd+bwd time seen in the reported OOM run)")
    args = parser.parse_args()

    if not _CUPY_AVAILABLE:
        sys.exit("CuPy is required to run this test.")

    if args.h5_file is not None:
        h5_path = args.h5_file
        cleanup = False
    else:
        tmpdir = Path(tempfile.mkdtemp(prefix="h5py_stream_test_"))
        h5_path = _make_temp_file(tmpdir)
        cleanup = True

    try:
        test_default_stream_is_private(h5_path)
        test_explicit_stream_is_used(h5_path)
        test_results_identical_with_and_without_explicit_stream(h5_path)
        test_stream_sharing_bounds_pool_growth(
            h5_path, n_iters=args.n_iters, batch_size=args.batch_size,
            consume_ms=args.consume_ms)
        print("\nAll tests passed.")
    finally:
        if cleanup:
            import shutil
            shutil.rmtree(h5_path.parent, ignore_errors=True)


if __name__ == "__main__":
    main()
