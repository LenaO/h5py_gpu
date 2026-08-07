#!/usr/bin/env python3
"""
train_fastmri_resnet.py – Train ResNet-18 on fastMRI slices, comparing
standard h5py loading against GPU-aware h5py loading.

Task
----
Coarse anatomical position classification: each MRI slice is labelled with
one of N_CLASSES bins representing its relative depth within the scan
(e.g., bin 0 = top of scan, bin 4 = bottom). This requires no external
annotations — labels are derived from the slice_index / file_index metadata
written by prepare_fastmri.py.

Loading modes
-------------
    baseline   h5py → numpy → DataLoader (pin_memory, num_workers=4)
               Standard PyTorch HDF5 pipeline.
    gpu        GPUDataset → cupy → torch (zero-copy via DLPack, num_workers=0)
               One contiguous batch read per step; read_selection_chunked
               pipelines all B chunks in a single call.

The two modes produce identical model outputs (same weights, same loss) —
only the data pipeline differs, so wall-clock training time directly reflects
I/O throughput.

Usage
-----
    # Baseline
    python train_fastmri_resnet.py knee_512.h5 --mode baseline

    # GPU-aware
    python train_fastmri_resnet.py knee_512.h5 --mode gpu

    # Compare both in one run (runs baseline first, then gpu, same epochs)
    python train_fastmri_resnet.py knee_512.h5 --mode both

    # Quick smoke-test
    python train_fastmri_resnet.py knee_512.h5 --mode both --epochs 1 --steps 20

Dependencies
------------
    h5py, cupy, numpy, torch, torchvision
"""

import argparse
import os
import queue as _queue
import socket as _socket
import subprocess
import sys
import threading
import time
from pathlib import Path

# Prefer the local h5py fork (contains GPU extensions) over any installed copy.
# Must happen before "import h5py" so the module cache is populated correctly.
_local_h5py = Path(__file__).resolve().parent / "h5py"
if _local_h5py.is_dir() and str(_local_h5py) not in sys.path:
    sys.path.insert(0, str(_local_h5py))

import h5py
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torchvision.models as tv_models
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset

try:
    from h5py.gpu import GPUDataset as H5GPUDataset
    _GPU_AVAILABLE = True
except ImportError:
    _GPU_AVAILABLE = False

# POSIX_FADV_SEQUENTIAL = 2: tell the kernel (and GPFS) to prefetch ahead
# aggressively.  No-ops on non-Linux or if libc is unavailable.
try:
    import ctypes
    _libc = ctypes.CDLL("libc.so.6", use_errno=True)
    _libc.posix_fadvise.argtypes = [
        ctypes.c_int, ctypes.c_int64, ctypes.c_int64, ctypes.c_int
    ]
    _FADV_SEQUENTIAL = 2
    def _fadvise_sequential(fd: int, size: int = 0) -> None:
        _libc.posix_fadvise(fd, ctypes.c_int64(0), ctypes.c_int64(size),
                            _FADV_SEQUENTIAL)
except Exception:
    def _fadvise_sequential(fd: int, size: int = 0) -> None:
        pass

N_CLASSES = 5          # anatomical bins (top → bottom of scan)
DEVICE    = torch.device("cpu")  # overwritten in main() once the local rank is known


def _windowed_shuffle(n: int, window: int, rng) -> list:
    """Return indices 0..n-1 shuffled within consecutive windows of `window`.

    window=n  → full random shuffle (maximum randomness, worst seek pattern)
    window=1  → sequential order   (no shuffle, best seek pattern)
    Intermediate values trade off between the two extremes.
    """
    indices = list(range(n))
    for s in range(0, n, window):
        block = indices[s:s + window]
        rng.shuffle(block)
        indices[s:s + len(block)] = block
    return indices


# ---------------------------------------------------------------------------
# Label computation
# ---------------------------------------------------------------------------

def _compute_labels(file_index: np.ndarray, slice_index: np.ndarray) -> np.ndarray:
    """Bin each slice into one of N_CLASSES anatomical position classes."""
    max_per_file = {}
    for fi, si in zip(file_index, slice_index):
        max_per_file[int(fi)] = max(max_per_file.get(int(fi), 0), int(si))

    labels = np.empty(len(file_index), dtype=np.int64)
    for i, (fi, si) in enumerate(zip(file_index, slice_index)):
        n = max_per_file[int(fi)] + 1
        labels[i] = min(int(si / n * N_CLASSES), N_CLASSES - 1)
    return labels


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------

class BaselineDataset(Dataset):
    """Standard h5py dataset: returns (float32 tensor, int64 label).

    Opens the HDF5 file lazily per worker (required for num_workers > 0).
    Each __getitem__ reads ONE full batch so that the DataLoader is used
    with batch_size=1 and a passthrough collate — this matches the GPU
    dataset's batch-level interface and makes the timing comparison fair.
    """

    def __init__(self, h5_path: Path, batch_list: list, labels: np.ndarray):
        self.h5_path    = str(h5_path)
        self.batch_list = batch_list        # list of (start, stop) tuples
        self.labels     = labels            # int64 array, length = n_slices
        self._file      = None             # opened lazily

    def _open(self):
        if self._file is None:
            self._file = h5py.File(self.h5_path, "r")
            _fadvise_sequential(self._file.id.get_vfd_handle())

    def __len__(self):
        return len(self.batch_list)

    def __getitem__(self, idx):
        self._open()
        start, stop = self.batch_list[idx]
        imgs_np  = self.file["images"][start:stop].astype(np.float32)   # (B, H, W)
        imgs     = torch.from_numpy(imgs_np).unsqueeze(1)                 # (B, 1, H, W)
        lbl      = torch.from_numpy(self.labels[start:stop])              # (B,)
        return imgs, lbl

    @property
    def file(self):
        self._open()
        return self._file

    def __del__(self):
        if self._file is not None:
            try:
                self._file.close()
            except Exception:
                pass


# torch.from_dlpack()'s zero-copy import of a CuPy array was observed to
# leak GPU memory in practice (confirmed with --readers 0, 1, and >1, so
# it is not specific to any one loader path; and confirmed to survive
# stream-sharing, gc.collect(), and routing CuPy's own allocations through
# torch's caching allocator via pytorch-pfn-extras -- none of which touch
# it, since the problem appears to be on DLPack's *free* side, not the
# allocation side. This is a known, long-standing bug class at the
# CuPy/PyTorch DLPack boundary; see e.g. pytorch/pytorch#9261 and
# pytorch/pytorch#117273 -- note #9261 is the *reverse* direction from what
# we were doing, so simply swapping which side exports is not automatically
# safe either).
#
# GPUBatchDataset instead never lets CuPy export anything via DLPack at
# all: torch.empty() allocates and owns the destination tensor from the
# start, cp.asarray() borrows a view into it via the CUDA Array Interface
# (a much simpler protocol than DLPack -- a plain {pointer, shape, dtype}
# dict with no deleter and no ownership-transfer bookkeeping at all, so
# there is nothing for that class of bug to hide in), and
# read_batch_async(out=view) transfers directly into torch's memory. This
# is true zero-copy (no extra device-to-device copy, unlike an explicit-copy
# workaround) with no cross-framework ownership contract in the loop.
_CUPY_TO_TORCH_DTYPE = {}
if _GPU_AVAILABLE:
    import cupy as _cp_dtypes
    _CUPY_TO_TORCH_DTYPE = {
        _cp_dtypes.dtype("float32"): torch.float32,
        _cp_dtypes.dtype("float64"): torch.float64,
        _cp_dtypes.dtype("float16"): torch.float16,
        _cp_dtypes.dtype("int64"):   torch.int64,
        _cp_dtypes.dtype("int32"):   torch.int32,
    }


class GPUBatchDataset(Dataset):
    """GPU-aware dataset: reads a whole batch in one GPUDataset call.

    Returns (cuda float32 tensor, cuda int64 label) — already on the device,
    so pin_memory and device transfers in the DataLoader are both skipped.
    Must run with num_workers=0 (CUDA contexts cannot be forked).

    See the module-level comment above _CUPY_TO_TORCH_DTYPE for why this
    reads directly into a torch-allocated tensor via the CUDA Array
    Interface instead of relying on gpu.py's backend="torch" DLPack wrap.
    """

    def __init__(self, h5_path: Path, batch_list: list, labels: np.ndarray):
        if not _GPU_AVAILABLE:
            raise RuntimeError("CuPy / h5py.gpu not available")
        self.h5_path      = str(h5_path)
        self.batch_list   = batch_list
        # Move labels to GPU once; index into them per batch
        self.labels_gpu   = torch.as_tensor(labels, device=DEVICE)
        self._file        = None
        self._gpu_ds      = None
        self._row_shape   = None
        self._torch_dtype = None

    def _open(self):
        if self._file is None:
            self._file  = h5py.File(self.h5_path, "r")
            _fadvise_sequential(self._file.id.get_vfd_handle())
            # Share torch's current stream with GPUDataset instead of letting
            # it create its own, so the H2D transfer and the model's forward/
            # backward pass happen on the same stream.
            stream = None
            if DEVICE.type == "cuda":
                import cupy as _cp
                stream = _cp.cuda.ExternalStream(
                    torch.cuda.current_stream().cuda_stream)
            # backend="cupy": we manage the torch-side tensor ourselves (see
            # _read_into_torch), so gpu.py never needs to DLPack-wrap
            # anything -- it just transfers into a CuPy view we hand it.
            self._gpu_ds = H5GPUDataset(self._file["images"], backend="cupy",
                                        stream=stream)
            ds = self._file["images"]
            self._row_shape   = ds.shape[1:]
            self._torch_dtype = _CUPY_TO_TORCH_DTYPE[ds.dtype]

    def _read_into_torch(self, start, stop, buf_idx=0):
        """Read dataset[start:stop] directly into a freshly torch-allocated
        tensor. Returns (tensor, event_or_None); event is unsynchronized --
        callers must wait on it (if not None) before reading the tensor,
        same contract read_batch_async itself has.

        buf_idx must alternate (0, 1, 0, 1, ...) across calls that may be
        in flight at the same time on this instance -- i.e. issued before
        the previous call's event has been synchronized. read_batch_async
        stages each transfer through a pinned host buffer selected by
        buf_idx; passing the same value to two overlapping calls would let
        the second one's CPU-side read overwrite the pinned buffer while
        the first call's H2D transfer might still be reading from it.
        """
        import cupy as _cp
        B = stop - start
        out_torch = torch.empty((B,) + self._row_shape,
                                dtype=self._torch_dtype, device=DEVICE)
        out_cupy = _cp.asarray(out_torch)  # CUDA Array Interface view, not DLPack
        if hasattr(self._gpu_ds, "read_batch_async"):
            _, event = self._gpu_ds.read_batch_async(start, stop, out=out_cupy,
                                                      buf_idx=buf_idx)
        else:
            # Rare fallback for layouts that miss the fast path: gpu.py
            # allocates its own array here since there is no out= for plain
            # indexing, so an explicit copy into out_cupy is unavoidable.
            out_cupy[...] = self._gpu_ds[start:stop]
            event = None
        return out_torch, event

    def __len__(self):
        return len(self.batch_list)

    def __getitem__(self, idx):
        self._open()
        start, stop = self.batch_list[idx]
        out_torch, event = self._read_into_torch(start, stop)
        if event is not None:
            event.synchronize()
        imgs = out_torch.unsqueeze(1)                                # (B, 1, H, W) float32 cuda
        lbl  = self.labels_gpu[start:stop]                            # (B,) cuda
        return imgs, lbl

    def _getitem_async(self, idx):
        """Return (imgs, lbl, event_or_None) without waiting for H2D to finish.

        The caller must call event.synchronize() before using imgs, same
        contract as before this change. idx alternates buf_idx (see
        _read_into_torch) since _iter_single calls this once per idx,
        in order, without synchronizing between consecutive calls.
        """
        self._open()
        start, stop = self.batch_list[idx]
        out_torch, event = self._read_into_torch(start, stop, buf_idx=idx % 2)
        imgs = out_torch.unsqueeze(1)
        lbl  = self.labels_gpu[start:stop]
        return imgs, lbl, event

    def clone(self):
        """Return a new instance with its own file handle but shared labels_gpu.

        Each reader thread must own its own h5py.File + GPUDataset because
        neither is thread-safe.  labels_gpu is a read-only GPU tensor and is
        safe to share across threads.
        """
        inst = object.__new__(GPUBatchDataset)
        inst.h5_path      = self.h5_path
        inst.batch_list   = self.batch_list   # not used by workers directly
        inst.labels_gpu   = self.labels_gpu   # read-only, safe to share
        inst._row_shape   = None
        inst._torch_dtype = None
        inst._file      = None
        inst._gpu_ds    = None
        return inst

    def __del__(self):
        if self._file is not None:
            try:
                self._file.close()
            except Exception:
                pass


class _GPUPrefetchLoader:
    """Iterate GPUBatchDataset with background reader thread(s).

    readers=1  (default)
        Single background thread: submit H2D async, main thread waits on the
        CUDA event while the worker reads the next batch from HDF5.

    readers=N  (N > 1)
        N independent threads, each owning its own h5py.File + GPUDataset.
        They pull work from a shared queue and push (order_idx, tensor, event)
        results to a shared result queue.  The main thread re-sequences results
        in epoch order before yielding, so the training loop sees a determistic
        batch stream regardless of which reader finishes first.

    prefetch controls the per-reader result buffer depth (total buffer ≈
    prefetch × readers batches ahead).
    """

    def __init__(self, dataset: "GPUBatchDataset", prefetch: int = 2,
                 readers: int = 1):
        self._dataset  = dataset
        self._prefetch = prefetch
        self._readers  = readers

    def __len__(self):
        return len(self._dataset)

    def __iter__(self):
        if self._readers <= 1:
            yield from self._iter_single()
        else:
            yield from self._iter_multi()

    # ------------------------------------------------------------------
    # Single-reader path
    # ------------------------------------------------------------------

    def _iter_single(self):
        ds        = self._dataset
        n         = len(ds)
        # queue.Queue(maxsize=0) means UNBOUNDED in Python, not "no
        # prefetching" -- the opposite of what --prefetch 0 sounds like it
        # should mean. Without this guard the worker thread races arbitrarily
        # far ahead of the consumer, each batch holding its own live GPU
        # tensor, which is exactly the "still growing, but the script runs"
        # symptom reported with --prefetch 0 (bounded only by how many
        # batches are left in the epoch, not by anything intentional).
        q         = _queue.Queue(maxsize=max(1, self._prefetch))
        stop_flag = threading.Event()
        _async    = hasattr(ds, "_getitem_async")

        def _worker():
            try:
                for idx in range(n):
                    if stop_flag.is_set():
                        return
                    if _async:
                        q.put(ds._getitem_async(idx))   # (imgs, lbl, event)
                    else:
                        imgs, lbl = ds[idx]
                        q.put((imgs, lbl, None))
            finally:
                q.put(None)

        t = threading.Thread(target=_worker, daemon=True)
        t.start()
        try:
            while True:
                item = q.get()
                if item is None:
                    break
                imgs, lbl, event = item
                if event is not None:
                    event.synchronize()
                yield imgs, lbl
        finally:
            stop_flag.set()
            # Drain so the worker is never blocked on q.put() and can reach
            # the None sentinel, avoiding a t.join() deadlock.
            while t.is_alive():
                try:
                    q.get_nowait()
                except _queue.Empty:
                    threading.Event().wait(0.001)
            while True:
                try:
                    q.get_nowait()
                except _queue.Empty:
                    break
            t.join()

    # ------------------------------------------------------------------
    # Multi-reader path
    # ------------------------------------------------------------------

    def _iter_multi(self):
        ds        = self._dataset
        n         = len(ds)
        n_readers = self._readers

        # Snapshot the batch list now so epoch-shuffle can't race with workers.
        batch_list = list(ds.batch_list)

        # Bound on how far any reader is allowed to race ahead of next_order:
        # without this, independent reader threads finish batches out of
        # order as a matter of course (any timing skew between threads makes
        # this certain over a real run), and every batch finished ahead of
        # next_order piles up in `pending` waiting for a slower reader to
        # catch up. result_q's own maxsize does NOT protect against this --
        # the main loop below drains every item out of result_q into pending
        # unconditionally, so a bounded queue feeding an unbounded dict is
        # still unbounded overall. Feeding work_q incrementally, one new item
        # per batch actually consumed, keeps the total number of batches
        # in flight or waiting in `pending` capped at `window` at all times.
        window = max(1, self._prefetch) * n_readers

        # work_q  : (order_idx, start, stop)  then one None per reader
        # result_q: (order_idx, imgs, lbl, event)  then one None per reader
        work_q   = _queue.Queue()
        result_q = _queue.Queue(maxsize=window)
        stop_flag = threading.Event()

        n_submitted = min(window, n)
        for order_idx in range(n_submitted):
            start, stop = batch_list[order_idx]
            work_q.put((order_idx, start, stop))
        # Sentinels are appended once every batch has been submitted (see
        # the replenishment step below); with n <= window they belong here.
        if n_submitted >= n:
            for _ in range(n_readers):
                work_q.put(None)

        def _reader(reader):
            # Local to this thread: alternates the pinned-buffer slot across
            # this reader's own consecutive calls, which are never
            # synchronized against each other before the next one starts
            # (see _read_into_torch's buf_idx note). order_idx is not usable
            # for this -- it is a global batch index, not guaranteed to
            # alternate parity across the specific items this thread picks
            # up from the shared work_q.
            local_count = 0
            try:
                reader._open()
                while True:
                    item = work_q.get()
                    if item is None:
                        return
                    if stop_flag.is_set():
                        continue        # drain work_q without doing I/O
                    order_idx, start, stop = item
                    out_torch, event = reader._read_into_torch(
                        start, stop, buf_idx=local_count % 2)
                    local_count += 1
                    imgs = out_torch.unsqueeze(1)
                    lbl  = reader.labels_gpu[start:stop]
                    result_q.put((order_idx, imgs, lbl, event))
            except Exception:
                import traceback
                traceback.print_exc()
            finally:
                result_q.put(None)      # signal that this worker is done

        threads = []
        for _ in range(n_readers):
            t = threading.Thread(target=_reader, args=(ds.clone(),), daemon=True)
            t.start()
            threads.append(t)

        pending      = {}               # order_idx → (imgs, lbl, event)
        next_order   = 0
        next_to_submit = n_submitted
        done_workers = 0

        try:
            while next_order < n or done_workers < n_readers:
                item = result_q.get()
                if item is None:
                    done_workers += 1
                    continue
                order_idx, imgs, lbl, event = item
                pending[order_idx] = (imgs, lbl, event)
                # Flush all consecutive ready batches in epoch order.
                while next_order in pending:
                    p_imgs, p_lbl, p_event = pending.pop(next_order)
                    if p_event is not None:
                        p_event.synchronize()
                    yield p_imgs, p_lbl
                    next_order += 1
                    # Replenish the lookahead window by exactly one batch
                    # per batch actually consumed, so the number in flight
                    # or pending never exceeds `window`.
                    if next_to_submit < n:
                        s, e = batch_list[next_to_submit]
                        work_q.put((next_to_submit, s, e))
                        next_to_submit += 1
                        if next_to_submit >= n:
                            for _ in range(n_readers):
                                work_q.put(None)
        finally:
            stop_flag.set()
            # On an early exit (e.g. --steps cutting an epoch short), not all
            # n batches may have been submitted yet, so the sentinels that
            # normally follow the last submitted batch may never be pushed.
            # A reader currently blocked on work_q.get() would then wait
            # forever. Push one sentinel per reader unconditionally -- extra,
            # unclaimed sentinels left over from the normal path are harmless.
            for _ in range(n_readers):
                work_q.put(None)
            # Drain result_q until every worker has sent its None sentinel.
            while done_workers < n_readers:
                try:
                    item = result_q.get(timeout=0.05)
                except _queue.Empty:
                    continue
                if item is None:
                    done_workers += 1
            for t in threads:
                t.join()


def _passthrough_collate(batch):
    """DataLoader collate that unwraps the single-element list."""
    assert len(batch) == 1
    return batch[0]


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def build_model(img_size: int) -> nn.Module:
    """ResNet-18 adapted for 1-channel input and N_CLASSES output."""
    model = tv_models.resnet18(weights=None)
    # Replace first conv: 3-channel → 1-channel, keep everything else
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    # Replace final FC: 512 → N_CLASSES
    model.fc    = nn.Linear(model.fc.in_features, N_CLASSES)
    return model.to(DEVICE)


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

class AverageMeter:
    def __init__(self, keep_samples: bool = False):
        self._keep = keep_samples
        self.reset()

    def reset(self):
        self.sum = self.count = self.max = 0.0
        self._samples: list = [] if self._keep else None

    def update(self, val, n=1):
        self.sum   += val * n
        self.count += n
        if val > self.max:
            self.max = val
        if self._keep:
            self._samples.append(val)

    @property
    def avg(self):
        return self.sum / self.count if self.count else 0.0

    def percentile(self, p: float) -> float:
        if not self._keep or not self._samples:
            return float("nan")
        s = sorted(self._samples)
        idx = max(0, int(len(s) * p / 100) - 1)
        return s[idx]

    def stall_count(self, threshold_x: float = 5.0) -> int:
        """Steps where the value exceeded threshold_x × median."""
        if not self._keep or not self._samples:
            return 0
        med = self.percentile(50)
        return sum(1 for v in self._samples if v > threshold_x * med)


def _to_device(imgs, lbl):
    """Move tensors to DEVICE if not already there."""
    if imgs.device != DEVICE:
        imgs = imgs.to(DEVICE, non_blocking=True)
    if lbl.device != DEVICE:
        lbl  = lbl.to(DEVICE, non_blocking=True)
    return imgs, lbl


def _ddp_aggregate(metrics: dict) -> dict:
    """Average scalar metrics across all DDP ranks via all_reduce."""
    keys = ["loss", "acc", "io_ms", "fwd_ms", "io_pct", "bw_gbs"]
    t = torch.tensor([metrics[k] for k in keys], dtype=torch.float64, device=DEVICE)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    t /= dist.get_world_size()
    return {**metrics, **{k: t[i].item() for i, k in enumerate(keys)}}


def train_epoch(
    loader: DataLoader,
    model: nn.Module,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    max_steps: int | None,
    verbose: bool = True,
    mem_flush_steps: int = 0,
) -> dict:
    model.train()
    loss_m   = AverageMeter()
    acc_m    = AverageMeter()
    io_m     = AverageMeter(keep_samples=True)
    fwd_m    = AverageMeter()

    step = 0
    t_io_start = time.perf_counter()

    for imgs, lbl in loader:
        io_time = time.perf_counter() - t_io_start
        io_m.update(io_time)

        imgs, lbl = _to_device(imgs, lbl)
        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
        t_fwd = time.perf_counter()

        logits = model(imgs)
        loss   = criterion(logits, lbl)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
        fwd_m.update(time.perf_counter() - t_fwd)

        with torch.no_grad():
            acc = (logits.argmax(1) == lbl).float().mean().item()
        loss_m.update(loss.item(), imgs.shape[0])
        acc_m.update(acc,          imgs.shape[0])

        step += 1
        if verbose and step % 10 == 0:
            total_s = (io_m.sum + fwd_m.sum)
            io_pct  = 100 * io_m.sum / total_s if total_s else 0
            mem_dbg = ""
            if DEVICE.type == "cuda":
                # TEMPORARY diagnostic for the OOM investigation -- remove once
                # resolved. Reports both allocators separately: torch's own
                # caching allocator (model/activations/gradients) vs CuPy's
                # memory pool (DLPack-imported batch tensors from GPUDataset).
                # If only one of these grows across steps, that tells us which
                # side is holding on to memory it should be releasing.
                torch_alloc = torch.cuda.memory_allocated() / 1e6
                torch_resv  = torch.cuda.memory_reserved()  / 1e6
                try:
                    import cupy as _cp
                    cp_pool = _cp.get_default_memory_pool()
                    cp_used  = cp_pool.used_bytes()  / 1e6
                    cp_total = cp_pool.total_bytes() / 1e6
                except ImportError:
                    cp_used = cp_total = float("nan")
                mem_dbg = (
                    f"  [torch alloc={torch_alloc:.1f}MB resv={torch_resv:.1f}MB"
                    f"  cupy used={cp_used:.1f}MB pool={cp_total:.1f}MB]"
                )
            print(
                f"    step {step:>4d}  loss {loss_m.avg:.4f}  acc {acc_m.avg:.3f}"
                f"  io {io_m.avg*1e3:.1f}ms  fwd+bwd {fwd_m.avg*1e3:.1f}ms"
                f"  [I/O {io_pct:.0f}% of wall]{mem_dbg}",
                flush=True,
            )

        if mem_flush_steps and step % mem_flush_steps == 0 and DEVICE.type == "cuda":
            # TEMPORARY diagnostic for the OOM investigation: flushing at
            # this granularity instead of once per epoch tells us whether
            # the pool growth is genuinely reclaimable (growth rate should
            # drop roughly in proportion to how much more often we flush)
            # or a true leak (flushing more often would not help at all,
            # since free_all_blocks() only returns blocks the pool already
            # considers free -- it cannot touch a block still referenced by
            # something).
            import gc
            import cupy as _cp
            gc.collect()
            _cp.get_default_memory_pool().free_all_blocks()

        if max_steps and step >= max_steps:
            break

        t_io_start = time.perf_counter()

    bytes_per_batch = imgs.shape[0] * imgs.shape[2] * imgs.shape[3] * 4
    bw_gbs = (bytes_per_batch * step) / (io_m.sum * 1e9) if io_m.sum > 0 else 0.0

    return {
        "loss":        loss_m.avg,
        "acc":         acc_m.avg,
        "io_ms":       io_m.avg  * 1e3,
        "io_p50_ms":   io_m.percentile(50)  * 1e3,
        "io_p95_ms":   io_m.percentile(95)  * 1e3,
        "io_p99_ms":   io_m.percentile(99)  * 1e3,
        "io_max_ms":   io_m.max  * 1e3,
        "io_stalls":   io_m.stall_count(threshold_x=5.0),
        "fwd_ms":      fwd_m.avg * 1e3,
        "io_pct":      100 * io_m.sum / (io_m.sum + fwd_m.sum),
        "bw_gbs":      bw_gbs,
        "steps":       step,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _run(mode: str, h5_path: Path, args,
         rank: int = 0, world_size: int = 1, is_ddp: bool = False) -> dict:
    """Set up and train for one mode ('baseline' or 'gpu').

    In DDP mode each rank receives a contiguous shard of the batch list so
    that every process reads a sequential region of the HDF5 file.  Metrics
    are all-reduced to rank-0 after every epoch; only rank 0 prints.
    """
    is_main = (rank == 0)

    if is_main:
        print(f"\n{'='*60}")
        ddp_tag = f"  [DDP {world_size}×GPU]" if is_ddp else ""
        print(f"  Mode: {mode.upper()}{ddp_tag}")
        print(f"{'='*60}")

    with h5py.File(h5_path, "r") as f:
        n_slices, H, W = f["images"].shape
        chunk_shape    = f["images"].chunks
        chunk_mb       = chunk_shape[1] * chunk_shape[2] * 4 / 1e6
        file_index     = f["file_index"][:]
        slice_index    = f["slice_index"][:]

    if is_main:
        print(f"  Dataset  : {n_slices:,} slices  {H}×{W}  "
              f"chunk={chunk_shape}  ({chunk_mb:.2f} MB/chunk)")

    labels = _compute_labels(file_index, slice_index)

    # Build the full batch lists, then give each rank a contiguous shard.
    # Contiguous sharding keeps each GPU's reads sequential in the HDF5 file.
    n_train = int(n_slices * 0.8)
    all_train = [(s, s + args.batch_size)
                 for s in range(0, n_train - args.batch_size + 1, args.batch_size)]
    all_val   = [(s, s + args.batch_size)
                 for s in range(n_train, n_slices - args.batch_size + 1, args.batch_size)]

    def _shard(lst):
        # Floor division: every rank gets exactly the same number of batches.
        # At most (world_size - 1) batches are dropped per epoch — negligible.
        # Ceiling division would give rank N-1 one fewer batch, causing every
        # all_reduce and DDP gradient sync to deadlock on the last step.
        sz    = len(lst) // world_size
        start = rank * sz
        return lst[start: start + sz]

    train_batches = _shard(all_train)
    val_batches   = _shard(all_val)

    # Per-rank RNG so different ranks shuffle their shard differently.
    rng = np.random.default_rng(args.seed + rank)

    if mode == "baseline":
        train_ds = BaselineDataset(h5_path, train_batches, labels)
        val_ds   = BaselineDataset(h5_path, val_batches,   labels)
        train_loader = DataLoader(
            train_ds, batch_size=1, shuffle=False,
            num_workers=args.workers, pin_memory=(DEVICE.type == "cuda"),
            collate_fn=_passthrough_collate,
        )
        val_loader = DataLoader(
            val_ds, batch_size=1, shuffle=False,
            num_workers=args.workers, pin_memory=(DEVICE.type == "cuda"),
            collate_fn=_passthrough_collate,
        )
    else:
        if not _GPU_AVAILABLE:
            if is_main:
                print("  [SKIP] CuPy / h5py.gpu not available — skipping gpu mode")
            return {}
        train_ds = GPUBatchDataset(h5_path, train_batches, labels)
        val_ds   = GPUBatchDataset(h5_path, val_batches,   labels)
        train_loader = _GPUPrefetchLoader(train_ds, prefetch=args.prefetch,
                                          readers=args.readers)
        val_loader   = _GPUPrefetchLoader(val_ds,   prefetch=args.prefetch,
                                          readers=args.readers)

    model     = build_model(H)
    if is_ddp:
        model = DDP(model, device_ids=[DEVICE.index])
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs * len(train_loader)
    )

    if is_main:
        n_params = sum(p.numel() for p in model.parameters())
        print(f"  Model    : ResNet-18  ({n_params/1e6:.1f}M params)  "
              f"→ {N_CLASSES}-class anatomical position")
        print(f"  Batches  : {len(all_train):,} train  {len(all_val):,} val  "
              f"(batch_size={args.batch_size}  per-rank: {len(train_batches):,})")
        if mode == "baseline":
            print(f"  Workers  : {args.workers}  pin_memory={DEVICE.type == 'cuda'}")
        else:
            print(f"  Workers  : {args.readers} reader thread(s)  "
                  f"prefetch_depth={args.prefetch}")
        print()

    epoch_results = []
    t_total = time.perf_counter()

    window = args.shuffle_window if args.shuffle_window else len(train_batches)

    for epoch in range(1, args.epochs + 1):
        perm = _windowed_shuffle(len(train_batches), window, rng)
        train_ds.batch_list = [train_batches[i] for i in perm]

        if is_main:
            print(f"  Epoch {epoch}/{args.epochs}")

        r = train_epoch(train_loader, model, criterion, optimizer,
                        args.steps, verbose=is_main,
                        mem_flush_steps=args.mem_flush_steps)
        scheduler.step()

        if mode == "gpu" and DEVICE.type == "cuda":
            # If free_all_blocks() alone only slows (rather than stops) pool
            # growth, some blocks are not being marked free at all -- i.e.
            # something is still holding a live reference to them, which
            # free_all_blocks() cannot touch (it only returns blocks the pool
            # already considers free). gc.collect() first catches the case
            # where that reference is part of an unreachable *cycle*: CPython
            # only reclaims cycles during a cyclic-GC pass, not via plain
            # refcounting, so a cycle involving a batch tensor would sit
            # uncollected between passes even though nothing "wants" it
            # anymore.
            import gc
            import cupy as _cp
            gc.collect()
            _cp.get_default_memory_pool().free_all_blocks()

        if is_ddp:
            r = _ddp_aggregate(r)

        epoch_results.append(r)

        if is_main:
            stall_warn = (f"  *** {r['io_stalls']} stall(s) >5×median"
                          if r["io_stalls"] else "")
            print(
                f"  → loss {r['loss']:.4f}  acc {r['acc']:.3f}"
                f"  |  io avg {r['io_ms']:.1f}  p50 {r['io_p50_ms']:.1f}"
                f"  p95 {r['io_p95_ms']:.1f}  p99 {r['io_p99_ms']:.1f}"
                f"  max {r['io_max_ms']:.1f} ms"
                f"  |  fwd+bwd {r['fwd_ms']:.1f}ms"
                f"  |  I/O {r['io_pct']:.0f}%  BW {r['bw_gbs']:.2f} GB/s"
                + stall_warn
            )

    total_s = time.perf_counter() - t_total
    mean_bw = np.mean([r["bw_gbs"] for r in epoch_results])

    if is_main:
        print(f"\n  Total training time : {total_s:.1f}s")
        print(f"  Mean I/O bandwidth  : {mean_bw:.2f} GB/s")
    return {"mode": mode, "epoch_results": epoch_results,
            "total_s": total_s, "mean_bw": mean_bw}


def _print_comparison(results: dict) -> None:
    if len(results) < 2:
        return
    base = results.get("baseline", {})
    gpu  = results.get("gpu",      {})
    if not base or not gpu:
        return

    base_io  = np.mean([r["io_ms"]  for r in base["epoch_results"]])
    gpu_io   = np.mean([r["io_ms"]  for r in gpu["epoch_results"]])
    base_bw  = base["mean_bw"]
    gpu_bw   = gpu["mean_bw"]

    print(f"\n{'='*60}")
    print(f"  COMPARISON SUMMARY")
    print(f"{'─'*60}")
    print(f"  {'':20s}  {'baseline':>10s}  {'gpu':>10s}  {'speedup':>8s}")
    print(f"  {'I/O ms/batch':20s}  {base_io:>10.1f}  {gpu_io:>10.1f}"
          f"  {base_io/gpu_io:>7.2f}×")
    print(f"  {'BW (GB/s)':20s}  {base_bw:>10.2f}  {gpu_bw:>10.2f}"
          f"  {gpu_bw/base_bw:>7.2f}×")
    print(f"  {'Total time (s)':20s}  {base['total_s']:>10.1f}  {gpu['total_s']:>10.1f}"
          f"  {base['total_s']/gpu['total_s']:>7.2f}×")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Train ResNet-18 on fastMRI slices (baseline vs GPU-aware loading).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("h5_file",    type=Path)
    parser.add_argument("--mode",     choices=["baseline", "gpu", "both"],
                        default="both",
                        help="Loading mode (default: both)")
    parser.add_argument("--epochs",   type=int,   default=3)
    parser.add_argument("--steps",    type=int,   default=None,
                        help="Max steps per epoch (omit = full epoch)")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Images per batch (default: 16)")
    parser.add_argument("--lr",       type=float, default=1e-3)
    parser.add_argument("--workers",  type=int,   default=4,
                        help="num_workers for baseline DataLoader (default: 4)")
    parser.add_argument("--prefetch", type=int,   default=2,
                        help="Batches to prefetch ahead in GPU mode (default: 2). "
                             "Values <= 0 are treated the same as 1, not as "
                             "'no prefetching': Python's queue.Queue(maxsize=0) "
                             "means unbounded, which would let the reader "
                             "thread race arbitrarily far ahead instead.")
    parser.add_argument("--readers",  type=int,   default=1,
                        help="Number of parallel HDF5 reader threads in GPU mode. "
                             "Each thread owns its own file handle + CUDA stream. "
                             "(default: 1)")
    parser.add_argument("--shuffle-window", type=int, default=None,
                        metavar="W",
                        help="Shuffle within windows of W consecutive batches "
                             "instead of fully random shuffle. Smaller W = "
                             "more sequential reads (faster on HPC filesystems). "
                             "Default: full random shuffle.")
    parser.add_argument("--seed",     type=int,   default=42)
    parser.add_argument("--mem-flush-steps", type=int, default=0, metavar="N",
                        help="TEMPORARY diagnostic for the GPU-mode OOM "
                             "investigation: every N steps, run gc.collect() "
                             "then cupy's free_all_blocks() (default: 0, "
                             "disabled -- only the once-per-epoch flush "
                             "runs). Compare growth rate at a few values of "
                             "N to tell reclaimable-but-fragmented pool "
                             "growth (rate drops with smaller N) from a true "
                             "reference leak (rate is unaffected).")
    parser.add_argument("--dist-init-file", default=None, metavar="HOST",
                        help="Override MASTER_ADDR for the SLURM env:// rendezvous "
                             "(default: first host from scontrol show hostnames). "
                             "Set MASTER_ADDR / MASTER_PORT in the job script instead "
                             "if the cluster doesn't have scontrol in PATH.")
    args = parser.parse_args()

    # -----------------------------------------------------------------------
    # DDP initialisation — three supported launch methods:
    #
    #   torchrun  (local / non-HPC)
    #       torchrun --nproc_per_node=4 train_fastmri_resnet.py ...
    #       Sets RANK / LOCAL_RANK / WORLD_SIZE; uses TCP rendezvous.
    #
    #   srun  (SLURM / HPC — recommended when TCP ports are blocked)
    #       srun --ntasks-per-node=4 --gres=gpu:4 python train_fastmri_resnet.py ...
    #       Sets SLURM_PROCID / SLURM_LOCALID / SLURM_NTASKS; uses a shared
    #       filesystem file for rendezvous so no open TCP port is required.
    #
    #   single GPU  (no launcher)
    #       python train_fastmri_resnet.py ...
    # -----------------------------------------------------------------------
    _torchrun = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    _slurm    = "SLURM_PROCID" in os.environ and not _torchrun
    is_ddp    = _torchrun or _slurm

    if _torchrun:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        rank       = dist.get_rank()
        world_size = dist.get_world_size()

    elif _slurm:
        rank       = int(os.environ["SLURM_PROCID"])
        local_rank = int(os.environ["SLURM_LOCALID"])
        world_size = int(os.environ["SLURM_NTASKS"])

        # Resolve MASTER_ADDR from SLURM_NODELIST unless already set by the
        # job script.  scontrol expands bracket notation (jwb[0418,0422])
        # into one hostname per line; we take the first as the rendezvous host.
        if "MASTER_ADDR" not in os.environ:
            if args.dist_init_file:
                # Legacy: caller supplied an explicit host via --dist-init-file.
                os.environ["MASTER_ADDR"] = args.dist_init_file
            else:
                try:
                    master = subprocess.check_output(
                        ["scontrol", "show", "hostnames",
                         os.environ["SLURM_NODELIST"]],
                        text=True, stderr=subprocess.DEVNULL,
                    ).splitlines()[0].strip()
                except Exception:
                    master = os.environ.get("SLURM_LAUNCH_NODE_IPADDR",
                                            "localhost")
                # Resolve to an IPv4 address — FQDNs on some clusters resolve
                # to IPv6, which causes EAFNOSUPPORT in NCCL's IPv4 socket.
                try:
                    master = _socket.getaddrinfo(
                        master, None, _socket.AF_INET
                    )[0][4][0]
                except Exception:
                    pass
                os.environ["MASTER_ADDR"] = master

        if "MASTER_PORT" not in os.environ:
            # Derive a port from the job-id so concurrent jobs don't collide.
            job_id = int(os.environ.get("SLURM_JOB_ID", "0"))
            os.environ["MASTER_PORT"] = str(20000 + job_id % 10000)

        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            rank=rank,
            world_size=world_size,
        )

    else:
        local_rank = 0
        rank       = 0
        world_size = 1

    # Set the CUDA device for this process.  Must happen before any CUDA
    # allocation, including CuPy streams created inside GPUBatchDataset.
    global DEVICE
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        DEVICE = torch.device(f"cuda:{local_rank}")
        try:
            import cupy as cp
            cp.cuda.Device(local_rank).use()
        except ImportError:
            pass

        if args.mode in ("gpu", "both"):
            # CuPy and PyTorch normally maintain separate, non-coordinating
            # CUDA memory allocators. The batch tensors GPUBatchDataset
            # creates are CuPy-allocated but DLPack-imported into torch, and
            # that cross-framework ownership handoff has long-standing,
            # documented leak issues (e.g. pytorch/pytorch#9261,
            # pytorch/pytorch#117273) -- consistent with the unbounded CuPy
            # pool growth seen in --mode gpu training that neither
            # gc.collect() nor explicit stream-sharing resolved. Routing
            # CuPy's allocations through torch's own caching allocator
            # removes the cross-framework handoff (and its bugs) entirely,
            # since only one allocator ever owns the memory.
            try:
                import pytorch_pfn_extras as ppe
                ppe.cuda.use_torch_mempool_in_cupy()
                if rank == 0:
                    print("  Routing CuPy allocations through torch's "
                          "caching allocator (pytorch-pfn-extras)")
            except ImportError:
                if rank == 0:
                    print("  NOTE: pytorch-pfn-extras not installed -- CuPy "
                          "and torch will use separate memory pools, which "
                          "is the suspected cause of unbounded GPU memory "
                          "growth in --mode gpu. Install with:\n"
                          "      pip install pytorch-pfn-extras")
    else:
        if rank == 0:
            print("Warning: CUDA not available — running on CPU")

    if not args.h5_file.exists():
        sys.exit(f"File not found: {args.h5_file}")

    modes = ["baseline", "gpu"] if args.mode == "both" else [args.mode]
    all_results = {}

    for mode in modes:
        res = _run(mode, args.h5_file, args,
                   rank=rank, world_size=world_size, is_ddp=is_ddp)
        if res and rank == 0:
            all_results[mode] = res

    if rank == 0:
        _print_comparison(all_results)

    if is_ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
