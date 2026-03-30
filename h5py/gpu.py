# This file is part of h5py, a Python interface to the HDF5 library.
#
# http://www.h5py.org
#
# Copyright 2008-2013 Andrew Collette and contributors
#
# License:  Standard 3-clause BSD; see "license.txt" for full license terms
#           and contributor agreement.

"""
GPU support for h5py via CuPy.

This module provides wrappers around h5py Dataset and Group objects that
transparently read HDF5 data into GPU memory.

The read path is:
  HDF5 file  →  pinned (page-locked) host memory  →  GPU device memory

Using pinned memory avoids an extra pageable-to-pinned copy inside CUDA's
DMA engine and typically yields higher host-to-device bandwidth than
transferring from ordinary numpy arrays.

Basic usage::

    import h5py
    from h5py.gpu import GPUFile

    with GPUFile("data.h5") as f:
        arr = f["dataset"][:]        # returns a cupy.ndarray on the GPU
        arr = f["dataset"][0:100]    # partial read, also on GPU

    # Wrap an already-open h5py.File or h5py.Dataset
    from h5py.gpu import GPUGroup, GPUDataset

    with h5py.File("data.h5") as f:
        gpu_f   = GPUGroup(f)
        gpu_ds  = gpu_f["dataset"]   # GPUDataset
        arr     = gpu_ds[:]          # cupy.ndarray

        # Or wrap a Dataset directly:
        gpu_ds2 = GPUDataset(f["dataset"])
        arr2    = gpu_ds2[0:10]

    # Read into a pre-allocated CuPy array (avoids one temporary allocation)
    import cupy as cp
    with h5py.File("data.h5") as f:
        gpu_ds = GPUDataset(f["dataset"])
        dest   = cp.empty(f["dataset"].shape, dtype=f["dataset"].dtype)
        gpu_ds.read_direct_gpu(dest)
"""

import itertools
import numpy as np

try:
    import cupy as cp
    _CUPY_AVAILABLE = True
except ImportError:
    _CUPY_AVAILABLE = False

from ._hl.dataset import Dataset
from ._hl.group import Group
from ._hl.files import File


def _require_cupy():
    if not _CUPY_AVAILABLE:
        raise ImportError(
            "CuPy is required for GPU support but could not be imported. "
            "Install it with: pip install cupy-cuda12x  "
            "(adjust the CUDA suffix to match your toolkit version)"
        )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _numpy_to_gpu(arr):
    """Copy a numpy array to the GPU via pinned (page-locked) memory.

    Allocates a page-locked host buffer, copies *arr* into it, then issues
    a DMA transfer to the current CUDA device.  This is faster than passing
    a pageable numpy array to :func:`cupy.array` because the CUDA driver does
    not need to stage through a hidden pinned buffer of its own.

    Parameters
    ----------
    arr : numpy.ndarray
        Source array.  Any dtype and shape are supported.

    Returns
    -------
    cupy.ndarray
        A new device array containing the same data.
    """
    _require_cupy()
    # Fast path for empty arrays (nbytes == 0)
    if arr.nbytes == 0:
        return cp.empty(arr.shape, dtype=arr.dtype)
    # Allocate pinned memory and create a numpy view of it
    pinned_mem = cp.cuda.alloc_pinned_memory(arr.nbytes)
    pinned = np.frombuffer(pinned_mem, dtype=arr.dtype, count=arr.size).reshape(arr.shape)
    np.copyto(pinned, arr)
    # Transfer to GPU (cupy.array understands __cuda_array_interface__ and
    # also plain numpy arrays backed by pinned memory)
    return cp.array(pinned)


def _alloc_pinned_like(shape, dtype):
    """Return a C-contiguous numpy array backed by pinned memory.

    Parameters
    ----------
    shape : tuple of int
    dtype : numpy dtype or compatible

    Returns
    -------
    (pinned_mem, numpy.ndarray)
        *pinned_mem* keeps the allocation alive; *numpy.ndarray* is a view.
    """
    _require_cupy()
    dtype = np.dtype(dtype)
    size = int(np.prod(shape))
    nbytes = size * dtype.itemsize
    pinned_mem = cp.cuda.alloc_pinned_memory(nbytes)
    arr = np.frombuffer(pinned_mem, dtype=dtype, count=size).reshape(shape)
    return pinned_mem, arr


# ---------------------------------------------------------------------------
# Selection helpers (for read_selection_chunked)
# ---------------------------------------------------------------------------

def _normalize_sel(args, shape):
    """Convert *args* to a plain tuple of slices, or return ``(None, None)``.

    Handles ``slice``, ``Ellipsis``, and missing trailing dimensions.
    Returns ``(None, None)`` for integer indices, stepped slices, or any form
    of fancy indexing — the caller should fall back to the simple path.

    Returns
    -------
    (tuple[slice], tuple[int])
        Normalized selection and the resulting output shape.
    """
    ndim = len(shape)

    # Expand a single Ellipsis into the right number of full slices
    n_ellipsis = sum(1 for a in args if a is Ellipsis)
    if n_ellipsis > 1:
        return None, None
    expanded = []
    for a in args:
        if a is Ellipsis:
            expanded.extend([slice(None)] * (ndim - (len(args) - 1)))
        else:
            expanded.append(a)
    # Pad missing trailing dims
    while len(expanded) < ndim:
        expanded.append(slice(None))
    if len(expanded) != ndim:
        return None, None

    result, out_shape = [], []
    for a, n in zip(expanded, shape):
        if not isinstance(a, slice):
            return None, None          # integer / array index → fall back
        start, stop, step = a.indices(n)
        if step != 1:
            return None, None          # stepped slice → fall back
        result.append(slice(start, stop))
        out_shape.append(max(0, stop - start))

    return tuple(result), tuple(out_shape)


def _iter_touched_chunks(shape, chunks, sel):
    """Yield info for every HDF5 chunk that overlaps *sel*.

    Parameters
    ----------
    shape  : tuple[int]  — dataset shape
    chunks : tuple[int]  — HDF5 chunk shape
    sel    : tuple[slice] — normalized selection (output of :func:`_normalize_sel`)

    Yields
    ------
    (chunk_file_sel, actual_chunk_shape, local_sel, out_sel)
        chunk_file_sel    : slices selecting the full chunk from the dataset
        actual_chunk_shape: actual size of this chunk (clipped at dataset edge)
        local_sel         : slices into the chunk for the requested sub-region
        out_sel           : slices into the output array for this sub-region
    """
    dim_info = []
    for dim_size, chunk_size, s in zip(shape, chunks, sel):
        sel_start, sel_stop = s.start, s.stop
        if sel_start >= sel_stop:          # empty selection in this dim
            dim_info.append([])
            break
        first_ci = sel_start // chunk_size
        last_ci  = (sel_stop - 1) // chunk_size

        d_info = []
        for ci in range(first_ci, last_ci + 1):
            chunk_start = ci * chunk_size
            chunk_stop  = min(chunk_start + chunk_size, dim_size)

            overlap_start = max(chunk_start, sel_start)
            overlap_stop  = min(chunk_stop,  sel_stop)

            d_info.append({
                'chunk_file':  slice(chunk_start, chunk_stop),
                'actual_size': chunk_stop - chunk_start,
                'local':       slice(overlap_start - chunk_start,
                                     overlap_stop  - chunk_start),
                'out':         slice(overlap_start - sel_start,
                                     overlap_stop  - sel_start),
            })
        dim_info.append(d_info)

    for combo in itertools.product(*dim_info):
        yield (
            tuple(d['chunk_file']  for d in combo),
            tuple(d['actual_size'] for d in combo),
            tuple(d['local']       for d in combo),
            tuple(d['out']         for d in combo),
        )


def _async_h2d_subtile(chunk_ptr, actual_chunk_shape, local_sel, out, out_sel, stream):
    """Submit async H2D DMA for a sub-region of a pinned chunk buffer into ``out``.

    The source is *not* the whole chunk but the sub-region ``local_sel`` within
    it.  ``memcpy2DAsync`` is used with the chunk's row pitch as the source
    pitch, so no intermediate CPU extraction is needed.

    Parameters
    ----------
    chunk_ptr         : int  — host pointer to the start of the pinned chunk data
    actual_chunk_shape: tuple[int]  — shape of the full chunk in pinned memory
    local_sel         : tuple[slice] — sub-region within the chunk to copy
    out               : cupy.ndarray (C-contiguous)
    out_sel           : tuple[slice] — where the sub-region goes in *out*
    stream            : cupy.cuda.Stream
    """
    itemsize = out.dtype.itemsize
    H2D = cp.cuda.runtime.memcpyHostToDevice

    if len(actual_chunk_shape) == 2:
        act_r, act_c = actual_chunk_shape
        r0l, r1l = local_sel[0].start, local_sel[0].stop
        c0l, c1l = local_sel[1].start, local_sel[1].stop
        R_out, C_out = out.shape
        r0o, c0o = out_sel[0].start, out_sel[1].start

        cp.cuda.runtime.memcpy2DAsync(
            out.data.ptr + (r0o * C_out + c0o) * itemsize,   # dst
            C_out * itemsize,                                   # dst pitch
            chunk_ptr + (r0l * act_c + c0l) * itemsize,        # src (offset into chunk)
            act_c * itemsize,                                   # src pitch (chunk row)
            (c1l - c0l) * itemsize,                             # copy width
            r1l - r0l,                                          # copy height (rows)
            H2D, stream.ptr,
        )

    elif len(actual_chunk_shape) == 3:
        act_d, act_r, act_c = actual_chunk_shape
        d0l, d1l = local_sel[0].start, local_sel[0].stop
        r0l, r1l = local_sel[1].start, local_sel[1].stop
        c0l, c1l = local_sel[2].start, local_sel[2].stop
        D_out, R_out, C_out = out.shape
        d0o, r0o, c0o = out_sel[0].start, out_sel[1].start, out_sel[2].start
        slice_elems = act_r * act_c

        for d in range(d1l - d0l):
            src = chunk_ptr + ((d0l + d) * slice_elems + r0l * act_c + c0l) * itemsize
            dst = out.data.ptr + ((d0o + d) * R_out * C_out + r0o * C_out + c0o) * itemsize
            cp.cuda.runtime.memcpy2DAsync(
                dst, C_out * itemsize,
                src, act_c * itemsize,
                (c1l - c0l) * itemsize,
                r1l - r0l,
                H2D, stream.ptr,
            )

    else:
        raise ValueError(
            f"_async_h2d_subtile: unsupported ndim={len(actual_chunk_shape)} (2 or 3 only)"
        )


# ---------------------------------------------------------------------------
# D2H helpers (mirror of H2D helpers, for write path)
# ---------------------------------------------------------------------------

def _async_d2h_tile(dst_ptr, tile_shape, src, sel, stream):
    """Submit async D2H DMA: extract ``src[sel]`` into a compact pinned buffer.

    This is the write-side mirror of :func:`_async_h2d_tile`.  The GPU source
    is strided (full array row pitch); the pinned destination is compact (no
    padding between rows of the tile).

    Parameters
    ----------
    dst_ptr    : int  — host pointer to the start of the compact pinned buffer
    tile_shape : tuple[int]  — actual shape of the tile (= shape of src[sel])
    src        : cupy.ndarray (C-contiguous)
    sel        : tuple[slice] — selection into *src* for this tile
    stream     : cupy.cuda.Stream
    """
    itemsize = src.dtype.itemsize
    D2H = cp.cuda.runtime.memcpyDeviceToHost

    if len(tile_shape) == 2:
        R, C = src.shape
        ar, ac = tile_shape
        r0, c0 = sel[0].start, sel[1].start
        cp.cuda.runtime.memcpy2DAsync(
            dst_ptr,                                     # dst (compact pinned)
            ac * itemsize,                               # dst pitch
            src.data.ptr + (r0 * C + c0) * itemsize,    # src (strided GPU)
            C * itemsize,                                # src pitch (full row)
            ac * itemsize,                               # copy width
            ar,                                          # copy height
            D2H, stream.ptr,
        )

    elif len(tile_shape) == 3:
        D, R, C = src.shape
        ad, ar, ac = tile_shape
        d0, r0, c0 = sel[0].start, sel[1].start, sel[2].start
        slice_bytes = ar * ac * itemsize
        for d in range(ad):
            cp.cuda.runtime.memcpy2DAsync(
                dst_ptr + d * slice_bytes,
                ac * itemsize,
                src.data.ptr + ((d0 + d) * R * C + r0 * C + c0) * itemsize,
                C * itemsize,
                ac * itemsize,
                ar,
                D2H, stream.ptr,
            )

    else:
        raise ValueError(
            f"_async_d2h_tile: unsupported ndim={len(tile_shape)} (2 or 3 only)"
        )


def _async_d2h_subtile(dst_ptr, compact_shape, src, src_sel, stream):
    """Submit async D2H DMA for a sub-region of *src* into a compact pinned buffer.

    This is the write-side mirror of :func:`_async_h2d_subtile`.

    Parameters
    ----------
    dst_ptr       : int  — host pointer to compact pinned destination
    compact_shape : tuple[int]  — shape of the sub-region being extracted
    src           : cupy.ndarray (C-contiguous)
    src_sel       : tuple[slice] — sub-region of *src* to copy
    stream        : cupy.cuda.Stream
    """
    itemsize = src.dtype.itemsize
    D2H = cp.cuda.runtime.memcpyDeviceToHost

    if len(compact_shape) == 2:
        R_src, C_src = src.shape
        ar, ac = compact_shape
        r0s, c0s = src_sel[0].start, src_sel[1].start
        cp.cuda.runtime.memcpy2DAsync(
            dst_ptr,                                          # dst (compact)
            ac * itemsize,                                    # dst pitch
            src.data.ptr + (r0s * C_src + c0s) * itemsize,   # src (strided)
            C_src * itemsize,                                 # src pitch
            ac * itemsize,                                    # copy width
            ar,                                               # copy height
            D2H, stream.ptr,
        )

    elif len(compact_shape) == 3:
        D_src, R_src, C_src = src.shape
        ad, ar, ac = compact_shape
        d0s, r0s, c0s = src_sel[0].start, src_sel[1].start, src_sel[2].start
        slice_bytes = ar * ac * itemsize
        for d in range(ad):
            cp.cuda.runtime.memcpy2DAsync(
                dst_ptr + d * slice_bytes,
                ac * itemsize,
                src.data.ptr + ((d0s + d) * R_src * C_src + r0s * C_src + c0s) * itemsize,
                C_src * itemsize,
                ac * itemsize,
                ar,
                D2H, stream.ptr,
            )

    else:
        raise ValueError(
            f"_async_d2h_subtile: unsupported ndim={len(compact_shape)} (2 or 3 only)"
        )


# ---------------------------------------------------------------------------
# Tile-based helpers (for read_chunks_to_gpu / write_chunks_from_gpu)
# ---------------------------------------------------------------------------

def _iter_tiles(shape, chunks):
    """Yield ``(file_sel, tile_shape)`` for every HDF5 chunk in row-major order.

    *file_sel* is a tuple of slices that selects the chunk from the dataset.
    *tile_shape* is the actual shape of that chunk (may be smaller than
    *chunks* at the edge of each dimension).
    """
    dim_ranges = [
        [(s, min(s + c, n)) for s in range(0, n, c)]
        for n, c in zip(shape, chunks)
    ]
    for corners in itertools.product(*dim_ranges):
        sel = tuple(slice(s, e) for s, e in corners)
        tile_shape = tuple(e - s for s, e in corners)
        yield sel, tile_shape


def _async_h2d_tile(src_ptr, tile_shape, out, sel, stream):
    """Submit async H2D DMA for one tile from pinned memory into ``out[sel]``.

    Uses ``memcpy2DAsync`` so the tile lands directly at its final location
    in *out* without requiring a GPU scatter kernel.  For 3-D tiles a separate
    ``memcpy2DAsync`` is issued per depth-slice (all non-blocking, on *stream*).

    Parameters
    ----------
    src_ptr : int
        Host pointer to the start of the pinned tile data (C-contiguous,
        row-major order matching *tile_shape*).
    tile_shape : tuple of int  (length 2 or 3)
        Actual shape of the tile.
    out : cupy.ndarray
        C-contiguous destination array on the GPU.
    sel : tuple of slice
        Selection into *out* where the tile should be placed.
    stream : cupy.cuda.Stream
    """
    itemsize = out.dtype.itemsize
    H2D = cp.cuda.runtime.memcpyHostToDevice

    if len(tile_shape) == 2:
        R, C = out.shape
        ar, ac = tile_shape
        r0, c0 = sel[0].start, sel[1].start
        cp.cuda.runtime.memcpy2DAsync(
            out.data.ptr + (r0 * C + c0) * itemsize,   # dst: out[r0, c0]
            C * itemsize,                                # dst pitch (full row)
            src_ptr,                                     # src
            ac * itemsize,                               # src pitch (tile row)
            ac * itemsize,                               # copy width in bytes
            ar,                                          # copy height in rows
            H2D, stream.ptr,
        )

    elif len(tile_shape) == 3:
        D, R, C = out.shape
        ad, ar, ac = tile_shape
        d0, r0, c0 = sel[0].start, sel[1].start, sel[2].start
        slice_bytes = ar * ac * itemsize          # bytes per depth-slice in tile
        for d in range(ad):
            cp.cuda.runtime.memcpy2DAsync(
                out.data.ptr + ((d0 + d) * R * C + r0 * C + c0) * itemsize,
                C * itemsize,
                src_ptr + d * slice_bytes,
                ac * itemsize,
                ac * itemsize,
                ar,
                H2D, stream.ptr,
            )

    else:
        raise ValueError(
            f"_async_h2d_tile: unsupported ndim={len(tile_shape)} (2 or 3 only)"
        )


# ---------------------------------------------------------------------------
# GPUDataset
# ---------------------------------------------------------------------------

class GPUDataset:
    """Wrapper around :class:`h5py.Dataset` that returns :class:`cupy.ndarray`.

    All indexing and attribute access is delegated to the underlying
    :class:`~h5py.Dataset`.  Only the *output* is redirected to GPU memory.

    Parameters
    ----------
    dataset : h5py.Dataset
        An open h5py dataset.
    """

    def __init__(self, dataset):
        if not isinstance(dataset, Dataset):
            raise TypeError(
                f"GPUDataset requires an h5py.Dataset, got {type(dataset)!r}"
            )
        _require_cupy()
        # Store under a mangled name so __getattr__ does not recurse
        object.__setattr__(self, "_gpu_dataset", dataset)

    # ------------------------------------------------------------------
    # Core read interface
    # ------------------------------------------------------------------

    def __getitem__(self, args):
        """Read a slice from HDF5 into GPU memory.

        Accepts the same indexing syntax as :class:`h5py.Dataset`.

        For HDF5-chunked 2-D and 3-D datasets with simple slice selections,
        reading is automatically dispatched to :meth:`read_selection_chunked`,
        which reads each touched HDF5 chunk in full (mirroring HDF5's own
        behaviour) and uses double-buffering to overlap storage reads with
        host-to-device transfers.  All other cases use the simple path.

        Returns
        -------
        cupy.ndarray
            The requested data on the current CUDA device.
        """
        dataset = object.__getattribute__(self, "_gpu_dataset")

        # Fast path: chunked 2-D/3-D with a plain slice selection
        if dataset.chunks is not None and dataset.ndim in (2, 3):
            _args = args if isinstance(args, tuple) else (args,)
            sel, out_shape = _normalize_sel(_args, dataset.shape)
            if sel is not None:
                return self.read_selection_chunked(sel)

        # Fall back: let h5py handle the selection, then transfer to GPU
        arr = dataset[args]
        if not isinstance(arr, np.ndarray):
            return arr
        return _numpy_to_gpu(arr)

    def read_selection_chunked(self, sel, out=None, stream=None, transform=None):
        """Read a slice selection from a chunked dataset to the GPU,
        processing one HDF5 chunk at a time with double-buffering.

        HDF5 always decompresses a full chunk even when only part of it is
        needed.  This method mirrors that behaviour explicitly:

        1. Determine which HDF5 chunks overlap the requested selection.
        2. For each touched chunk: read the **full** chunk into a pinned host
           buffer (one HDF5 I/O call, one decompression pass).
        3. Extract the requested sub-region from the chunk and copy it
           directly to its correct position in the output array using
           ``memcpy2DAsync`` with the chunk row-pitch as the source stride —
           no intermediate CPU extraction step is needed.
        4. If *transform* is provided, it is enqueued on the same CUDA stream
           immediately after the H2D copy, so compute overlaps with the CPU
           reading the next chunk.

        Double-buffering overlaps step 2 for chunk *i+1* with step 3 for
        chunk *i*.

        Parameters
        ----------
        sel : tuple[slice]
            Selection as a tuple of plain (step-1) slices, one per dataset
            dimension.  Produce it with ``numpy.s_[10:50, 20:80]`` or pass
            the normalized output of :func:`_normalize_sel`.
        out : cupy.ndarray, optional
            Pre-allocated C-contiguous output array of the correct shape and
            dtype.  Allocated automatically when *None*.
        stream : cupy.cuda.Stream, optional
            CUDA stream for H2D transfers.  A new non-blocking stream is
            created when *None*.
        transform : callable, optional
            Element-wise operation applied to each tile on the GPU **after**
            the H2D transfer, on the same stream.  Called as
            ``out[out_sel] = transform(out[out_sel])`` inside a
            ``with stream:`` block.  Any CuPy ufunc or lambda works::

                gpu_ds.read_selection_chunked(sel, transform=cp.sqrt)
                gpu_ds.read_selection_chunked(sel, transform=lambda x: x * 2.0)

        Returns
        -------
        cupy.ndarray

        Raises
        ------
        ValueError
            If the dataset is not HDF5-chunked, has ``ndim`` other than 2 or
            3, or *sel* contains stepped slices.
        """
        dataset = object.__getattribute__(self, "_gpu_dataset")

        if dataset.chunks is None:
            raise ValueError(
                "Dataset is not HDF5-chunked. "
                "Use read_double_buffered() for contiguous datasets."
            )
        if dataset.ndim not in (2, 3):
            raise ValueError(
                f"read_selection_chunked supports 2-D and 3-D datasets, "
                f"got ndim={dataset.ndim}"
            )

        # Accept numpy.s_-style tuples as well as already-normalised selections
        sel = sel if isinstance(sel, tuple) else (sel,)
        sel, out_shape = _normalize_sel(sel, dataset.shape)
        if sel is None:
            raise ValueError(
                "sel must be a tuple of plain (step-1) slices. "
                "For fancy indexing use GPUDataset[...] which falls back "
                "to the simple path."
            )

        dtype  = np.dtype(dataset.dtype)
        chunks = dataset.chunks

        if out is None:
            out = cp.empty(out_shape, dtype=dtype)
        else:
            if not isinstance(out, cp.ndarray):
                raise TypeError(f"out must be a cupy.ndarray, got {type(out)!r}")
            if out.shape != out_shape or out.dtype != dtype:
                raise ValueError(
                    f"out shape/dtype {out.shape}/{out.dtype} does not match "
                    f"expected {out_shape}/{dtype}"
                )
            if not out.flags["C_CONTIGUOUS"]:
                raise ValueError("out must be C-contiguous")

        if stream is None:
            stream = cp.cuda.Stream(non_blocking=True)

        # Two pinned host buffers — each big enough for one full (max) chunk
        max_chunk_elems = int(np.prod(chunks))
        pms  = [cp.cuda.alloc_pinned_memory(max_chunk_elems * dtype.itemsize)
                for _ in range(2)]
        bufs = [np.frombuffer(pm, dtype=dtype, count=max_chunk_elems) for pm in pms]

        touched = list(_iter_touched_chunks(dataset.shape, chunks, sel))
        if not touched:
            return out

        def _fill_buf(idx, chunk_file_sel, actual_chunk_shape):
            """Read the full HDF5 chunk into pinned buffer *idx*."""
            n = int(np.prod(actual_chunk_shape))
            view = np.frombuffer(pms[idx], dtype=dtype, count=n).reshape(actual_chunk_shape)
            dataset.read_direct(view, source_sel=chunk_file_sel)

        # Prime: read first chunk into buf[0]
        _fill_buf(0, touched[0][0], touched[0][1])

        for i, (chunk_file_sel, actual_chunk_shape, local_sel, out_sel) in enumerate(touched):
            cur = i % 2
            nxt = 1 - cur

            # 1. Async H2D: sub-region of cur chunk → out[out_sel]
            _async_h2d_subtile(
                bufs[cur].ctypes.data, actual_chunk_shape,
                local_sel, out, out_sel, stream,
            )

            # 2. Enqueue optional element-wise transform (ordered after H2D
            #    on the same stream; runs while CPU reads the next chunk)
            if transform is not None:
                with stream:
                    out[out_sel] = transform(out[out_sel])

            # 3. While H2D + transform run, read the next chunk on CPU
            if i + 1 < len(touched):
                nxt_file_sel, nxt_shape = touched[i + 1][0], touched[i + 1][1]
                _fill_buf(nxt, nxt_file_sel, nxt_shape)

            # 4. Wait for H2D (and transform) before reusing cur buffer
            stream.synchronize()

        return out

    def read_double_buffered(self, chunk_size=None, out=None, stream=None,
                             transform=None):
        """Read the entire dataset to the GPU using double-buffered I/O.

        Overlaps HDF5 storage reads with host-to-device (H2D) DMA transfers
        by keeping two pinned host buffers and a CUDA stream:

        .. code-block:: text

            Iteration i:
              CPU  ──▶  [submit async H2D buf_i → GPU]
                        [optional transform(out[i]) on stream]    (non-blocking)
                        [read chunk i+1 from HDF5 → buf_{1-i}]   ← overlaps DMA+compute
                        [stream.synchronize()]

        Because the source buffers are page-locked (pinned), the CUDA DMA
        engine can carry out the H2D transfer autonomously while the CPU
        thread is busy reading the next HDF5 chunk.  The effective transfer
        time per chunk approaches ``max(T_io, T_h2d + T_compute)`` rather than
        ``T_io + T_h2d + T_compute``.

        This method reads along the *first* axis in equal-sized chunks and
        assembles the result into a single output array.  Works for any
        number of dimensions, including 1-D datasets.

        Parameters
        ----------
        chunk_size : int, optional
            Number of rows (along axis 0) per I/O chunk.  Larger values
            reduce Python overhead; smaller values increase overlap
            opportunities.  Defaults to ``dataset.chunks[0]`` for HDF5-chunked
            datasets (aligns reads to chunk boundaries, avoiding partial-chunk
            decompression), or ``max(1, nrows // 8)`` for contiguous datasets.
        out : cupy.ndarray, optional
            Pre-allocated output array with the same shape and dtype as the
            dataset.  If *None* a new array is allocated.
        stream : cupy.cuda.Stream, optional
            CUDA stream used for async H2D transfers.  If *None* a new
            non-blocking stream is created.
        transform : callable, optional
            Element-wise operation applied to each row-band on the GPU
            **after** its H2D transfer, on the same stream.  Called as
            ``out[start:end] = transform(out[start:end])`` inside a
            ``with stream:`` block::

                gpu_ds.read_double_buffered(transform=cp.sqrt)
                gpu_ds.read_double_buffered(transform=lambda x: x * 2.0)

        Returns
        -------
        cupy.ndarray
            The full dataset on the current CUDA device.

        Raises
        ------
        ValueError
            If the dataset has zero dimensions (use ``__getitem__`` instead).
        """
        dataset = object.__getattribute__(self, "_gpu_dataset")

        if dataset.ndim == 0:
            raise ValueError(
                "read_double_buffered requires at least 1 dimension; "
                "use dataset[()] for scalar datasets"
            )

        n_rows = dataset.shape[0]
        row_shape = dataset.shape[1:]
        dtype = np.dtype(dataset.dtype)
        row_nbytes = int(np.prod(row_shape, dtype=np.intp)) * dtype.itemsize

        if chunk_size is None:
            hdf5_chunks = dataset.chunks
            if hdf5_chunks is not None:
                # Align reads to the HDF5 chunk boundary along axis 0.
                # This avoids partial-chunk reads, which would force HDF5 to
                # decompress a chunk and discard part of it.
                chunk_size = hdf5_chunks[0]
            else:
                chunk_size = max(1, n_rows // 8)
        chunk_size = min(chunk_size, n_rows)

        if stream is None:
            stream = cp.cuda.Stream(non_blocking=True)

        # Allocate (or validate) output GPU array
        if out is None:
            out = cp.empty(dataset.shape, dtype=dtype)
        else:
            if not isinstance(out, cp.ndarray):
                raise TypeError(f"out must be a cupy.ndarray, got {type(out)!r}")
            if out.shape != dataset.shape or out.dtype != dtype:
                raise ValueError(
                    f"out shape/dtype {out.shape}/{out.dtype} does not match "
                    f"dataset {dataset.shape}/{dtype}"
                )

        # Two pinned host buffers — each sized for one full chunk
        buf_shape = (chunk_size,) + row_shape
        _pm = [None, None]
        bufs = [None, None]
        for k in range(2):
            _pm[k], bufs[k] = _alloc_pinned_like(buf_shape, dtype)

        chunk_starts = list(range(0, n_rows, chunk_size))

        # --- Prime the pipeline: fill buf[0] with chunk 0 ---
        end0 = min(chunk_size, n_rows)
        dataset.read_direct(bufs[0][:end0], source_sel=np.s_[0:end0])

        for i, start in enumerate(chunk_starts):
            end = min(start + chunk_size, n_rows)
            actual_rows = end - start
            cur = i % 2
            nxt = 1 - cur

            # 1. Submit async H2D: cur pinned buf → out[start:end]
            nbytes = actual_rows * row_nbytes
            cp.cuda.runtime.memcpyAsync(
                out[start:end].data.ptr,
                bufs[cur][:actual_rows].ctypes.data,
                nbytes,
                cp.cuda.runtime.memcpyHostToDevice,
                stream.ptr,
            )

            # 2. Enqueue optional element-wise transform on the same stream
            if transform is not None:
                with stream:
                    out[start:end] = transform(out[start:end])

            # 3. While H2D + transform run, read the next chunk into the other buffer
            if i + 1 < len(chunk_starts):
                next_start = chunk_starts[i + 1]
                next_end = min(next_start + chunk_size, n_rows)
                next_rows = next_end - next_start
                dataset.read_direct(
                    bufs[nxt][:next_rows],
                    source_sel=np.s_[next_start:next_end],
                )

            # 4. Wait for H2D (and transform) before cur buffer can be reused
            stream.synchronize()

        return out

    def read_chunks_to_gpu(self, out=None, stream=None, transform=None):
        """Read a chunked HDF5 dataset to the GPU one tile at a time,
        with double-buffering.

        Each HDF5 chunk is treated as an independent I/O unit.  Two pinned
        host buffers alternate between roles — while the GPU DMA engine
        transfers chunk *i* to its final position in the output array, the
        CPU reads chunk *i+1* from storage:

        .. code-block:: text

            iteration i:
              CPU  ──▶  [submit memcpy2DAsync  tile_i → out[sel_i]]  (non-blocking)
                        [optional transform(out[sel_i]) on stream]    (non-blocking)
                        [dataset[sel_{i+1}] → pinned_{nxt}]           ← overlaps DMA+compute
                        [stream.synchronize()]

        ``memcpy2DAsync`` is used for 2-D tiles (and once per depth-slice for
        3-D tiles), so each tile is written directly to its correct location
        in *out* without a GPU scatter kernel.

        Supports 2-D and 3-D datasets only.  The dataset must be HDF5-chunked;
        for contiguous datasets use :meth:`read_double_buffered` instead.

        Parameters
        ----------
        out : cupy.ndarray, optional
            Pre-allocated C-contiguous output array matching the dataset's
            shape and dtype.  Allocated automatically if *None*.
        stream : cupy.cuda.Stream, optional
            CUDA stream for H2D transfers.  A new non-blocking stream is
            created when *None*.
        transform : callable, optional
            Element-wise operation applied to each tile on the GPU **after**
            the H2D transfer, on the same stream.  Called as
            ``out[sel] = transform(out[sel])`` inside a ``with stream:``
            block.  Any CuPy ufunc or lambda works::

                gpu_ds.read_chunks_to_gpu(transform=cp.sqrt)
                gpu_ds.read_chunks_to_gpu(transform=lambda x: x * 2.0)

        Returns
        -------
        cupy.ndarray

        Raises
        ------
        ValueError
            If the dataset is not HDF5-chunked, or has ``ndim`` other than
            2 or 3.
        """
        dataset = object.__getattribute__(self, "_gpu_dataset")

        if dataset.chunks is None:
            raise ValueError(
                "Dataset is not HDF5-chunked. "
                "Use read_double_buffered() for contiguous datasets."
            )
        if dataset.ndim not in (2, 3):
            raise ValueError(
                f"read_chunks_to_gpu supports 2-D and 3-D datasets, "
                f"got ndim={dataset.ndim}"
            )

        shape  = dataset.shape
        chunks = dataset.chunks
        dtype  = np.dtype(dataset.dtype)

        if out is None:
            out = cp.empty(shape, dtype=dtype)
        else:
            if not isinstance(out, cp.ndarray):
                raise TypeError(f"out must be a cupy.ndarray, got {type(out)!r}")
            if out.shape != shape or out.dtype != dtype:
                raise ValueError(
                    f"out shape/dtype {out.shape}/{out.dtype} does not match "
                    f"dataset {shape}/{dtype}"
                )
            if not out.flags["C_CONTIGUOUS"]:
                raise ValueError("out must be C-contiguous")

        if stream is None:
            stream = cp.cuda.Stream(non_blocking=True)

        # Two pinned host buffers — each sized for one full (max) chunk.
        # Edge tiles are smaller but always fit.
        max_elems = int(np.prod(chunks))
        pms  = [cp.cuda.alloc_pinned_memory(max_elems * dtype.itemsize)
                for _ in range(2)]
        bufs = [np.frombuffer(pm, dtype=dtype, count=max_elems) for pm in pms]

        tiles = list(_iter_tiles(shape, chunks))
        if not tiles:
            return out

        # Prime the pipeline: read first tile into buf[0]
        first_sel, _ = tiles[0]
        first_tile = dataset[first_sel]
        np.copyto(bufs[0][:first_tile.size], first_tile.ravel())

        for i, (sel, tile_shape) in enumerate(tiles):
            cur = i % 2
            nxt = 1 - cur

            # 1. Submit async H2D: cur pinned buf → out[sel]
            _async_h2d_tile(bufs[cur].ctypes.data, tile_shape, out, sel, stream)

            # 2. Enqueue optional element-wise transform on the same stream
            #    (ordered after H2D; runs while CPU reads the next tile)
            if transform is not None:
                with stream:
                    out[sel] = transform(out[sel])

            # 3. While H2D + transform run, read the next tile from HDF5 on CPU
            if i + 1 < len(tiles):
                next_sel, _ = tiles[i + 1]
                next_tile = dataset[next_sel]
                np.copyto(bufs[nxt][:next_tile.size], next_tile.ravel())

            # 4. Wait for H2D (and transform) before cur buffer can be reused
            stream.synchronize()

        return out

    def read_chunks_parallel(self, out=None, n_streams=2, transform=None):
        """Read a chunked HDF5 dataset to the GPU using multiple CUDA streams.

        Distributes HDF5 chunks across *n_streams* independent CUDA streams in
        round-robin order.  Each stream independently pipelines its H2D
        transfer and optional compute, so up to *n_streams* tiles can have
        active GPU work simultaneously:

        .. code-block:: text

            stream 0: [H2D tile0] [compute tile0]               [H2D tile2] ...
            stream 1:             [H2D tile1]   [compute tile1]             ...
            CPU:      [read0]  [read1]  [read2]  [read3] ...

        Compared to :meth:`read_chunks_to_gpu` (single stream), this better
        utilises GPUs with multiple DMA copy engines and hides compute latency
        behind concurrent transfers on other streams.

        Supports 2-D and 3-D HDF5-chunked datasets.

        Parameters
        ----------
        out : cupy.ndarray, optional
            Pre-allocated C-contiguous output array matching the dataset's
            shape and dtype.  Allocated automatically if *None*.
        n_streams : int, optional
            Number of independent CUDA streams.  Default is 2.  Values of
            4–8 can improve throughput when *transform* is compute-heavy.
        transform : callable, optional
            Element-wise operation applied to each tile after its H2D
            transfer, on the same stream as the transfer.  Called as
            ``out[sel] = transform(out[sel])`` inside a ``with stream:``
            block::

                gpu_ds.read_chunks_parallel(n_streams=4, transform=cp.sqrt)

        Returns
        -------
        cupy.ndarray

        Raises
        ------
        ValueError
            If the dataset is not HDF5-chunked, or has ``ndim`` other than
            2 or 3.
        """
        dataset = object.__getattribute__(self, "_gpu_dataset")

        if dataset.chunks is None:
            raise ValueError(
                "Dataset is not HDF5-chunked. "
                "Use read_double_buffered() for contiguous datasets."
            )
        if dataset.ndim not in (2, 3):
            raise ValueError(
                f"read_chunks_parallel supports 2-D and 3-D datasets, "
                f"got ndim={dataset.ndim}"
            )

        shape  = dataset.shape
        chunks = dataset.chunks
        dtype  = np.dtype(dataset.dtype)

        if out is None:
            out = cp.empty(shape, dtype=dtype)
        else:
            if not isinstance(out, cp.ndarray):
                raise TypeError(f"out must be a cupy.ndarray, got {type(out)!r}")
            if out.shape != shape or out.dtype != dtype:
                raise ValueError(
                    f"out shape/dtype {out.shape}/{out.dtype} does not match "
                    f"dataset {shape}/{dtype}"
                )
            if not out.flags["C_CONTIGUOUS"]:
                raise ValueError("out must be C-contiguous")

        n_streams = max(1, int(n_streams))
        streams = [cp.cuda.Stream(non_blocking=True) for _ in range(n_streams)]

        # One pinned buffer per stream — each sized for one full (max) chunk
        max_elems = int(np.prod(chunks))
        pms  = [cp.cuda.alloc_pinned_memory(max_elems * dtype.itemsize)
                for _ in range(n_streams)]
        bufs = [np.frombuffer(pm, dtype=dtype, count=max_elems) for pm in pms]

        tiles = list(_iter_tiles(shape, chunks))
        if not tiles:
            return out

        for i, (sel, tile_shape) in enumerate(tiles):
            sid    = i % n_streams
            stream = streams[sid]
            buf    = bufs[sid]

            # Wait for this stream's previous work (H2D + compute from n_streams
            # iterations ago) before reusing its pinned buffer
            stream.synchronize()

            # Read this tile from HDF5 into the stream's pinned buffer (CPU)
            tile = dataset[sel]
            np.copyto(buf[:tile.size], tile.ravel())

            # Submit async H2D on this stream
            _async_h2d_tile(buf.ctypes.data, tile_shape, out, sel, stream)

            # Enqueue optional transform immediately after H2D on the same stream
            if transform is not None:
                with stream:
                    out[sel] = transform(out[sel])

        # Drain all streams
        for s in streams:
            s.synchronize()

        return out

    # ------------------------------------------------------------------
    # Write methods  (GPU → HDF5)
    # ------------------------------------------------------------------

    def __setitem__(self, args, val):
        """Write to the HDF5 dataset from a CuPy or NumPy array.

        For HDF5-chunked 2-D and 3-D datasets with a plain slice selection
        and a CuPy source array, the write is dispatched to
        :meth:`write_selection_chunked`, which uses double-buffered D2H
        transfers.  All other cases fall back to h5py's own ``__setitem__``.
        """
        dataset = object.__getattribute__(self, "_gpu_dataset")

        if (isinstance(val, cp.ndarray)
                and dataset.chunks is not None
                and dataset.ndim in (2, 3)):
            _args = args if isinstance(args, tuple) else (args,)
            sel, _ = _normalize_sel(_args, dataset.shape)
            if sel is not None:
                self.write_selection_chunked(val, sel)
                return

        # Fall back: move to CPU if needed, let h5py handle the rest
        if isinstance(val, cp.ndarray):
            val = cp.asnumpy(val)
        dataset[args] = val

    def write_double_buffered(self, src, chunk_size=None, stream=None):
        """Write a CuPy array to the entire dataset using double-buffered I/O.

        Mirrors :meth:`read_double_buffered`: while HDF5 writes row-band *i*
        from pinned memory, the CUDA DMA engine asynchronously transfers
        row-band *i+1* from the GPU to the next pinned buffer:

        .. code-block:: text

            prime:  D2H src[0:chunk] → pinned[0]  (sync)

            iteration i:
              GPU  ──▶  [memcpyAsync src[i+1] → pinned[nxt]]  (D2H, non-blocking)
                        [dataset.write_direct(pinned[cur])]     ← overlaps D2H
                        [stream.synchronize()]

        Parameters
        ----------
        src : cupy.ndarray
            Source array on the GPU.  Must match the dataset's shape and dtype.
        chunk_size : int, optional
            Rows per I/O band.  Defaults to ``dataset.chunks[0]`` for chunked
            datasets, or ``max(1, nrows // 8)`` for contiguous ones.
        stream : cupy.cuda.Stream, optional
            CUDA stream for D2H transfers.  A new non-blocking stream is
            created when *None*.
        """
        dataset = object.__getattribute__(self, "_gpu_dataset")

        if dataset.ndim == 0:
            raise ValueError(
                "write_double_buffered requires at least 1 dimension"
            )
        if not isinstance(src, cp.ndarray):
            raise TypeError(f"src must be a cupy.ndarray, got {type(src)!r}")

        n_rows    = dataset.shape[0]
        row_shape = dataset.shape[1:]
        dtype     = np.dtype(dataset.dtype)
        row_nbytes = int(np.prod(row_shape, dtype=np.intp)) * dtype.itemsize

        if src.shape != dataset.shape or src.dtype != dtype:
            raise ValueError(
                f"src shape/dtype {src.shape}/{src.dtype} does not match "
                f"dataset {dataset.shape}/{dtype}"
            )
        if not src.flags["C_CONTIGUOUS"]:
            raise ValueError("src must be C-contiguous")

        if chunk_size is None:
            hdf5_chunks = dataset.chunks
            chunk_size = hdf5_chunks[0] if hdf5_chunks else max(1, n_rows // 8)
        chunk_size = min(chunk_size, n_rows)

        if stream is None:
            stream = cp.cuda.Stream(non_blocking=True)

        # Two pinned host buffers
        buf_shape = (chunk_size,) + row_shape
        _pm = [None, None]
        bufs = [None, None]
        for k in range(2):
            _pm[k], bufs[k] = _alloc_pinned_like(buf_shape, dtype)

        chunk_starts = list(range(0, n_rows, chunk_size))

        # Prime: D2H first band
        end0 = min(chunk_size, n_rows)
        nbytes0 = end0 * row_nbytes
        cp.cuda.runtime.memcpyAsync(
            bufs[0].ctypes.data,
            src[0:end0].data.ptr,
            nbytes0,
            cp.cuda.runtime.memcpyDeviceToHost,
            stream.ptr,
        )
        stream.synchronize()

        for i, start in enumerate(chunk_starts):
            end = min(start + chunk_size, n_rows)
            actual_rows = end - start
            cur = i % 2
            nxt = 1 - cur

            # 1. Submit async D2H for the NEXT band while we write current
            if i + 1 < len(chunk_starts):
                next_start = chunk_starts[i + 1]
                next_end   = min(next_start + chunk_size, n_rows)
                next_rows  = next_end - next_start
                cp.cuda.runtime.memcpyAsync(
                    bufs[nxt].ctypes.data,
                    src[next_start:next_end].data.ptr,
                    next_rows * row_nbytes,
                    cp.cuda.runtime.memcpyDeviceToHost,
                    stream.ptr,
                )

            # 2. Write current pinned band to HDF5 (overlaps with D2H)
            dataset.write_direct(
                bufs[cur][:actual_rows],
                dest_sel=np.s_[start:end],
            )

            # 3. Wait for D2H before next iteration reuses nxt buffer
            stream.synchronize()

    def write_chunks_from_gpu(self, src, stream=None):
        """Write a CuPy array to a chunked HDF5 dataset one tile at a time,
        with double-buffering.

        Mirrors :meth:`read_chunks_to_gpu`: while HDF5 writes chunk *i* from
        pinned memory, ``memcpy2DAsync`` asynchronously extracts chunk *i+1*
        from the GPU into the next pinned buffer.

        Supports 2-D and 3-D datasets only.  The dataset must be HDF5-chunked.

        Parameters
        ----------
        src : cupy.ndarray
            Source GPU array matching the dataset shape and dtype.
        stream : cupy.cuda.Stream, optional
        """
        dataset = object.__getattribute__(self, "_gpu_dataset")

        if dataset.chunks is None:
            raise ValueError(
                "Dataset is not HDF5-chunked. "
                "Use write_double_buffered() for contiguous datasets."
            )
        if dataset.ndim not in (2, 3):
            raise ValueError(
                f"write_chunks_from_gpu supports 2-D and 3-D datasets, "
                f"got ndim={dataset.ndim}"
            )
        if not isinstance(src, cp.ndarray):
            raise TypeError(f"src must be a cupy.ndarray, got {type(src)!r}")

        shape  = dataset.shape
        chunks = dataset.chunks
        dtype  = np.dtype(dataset.dtype)

        if src.shape != shape or src.dtype != dtype:
            raise ValueError(
                f"src shape/dtype {src.shape}/{src.dtype} does not match "
                f"dataset {shape}/{dtype}"
            )
        if not src.flags["C_CONTIGUOUS"]:
            raise ValueError("src must be C-contiguous")

        if stream is None:
            stream = cp.cuda.Stream(non_blocking=True)

        max_elems = int(np.prod(chunks))
        pms  = [cp.cuda.alloc_pinned_memory(max_elems * dtype.itemsize)
                for _ in range(2)]
        bufs = [np.frombuffer(pm, dtype=dtype, count=max_elems) for pm in pms]

        tiles = list(_iter_tiles(shape, chunks))
        if not tiles:
            return

        # Prime: D2H first tile
        first_sel, first_shape = tiles[0]
        n0 = int(np.prod(first_shape))
        _async_d2h_tile(bufs[0].ctypes.data, first_shape, src, first_sel, stream)
        stream.synchronize()

        for i, (sel, tile_shape) in enumerate(tiles):
            cur = i % 2
            nxt = 1 - cur
            n = int(np.prod(tile_shape))

            # 1. Submit async D2H for the NEXT tile
            if i + 1 < len(tiles):
                next_sel, next_shape = tiles[i + 1]
                _async_d2h_tile(bufs[nxt].ctypes.data, next_shape, src, next_sel, stream)

            # 2. Write current pinned tile to HDF5 (overlaps with D2H)
            chunk_view = np.frombuffer(pms[cur], dtype=dtype, count=n).reshape(tile_shape)
            dataset.write_direct(chunk_view, dest_sel=sel)

            # 3. Wait for D2H before reusing nxt buffer
            stream.synchronize()

    def write_selection_chunked(self, src, sel, stream=None):
        """Write a CuPy array to a selection in a chunked HDF5 dataset,
        processing one HDF5 chunk at a time with double-buffering.

        Mirrors :meth:`read_selection_chunked`: for each HDF5 chunk that
        overlaps the selection, ``memcpy2DAsync`` (D2H) extracts the relevant
        sub-region from the GPU source array into a compact pinned buffer,
        then h5py writes that buffer to the dataset (HDF5 handles any
        read-modify-write internally for partial chunks):

        .. code-block:: text

            prime:  D2H src[src_sel_0] → pinned[0]  (sync)

            iteration i:
              GPU  ──▶  [D2H src[src_sel_{i+1}] → pinned[nxt]]  (non-blocking)
                        [dataset.write_direct(pinned[cur])]       ← overlaps D2H
                        [stream.synchronize()]

        Parameters
        ----------
        src : cupy.ndarray
            Source GPU array whose shape matches the selection's output shape.
        sel : tuple[slice]
            Plain (step-1) slice selection into the *dataset* (not into *src*).
            Produce with ``numpy.s_[10:50, 20:80]``.
        stream : cupy.cuda.Stream, optional
        """
        dataset = object.__getattribute__(self, "_gpu_dataset")

        if dataset.chunks is None:
            raise ValueError(
                "Dataset is not HDF5-chunked. "
                "Use write_double_buffered() for contiguous datasets."
            )
        if dataset.ndim not in (2, 3):
            raise ValueError(
                f"write_selection_chunked supports 2-D and 3-D datasets, "
                f"got ndim={dataset.ndim}"
            )
        if not isinstance(src, cp.ndarray):
            raise TypeError(f"src must be a cupy.ndarray, got {type(src)!r}")

        sel = sel if isinstance(sel, tuple) else (sel,)
        sel, out_shape = _normalize_sel(sel, dataset.shape)
        if sel is None:
            raise ValueError(
                "sel must be a tuple of plain (step-1) slices."
            )

        dtype  = np.dtype(dataset.dtype)
        chunks = dataset.chunks

        if src.shape != out_shape or src.dtype != dtype:
            raise ValueError(
                f"src shape/dtype {src.shape}/{src.dtype} does not match "
                f"selection shape/dtype {out_shape}/{dtype}"
            )
        if not src.flags["C_CONTIGUOUS"]:
            raise ValueError("src must be C-contiguous")

        if stream is None:
            stream = cp.cuda.Stream(non_blocking=True)

        max_elems = int(np.prod(chunks))
        pms  = [cp.cuda.alloc_pinned_memory(max_elems * dtype.itemsize)
                for _ in range(2)]
        bufs = [np.frombuffer(pm, dtype=dtype, count=max_elems) for pm in pms]

        touched = list(_iter_touched_chunks(dataset.shape, chunks, sel))
        if not touched:
            return

        def _fill_buf_d2h(idx, src_sel, compact_shape):
            """Async D2H: src[src_sel] → compact pinned buf[idx]."""
            _async_d2h_subtile(bufs[idx].ctypes.data, compact_shape, src, src_sel, stream)

        def _write_chunk(idx, chunk_file_sel, local_sel):
            """Write compact pinned buf[idx] to the dataset intersection."""
            compact_shape = tuple(s.stop - s.start for s in local_sel)
            n = int(np.prod(compact_shape))
            view = np.frombuffer(pms[idx], dtype=dtype, count=n).reshape(compact_shape)
            # Dataset coordinates of the intersection
            dataset_sel = tuple(
                slice(cf.start + lc.start, cf.start + lc.stop)
                for cf, lc in zip(chunk_file_sel, local_sel)
            )
            dataset.write_direct(view, dest_sel=dataset_sel)

        # Prime: D2H first chunk's sub-region
        cf0, _, lc0, src_sel0 = touched[0]
        compact0 = tuple(s.stop - s.start for s in lc0)
        _fill_buf_d2h(0, src_sel0, compact0)
        stream.synchronize()

        for i, (chunk_file_sel, _, local_sel, src_sel) in enumerate(touched):
            cur = i % 2
            nxt = 1 - cur

            # 1. Async D2H for the NEXT chunk's sub-region
            if i + 1 < len(touched):
                cf_nxt, _, lc_nxt, ss_nxt = touched[i + 1]
                compact_nxt = tuple(s.stop - s.start for s in lc_nxt)
                _fill_buf_d2h(nxt, ss_nxt, compact_nxt)

            # 2. Write current pinned buffer to HDF5 (overlaps D2H)
            _write_chunk(cur, chunk_file_sel, local_sel)

            # 3. Wait for D2H before reusing nxt buffer
            stream.synchronize()

    def read_direct_gpu(self, dest, source_sel=None, dest_sel=None):
        """Read HDF5 data directly into a pre-allocated :class:`cupy.ndarray`.

        Data is read from HDF5 into a pinned host buffer (matching *dest*'s
        shape and dtype), then copied to *dest* on the GPU.  This avoids
        allocating a throw-away device array when the caller already has a
        suitably shaped output buffer.

        Parameters
        ----------
        dest : cupy.ndarray
            Destination GPU array.  Must be C-contiguous and writable.
        source_sel : numpy slice, optional
            Selection within the HDF5 dataset (output of ``numpy.s_[...]``).
        dest_sel : numpy slice, optional
            Selection within *dest* to write to.

        Notes
        -----
        When *dest_sel* is ``None`` the entire *dest* array is overwritten.
        The pinned buffer is allocated to match the selected sub-region of
        *dest*, so memory usage is proportional to the transferred data, not
        the full *dest* size.
        """
        if not isinstance(dest, cp.ndarray):
            raise TypeError(
                f"dest must be a cupy.ndarray, got {type(dest)!r}"
            )

        dataset = object.__getattribute__(self, "_gpu_dataset")

        # Determine the shape of the region we are writing into
        if dest_sel is None:
            write_shape = dest.shape
        else:
            # Materialise the selection against dest's shape to get the shape
            import h5py._hl.selections as _sel
            write_shape = _sel.select(dest.shape, dest_sel).array_shape

        # Allocate pinned host buffer and read HDF5 data into it
        pinned_mem, pinned_arr = _alloc_pinned_like(write_shape, dest.dtype)
        dataset.read_direct(pinned_arr, source_sel=source_sel)

        # Transfer to GPU
        gpu_chunk = cp.array(pinned_arr)

        if dest_sel is None:
            cp.copyto(dest, gpu_chunk)
        else:
            dest[dest_sel] = gpu_chunk

    # ------------------------------------------------------------------
    # Attribute delegation
    # ------------------------------------------------------------------

    def __getattr__(self, name):
        dataset = object.__getattribute__(self, "_gpu_dataset")
        return getattr(dataset, name)

    def __repr__(self):
        dataset = object.__getattribute__(self, "_gpu_dataset")
        return f"<GPUDataset wrapping {dataset!r}>"

    def __len__(self):
        dataset = object.__getattribute__(self, "_gpu_dataset")
        return len(dataset)

    def __iter__(self):
        dataset = object.__getattribute__(self, "_gpu_dataset")
        return iter(dataset)

    def __contains__(self, item):
        dataset = object.__getattribute__(self, "_gpu_dataset")
        return item in dataset


# ---------------------------------------------------------------------------
# GPUGroup / GPUFile
# ---------------------------------------------------------------------------

class GPUGroup:
    """Wrapper around :class:`h5py.Group` that returns :class:`GPUDataset`
    for dataset items and nested :class:`GPUGroup` for sub-groups.

    Parameters
    ----------
    group : h5py.Group or h5py.File
        An open h5py group (or file, which is a subclass of group).
    """

    def __init__(self, group):
        if not isinstance(group, Group):
            raise TypeError(
                f"GPUGroup requires an h5py.Group (or File), got {type(group)!r}"
            )
        _require_cupy()
        object.__setattr__(self, "_gpu_group", group)

    # ------------------------------------------------------------------
    # Item access
    # ------------------------------------------------------------------

    def __getitem__(self, name):
        group = object.__getattribute__(self, "_gpu_group")
        item = group[name]
        if isinstance(item, Dataset):
            return GPUDataset(item)
        if isinstance(item, Group):
            return GPUGroup(item)
        return item

    # ------------------------------------------------------------------
    # Iteration / membership
    # ------------------------------------------------------------------

    def __iter__(self):
        group = object.__getattribute__(self, "_gpu_group")
        return iter(group)

    def __len__(self):
        group = object.__getattribute__(self, "_gpu_group")
        return len(group)

    def __contains__(self, name):
        group = object.__getattribute__(self, "_gpu_group")
        return name in group

    # ------------------------------------------------------------------
    # Attribute delegation
    # ------------------------------------------------------------------

    def __getattr__(self, name):
        group = object.__getattribute__(self, "_gpu_group")
        return getattr(group, name)

    def __repr__(self):
        group = object.__getattribute__(self, "_gpu_group")
        return f"<GPUGroup wrapping {group!r}>"

    # ------------------------------------------------------------------
    # Context-manager support (mirrors h5py.File)
    # ------------------------------------------------------------------

    def __enter__(self):
        return self

    def __exit__(self, *args):
        group = object.__getattribute__(self, "_gpu_group")
        group.__exit__(*args)


class GPUFile(GPUGroup):
    """Convenience wrapper: open an HDF5 file and expose it as a
    :class:`GPUGroup`.

    Parameters are forwarded directly to :class:`h5py.File`.

    Example
    -------
    ::

        with GPUFile("data.h5") as f:
            arr = f["dataset"][:]   # cupy.ndarray
    """

    def __init__(self, *args, **kwargs):
        _require_cupy()
        file_obj = File(*args, **kwargs)
        # Bypass GPUGroup.__init__ to avoid the isinstance check on File
        object.__setattr__(self, "_gpu_group", file_obj)

    def __repr__(self):
        group = object.__getattribute__(self, "_gpu_group")
        return f"<GPUFile wrapping {group!r}>"
