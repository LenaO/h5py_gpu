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
import zlib as _zlib
import numpy as np

try:
    import cupy as cp
    _CUPY_AVAILABLE = True
except ImportError:
    _CUPY_AVAILABLE = False

# HDF5 filter codes
_FILTER_DEFLATE  = 1      # gzip / zlib
_FILTER_SHUFFLE  = 2      # byte-shuffle pre-filter
_FILTER_LZ4      = 32004  # LZ4 (HDF5 plugin)
_FILTER_ZSTD     = 32015  # Zstd (HDF5 plugin)

# Lazy-compiled CUDA unshuffle kernel (built on first use)
_UNSHUFFLE_KERNEL = None

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
# Compression helpers  (for read_chunks_compressed)
# ---------------------------------------------------------------------------

def _get_unshuffle_kernel():
    """Return the lazily-compiled CuPy RawKernel that reverses HDF5's shuffle filter.

    HDF5 shuffle rearranges bytes so that byte *b* of every element is grouped
    together before compression:

        shuffled = [b0_of_elem0, b0_of_elem1, ..., b1_of_elem0, b1_of_elem1, ...]

    This kernel writes the original interleaved byte order back::

        dst[elem_idx * element_size + byte_pos] = src[byte_pos * n_elements + elem_idx]
    """
    global _UNSHUFFLE_KERNEL
    if _UNSHUFFLE_KERNEL is None:
        _UNSHUFFLE_KERNEL = cp.RawKernel(r"""
extern "C" __global__ void unshuffle(
    const unsigned char* __restrict__ src,
    unsigned char*       __restrict__ dst,
    int n_elements,
    int element_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_elements * element_size) return;
    int elem_idx = idx / element_size;
    int byte_pos = idx % element_size;
    dst[idx] = src[byte_pos * n_elements + elem_idx];
}
""", "unshuffle")
    return _UNSHUFFLE_KERNEL


# Map filter_name → (nvcomp_algorithm, header_bytes_to_skip).
# Only filters where GPU decompression is possible are listed.
# Requires `nvidia-nvcomp-cu12` (or cu13) installed:
#   pip install nvidia-nvcomp-cu12  --extra-index-url https://pypi.nvidia.com
_NVCOMP_FILTER_MAP = {
    "lz4":  ("lz4",  16),   # HDF5 LZ4 plugin prepends 16 bytes of metadata
    "zstd": ("zstd",  0),   # Standard Zstandard frames, no header to skip
}


def _get_nvcomp_decompressor(filter_name, uncompressed_nb, stream_ptr):
    """Return a ``(nvidia.nvcomp.Codec, header_skip)`` pair for GPU decompression,
    or ``None`` if nvCOMP is not installed or the filter is not supported.

    GPU decompression is available for:

    * **LZ4** (HDF5 filter 32004, ``hdf5plugin.LZ4``): the standard LZ4 block
      format produced by the HDF5 LZ4 plugin is compatible with
      ``nvcomp.Codec(algorithm='lz4', bitstream_kind=BitstreamKind.RAW)``.
      The HDF5 LZ4 plugin prepends a 16-byte metadata header; we skip it.

    * **Zstd** (HDF5 filter 32015, ``hdf5plugin.Zstd``): HDF5 Zstd produces
      standard RFC 8878 frames that nvCOMP decompresses directly with
      ``nvcomp.Codec(algorithm='zstd', bitstream_kind=BitstreamKind.RAW)``.

    **gzip / deflate** (HDF5 filter 1) is *not* supported for GPU
    decompression — nvCOMP's ``deflate`` algorithm uses a GPU-specific
    bitstream that is incompatible with RFC 1951 DEFLATE / RFC 1950 zlib
    produced by HDF5.  Those chunks are decompressed on CPU with
    ``zlib.decompress``.
    """
    if filter_name not in _NVCOMP_FILTER_MAP:
        return None
    try:
        from nvidia import nvcomp as _nv  # noqa: F401
    except ImportError:
        return None
    alg, skip = _NVCOMP_FILTER_MAP[filter_name]
    from nvidia import nvcomp as _nv
    codec = _nv.Codec(
        algorithm=alg,
        bitstream_kind=_nv.BitstreamKind.RAW,
        uncomp_chunk_size=uncompressed_nb,
        cuda_stream=stream_ptr,
    )
    return codec, skip


def _detect_filters(dataset):
    """Inspect a dataset's creation property list and return filter information.

    Returns
    -------
    has_shuffle : bool
        Whether the shuffle pre-filter is active.
    decompress_fn : callable or None
        ``bytes -> bytes`` function that decompresses one raw chunk.
        *None* means no compression filter was found.
    filter_name : str or None
        Human-readable name of the compression filter, or *None*.

    Raises
    ------
    ImportError
        If a compression filter is present but the required Python package
        (``lz4``, ``zstd``) is not installed.
    ValueError
        If the compression filter code is not recognised.
    """
    dcpl = dataset.id.get_create_plist()
    n = dcpl.get_nfilters()
    has_shuffle  = False
    decompress_fn = None
    filter_name  = None

    for i in range(n):
        code, _flags, _aux, _name = dcpl.get_filter(i)

        if code == _FILTER_SHUFFLE:
            has_shuffle = True

        elif code == _FILTER_DEFLATE:
            decompress_fn = _zlib.decompress
            filter_name   = "deflate/gzip"

        elif code == _FILTER_LZ4:
            _lz4_available = False
            try:
                import lz4.block as _lz4, struct as _struct
                _lz4_available = True
            except ImportError:
                pass

            if _lz4_available:
                import lz4.block as _lz4, struct as _struct

                def _lz4_decompress(raw_bytes):
                    # HDF5 LZ4 plugin header (big-endian):
                    #   bytes 0-7  : uint64 total uncompressed size
                    #   bytes 8-11 : uint32 block size (uncompressed per block)
                    # Then for each block:
                    #   bytes n..n+3 : uint32 compressed block size
                    #   bytes n+4..  : block data (LZ4 or raw)
                    #
                    # IMPORTANT: when LZ4 cannot compress a block (data is
                    # incompressible), the HDF5 plugin stores the raw bytes
                    # directly and sets comp_size == block_size.  We detect
                    # this and copy the block bytes verbatim.
                    total_uncomp = _struct.unpack_from(">Q", raw_bytes, 0)[0]
                    block_size   = _struct.unpack_from(">I", raw_bytes, 8)[0]
                    offset = 12
                    out_buf = bytearray()
                    while offset < len(raw_bytes):
                        comp_size = _struct.unpack_from(">I", raw_bytes, offset)[0]
                        offset += 4
                        block = raw_bytes[offset : offset + comp_size]
                        if comp_size >= block_size:
                            # Stored as raw (incompressible)
                            out_buf.extend(block)
                        else:
                            out_buf.extend(_lz4.decompress(bytes(block),
                                                           uncompressed_size=block_size))
                        offset += comp_size
                    return bytes(out_buf[:total_uncomp])

                decompress_fn = _lz4_decompress
            else:
                # No CPU lz4 library — decompress_fn stays None.
                # GPU decompression via nvCOMP is tried in read_chunks_compressed.
                # If nvCOMP is also absent, the method raises at runtime.
                decompress_fn = None
            filter_name = "lz4"

        elif code == _FILTER_ZSTD:
            try:
                import zstd as _zstd
                decompress_fn = _zstd.decompress
            except ImportError:
                # No CPU zstd library — GPU path (nvCOMP) will be tried.
                decompress_fn = None
            filter_name = "zstd"

        elif code not in (_FILTER_SHUFFLE,):
            # Unknown filter — not shuffle, not a compression we know about.
            # Raise so the caller knows it cannot bypass HDF5's pipeline.
            raise ValueError(
                f"Unsupported HDF5 filter code {code!r}.  "
                f"read_chunks_compressed supports deflate/gzip ({_FILTER_DEFLATE}), "
                f"LZ4 ({_FILTER_LZ4}), and Zstd ({_FILTER_ZSTD})."
            )

    return has_shuffle, decompress_fn, filter_name


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

        # Two pinned host buffers — each big enough for one full (max) chunk.
        # Used for edge chunks in the 2D path and for all chunks in 3D.
        max_chunk_elems = int(np.prod(chunks))
        pms  = [cp.cuda.alloc_pinned_memory(max_chunk_elems * dtype.itemsize)
                for _ in range(2)]
        bufs = [np.frombuffer(pm, dtype=dtype, count=max_chunk_elems) for pm in pms]

        def _fill_buf(idx, chunk_file_sel, actual_chunk_shape):
            """Read one HDF5 chunk directly into pinned buffer *idx*."""
            n = int(np.prod(actual_chunk_shape))
            view = np.frombuffer(pms[idx], dtype=dtype, count=n).reshape(actual_chunk_shape)
            dataset.read_direct(view, source_sel=chunk_file_sel)

        if dataset.ndim == 2:
            # ── 2D optimized path ──────────────────────────────────────────
            # Classify touched chunks into edge and interior:
            #
            #   edge     — partial in ≥1 dimension (top/bottom row, left/right
            #              col); processed one at a time with _async_h2d_subtile
            #   interior — fully covered in both dimensions; batched by
            #              row-band so one read_direct + one memcpy2DAsync
            #              replaces n_interior_cols individual copies
            #
            # Ordering: top edge → bottom edge → left/right edges → interior.
            # Within each group double-buffering overlaps HDF5 reads with H2D.

            r0, r1 = sel[0].start, sel[0].stop
            c0, c1 = sel[1].start, sel[1].stop
            R, C   = dataset.shape
            cr, cc = chunks

            # Inclusive chunk-index bounds of the touched region
            ri0 = r0 // cr;  ri1 = (r1 - 1) // cr
            ci0 = c0 // cc;  ci1 = (c1 - 1) // cc

            # Interior chunk-index ranges: fully covered in both dims
            ri_in0 = ri0 + (1 if r0 % cr else 0)
            ri_in1 = ri1 - (1 if r1 % cr else 0)
            ci_in0 = ci0 + (1 if c0 % cc else 0)
            ci_in1 = ci1 - (1 if c1 % cc else 0)

            has_interior = ri_in0 <= ri_in1 and ci_in0 <= ci_in1

            # Edge = not inside the interior chunk-index box
            edge_chunks = [
                entry for entry in _iter_touched_chunks(dataset.shape, chunks, sel)
                if not (ri_in0 <= entry[0][0].start // cr <= ri_in1
                        and ci_in0 <= entry[0][1].start // cc <= ci_in1)
            ]

            # ── Phase 1: edge chunks ────────────────────────────────────────
            if edge_chunks:
                _fill_buf(0, edge_chunks[0][0], edge_chunks[0][1])
                for i, (cfs, acs, ls, os) in enumerate(edge_chunks):
                    cur = i % 2
                    nxt = 1 - cur
                    _async_h2d_subtile(bufs[cur].ctypes.data, acs, ls, out, os, stream)
                    if transform is not None:
                        with stream:
                            out[os] = transform(out[os])
                    if i + 1 < len(edge_chunks):
                        _fill_buf(nxt, edge_chunks[i + 1][0], edge_chunks[i + 1][1])
                    stream.synchronize()

            # ── Phase 2: interior row-bands ─────────────────────────────────
            # The interior col range is fixed across all row-bands, so the
            # pinned source and the output destination are both contiguous
            # rectangles — one memcpy2DAsync handles the entire band.
            if has_interior:
                c_in_start  = ci_in0 * cc
                c_in_end    = (ci_in1 + 1) * cc
                n_in_cols   = c_in_end - c_in_start
                oc_in_start = c_in_start - c0
                oc_in_end   = c_in_end   - c0

                # Two double-buffered pinned buffers, one row-band each
                row_pms  = [cp.cuda.alloc_pinned_memory(cr * n_in_cols * dtype.itemsize)
                            for _ in range(2)]
                row_bufs = [np.frombuffer(pm, dtype=dtype, count=cr * n_in_cols)
                            for pm in row_pms]

                n_row_bands = ri_in1 - ri_in0 + 1

                def _fill_row_band(buf_idx, ri):
                    r_s  = ri * cr
                    r_e  = min(r_s + cr, R)
                    view = np.frombuffer(row_pms[buf_idx], dtype=dtype,
                                        count=(r_e - r_s) * n_in_cols
                                        ).reshape(r_e - r_s, n_in_cols)
                    dataset.read_direct(view, source_sel=(slice(r_s, r_e),
                                                          slice(c_in_start, c_in_end)))
                    return r_s, r_e

                # Prime: read first interior row-band into row_bufs[0]
                _fill_row_band(0, ri_in0)

                for j in range(n_row_bands):
                    cur  = j % 2
                    nxt  = 1 - cur
                    ri   = ri_in0 + j
                    r_s  = ri * cr
                    r_e  = min(r_s + cr, R)
                    nr   = r_e - r_s
                    or_s = r_s - r0
                    or_e = r_e - r0

                    # One H2D for the full interior col width of this row-band
                    _async_h2d_tile(
                        row_bufs[cur].ctypes.data,
                        (nr, n_in_cols),
                        out,
                        (slice(or_s, or_e), slice(oc_in_start, oc_in_end)),
                        stream,
                    )
                    if transform is not None:
                        with stream:
                            out[or_s:or_e, oc_in_start:oc_in_end] = \
                                transform(out[or_s:or_e, oc_in_start:oc_in_end])

                    if j + 1 < n_row_bands:
                        _fill_row_band(nxt, ri_in0 + j + 1)

                    stream.synchronize()

        else:
            # ── 3D: one chunk at a time (existing approach) ────────────────
            touched = list(_iter_touched_chunks(dataset.shape, chunks, sel))
            if not touched:
                return out

            _fill_buf(0, touched[0][0], touched[0][1])
            for i, (cfs, acs, ls, os) in enumerate(touched):
                cur = i % 2
                nxt = 1 - cur
                _async_h2d_subtile(bufs[cur].ctypes.data, acs, ls, out, os, stream)
                if transform is not None:
                    with stream:
                        out[os] = transform(out[os])
                if i + 1 < len(touched):
                    _fill_buf(nxt, touched[i + 1][0], touched[i + 1][1])
                stream.synchronize()

        return out

    def read_double_buffered(self, chunk_size=None, out=None, stream=None,
                             transform=None, sel=None):
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
            decompression), or ``max(1, sel_rows // 8)`` for contiguous datasets.
        out : cupy.ndarray, optional
            Pre-allocated output array whose shape must match the selected
            region ``(sel_rows, *row_shape)`` and have the correct dtype.
            If *None* a new array is allocated.
        stream : cupy.cuda.Stream, optional
            CUDA stream used for async H2D transfers.  If *None* a new
            non-blocking stream is created.
        transform : callable, optional
            Element-wise operation applied to each row-band on the GPU
            **after** its H2D transfer, on the same stream.  Called as
            ``out[os:oe] = transform(out[os:oe])`` inside a
            ``with stream:`` block::

                gpu_ds.read_double_buffered(transform=cp.sqrt)
                gpu_ds.read_double_buffered(transform=lambda x: x * 2.0)

        sel : slice or tuple[slice, ...], optional
            Region to read.  *None* reads the entire dataset (default).

            * A plain ``slice`` selects rows along axis 0; all remaining
              dimensions are read in full::

                  gpu_ds.read_double_buffered(sel=np.s_[512:2048])

            * A tuple of slices selects each dimension independently.  The
              first slice is always the row axis; subsequent slices clip the
              remaining dimensions (e.g. columns for a 2-D dataset).  Every
              slice must have step 1::

                  gpu_ds.read_double_buffered(sel=np.s_[512:2048, 64:192])

        Returns
        -------
        cupy.ndarray
            The selected rows on the current CUDA device, shape
            ``(sel_rows, *row_shape)``.

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

        if sel is None:
            r0, r1      = 0, n_rows
            extra_slices = ()
        elif isinstance(sel, slice):
            r0, r1, step = sel.indices(n_rows)
            if step != 1:
                raise ValueError("sel row slice must have step 1")
            extra_slices = ()
        elif isinstance(sel, tuple):
            if not sel or not isinstance(sel[0], slice):
                raise TypeError("sel tuple must start with a row slice")
            r0, r1, step = sel[0].indices(n_rows)
            if step != 1:
                raise ValueError("sel row slice must have step 1")
            extra_slices = sel[1:]
            for i, s in enumerate(extra_slices, start=1):
                if not isinstance(s, slice):
                    raise TypeError(f"sel[{i}] must be a slice")
                if s.indices(dataset.shape[i])[2] != 1:
                    raise ValueError(f"sel[{i}] must have step 1")
        else:
            raise TypeError(
                "sel must be a slice, a tuple of slices, or None"
            )
        sel_rows = r1 - r0

        # Effective row_shape after applying extra_slices to trailing dims
        if extra_slices:
            row_shape = tuple(
                s.indices(n)[1] - s.indices(n)[0]
                for s, n in zip(extra_slices, dataset.shape[1:])
            ) + dataset.shape[1 + len(extra_slices):]
        # (row_shape stays as-is when extra_slices is empty)
        row_nbytes = int(np.prod(row_shape, dtype=np.intp)) * dtype.itemsize

        if chunk_size is None:
            hdf5_chunks = dataset.chunks
            if hdf5_chunks is not None:
                # Align reads to the HDF5 chunk boundary along axis 0.
                # This avoids partial-chunk reads, which would force HDF5 to
                # decompress a chunk and discard part of it.
                chunk_size = hdf5_chunks[0]
            else:
                chunk_size = max(1, sel_rows // 8)
        chunk_size = min(chunk_size, sel_rows)

        if stream is None:
            stream = cp.cuda.Stream(non_blocking=True)

        # Allocate (or validate) output GPU array — shape matches the selection
        out_shape = (sel_rows,) + row_shape
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

        # Two pinned host buffers — each sized for one full chunk
        buf_shape = (chunk_size,) + row_shape
        _pm = [None, None]
        bufs = [None, None]
        for k in range(2):
            _pm[k], bufs[k] = _alloc_pinned_like(buf_shape, dtype)

        chunk_starts = list(range(r0, r1, chunk_size))

        # --- Prime the pipeline: fill buf[0] with the first row-band ---
        end0 = min(r0 + chunk_size, r1)
        _nr0 = end0 - r0
        if extra_slices:
            _src0  = (slice(r0, end0),) + extra_slices
            _view0 = bufs[0][:_nr0].reshape((_nr0,) + row_shape)
        else:
            _src0  = np.s_[r0:end0]
            _view0 = bufs[0][:_nr0]
        dataset.read_direct(_view0, source_sel=_src0)

        for i, start in enumerate(chunk_starts):
            end         = min(start + chunk_size, r1)
            actual_rows = end - start
            cur = i % 2
            nxt = 1 - cur
            os  = start - r0    # offset into output array
            oe  = os + actual_rows

            # 1. Submit async H2D: cur pinned buf → out[os:oe]
            nbytes = actual_rows * row_nbytes
            cp.cuda.runtime.memcpyAsync(
                out[os:oe].data.ptr,
                bufs[cur][:actual_rows].ctypes.data,
                nbytes,
                cp.cuda.runtime.memcpyHostToDevice,
                stream.ptr,
            )

            # 2. Enqueue optional element-wise transform on the same stream
            if transform is not None:
                with stream:
                    out[os:oe] = transform(out[os:oe])

            # 3. While H2D + transform run, read the next row-band into the other buffer
            if i + 1 < len(chunk_starts):
                next_start = chunk_starts[i + 1]
                next_end   = min(next_start + chunk_size, r1)
                next_rows  = next_end - next_start
                if extra_slices:
                    _src_nxt  = (slice(next_start, next_end),) + extra_slices
                    _view_nxt = bufs[nxt][:next_rows].reshape((next_rows,) + row_shape)
                else:
                    _src_nxt  = np.s_[next_start:next_end]
                    _view_nxt = bufs[nxt][:next_rows]
                dataset.read_direct(_view_nxt, source_sel=_src_nxt)

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

        # Prime the pipeline: read first tile directly into pinned buf[0]
        first_sel, first_shape = tiles[0]
        dataset.read_direct(
            np.frombuffer(pms[0], dtype=dtype,
                          count=int(np.prod(first_shape))).reshape(first_shape),
            source_sel=first_sel,
        )

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

            # 3. While H2D + transform run, read the next tile directly into
            #    pinned buf[nxt] (no intermediate pageable allocation)
            if i + 1 < len(tiles):
                next_sel, next_shape = tiles[i + 1]
                dataset.read_direct(
                    np.frombuffer(pms[nxt], dtype=dtype,
                                  count=int(np.prod(next_shape))).reshape(next_shape),
                    source_sel=next_sel,
                )

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

            # Read this tile directly into the stream's pinned buffer (CPU)
            dataset.read_direct(
                np.frombuffer(pms[sid], dtype=dtype,
                              count=int(np.prod(tile_shape))).reshape(tile_shape),
                source_sel=sel,
            )

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

    def read_chunks_compressed(self, out=None, stream=None, transform=None):
        """Read a compressed, chunked HDF5 dataset to the GPU via a
        decompress-into-pinned-memory pipeline.

        Standard h5py decompresses each chunk into a pageable (ordinary) host
        buffer and then CUDA must stage through a hidden pinned buffer for the
        DMA.  This method bypasses that by:

        1. Reading the raw compressed chunk bytes with ``H5Dread_chunk``
           (no decompression by HDF5 itself).
        2. Decompressing each chunk directly into a **pinned** (page-locked)
           host buffer, eliminating the pageable → pinned staging copy.
        3. Issuing an async ``cudaMemcpyAsync`` (or ``memcpy2DAsync``) from the
           pinned buffer to the output array on the GPU.
        4. If the HDF5 *shuffle* pre-filter is present, reversing it on the GPU
           with a lightweight CUDA kernel — the unshuffle runs on the same
           stream as the H2D transfer, overlapping with the CPU decompressing
           the next chunk.
        5. Applying an optional element-wise *transform* on the same stream
           after the unshuffle.

        Double-buffering is used so that GPU DMA + unshuffle + transform
        for chunk *i* overlaps the CPU decompression of chunk *i+1*.

        .. code-block:: text

            iteration i:
              CPU  ──▶  [decompress chunk_i  → pinned_buf[cur]]   ← already done
                        [memcpy2DAsync pinned → out[sel_i]]         (non-blocking)
                        [GPU unshuffle out[sel_i] → gpu_tmp]        (non-blocking)
                        [cp.copyto(out[sel_i], gpu_tmp_view)]       (non-blocking)
                        [optional transform(out[sel_i])]            (non-blocking)
                        [decompress chunk_{i+1} → pinned_buf[nxt]] ← overlaps GPU
                        [stream.synchronize()]

        Supported compression filters
        ------------------------------
        * ``deflate`` / gzip (HDF5 filter code 1) — via Python's built-in
          :mod:`zlib`.
        * LZ4 (filter code 32004) — requires ``pip install lz4``.
        * Zstd (filter code 32015) — requires ``pip install zstd``.

        If the dataset is **not compressed**, the method falls back to
        :meth:`read_chunks_to_gpu` (which is already optimal for that case).

        Future: GPU decompression
        -------------------------
        When `nvCOMP <https://github.com/NVIDIA/nvcomp>`_ is available the
        pipeline can be extended so that *compressed* bytes are transferred to
        the GPU (smaller H2D payload) and decompressed entirely on the GPU,
        eliminating the CPU decompress step.  The current implementation is the
        correct staging ground for that upgrade.

        Parameters
        ----------
        out : cupy.ndarray, optional
            Pre-allocated C-contiguous output array matching the dataset's
            shape and dtype.  Allocated automatically if *None*.
        stream : cupy.cuda.Stream, optional
            CUDA stream for H2D transfers and GPU kernels.  A new non-blocking
            stream is created when *None*.
        transform : callable, optional
            Element-wise operation applied to each tile on the GPU **after**
            the unshuffle, on the same stream::

                gpu_ds.read_chunks_compressed(transform=cp.log1p)

        Returns
        -------
        cupy.ndarray

        Raises
        ------
        ValueError
            If the dataset is not HDF5-chunked, ``ndim`` is not 2 or 3, or the
            dataset uses an unsupported filter.
        ImportError
            If a third-party decompressor (``lz4``, ``zstd``) is required but
            not installed.
        """
        import cupy as cp  # already checked by _require_cupy in __init__

        dataset = object.__getattribute__(self, "_gpu_dataset")

        if dataset.chunks is None:
            raise ValueError(
                "Dataset is not HDF5-chunked. "
                "Use read_double_buffered() for contiguous datasets."
            )
        if dataset.ndim not in (2, 3):
            raise ValueError(
                f"read_chunks_compressed supports 2-D and 3-D datasets, "
                f"got ndim={dataset.ndim}"
            )

        has_shuffle, decompress_fn, filter_name = _detect_filters(dataset)

        # No compression at all → regular pipelined read is already optimal
        if decompress_fn is None and filter_name is None:
            return self.read_chunks_to_gpu(out=out, stream=stream,
                                           transform=transform)

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

        max_elems       = int(np.prod(chunks))
        uncompressed_nb = max_elems * dtype.itemsize   # bytes per full chunk
        itemsize        = dtype.itemsize
        H2D             = cp.cuda.runtime.memcpyHostToDevice

        # ── Choose CPU or GPU decompression ───────────────────────────────────
        # GPU decompression (nvCOMP) is available for LZ4 and Zstd.
        # gzip/deflate falls back to CPU (nvCOMP's deflate is a different
        # algorithm, incompatible with the standard zlib format HDF5 produces).
        nvcomp_result = _get_nvcomp_decompressor(filter_name, uncompressed_nb,
                                                  stream.ptr)
        use_gpu_decomp = nvcomp_result is not None

        # If neither a CPU decompressor nor nvCOMP is available, fail clearly.
        if not use_gpu_decomp and decompress_fn is None:
            raise ImportError(
                f"Cannot decompress '{filter_name}' HDF5 chunks: "
                f"neither a CPU decompressor nor nvCOMP is available.  "
                f"Install one of:\n"
                f"  pip install {'lz4' if filter_name == 'lz4' else 'zstd'}\n"
                f"  pip install nvidia-nvcomp-cu12  "
                f"--extra-index-url https://pypi.nvidia.com"
            )

        if use_gpu_decomp:
            nvcomp_codec, header_skip = nvcomp_result
            nvcomp_codec.__enter__()   # activate context manager

            # Find the largest compressed chunk to size the pinned buffers.
            n_stored = dataset.id.get_num_chunks()
            max_comp_nb = max(dataset.id.get_chunk_info(i).size
                              for i in range(n_stored))

            # Two pinned buffers for COMPRESSED bytes (much smaller than
            # uncompressed_nb for high-compression datasets).
            pms  = [cp.cuda.alloc_pinned_memory(max_comp_nb) for _ in range(2)]
            bufs = [np.frombuffer(pm, dtype=np.uint8, count=max_comp_nb)
                    for pm in pms]

            # One GPU buffer for the compressed payload (reused per chunk).
            gpu_comp = cp.empty(max_comp_nb, dtype=cp.uint8)

            from nvidia import nvcomp as _nv

            import struct as _struct

            def _read_chunk_into(pinned_buf, chunk_offset):
                """``read_direct_chunk`` → pinned.

                Returns ``(comp_size, is_raw_block)`` where ``is_raw_block``
                is ``True`` for LZ4 chunks that the plugin stored as raw
                bytes because the data was incompressible.
                """
                _fmask, raw = dataset.id.read_direct_chunk(chunk_offset)
                raw_np      = np.frombuffer(raw, dtype=np.uint8)
                pinned_buf[:len(raw_np)] = raw_np
                # Detect LZ4 "stored raw" blocks (incompressible data):
                # the plugin sets block_comp_size == block_size in that case.
                is_raw_block = False
                if header_skip == 16 and len(raw_np) >= 16:
                    block_comp = _struct.unpack_from(">I", raw_np, 12)[0]
                    block_unc  = _struct.unpack_from(">I", raw_np, 8)[0]
                    is_raw_block = (block_comp >= block_unc)
                return len(raw_np), is_raw_block

            def _gpu_decomp_and_place(pinned_buf, comp_size, is_raw_block,
                                       tile_shape, sel):
                """H2D compressed/raw bytes → (nvCOMP decode) → unshuffle → place."""
                payload_nb = comp_size - header_skip

                # 1. H2D payload (skip plugin header)
                cp.cuda.runtime.memcpyAsync(
                    gpu_comp.data.ptr,
                    pinned_buf.ctypes.data + header_skip,
                    payload_nb,
                    H2D, stream.ptr,
                )
                stream.synchronize()

                if is_raw_block:
                    # LZ4 "stored raw" block: payload IS the uncompressed data.
                    decomp_cp   = cp.empty(payload_nb, dtype=cp.uint8)
                    cp.cuda.runtime.memcpyAsync(
                        decomp_cp.data.ptr, gpu_comp.data.ptr,
                        payload_nb,
                        cp.cuda.runtime.memcpyDeviceToDevice, stream.ptr,
                    )
                    _decomp_ref = None
                else:
                    # 2. GPU decompress → uint8 GPU array.
                    # Use a D2D memcpy on `stream` to move nvCOMP's decompressed
                    # data into a CuPy-owned buffer.  Both decode and the D2D copy
                    # run on the same non-blocking `stream`, so the copy is
                    # naturally ordered after the decode without an extra sync.
                    # We keep `_decomp_ref` alive until after stream.synchronize()
                    # so that nvCOMP cannot free its buffer while the copy is in
                    # flight on a non-blocking stream.
                    comp_arr    = _nv.as_array(gpu_comp[:payload_nb],
                                               cuda_stream=stream.ptr)
                    _decomp_ref = nvcomp_codec.decode(comp_arr)
                    decomp_view = cp.from_dlpack(_decomp_ref)
                    decomp_cp   = cp.empty(decomp_view.shape, dtype=decomp_view.dtype)
                    cp.cuda.runtime.memcpyAsync(
                        decomp_cp.data.ptr, decomp_view.data.ptr,
                        decomp_view.nbytes,
                        cp.cuda.runtime.memcpyDeviceToDevice, stream.ptr,
                    )

                # 3. Unshuffle (if needed) and place in output
                with stream:
                    if has_shuffle:
                        shuffled = decomp_cp.view(cp.uint8)
                        gpu_full = cp.empty(max_elems, dtype=dtype)
                        total    = uncompressed_nb
                        thr      = 256
                        blk      = (total + thr - 1) // thr
                        _get_unshuffle_kernel()(
                            (blk,), (thr,),
                            (shuffled, gpu_full.view(cp.uint8),
                             np.int32(max_elems), np.int32(itemsize)),
                        )
                        sl = tuple(slice(0, t) for t in tile_shape)
                        out[sel] = gpu_full.reshape(chunks)[sl]
                    else:
                        sl = tuple(slice(0, t) for t in tile_shape)
                        out[sel] = decomp_cp.view(dtype).reshape(chunks)[sl]

                if transform is not None:
                    with stream:
                        out[sel] = transform(out[sel])

                # Synchronise stream so all GPU work (D2D copy + unshuffle) is
                # complete before _decomp_ref goes out of scope.  Without this,
                # nvCOMP could free its buffer while the D2D copy is still in
                # flight on the non-blocking stream (non-blocking streams are NOT
                # serialised with the null stream, so cp.from_dlpack().copy() on
                # the null stream would not help either).
                stream.synchronize()
                del _decomp_ref  # safe: stream is idle, nvCOMP buffer can be freed

            tiles = list(_iter_tiles(shape, chunks))
            if not tiles:
                nvcomp_codec.__exit__(None, None, None)
                return out

            # Prime: read first chunk into buf[0]
            first_sel, _  = tiles[0]
            comp_meta     = [(0, False)] * len(tiles)   # (comp_size, is_raw_block)
            comp_meta[0]  = _read_chunk_into(
                bufs[0], tuple(s.start for s in first_sel)
            )

            try:
                for i, (sel, tile_shape) in enumerate(tiles):
                    cur = i % 2
                    nxt = 1 - cur

                    comp_size, is_raw = comp_meta[i]
                    # GPU: decompress + unshuffle + transform for tile i
                    _gpu_decomp_and_place(bufs[cur], comp_size, is_raw,
                                          tile_shape, sel)

                    # CPU: read next compressed chunk while GPU works
                    if i + 1 < len(tiles):
                        next_sel          = tiles[i + 1][0]
                        next_offset       = tuple(s.start for s in next_sel)
                        comp_meta[i + 1]  = _read_chunk_into(bufs[nxt], next_offset)

                    stream.synchronize()
            finally:
                nvcomp_codec.__exit__(None, None, None)
            # Release the nvCOMP codec and GPU/pinned buffers explicitly before
            # returning.  Without this, Python's GC may interleave the destructor
            # of the nvCOMP Codec with the cleanup of CUDA arrays (CuPy pinned
            # memory, device arrays), causing an access violation on Windows.
            del nvcomp_codec, nvcomp_result
            del gpu_comp, bufs, pms

        else:
            # ── CPU decompression path ─────────────────────────────────────────
            # Two pinned host buffers — each holds one FULL decompressed chunk
            # (padded with fill values for edge tiles).
            pms  = [cp.cuda.alloc_pinned_memory(uncompressed_nb) for _ in range(2)]
            bufs = [np.frombuffer(pm, dtype=np.uint8, count=uncompressed_nb)
                    for pm in pms]

            # GPU scratch for the unshuffle path.
            gpu_scratch      = cp.empty(uncompressed_nb, dtype=cp.uint8) if has_shuffle else None
            unshuffle_kernel = _get_unshuffle_kernel() if has_shuffle else None
            threads          = 256

            def _decompress_into(pinned_buf, chunk_offset):
                _fmask, raw  = dataset.id.read_direct_chunk(chunk_offset)
                decompressed  = decompress_fn(raw)
                pinned_buf[:len(decompressed)] = np.frombuffer(decompressed, dtype=np.uint8)

            def _h2d_and_place(pinned_buf, tile_shape, sel):
                if has_shuffle:
                    cp.cuda.runtime.memcpyAsync(
                        gpu_scratch.data.ptr, pinned_buf.ctypes.data,
                        uncompressed_nb, H2D, stream.ptr,
                    )
                    gpu_full = cp.empty(max_elems, dtype=dtype)
                    total    = uncompressed_nb
                    blocks   = (total + threads - 1) // threads
                    with stream:
                        unshuffle_kernel(
                            (blocks,), (threads,),
                            (gpu_scratch, gpu_full.view(cp.uint8),
                             np.int32(max_elems), np.int32(itemsize)),
                        )
                        sl = tuple(slice(0, t) for t in tile_shape)
                        out[sel] = gpu_full.reshape(chunks)[sl]
                else:
                    if len(tile_shape) == 2:
                        out_R, out_C   = out.shape
                        tile_R, tile_C = tile_shape
                        chunk_C        = chunks[1]
                        r0, c0         = sel[0].start, sel[1].start
                        cp.cuda.runtime.memcpy2DAsync(
                            out.data.ptr + (r0 * out_C + c0) * itemsize,
                            out_C  * itemsize,
                            pinned_buf.ctypes.data,
                            chunk_C * itemsize,
                            tile_C  * itemsize,
                            tile_R,
                            H2D, stream.ptr,
                        )
                    else:
                        _, out_R, out_C          = out.shape
                        tile_D, tile_R, tile_C   = tile_shape
                        chunk_D, chunk_R, chunk_C = chunks
                        d0, r0, c0               = (sel[0].start, sel[1].start,
                                                    sel[2].start)
                        src_slice_nb = chunk_R * chunk_C * itemsize
                        for d in range(tile_D):
                            cp.cuda.runtime.memcpy2DAsync(
                                out.data.ptr + (
                                    (d0 + d) * out_R * out_C + r0 * out_C + c0
                                ) * itemsize,
                                out_C  * itemsize,
                                pinned_buf.ctypes.data + d * src_slice_nb,
                                chunk_C * itemsize,
                                tile_C  * itemsize,
                                tile_R,
                                H2D, stream.ptr,
                            )

                if transform is not None:
                    with stream:
                        out[sel] = transform(out[sel])

            tiles = list(_iter_tiles(shape, chunks))
            if not tiles:
                return out

            first_sel, _ = tiles[0]
            _decompress_into(bufs[0], tuple(s.start for s in first_sel))

            for i, (sel, tile_shape) in enumerate(tiles):
                cur = i % 2
                nxt = 1 - cur

                _h2d_and_place(bufs[cur], tile_shape, sel)

                if i + 1 < len(tiles):
                    next_sel    = tiles[i + 1][0]
                    _decompress_into(bufs[nxt], tuple(s.start for s in next_sel))

                stream.synchronize()

        return out

    # ------------------------------------------------------------------
    # Reduction methods  (HDF5 → per-chunk GPU reduction → scalar)
    # ------------------------------------------------------------------

    def reduce_chunks(self, reduce_fn, combine_fn=None, transform=None,
                      stream=None):
        """Compute a reduction over a chunked dataset without loading it fully
        to GPU memory.

        For each HDF5 chunk the method:

        1. Reads the full chunk into a pinned host buffer.
        2. H2D-transfers it asynchronously into a reusable GPU temp buffer.
        3. Optionally applies *transform* on the same stream.
        4. Applies *reduce_fn* and stores one value in a ``partial`` array.
        5. Reads the next chunk on the CPU while 2–4 run on the GPU.

        Finally *combine_fn* is applied to ``partial`` to produce the result.

        .. code-block:: text

            iteration i:
              [H2D chunk_i → gpu_temp]          (async)
              [transform(gpu_temp)]              (async, same stream)
              [partial[i] = reduce_fn(gpu_temp)] (async, same stream)
              [CPU reads chunk i+1]              ← overlaps all GPU work
              [stream.synchronize()]

        GPU memory usage is ``O(chunk_size)`` for the temp buffer plus
        ``O(n_chunks)`` for the partial results — the full dataset is never
        resident on the GPU at once.

        Parameters
        ----------
        reduce_fn : callable
            Applied to each tile (a ``cupy.ndarray``) and must return a
            scalar ``cupy.ndarray``  (0-D).  Any CuPy reducing ufunc works
            directly::

                gpu_ds.reduce_chunks(cp.sum)
                gpu_ds.reduce_chunks(cp.max)
                gpu_ds.reduce_chunks(cp.min)

        combine_fn : callable, optional
            Applied to the 1-D ``partial`` array (shape ``(n_chunks,)``) to
            produce the final result.  Defaults to *reduce_fn*, which is
            correct for *sum*, *max*, and *min*.

            For a global **mean** use::

                n = int(np.prod(dataset.shape))
                gpu_ds.reduce_chunks(cp.sum,
                                     combine_fn=lambda x: cp.sum(x) / n)

            .. warning::
                ``combine_fn=cp.mean`` gives a wrong result when chunks have
                different sizes (edge tiles), because it averages the per-chunk
                means rather than the per-element values.  Use the sum-then-
                divide pattern above instead.

        transform : callable, optional
            Applied to each tile before *reduce_fn* (on the same stream).

        stream : cupy.cuda.Stream, optional
            CUDA stream.  A new non-blocking stream is created when *None*.

        Returns
        -------
        cupy.ndarray
            0-D (scalar) result, or whatever shape *combine_fn* returns.

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
                "Use reduce_double_buffered() for contiguous datasets."
            )
        if dataset.ndim not in (2, 3):
            raise ValueError(
                f"reduce_chunks supports 2-D and 3-D datasets, "
                f"got ndim={dataset.ndim}"
            )

        if combine_fn is None:
            combine_fn = reduce_fn

        if stream is None:
            stream = cp.cuda.Stream(non_blocking=True)

        shape  = dataset.shape
        chunks = dataset.chunks
        dtype  = np.dtype(dataset.dtype)

        tiles = list(_iter_tiles(shape, chunks))
        if not tiles:
            return None

        # Probe the output dtype of reduce_fn
        _probe = reduce_fn(cp.zeros(1, dtype=dtype))
        out_dtype = _probe.dtype

        max_elems = int(np.prod(chunks))

        # Two pinned host buffers for double-buffered CPU reads
        pms  = [cp.cuda.alloc_pinned_memory(max_elems * dtype.itemsize)
                for _ in range(2)]
        bufs = [np.frombuffer(pm, dtype=dtype, count=max_elems) for pm in pms]

        # Single reusable GPU temp buffer (safe: always synced before reuse)
        gpu_temp = cp.empty(max_elems, dtype=dtype)

        # One partial result per tile
        partial = cp.empty(len(tiles), dtype=out_dtype)

        # Prime: read first tile into buf[0]
        first_sel, first_shape = tiles[0]
        first_tile = dataset[first_sel]
        np.copyto(bufs[0][:first_tile.size], first_tile.ravel())

        for i, (sel, tile_shape) in enumerate(tiles):
            cur = i % 2
            nxt = 1 - cur
            n = int(np.prod(tile_shape))

            # 1. H2D: pinned buf[cur] → gpu_temp[:n]  (async)
            cp.cuda.runtime.memcpyAsync(
                gpu_temp.data.ptr,
                bufs[cur].ctypes.data,
                n * dtype.itemsize,
                cp.cuda.runtime.memcpyHostToDevice,
                stream.ptr,
            )

            # 2. Enqueue transform (optional) + reduce → partial[i]
            #    All ordered after H2D on the same stream.
            with stream:
                tile_view = gpu_temp[:n].reshape(tile_shape)
                if transform is not None:
                    tile_view = transform(tile_view)
                partial[i] = reduce_fn(tile_view)

            # 3. Read next tile on CPU (overlaps H2D + compute on GPU)
            if i + 1 < len(tiles):
                next_sel, next_shape = tiles[i + 1]
                next_tile = dataset[next_sel]
                np.copyto(bufs[nxt][:next_tile.size], next_tile.ravel())

            # 4. Sync before reusing gpu_temp and pinned buf
            stream.synchronize()

        return combine_fn(partial)

    def reduce_double_buffered(self, reduce_fn, combine_fn=None, transform=None,
                               chunk_size=None, stream=None):
        """Compute a reduction over a dataset using row-band double-buffering.

        Mirrors :meth:`reduce_chunks` but works for any number of dimensions,
        including 1-D datasets and contiguous (non-chunked) datasets, by
        processing the data in equal-sized row bands along the first axis.

        Parameters
        ----------
        reduce_fn : callable
            Applied to each row band.  Must return a scalar ``cupy.ndarray``.
        combine_fn : callable, optional
            Applied to the partial-results array.  Defaults to *reduce_fn*.
        transform : callable, optional
            Applied to each band before *reduce_fn*, on the same stream.
        chunk_size : int, optional
            Number of rows per band.  Defaults to ``dataset.chunks[0]`` for
            HDF5-chunked datasets, or ``max(1, nrows // 8)`` otherwise.
        stream : cupy.cuda.Stream, optional

        Returns
        -------
        cupy.ndarray
            Scalar (0-D) result, or whatever shape *combine_fn* returns.

        Raises
        ------
        ValueError
            If the dataset has zero dimensions.
        """
        dataset = object.__getattribute__(self, "_gpu_dataset")

        if dataset.ndim == 0:
            raise ValueError("reduce_double_buffered requires at least 1 dimension")

        if combine_fn is None:
            combine_fn = reduce_fn

        n_rows    = dataset.shape[0]
        row_shape = dataset.shape[1:]
        dtype     = np.dtype(dataset.dtype)
        row_nbytes = int(np.prod(row_shape, dtype=np.intp)) * dtype.itemsize

        if chunk_size is None:
            hdf5_chunks = dataset.chunks
            chunk_size = hdf5_chunks[0] if hdf5_chunks else max(1, n_rows // 8)
        chunk_size = min(chunk_size, n_rows)

        if stream is None:
            stream = cp.cuda.Stream(non_blocking=True)

        chunk_starts = list(range(0, n_rows, chunk_size))

        # Probe output dtype
        _probe = reduce_fn(cp.zeros(1, dtype=dtype))
        out_dtype = _probe.dtype

        # Two pinned host buffers (double-buffered CPU reads)
        buf_shape = (chunk_size,) + row_shape
        _pm = [None, None]
        bufs = [None, None]
        for k in range(2):
            _pm[k], bufs[k] = _alloc_pinned_like(buf_shape, dtype)

        # Reusable GPU temp buffer (one band at a time)
        gpu_temp = cp.empty(buf_shape, dtype=dtype)

        # Partial results: one per band
        partial = cp.empty(len(chunk_starts), dtype=out_dtype)

        # Prime: read first band
        end0 = min(chunk_size, n_rows)
        dataset.read_direct(bufs[0][:end0], source_sel=np.s_[0:end0])

        for i, start in enumerate(chunk_starts):
            end = min(start + chunk_size, n_rows)
            actual_rows = end - start
            cur = i % 2
            nxt = 1 - cur
            nbytes = actual_rows * row_nbytes

            # 1. H2D: cur pinned band → gpu_temp  (async)
            cp.cuda.runtime.memcpyAsync(
                gpu_temp.data.ptr,
                bufs[cur].ctypes.data,
                nbytes,
                cp.cuda.runtime.memcpyHostToDevice,
                stream.ptr,
            )

            # 2. Enqueue transform (optional) + reduce → partial[i]
            with stream:
                band_view = gpu_temp[:actual_rows]
                if transform is not None:
                    band_view = transform(band_view)
                partial[i] = reduce_fn(band_view)

            # 3. Read next band on CPU (overlaps H2D + reduce)
            if i + 1 < len(chunk_starts):
                next_start = chunk_starts[i + 1]
                next_end   = min(next_start + chunk_size, n_rows)
                next_rows  = next_end - next_start
                dataset.read_direct(
                    bufs[nxt][:next_rows],
                    source_sel=np.s_[next_start:next_end],
                )

            # 4. Sync before reusing gpu_temp and pinned buf
            stream.synchronize()

        return combine_fn(partial)

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
# GPUCachedDataset
# ---------------------------------------------------------------------------

class GPUCachedDataset:
    """A :class:`GPUDataset` that keeps the full dataset resident in GPU memory.

    On first access (or when :meth:`preload` is called), the dataset is loaded
    to the GPU using the most efficient available read path.  All subsequent
    indexing, transforms, and reductions operate entirely on the GPU —
    **no disk I/O after the initial load**.

    Use :class:`GPUCachedDataset` when you need to:

    * Apply multiple transforms or reductions to the same data without
      re-reading from disk each time.
    * Repeatedly index into the dataset.
    * Pipeline several GPU operations after a single load.

    For datasets too large to fit in GPU memory, use the streaming methods
    on :class:`GPUDataset` (:meth:`~GPUDataset.reduce_chunks`,
    :meth:`~GPUDataset.read_chunks_to_gpu`, etc.) instead.

    Parameters
    ----------
    dataset : h5py.Dataset or GPUDataset
        The source dataset.
    preload : bool, optional
        If *True* (default), load to GPU immediately.  If *False*, defer
        loading until the first access.

    Examples
    --------
    Eager load and compute::

        with GPUCachedDataset(f["ds"]) as cached:
            total   = cached.reduce(cp.sum)
            maximum = cached.reduce(cp.max)
            subset  = cached[10:50, 5:55]   # pure GPU slice, no I/O

    Lazy load, chained transform + reduce::

        cached = GPUCachedDataset(f["ds"], preload=False)
        result = cached.transform(cp.sqrt).reduce(cp.sum)
        cached.free()
    """

    def __init__(self, dataset, preload=True):
        if isinstance(dataset, GPUDataset):
            self._gpu_ds = dataset
        elif isinstance(dataset, Dataset):
            self._gpu_ds = GPUDataset(dataset)
        else:
            raise TypeError(
                f"Expected GPUDataset or h5py.Dataset, got {type(dataset)!r}"
            )
        self._array = None
        if preload:
            self.preload()

    # ------------------------------------------------------------------
    # Loading / lifecycle
    # ------------------------------------------------------------------

    def preload(self):
        """Load the full dataset into GPU memory (no-op if already loaded).

        Uses :meth:`~GPUDataset.read_chunks_to_gpu` for 2-D/3-D HDF5-chunked
        datasets (tile-by-tile double-buffered), and
        :meth:`~GPUDataset.read_double_buffered` for everything else.

        Returns
        -------
        self
            For chaining.
        """
        if self._array is None:
            ds = object.__getattribute__(self._gpu_ds, "_gpu_dataset")
            if ds.chunks is not None and ds.ndim in (2, 3):
                self._array = self._gpu_ds.read_chunks_to_gpu()
            else:
                self._array = self._gpu_ds.read_double_buffered()
        return self

    def free(self):
        """Release the cached GPU array and free GPU memory."""
        self._array = None

    def reload(self):
        """Free the current cache and reload from disk.

        Returns
        -------
        self
        """
        self.free()
        return self.preload()

    # ------------------------------------------------------------------
    # Access to the cached array
    # ------------------------------------------------------------------

    @property
    def array(self):
        """The GPU-resident :class:`cupy.ndarray`.

        Triggers :meth:`preload` on first access if the data has not been
        loaded yet.
        """
        if self._array is None:
            self.preload()
        return self._array

    def __getitem__(self, args):
        """Index into the cached GPU array — pure GPU, no disk I/O."""
        return self.array[args]

    # ------------------------------------------------------------------
    # GPU-only compute (no I/O after initial load)
    # ------------------------------------------------------------------

    def reduce(self, reduce_fn, transform=None):
        """Apply *reduce_fn* to the cached array (pure GPU, no I/O).

        Parameters
        ----------
        reduce_fn : callable
            E.g. ``cp.sum``, ``cp.max``, ``cp.min``.  Applied to the full
            cached array (or to the result of *transform*).
        transform : callable, optional
            Applied to a **temporary view** of the cached array before the
            reduction.  The cache itself is not modified — use
            :meth:`transform` if you want to update it.

        Returns
        -------
        cupy.ndarray
            Scalar (0-D) result.

        Examples
        --------
        ::

            total = cached.reduce(cp.sum)
            rms   = cached.reduce(lambda x: cp.sqrt(cp.mean(x ** 2)))
            sum_sqrt = cached.reduce(cp.sum, transform=cp.sqrt)
        """
        arr = self.array
        if transform is not None:
            arr = transform(arr)
        return reduce_fn(arr)

    def transform(self, fn):
        """Apply *fn* to the cached array and update the cache with the result.

        Parameters
        ----------
        fn : callable
            Called as ``fn(array)`` and must return a :class:`cupy.ndarray`.
            The returned array replaces the cache.

        Returns
        -------
        self
            For chaining::

                cached.transform(cp.sqrt).transform(lambda x: x * 2.0)

        Notes
        -----
        Out-of-place transforms (e.g. ``cp.sqrt``) allocate a new GPU array
        and the old one is freed.  In-place operations (``array *= 2``) avoid
        the extra allocation but must be written as a lambda::

            cached.transform(lambda x: x.__imul__(2.0) or x)
        """
        self._array = fn(self.array)
        return self

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.free()

    # ------------------------------------------------------------------
    # Attribute delegation and representation
    # ------------------------------------------------------------------

    def __getattr__(self, name):
        return getattr(self._gpu_ds, name)

    def __repr__(self):
        loaded = self._array is not None
        status = f"loaded {list(self._array.shape)}" if loaded else "not loaded"
        return f"<GPUCachedDataset [{status}] wrapping {self._gpu_ds!r}>"


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
