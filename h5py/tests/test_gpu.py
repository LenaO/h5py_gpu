# This file is part of h5py, a Python interface to the HDF5 library.
#
# http://www.h5py.org
#
# Copyright 2008-2013 Andrew Collette and contributors
#
# License:  Standard 3-clause BSD; see "license.txt" for full license terms
#           and contributor agreement.

"""Tests for h5py.gpu — GPU read support via CuPy."""

import numpy as np
import pytest

import h5py
from .common import TestCase

cupy = pytest.importorskip("cupy", reason="CuPy not installed — skipping GPU tests")

from h5py.gpu import GPUDataset, GPUGroup, GPUFile  # noqa: E402

# Optional compression libraries — tests are skipped when absent
try:
    import hdf5plugin as _hdf5plugin
    _HDF5PLUGIN_AVAILABLE = True
except ImportError:
    _HDF5PLUGIN_AVAILABLE = False

try:
    import lz4.block  # noqa: F401
    _LZ4_AVAILABLE = True
except ImportError:
    _LZ4_AVAILABLE = False

try:
    import zstd  # noqa: F401
    _ZSTD_AVAILABLE = True
except ImportError:
    _ZSTD_AVAILABLE = False

try:
    from nvidia import nvcomp  # noqa: F401
    _NVCOMP_AVAILABLE = True
except ImportError:
    _NVCOMP_AVAILABLE = False

requires_hdf5plugin = pytest.mark.skipif(
    not _HDF5PLUGIN_AVAILABLE, reason="hdf5plugin not installed"
)
requires_lz4 = pytest.mark.skipif(
    not (_HDF5PLUGIN_AVAILABLE and _LZ4_AVAILABLE),
    reason="hdf5plugin or lz4 not installed",
)
requires_zstd = pytest.mark.skipif(
    not (_HDF5PLUGIN_AVAILABLE and _ZSTD_AVAILABLE),
    reason="hdf5plugin or zstd not installed",
)
requires_nvcomp_lz4 = pytest.mark.skipif(
    not (_HDF5PLUGIN_AVAILABLE and _NVCOMP_AVAILABLE),
    reason="hdf5plugin or nvidia-nvcomp not installed",
)
requires_nvcomp_zstd = pytest.mark.skipif(
    not (_HDF5PLUGIN_AVAILABLE and _NVCOMP_AVAILABLE),
    reason="hdf5plugin or nvidia-nvcomp not installed",
)


class TestGPUDataset(TestCase):
    def setUp(self):
        self.f = h5py.File(self.mktemp(), "w")

    def tearDown(self):
        if self.f:
            self.f.close()

    # ------------------------------------------------------------------
    # Basic read
    # ------------------------------------------------------------------

    def test_getitem_full(self):
        """Reading the whole dataset returns a cupy.ndarray with correct data."""
        data = np.arange(100, dtype=np.float32).reshape(10, 10)
        self.f.create_dataset("ds", data=data)
        gpu_ds = GPUDataset(self.f["ds"])
        result = gpu_ds[:]
        assert isinstance(result, cupy.ndarray)
        np.testing.assert_array_equal(cupy.asnumpy(result), data)

    def test_getitem_slice(self):
        """Partial slice returns the correct sub-array on the GPU."""
        data = np.arange(50, dtype=np.int32)
        self.f.create_dataset("ds", data=data)
        gpu_ds = GPUDataset(self.f["ds"])
        result = gpu_ds[10:20]
        assert isinstance(result, cupy.ndarray)
        np.testing.assert_array_equal(cupy.asnumpy(result), data[10:20])

    def test_getitem_multidim(self):
        """Multi-dimensional slicing works correctly."""
        data = np.random.rand(8, 12).astype(np.float64)
        self.f.create_dataset("ds", data=data)
        gpu_ds = GPUDataset(self.f["ds"])
        result = gpu_ds[2:5, 3:9]
        np.testing.assert_array_almost_equal(cupy.asnumpy(result), data[2:5, 3:9])

    # ------------------------------------------------------------------
    # dtype preservation
    # ------------------------------------------------------------------

    def test_dtype_preserved(self):
        """The GPU array has the same dtype as the HDF5 dataset."""
        for dt in [np.float32, np.float64, np.int16, np.int64, np.uint8]:
            data = np.ones(20, dtype=dt)
            name = f"ds_{dt.__name__}"
            self.f.create_dataset(name, data=data)
            gpu_ds = GPUDataset(self.f[name])
            result = gpu_ds[:]
            assert result.dtype == np.dtype(dt), f"dtype mismatch for {dt}"

    # ------------------------------------------------------------------
    # Empty dataset
    # ------------------------------------------------------------------

    def test_empty_dataset(self):
        """Reading an all-zero-sized dimension returns an empty cupy array."""
        data = np.empty((0, 5), dtype=np.float32)
        self.f.create_dataset("empty", data=data)
        gpu_ds = GPUDataset(self.f["empty"])
        result = gpu_ds[:]
        assert isinstance(result, cupy.ndarray)
        assert result.shape == (0, 5)

    # ------------------------------------------------------------------
    # read_direct_gpu
    # ------------------------------------------------------------------

    def test_read_direct_gpu_full(self):
        """read_direct_gpu fills a pre-allocated CuPy array correctly."""
        data = np.arange(60, dtype=np.float64).reshape(6, 10)
        self.f.create_dataset("ds", data=data)
        gpu_ds = GPUDataset(self.f["ds"])
        dest = cupy.empty((6, 10), dtype=np.float64)
        gpu_ds.read_direct_gpu(dest)
        np.testing.assert_array_equal(cupy.asnumpy(dest), data)

    def test_read_direct_gpu_source_sel(self):
        """source_sel restricts the HDF5 read region."""
        data = np.arange(100, dtype=np.int32)
        self.f.create_dataset("ds", data=data)
        gpu_ds = GPUDataset(self.f["ds"])
        dest = cupy.empty(20, dtype=np.int32)
        gpu_ds.read_direct_gpu(dest, source_sel=np.s_[30:50])
        np.testing.assert_array_equal(cupy.asnumpy(dest), data[30:50])

    # ------------------------------------------------------------------
    # read_double_buffered
    # ------------------------------------------------------------------

    def test_double_buffered_full_1d(self):
        """Double-buffered read of a 1-D dataset matches direct read."""
        data = np.arange(256, dtype=np.float64)
        self.f.create_dataset("ds", data=data)
        gpu_ds = GPUDataset(self.f["ds"])
        result = gpu_ds.read_double_buffered(chunk_size=32)
        assert isinstance(result, cupy.ndarray)
        np.testing.assert_array_equal(cupy.asnumpy(result), data)

    def test_double_buffered_full_2d(self):
        """Double-buffered read of a 2-D dataset matches direct read."""
        data = np.random.rand(64, 32).astype(np.float32)
        self.f.create_dataset("ds", data=data)
        gpu_ds = GPUDataset(self.f["ds"])
        result = gpu_ds.read_double_buffered(chunk_size=8)
        np.testing.assert_array_almost_equal(cupy.asnumpy(result), data)

    def test_double_buffered_uneven_chunks(self):
        """Correct result when dataset rows are not a multiple of chunk_size."""
        data = np.arange(100, dtype=np.int32)
        self.f.create_dataset("ds", data=data)
        gpu_ds = GPUDataset(self.f["ds"])
        result = gpu_ds.read_double_buffered(chunk_size=30)  # 30+30+30+10
        np.testing.assert_array_equal(cupy.asnumpy(result), data)

    def test_double_buffered_single_chunk(self):
        """Works when chunk_size >= dataset size (single iteration)."""
        data = np.ones(20, dtype=np.float32)
        self.f.create_dataset("ds", data=data)
        gpu_ds = GPUDataset(self.f["ds"])
        result = gpu_ds.read_double_buffered(chunk_size=100)
        np.testing.assert_array_equal(cupy.asnumpy(result), data)

    def test_double_buffered_preallocated_out(self):
        """Fills a pre-allocated output array correctly."""
        data = np.arange(64, dtype=np.float64)
        self.f.create_dataset("ds", data=data)
        gpu_ds = GPUDataset(self.f["ds"])
        out = cupy.empty(64, dtype=np.float64)
        result = gpu_ds.read_double_buffered(chunk_size=16, out=out)
        assert result is out
        np.testing.assert_array_equal(cupy.asnumpy(out), data)

    def test_double_buffered_custom_stream(self):
        """Accepts a caller-provided CUDA stream."""
        data = np.arange(50, dtype=np.int32)
        self.f.create_dataset("ds", data=data)
        gpu_ds = GPUDataset(self.f["ds"])
        stream = cupy.cuda.Stream(non_blocking=True)
        result = gpu_ds.read_double_buffered(chunk_size=10, stream=stream)
        np.testing.assert_array_equal(cupy.asnumpy(result), data)

    def test_double_buffered_transform_1d(self):
        """transform applied correctly on a 1-D dataset."""
        data = np.arange(1, 257, dtype=np.float32)
        self.f.create_dataset("ds", data=data)
        result = GPUDataset(self.f["ds"]).read_double_buffered(
            chunk_size=32, transform=lambda x: x * 2.0
        )
        np.testing.assert_array_almost_equal(cupy.asnumpy(result), data * 2.0)

    def test_double_buffered_transform_sqrt(self):
        """CuPy ufunc (sqrt) works as transform on 2-D dataset."""
        data = np.arange(1, 65, dtype=np.float32).reshape(8, 8)
        self.f.create_dataset("ds", data=data)
        result = GPUDataset(self.f["ds"]).read_double_buffered(
            chunk_size=2, transform=cupy.sqrt
        )
        np.testing.assert_array_almost_equal(cupy.asnumpy(result), np.sqrt(data))

    def test_double_buffered_scalar_raises(self):
        """Scalar datasets raise ValueError."""
        self.f.create_dataset("sc", data=np.float32(3.14))
        gpu_ds = GPUDataset(self.f["sc"])
        with pytest.raises(ValueError, match="scalar"):
            gpu_ds.read_double_buffered()

    def test_double_buffered_matches_getitem(self):
        """read_double_buffered and __getitem__[:] return the same data."""
        data = np.random.rand(128, 16).astype(np.float32)
        self.f.create_dataset("ds", data=data)
        gpu_ds = GPUDataset(self.f["ds"])
        via_dbl = gpu_ds.read_double_buffered(chunk_size=16)
        via_idx = gpu_ds[:]
        np.testing.assert_array_almost_equal(cupy.asnumpy(via_dbl),
                                              cupy.asnumpy(via_idx))

    def test_read_direct_gpu_wrong_type(self):
        """Passing a non-cupy array to read_direct_gpu raises TypeError."""
        data = np.ones(10, dtype=np.float32)
        self.f.create_dataset("ds", data=data)
        gpu_ds = GPUDataset(self.f["ds"])
        with pytest.raises(TypeError):
            gpu_ds.read_direct_gpu(np.empty(10, dtype=np.float32))

    # ------------------------------------------------------------------
    # Attribute delegation
    # ------------------------------------------------------------------

    def test_attribute_delegation(self):
        """shape, dtype, and other Dataset attributes are accessible."""
        data = np.zeros((3, 4, 5), dtype=np.float32)
        self.f.create_dataset("ds", data=data)
        gpu_ds = GPUDataset(self.f["ds"])
        assert gpu_ds.shape == (3, 4, 5)
        assert gpu_ds.dtype == np.float32
        assert gpu_ds.ndim == 3

    def test_repr(self):
        self.f.create_dataset("ds", data=np.zeros(1))
        gpu_ds = GPUDataset(self.f["ds"])
        assert "GPUDataset" in repr(gpu_ds)

    # ------------------------------------------------------------------
    # Wrong input type
    # ------------------------------------------------------------------

    def test_requires_dataset(self):
        """Passing a non-Dataset raises TypeError."""
        with pytest.raises(TypeError):
            GPUDataset(self.f)


class TestReadSelectionChunked(TestCase):
    def setUp(self):
        self.f = h5py.File(self.mktemp(), "w")

    def tearDown(self):
        if self.f:
            self.f.close()

    # ------------------------------------------------------------------
    # 2-D: correctness across different selection / chunk combinations
    # ------------------------------------------------------------------

    def test_2d_interior_sel(self):
        """Selection entirely inside the dataset, crossing chunk boundaries."""
        data = np.arange(128 * 128, dtype=np.float32).reshape(128, 128)
        self.f.create_dataset("ds", data=data, chunks=(32, 32))
        gpu_ds = GPUDataset(self.f["ds"])
        sel = np.s_[10:90, 20:110]
        result = gpu_ds.read_selection_chunked(sel)
        np.testing.assert_array_equal(cupy.asnumpy(result), data[sel])

    def test_2d_sel_aligned_to_chunks(self):
        """Selection that aligns exactly to chunk boundaries."""
        data = np.arange(64 * 64, dtype=np.float32).reshape(64, 64)
        self.f.create_dataset("ds", data=data, chunks=(16, 16))
        gpu_ds = GPUDataset(self.f["ds"])
        sel = np.s_[16:48, 32:64]
        result = gpu_ds.read_selection_chunked(sel)
        np.testing.assert_array_equal(cupy.asnumpy(result), data[sel])

    def test_2d_sel_partial_first_last_chunk(self):
        """Selection that only partially covers the first and last chunk."""
        data = np.arange(100 * 80, dtype=np.int32).reshape(100, 80)
        self.f.create_dataset("ds", data=data, chunks=(16, 16))
        gpu_ds = GPUDataset(self.f["ds"])
        sel = np.s_[5:85, 3:70]
        result = gpu_ds.read_selection_chunked(sel)
        np.testing.assert_array_equal(cupy.asnumpy(result), data[sel])

    def test_2d_full_dataset_via_sel(self):
        """Selecting the full dataset returns the same result as [:]."""
        data = np.random.rand(64, 64).astype(np.float32)
        self.f.create_dataset("ds", data=data, chunks=(16, 16))
        gpu_ds = GPUDataset(self.f["ds"])
        result = gpu_ds.read_selection_chunked(np.s_[:, :])
        np.testing.assert_array_almost_equal(cupy.asnumpy(result), data)

    def test_2d_single_row(self):
        """Selecting a single row (all columns)."""
        data = np.arange(64 * 64, dtype=np.float64).reshape(64, 64)
        self.f.create_dataset("ds", data=data, chunks=(8, 8))
        gpu_ds = GPUDataset(self.f["ds"])
        result = gpu_ds.read_selection_chunked(np.s_[17:18, :])
        np.testing.assert_array_equal(cupy.asnumpy(result), data[17:18, :])

    def test_2d_single_column(self):
        """Selecting a single column (all rows)."""
        data = np.arange(64 * 64, dtype=np.float64).reshape(64, 64)
        self.f.create_dataset("ds", data=data, chunks=(8, 8))
        gpu_ds = GPUDataset(self.f["ds"])
        result = gpu_ds.read_selection_chunked(np.s_[:, 25:26])
        np.testing.assert_array_equal(cupy.asnumpy(result), data[:, 25:26])

    # ------------------------------------------------------------------
    # 2-D: __getitem__ dispatch
    # ------------------------------------------------------------------

    def test_getitem_dispatches_for_chunked_2d(self):
        """__getitem__ with slices on a chunked 2-D dataset uses chunk path."""
        data = np.arange(64 * 64, dtype=np.float32).reshape(64, 64)
        self.f.create_dataset("ds", data=data, chunks=(16, 16))
        gpu_ds = GPUDataset(self.f["ds"])
        result = gpu_ds[10:50, 5:55]
        np.testing.assert_array_equal(cupy.asnumpy(result), data[10:50, 5:55])

    def test_getitem_fallback_for_unchunked(self):
        """__getitem__ on a contiguous dataset still returns correct data."""
        data = np.arange(64, dtype=np.float32)
        self.f.create_dataset("ds", data=data)   # contiguous
        gpu_ds = GPUDataset(self.f["ds"])
        np.testing.assert_array_equal(cupy.asnumpy(gpu_ds[:]), data)

    # ------------------------------------------------------------------
    # 3-D
    # ------------------------------------------------------------------

    def test_3d_interior_sel(self):
        """3-D selection crossing chunk boundaries in all three axes."""
        data = np.arange(32 * 32 * 32, dtype=np.float32).reshape(32, 32, 32)
        self.f.create_dataset("ds", data=data, chunks=(8, 8, 8))
        gpu_ds = GPUDataset(self.f["ds"])
        sel = np.s_[3:25, 5:27, 2:30]
        result = gpu_ds.read_selection_chunked(sel)
        np.testing.assert_array_equal(cupy.asnumpy(result), data[sel])

    def test_3d_matches_getitem(self):
        data = np.random.rand(16, 24, 32).astype(np.float32)
        self.f.create_dataset("ds", data=data, chunks=(4, 8, 8))
        gpu_ds = GPUDataset(self.f["ds"])
        sel = np.s_[1:14, 3:20, 5:28]
        np.testing.assert_array_almost_equal(
            cupy.asnumpy(gpu_ds.read_selection_chunked(sel)),
            data[sel],
        )

    # ------------------------------------------------------------------
    # Edge / error cases
    # ------------------------------------------------------------------

    def test_preallocated_out(self):
        data = np.arange(64 * 64, dtype=np.float32).reshape(64, 64)
        self.f.create_dataset("ds", data=data, chunks=(16, 16))
        gpu_ds = GPUDataset(self.f["ds"])
        sel = np.s_[8:40, 8:40]
        out = cupy.empty((32, 32), dtype=np.float32)
        result = gpu_ds.read_selection_chunked(sel, out=out)
        assert result is out
        np.testing.assert_array_equal(cupy.asnumpy(out), data[sel])

    def test_raises_on_unchunked(self):
        self.f.create_dataset("ds", data=np.zeros((16, 16)))
        with pytest.raises(ValueError, match="not HDF5-chunked"):
            GPUDataset(self.f["ds"]).read_selection_chunked(np.s_[:, :])

    def test_raises_on_stepped_slice(self):
        data = np.zeros((32, 32))
        self.f.create_dataset("ds", data=data, chunks=(8, 8))
        with pytest.raises(ValueError):
            GPUDataset(self.f["ds"]).read_selection_chunked(np.s_[::2, :])

    def test_ellipsis_selection(self):
        """Ellipsis in selection is expanded correctly."""
        data = np.arange(64 * 64, dtype=np.float32).reshape(64, 64)
        self.f.create_dataset("ds", data=data, chunks=(16, 16))
        gpu_ds = GPUDataset(self.f["ds"])
        result = gpu_ds.read_selection_chunked(np.s_[..., 10:50])
        np.testing.assert_array_equal(cupy.asnumpy(result), data[..., 10:50])


class TestReadChunksToGPU(TestCase):
    def setUp(self):
        self.f = h5py.File(self.mktemp(), "w")

    def tearDown(self):
        if self.f:
            self.f.close()

    # ------------------------------------------------------------------
    # 2-D
    # ------------------------------------------------------------------

    def test_2d_divisible(self):
        """2-D dataset whose dims are exact multiples of chunk size."""
        data = np.arange(64 * 64, dtype=np.float32).reshape(64, 64)
        self.f.create_dataset("ds", data=data, chunks=(16, 16))
        result = GPUDataset(self.f["ds"]).read_chunks_to_gpu()
        assert isinstance(result, cupy.ndarray)
        np.testing.assert_array_equal(cupy.asnumpy(result), data)

    def test_2d_non_divisible(self):
        """Edge tiles (rows/cols not divisible by chunk size) are correct."""
        data = np.arange(50 * 70, dtype=np.int32).reshape(50, 70)
        self.f.create_dataset("ds", data=data, chunks=(16, 16))
        result = GPUDataset(self.f["ds"]).read_chunks_to_gpu()
        np.testing.assert_array_equal(cupy.asnumpy(result), data)

    def test_2d_single_chunk(self):
        """Dataset fits in a single HDF5 chunk (chunk shape == dataset shape)."""
        data = np.ones((8, 8), dtype=np.float64)
        self.f.create_dataset("ds", data=data, chunks=(8, 8))
        result = GPUDataset(self.f["ds"]).read_chunks_to_gpu()
        np.testing.assert_array_equal(cupy.asnumpy(result), data)

    def test_2d_matches_getitem(self):
        """read_chunks_to_gpu and __getitem__[:] return identical data."""
        data = np.random.rand(96, 128).astype(np.float32)
        self.f.create_dataset("ds", data=data, chunks=(32, 32))
        gpu_ds = GPUDataset(self.f["ds"])
        via_tiles = gpu_ds.read_chunks_to_gpu()
        via_idx   = gpu_ds[:]
        np.testing.assert_array_almost_equal(cupy.asnumpy(via_tiles),
                                              cupy.asnumpy(via_idx))

    def test_2d_preallocated_out(self):
        """Fills a pre-allocated output array and returns it."""
        data = np.arange(32 * 32, dtype=np.float32).reshape(32, 32)
        self.f.create_dataset("ds", data=data, chunks=(8, 8))
        gpu_ds = GPUDataset(self.f["ds"])
        out = cupy.empty((32, 32), dtype=np.float32)
        result = gpu_ds.read_chunks_to_gpu(out=out)
        assert result is out
        np.testing.assert_array_equal(cupy.asnumpy(out), data)

    # ------------------------------------------------------------------
    # 3-D
    # ------------------------------------------------------------------

    def test_3d_divisible(self):
        """3-D dataset whose dims are exact multiples of chunk size."""
        data = np.arange(8 * 16 * 16, dtype=np.float32).reshape(8, 16, 16)
        self.f.create_dataset("ds", data=data, chunks=(2, 8, 8))
        result = GPUDataset(self.f["ds"]).read_chunks_to_gpu()
        np.testing.assert_array_equal(cupy.asnumpy(result), data)

    def test_3d_non_divisible(self):
        """3-D edge tiles (dims not divisible by chunk size)."""
        data = np.random.rand(10, 15, 20).astype(np.float64)
        self.f.create_dataset("ds", data=data, chunks=(3, 4, 6))
        result = GPUDataset(self.f["ds"]).read_chunks_to_gpu()
        np.testing.assert_array_almost_equal(cupy.asnumpy(result), data)

    def test_3d_matches_getitem(self):
        data = np.random.rand(12, 24, 32).astype(np.float32)
        self.f.create_dataset("ds", data=data, chunks=(4, 8, 8))
        gpu_ds = GPUDataset(self.f["ds"])
        np.testing.assert_array_almost_equal(
            cupy.asnumpy(gpu_ds.read_chunks_to_gpu()),
            cupy.asnumpy(gpu_ds[:]),
        )

    # ------------------------------------------------------------------
    # Error cases
    # ------------------------------------------------------------------

    def test_raises_on_unchunked(self):
        self.f.create_dataset("ds", data=np.zeros((8, 8)))  # contiguous
        with pytest.raises(ValueError, match="not HDF5-chunked"):
            GPUDataset(self.f["ds"]).read_chunks_to_gpu()

    def test_raises_on_1d(self):
        self.f.create_dataset("ds", data=np.zeros(32), chunks=(8,))
        with pytest.raises(ValueError, match="ndim"):
            GPUDataset(self.f["ds"]).read_chunks_to_gpu()

    def test_raises_on_wrong_out_shape(self):
        data = np.zeros((8, 8), dtype=np.float32)
        self.f.create_dataset("ds", data=data, chunks=(4, 4))
        out = cupy.empty((4, 8), dtype=np.float32)
        with pytest.raises(ValueError):
            GPUDataset(self.f["ds"]).read_chunks_to_gpu(out=out)

    def test_custom_stream(self):
        data = np.arange(16 * 16, dtype=np.int32).reshape(16, 16)
        self.f.create_dataset("ds", data=data, chunks=(4, 4))
        stream = cupy.cuda.Stream(non_blocking=True)
        result = GPUDataset(self.f["ds"]).read_chunks_to_gpu(stream=stream)
        np.testing.assert_array_equal(cupy.asnumpy(result), data)


class TestGPUGroup(TestCase):
    def setUp(self):
        self.f = h5py.File(self.mktemp(), "w")
        self.f.create_dataset("top", data=np.arange(10, dtype=np.int32))
        grp = self.f.create_group("sub")
        grp.create_dataset("nested", data=np.ones((4, 4), dtype=np.float32))

    def tearDown(self):
        if self.f:
            self.f.close()

    def test_dataset_access_returns_gpu_dataset(self):
        gpu_f = GPUGroup(self.f)
        item = gpu_f["top"]
        assert isinstance(item, GPUDataset)

    def test_nested_group_returns_gpu_group(self):
        gpu_f = GPUGroup(self.f)
        sub = gpu_f["sub"]
        assert isinstance(sub, GPUGroup)

    def test_nested_dataset_read(self):
        gpu_f = GPUGroup(self.f)
        result = gpu_f["sub"]["nested"][:]
        assert isinstance(result, cupy.ndarray)
        np.testing.assert_array_equal(cupy.asnumpy(result), np.ones((4, 4), dtype=np.float32))

    def test_iteration(self):
        gpu_f = GPUGroup(self.f)
        keys = list(gpu_f)
        assert set(keys) == {"top", "sub"}

    def test_contains(self):
        gpu_f = GPUGroup(self.f)
        assert "top" in gpu_f
        assert "missing" not in gpu_f

    def test_len(self):
        gpu_f = GPUGroup(self.f)
        assert len(gpu_f) == 2

    def test_repr(self):
        gpu_f = GPUGroup(self.f)
        assert "GPUGroup" in repr(gpu_f)


class TestTransformAndParallel(TestCase):
    """Tests for the transform parameter and read_chunks_parallel."""

    def setUp(self):
        self.f = h5py.File(self.mktemp(), "w")

    def tearDown(self):
        if self.f:
            self.f.close()

    # ------------------------------------------------------------------
    # read_chunks_to_gpu  with transform
    # ------------------------------------------------------------------

    def test_chunks_transform_scale(self):
        """transform applied after H2D produces correct scaled values."""
        data = np.arange(64 * 64, dtype=np.float32).reshape(64, 64)
        self.f.create_dataset("ds", data=data, chunks=(16, 16))
        result = GPUDataset(self.f["ds"]).read_chunks_to_gpu(
            transform=lambda x: x * 2.0
        )
        np.testing.assert_array_almost_equal(cupy.asnumpy(result), data * 2.0)

    def test_chunks_transform_sqrt(self):
        """CuPy ufunc (cp.sqrt) works as transform."""
        data = np.arange(1, 64 * 64 + 1, dtype=np.float32).reshape(64, 64)
        self.f.create_dataset("ds", data=data, chunks=(16, 16))
        result = GPUDataset(self.f["ds"]).read_chunks_to_gpu(
            transform=cupy.sqrt
        )
        np.testing.assert_array_almost_equal(cupy.asnumpy(result), np.sqrt(data))

    def test_chunks_transform_none_unchanged(self):
        """transform=None (default) leaves data unchanged."""
        data = np.random.rand(32, 32).astype(np.float32)
        self.f.create_dataset("ds", data=data, chunks=(8, 8))
        result = GPUDataset(self.f["ds"]).read_chunks_to_gpu(transform=None)
        np.testing.assert_array_almost_equal(cupy.asnumpy(result), data)

    def test_chunks_transform_3d(self):
        """transform works on 3-D datasets."""
        data = np.arange(8 * 16 * 16, dtype=np.float32).reshape(8, 16, 16) + 1.0
        self.f.create_dataset("ds", data=data, chunks=(2, 8, 8))
        result = GPUDataset(self.f["ds"]).read_chunks_to_gpu(
            transform=lambda x: x + 10.0
        )
        np.testing.assert_array_almost_equal(cupy.asnumpy(result), data + 10.0)

    # ------------------------------------------------------------------
    # read_selection_chunked  with transform
    # ------------------------------------------------------------------

    def test_sel_transform_scale(self):
        """transform applied to each touched chunk sub-region."""
        data = np.arange(64 * 64, dtype=np.float32).reshape(64, 64)
        self.f.create_dataset("ds", data=data, chunks=(16, 16))
        sel = np.s_[10:50, 5:55]
        result = GPUDataset(self.f["ds"]).read_selection_chunked(
            sel, transform=lambda x: x * 3.0
        )
        np.testing.assert_array_almost_equal(
            cupy.asnumpy(result), data[sel] * 3.0
        )

    def test_sel_transform_sqrt(self):
        data = np.arange(1, 64 * 64 + 1, dtype=np.float64).reshape(64, 64)
        self.f.create_dataset("ds", data=data, chunks=(16, 16))
        sel = np.s_[8:40, 8:40]
        result = GPUDataset(self.f["ds"]).read_selection_chunked(
            sel, transform=cupy.sqrt
        )
        np.testing.assert_array_almost_equal(
            cupy.asnumpy(result), np.sqrt(data[sel])
        )

    # ------------------------------------------------------------------
    # read_chunks_parallel
    # ------------------------------------------------------------------

    def test_parallel_2d_n2(self):
        """2-D dataset, n_streams=2, no transform — matches direct read."""
        data = np.arange(64 * 64, dtype=np.float32).reshape(64, 64)
        self.f.create_dataset("ds", data=data, chunks=(16, 16))
        result = GPUDataset(self.f["ds"]).read_chunks_parallel(n_streams=2)
        np.testing.assert_array_equal(cupy.asnumpy(result), data)

    def test_parallel_2d_n4(self):
        """n_streams=4 still produces correct result."""
        data = np.arange(64 * 64, dtype=np.float32).reshape(64, 64)
        self.f.create_dataset("ds", data=data, chunks=(16, 16))
        result = GPUDataset(self.f["ds"]).read_chunks_parallel(n_streams=4)
        np.testing.assert_array_equal(cupy.asnumpy(result), data)

    def test_parallel_non_divisible(self):
        """Edge tiles handled correctly with multiple streams."""
        data = np.arange(50 * 70, dtype=np.int32).reshape(50, 70)
        self.f.create_dataset("ds", data=data, chunks=(16, 16))
        result = GPUDataset(self.f["ds"]).read_chunks_parallel(n_streams=3)
        np.testing.assert_array_equal(cupy.asnumpy(result), data)

    def test_parallel_3d(self):
        """3-D dataset with multiple streams."""
        data = np.arange(8 * 16 * 16, dtype=np.float32).reshape(8, 16, 16)
        self.f.create_dataset("ds", data=data, chunks=(2, 8, 8))
        result = GPUDataset(self.f["ds"]).read_chunks_parallel(n_streams=2)
        np.testing.assert_array_equal(cupy.asnumpy(result), data)

    def test_parallel_transform(self):
        """transform applied correctly with multiple streams."""
        data = np.arange(1, 64 * 64 + 1, dtype=np.float32).reshape(64, 64)
        self.f.create_dataset("ds", data=data, chunks=(16, 16))
        result = GPUDataset(self.f["ds"]).read_chunks_parallel(
            n_streams=2, transform=cupy.sqrt
        )
        np.testing.assert_array_almost_equal(cupy.asnumpy(result), np.sqrt(data))

    def test_parallel_n1_same_as_single(self):
        """n_streams=1 gives the same result as read_chunks_to_gpu."""
        data = np.random.rand(32, 32).astype(np.float32)
        self.f.create_dataset("ds", data=data, chunks=(8, 8))
        gpu_ds = GPUDataset(self.f["ds"])
        r1 = gpu_ds.read_chunks_to_gpu()
        r2 = gpu_ds.read_chunks_parallel(n_streams=1)
        np.testing.assert_array_almost_equal(cupy.asnumpy(r1), cupy.asnumpy(r2))

    def test_parallel_preallocated_out(self):
        """Fills a pre-allocated output array."""
        data = np.arange(32 * 32, dtype=np.float32).reshape(32, 32)
        self.f.create_dataset("ds", data=data, chunks=(8, 8))
        gpu_ds = GPUDataset(self.f["ds"])
        out = cupy.empty((32, 32), dtype=np.float32)
        result = gpu_ds.read_chunks_parallel(out=out, n_streams=2)
        assert result is out
        np.testing.assert_array_equal(cupy.asnumpy(out), data)

    def test_parallel_raises_on_unchunked(self):
        self.f.create_dataset("ds", data=np.zeros((8, 8)))
        with pytest.raises(ValueError, match="not HDF5-chunked"):
            GPUDataset(self.f["ds"]).read_chunks_parallel()

    def test_parallel_raises_on_1d(self):
        self.f.create_dataset("ds", data=np.zeros(32), chunks=(8,))
        with pytest.raises(ValueError, match="ndim"):
            GPUDataset(self.f["ds"]).read_chunks_parallel()


class TestReductions(TestCase):
    """Tests for reduce_chunks and reduce_double_buffered."""

    def setUp(self):
        self.f = h5py.File(self.mktemp(), "w")

    def tearDown(self):
        if self.f:
            self.f.close()

    # ------------------------------------------------------------------
    # reduce_chunks  (2-D)
    # ------------------------------------------------------------------

    def test_reduce_chunks_sum_2d(self):
        data = np.arange(1, 64 * 64 + 1, dtype=np.float32).reshape(64, 64)
        self.f.create_dataset("ds", data=data, chunks=(16, 16))
        result = GPUDataset(self.f["ds"]).reduce_chunks(cupy.sum)
        np.testing.assert_allclose(float(result), float(data.sum()), rtol=1e-5)

    def test_reduce_chunks_max_2d(self):
        data = np.random.rand(64, 64).astype(np.float32)
        self.f.create_dataset("ds", data=data, chunks=(16, 16))
        result = GPUDataset(self.f["ds"]).reduce_chunks(cupy.max)
        np.testing.assert_allclose(float(result), float(data.max()), rtol=1e-5)

    def test_reduce_chunks_min_2d(self):
        data = np.random.rand(64, 64).astype(np.float32)
        self.f.create_dataset("ds", data=data, chunks=(16, 16))
        result = GPUDataset(self.f["ds"]).reduce_chunks(cupy.min)
        np.testing.assert_allclose(float(result), float(data.min()), rtol=1e-5)

    def test_reduce_chunks_mean_2d(self):
        """Global mean via sum / n_elements (correct for unequal edge chunks)."""
        data = np.random.rand(50, 70).astype(np.float64)
        self.f.create_dataset("ds", data=data, chunks=(16, 16))
        n = int(np.prod(data.shape))
        result = GPUDataset(self.f["ds"]).reduce_chunks(
            cupy.sum, combine_fn=lambda x: cupy.sum(x) / n
        )
        np.testing.assert_allclose(float(result), float(data.mean()), rtol=1e-5)

    def test_reduce_chunks_sum_3d(self):
        data = np.random.rand(8, 16, 16).astype(np.float32)
        self.f.create_dataset("ds", data=data, chunks=(2, 8, 8))
        result = GPUDataset(self.f["ds"]).reduce_chunks(cupy.sum)
        np.testing.assert_allclose(float(result), float(data.sum()), rtol=1e-4)

    def test_reduce_chunks_max_3d(self):
        data = np.random.rand(8, 16, 16).astype(np.float32)
        self.f.create_dataset("ds", data=data, chunks=(2, 8, 8))
        result = GPUDataset(self.f["ds"]).reduce_chunks(cupy.max)
        np.testing.assert_allclose(float(result), float(data.max()), rtol=1e-5)

    def test_reduce_chunks_non_divisible(self):
        """Edge tiles (dataset not divisible by chunk size)."""
        data = np.random.rand(50, 70).astype(np.float32)
        self.f.create_dataset("ds", data=data, chunks=(16, 16))
        result = GPUDataset(self.f["ds"]).reduce_chunks(cupy.sum)
        np.testing.assert_allclose(float(result), float(data.sum()), rtol=1e-4)

    def test_reduce_chunks_single_tile(self):
        """Dataset fits in one chunk."""
        data = np.arange(1, 17, dtype=np.float32).reshape(4, 4)
        self.f.create_dataset("ds", data=data, chunks=(4, 4))
        result = GPUDataset(self.f["ds"]).reduce_chunks(cupy.sum)
        np.testing.assert_allclose(float(result), float(data.sum()), rtol=1e-5)

    def test_reduce_chunks_with_transform(self):
        """Sum of sqrt(data) via transform."""
        data = np.random.rand(64, 64).astype(np.float32) + 0.01
        self.f.create_dataset("ds", data=data, chunks=(16, 16))
        result = GPUDataset(self.f["ds"]).reduce_chunks(
            cupy.sum, transform=cupy.sqrt
        )
        np.testing.assert_allclose(
            float(result), float(np.sqrt(data).sum()), rtol=1e-4
        )

    def test_reduce_chunks_raises_on_unchunked(self):
        self.f.create_dataset("ds", data=np.zeros((8, 8)))
        with pytest.raises(ValueError, match="not HDF5-chunked"):
            GPUDataset(self.f["ds"]).reduce_chunks(cupy.sum)

    def test_reduce_chunks_raises_on_1d(self):
        self.f.create_dataset("ds", data=np.zeros(32), chunks=(8,))
        with pytest.raises(ValueError, match="ndim"):
            GPUDataset(self.f["ds"]).reduce_chunks(cupy.sum)

    # ------------------------------------------------------------------
    # reduce_double_buffered  (any ndim, including 1-D)
    # ------------------------------------------------------------------

    def test_reduce_dbl_sum_1d(self):
        data = np.arange(1, 257, dtype=np.float32)
        self.f.create_dataset("ds", data=data)
        result = GPUDataset(self.f["ds"]).reduce_double_buffered(
            cupy.sum, chunk_size=32
        )
        np.testing.assert_allclose(float(result), float(data.sum()), rtol=1e-5)

    def test_reduce_dbl_max_1d(self):
        data = np.random.rand(256).astype(np.float32)
        self.f.create_dataset("ds", data=data)
        result = GPUDataset(self.f["ds"]).reduce_double_buffered(cupy.max)
        np.testing.assert_allclose(float(result), float(data.max()), rtol=1e-5)

    def test_reduce_dbl_min_1d(self):
        data = np.random.rand(256).astype(np.float32)
        self.f.create_dataset("ds", data=data)
        result = GPUDataset(self.f["ds"]).reduce_double_buffered(cupy.min)
        np.testing.assert_allclose(float(result), float(data.min()), rtol=1e-5)

    def test_reduce_dbl_sum_2d(self):
        data = np.random.rand(64, 32).astype(np.float64)
        self.f.create_dataset("ds", data=data)
        result = GPUDataset(self.f["ds"]).reduce_double_buffered(
            cupy.sum, chunk_size=8
        )
        np.testing.assert_allclose(float(result), float(data.sum()), rtol=1e-10)

    def test_reduce_dbl_mean_1d(self):
        """Global mean via sum/n."""
        data = np.random.rand(256).astype(np.float64)
        self.f.create_dataset("ds", data=data)
        n = data.size
        result = GPUDataset(self.f["ds"]).reduce_double_buffered(
            cupy.sum, combine_fn=lambda x: cupy.sum(x) / n, chunk_size=32
        )
        np.testing.assert_allclose(float(result), float(data.mean()), rtol=1e-10)

    def test_reduce_dbl_uneven_chunks(self):
        """Correct when data length is not a multiple of chunk_size."""
        data = np.arange(100, dtype=np.float32)
        self.f.create_dataset("ds", data=data)
        result = GPUDataset(self.f["ds"]).reduce_double_buffered(
            cupy.sum, chunk_size=30
        )
        np.testing.assert_allclose(float(result), float(data.sum()), rtol=1e-5)

    def test_reduce_dbl_with_transform(self):
        """Sum of squares via transform=lambda x: x**2."""
        data = np.random.rand(128).astype(np.float32) + 0.01
        self.f.create_dataset("ds", data=data)
        result = GPUDataset(self.f["ds"]).reduce_double_buffered(
            cupy.sum, transform=lambda x: x ** 2, chunk_size=16
        )
        np.testing.assert_allclose(
            float(result), float((data ** 2).sum()), rtol=1e-4
        )

    def test_reduce_dbl_scalar_raises(self):
        self.f.create_dataset("sc", data=np.float32(1.0))
        with pytest.raises(ValueError, match="1 dimension"):
            GPUDataset(self.f["sc"]).reduce_double_buffered(cupy.sum)


class TestGPUWrite(TestCase):
    """Tests for the GPU → HDF5 write methods."""

    def setUp(self):
        self.f = h5py.File(self.mktemp(), "w")

    def tearDown(self):
        if self.f:
            self.f.close()

    # ------------------------------------------------------------------
    # write_double_buffered
    # ------------------------------------------------------------------

    def test_write_double_buffered_1d(self):
        """Round-trip 1-D: write then read back matches original."""
        data = np.arange(256, dtype=np.float64)
        self.f.create_dataset("ds", shape=data.shape, dtype=data.dtype)
        gpu_ds = GPUDataset(self.f["ds"])
        src = cupy.asarray(data)
        gpu_ds.write_double_buffered(src, chunk_size=32)
        result = cupy.asnumpy(gpu_ds[:])
        np.testing.assert_array_equal(result, data)

    def test_write_double_buffered_2d(self):
        """Round-trip 2-D: write then read back."""
        data = np.random.rand(64, 32).astype(np.float32)
        self.f.create_dataset("ds", shape=data.shape, dtype=data.dtype)
        gpu_ds = GPUDataset(self.f["ds"])
        gpu_ds.write_double_buffered(cupy.asarray(data), chunk_size=8)
        np.testing.assert_array_almost_equal(cupy.asnumpy(gpu_ds[:]), data)

    def test_write_double_buffered_uneven_chunks(self):
        """Correct when rows are not a multiple of chunk_size."""
        data = np.arange(100, dtype=np.int32)
        self.f.create_dataset("ds", shape=data.shape, dtype=data.dtype)
        gpu_ds = GPUDataset(self.f["ds"])
        gpu_ds.write_double_buffered(cupy.asarray(data), chunk_size=30)
        np.testing.assert_array_equal(cupy.asnumpy(gpu_ds[:]), data)

    def test_write_double_buffered_single_chunk(self):
        """Works when chunk_size >= dataset size."""
        data = np.ones(20, dtype=np.float32)
        self.f.create_dataset("ds", shape=data.shape, dtype=data.dtype)
        gpu_ds = GPUDataset(self.f["ds"])
        gpu_ds.write_double_buffered(cupy.asarray(data), chunk_size=100)
        np.testing.assert_array_equal(cupy.asnumpy(gpu_ds[:]), data)

    def test_write_double_buffered_custom_stream(self):
        """Accepts a caller-provided CUDA stream."""
        data = np.arange(50, dtype=np.int32)
        self.f.create_dataset("ds", shape=data.shape, dtype=data.dtype)
        gpu_ds = GPUDataset(self.f["ds"])
        stream = cupy.cuda.Stream(non_blocking=True)
        gpu_ds.write_double_buffered(cupy.asarray(data), chunk_size=10, stream=stream)
        np.testing.assert_array_equal(cupy.asnumpy(gpu_ds[:]), data)

    def test_write_double_buffered_wrong_shape_raises(self):
        """Mismatched src shape raises ValueError."""
        self.f.create_dataset("ds", shape=(10,), dtype=np.float32)
        gpu_ds = GPUDataset(self.f["ds"])
        with pytest.raises(ValueError, match="shape"):
            gpu_ds.write_double_buffered(cupy.zeros(5, dtype=np.float32))

    def test_write_double_buffered_not_cupy_raises(self):
        """Passing a numpy array raises TypeError."""
        self.f.create_dataset("ds", shape=(10,), dtype=np.float32)
        gpu_ds = GPUDataset(self.f["ds"])
        with pytest.raises(TypeError):
            gpu_ds.write_double_buffered(np.zeros(10, dtype=np.float32))

    # ------------------------------------------------------------------
    # write_chunks_from_gpu
    # ------------------------------------------------------------------

    def test_write_chunks_2d_divisible(self):
        """Round-trip 2-D chunked (dims divisible by chunk size)."""
        data = np.arange(64 * 64, dtype=np.float32).reshape(64, 64)
        self.f.create_dataset("ds", shape=data.shape, dtype=data.dtype, chunks=(16, 16))
        gpu_ds = GPUDataset(self.f["ds"])
        gpu_ds.write_chunks_from_gpu(cupy.asarray(data))
        np.testing.assert_array_equal(cupy.asnumpy(gpu_ds.read_chunks_to_gpu()), data)

    def test_write_chunks_2d_non_divisible(self):
        """Round-trip 2-D chunked with edge tiles."""
        data = np.arange(50 * 70, dtype=np.int32).reshape(50, 70)
        self.f.create_dataset("ds", shape=data.shape, dtype=data.dtype, chunks=(16, 16))
        gpu_ds = GPUDataset(self.f["ds"])
        gpu_ds.write_chunks_from_gpu(cupy.asarray(data))
        np.testing.assert_array_equal(cupy.asnumpy(gpu_ds[:]), data)

    def test_write_chunks_3d(self):
        """Round-trip 3-D chunked dataset."""
        data = np.arange(8 * 16 * 16, dtype=np.float32).reshape(8, 16, 16)
        self.f.create_dataset("ds", shape=data.shape, dtype=data.dtype, chunks=(2, 8, 8))
        gpu_ds = GPUDataset(self.f["ds"])
        gpu_ds.write_chunks_from_gpu(cupy.asarray(data))
        np.testing.assert_array_equal(cupy.asnumpy(gpu_ds[:]), data)

    def test_write_chunks_raises_on_unchunked(self):
        self.f.create_dataset("ds", data=np.zeros((8, 8)))
        with pytest.raises(ValueError, match="not HDF5-chunked"):
            GPUDataset(self.f["ds"]).write_chunks_from_gpu(
                cupy.zeros((8, 8), dtype=np.float64)
            )

    def test_write_chunks_raises_on_1d(self):
        self.f.create_dataset("ds", shape=(32,), dtype=np.float32, chunks=(8,))
        with pytest.raises(ValueError, match="ndim"):
            GPUDataset(self.f["ds"]).write_chunks_from_gpu(
                cupy.zeros(32, dtype=np.float32)
            )

    def test_write_chunks_custom_stream(self):
        data = np.arange(16 * 16, dtype=np.int32).reshape(16, 16)
        self.f.create_dataset("ds", shape=data.shape, dtype=data.dtype, chunks=(4, 4))
        stream = cupy.cuda.Stream(non_blocking=True)
        GPUDataset(self.f["ds"]).write_chunks_from_gpu(cupy.asarray(data), stream=stream)
        np.testing.assert_array_equal(
            cupy.asnumpy(GPUDataset(self.f["ds"])[:]), data
        )

    # ------------------------------------------------------------------
    # write_selection_chunked
    # ------------------------------------------------------------------

    def test_write_sel_2d_interior(self):
        """Write a selection crossing chunk boundaries."""
        data = np.zeros((64, 64), dtype=np.float32)
        self.f.create_dataset("ds", data=data, chunks=(16, 16))
        gpu_ds = GPUDataset(self.f["ds"])
        patch = np.arange(80 * 60, dtype=np.float32).reshape(80, 60) + 1.0
        sel = np.s_[0:80, 0:60]  # only fits in a smaller dataset — use 64x64
        patch = np.arange(40 * 50, dtype=np.float32).reshape(40, 50) + 1.0
        sel = np.s_[10:50, 5:55]
        gpu_ds.write_selection_chunked(cupy.asarray(patch), sel)
        result = cupy.asnumpy(gpu_ds[:])
        expected = data.copy()
        expected[sel] = patch
        np.testing.assert_array_equal(result, expected)

    def test_write_sel_2d_aligned(self):
        """Write a chunk-aligned selection."""
        data = np.zeros((64, 64), dtype=np.float32)
        self.f.create_dataset("ds", data=data, chunks=(16, 16))
        gpu_ds = GPUDataset(self.f["ds"])
        sel = np.s_[16:48, 32:64]
        patch = np.random.rand(32, 32).astype(np.float32)
        gpu_ds.write_selection_chunked(cupy.asarray(patch), sel)
        result = cupy.asnumpy(gpu_ds[:])
        expected = data.copy()
        expected[sel] = patch
        np.testing.assert_array_almost_equal(result, expected)

    def test_write_sel_3d(self):
        """3-D partial write, crossing chunk boundaries."""
        data = np.zeros((16, 24, 32), dtype=np.float32)
        self.f.create_dataset("ds", data=data, chunks=(4, 8, 8))
        gpu_ds = GPUDataset(self.f["ds"])
        sel = np.s_[2:12, 4:20, 8:28]
        patch = np.random.rand(10, 16, 20).astype(np.float32)
        gpu_ds.write_selection_chunked(cupy.asarray(patch), sel)
        result = cupy.asnumpy(gpu_ds[:])
        expected = data.copy()
        expected[sel] = patch
        np.testing.assert_array_almost_equal(result, expected)

    def test_write_sel_raises_on_unchunked(self):
        self.f.create_dataset("ds", data=np.zeros((16, 16)))
        with pytest.raises(ValueError, match="not HDF5-chunked"):
            GPUDataset(self.f["ds"]).write_selection_chunked(
                cupy.zeros((8, 8), dtype=np.float64), np.s_[4:12, 4:12]
            )

    def test_write_sel_raises_wrong_src_shape(self):
        data = np.zeros((32, 32), dtype=np.float32)
        self.f.create_dataset("ds", data=data, chunks=(8, 8))
        gpu_ds = GPUDataset(self.f["ds"])
        with pytest.raises(ValueError, match="shape"):
            gpu_ds.write_selection_chunked(
                cupy.zeros((5, 5), dtype=np.float32), np.s_[0:10, 0:10]
            )

    # ------------------------------------------------------------------
    # __setitem__ dispatch
    # ------------------------------------------------------------------

    def test_setitem_dispatches_for_chunked_2d(self):
        """__setitem__ with a CuPy array on a chunked 2-D dataset dispatches
        to write_selection_chunked."""
        data = np.zeros((64, 64), dtype=np.float32)
        self.f.create_dataset("ds", data=data, chunks=(16, 16))
        gpu_ds = GPUDataset(self.f["ds"])
        patch = np.random.rand(20, 30).astype(np.float32)
        gpu_ds[10:30, 5:35] = cupy.asarray(patch)
        result = cupy.asnumpy(gpu_ds[:])
        expected = data.copy()
        expected[10:30, 5:35] = patch
        np.testing.assert_array_almost_equal(result, expected)

    def test_setitem_fallback_numpy_src(self):
        """__setitem__ with a numpy src falls back to h5py (no dispatch)."""
        data = np.zeros(20, dtype=np.int32)
        self.f.create_dataset("ds", data=data)
        gpu_ds = GPUDataset(self.f["ds"])
        patch = np.arange(5, dtype=np.int32) + 10
        gpu_ds[3:8] = patch
        result = cupy.asnumpy(gpu_ds[:])
        expected = data.copy()
        expected[3:8] = patch
        np.testing.assert_array_equal(result, expected)

    def test_setitem_fallback_cupy_unchunked(self):
        """__setitem__ with CuPy on a contiguous dataset converts to numpy."""
        data = np.zeros(20, dtype=np.float32)
        self.f.create_dataset("ds", data=data)
        gpu_ds = GPUDataset(self.f["ds"])
        patch = cupy.ones(20, dtype=np.float32)
        gpu_ds[:] = patch
        np.testing.assert_array_equal(cupy.asnumpy(gpu_ds[:]), np.ones(20, dtype=np.float32))


class TestGPUFile(TestCase):
    def test_open_and_read(self):
        """GPUFile opens a file and reads a dataset to the GPU."""
        path = self.mktemp()
        data = np.linspace(0, 1, 50, dtype=np.float64)
        with h5py.File(path, "w") as f:
            f.create_dataset("ds", data=data)

        with GPUFile(path) as f:
            result = f["ds"][:]

        assert isinstance(result, cupy.ndarray)
        np.testing.assert_array_almost_equal(cupy.asnumpy(result), data)

    def test_repr(self):
        path = self.mktemp()
        with h5py.File(path, "w") as f:
            pass
        with GPUFile(path) as f:
            assert "GPUFile" in repr(f)


# ---------------------------------------------------------------------------
# GPUCachedDataset
# ---------------------------------------------------------------------------

from h5py.gpu import GPUCachedDataset  # noqa: E402


class TestGPUCachedDataset(TestCase):
    def setUp(self):
        self.f = h5py.File(self.mktemp(), "w")

    def tearDown(self):
        if self.f:
            self.f.close()

    # ------------------------------------------------------------------
    # Preload behaviour
    # ------------------------------------------------------------------

    def test_eager_preload_2d_chunked(self):
        """Eager preload (default) fills _array via read_chunks_to_gpu."""
        data = np.arange(64 * 64, dtype=np.float32).reshape(64, 64)
        self.f.create_dataset("ds", data=data, chunks=(16, 16))
        cached = GPUCachedDataset(self.f["ds"])
        assert cached._array is not None
        np.testing.assert_array_equal(cupy.asnumpy(cached._array), data)

    def test_eager_preload_1d(self):
        """1-D unchunked dataset is loaded via read_double_buffered."""
        data = np.arange(256, dtype=np.float64)
        self.f.create_dataset("ds", data=data)
        cached = GPUCachedDataset(self.f["ds"])
        assert cached._array is not None
        np.testing.assert_array_almost_equal(cupy.asnumpy(cached._array), data)

    def test_lazy_preload(self):
        """preload=False defers loading until first access."""
        data = np.arange(32, dtype=np.float32)
        self.f.create_dataset("ds", data=data)
        cached = GPUCachedDataset(self.f["ds"], preload=False)
        assert cached._array is None
        # Trigger load via .array property
        arr = cached.array
        assert cached._array is not None
        np.testing.assert_array_almost_equal(cupy.asnumpy(arr), data)

    def test_preload_noop_if_already_loaded(self):
        """Calling preload() again does not replace the existing array."""
        data = np.ones(32, dtype=np.float32)
        self.f.create_dataset("ds", data=data)
        cached = GPUCachedDataset(self.f["ds"])
        first = cached._array
        cached.preload()   # second call
        assert cached._array is first  # same object, no reload

    # ------------------------------------------------------------------
    # __getitem__
    # ------------------------------------------------------------------

    def test_getitem_slice(self):
        """Indexing into the cache returns the correct sub-array."""
        data = np.arange(64, dtype=np.float32)
        self.f.create_dataset("ds", data=data)
        cached = GPUCachedDataset(self.f["ds"])
        result = cached[10:30]
        assert isinstance(result, cupy.ndarray)
        np.testing.assert_array_equal(cupy.asnumpy(result), data[10:30])

    def test_getitem_2d(self):
        """2-D slice on a 2-D dataset."""
        data = np.arange(8 * 8, dtype=np.float32).reshape(8, 8)
        self.f.create_dataset("ds", data=data, chunks=(4, 4))
        cached = GPUCachedDataset(self.f["ds"])
        result = cached[2:5, 1:6]
        np.testing.assert_array_equal(cupy.asnumpy(result), data[2:5, 1:6])

    # ------------------------------------------------------------------
    # reduce
    # ------------------------------------------------------------------

    def test_reduce_sum(self):
        """reduce(cp.sum) returns the correct total."""
        data = np.arange(1, 65, dtype=np.float64)
        self.f.create_dataset("ds", data=data)
        cached = GPUCachedDataset(self.f["ds"])
        result = float(cached.reduce(cupy.sum))
        assert abs(result - float(np.sum(data))) < 1e-6

    def test_reduce_max(self):
        data = np.arange(100, dtype=np.float32)
        self.f.create_dataset("ds", data=data)
        cached = GPUCachedDataset(self.f["ds"])
        assert float(cached.reduce(cupy.max)) == pytest.approx(99.0)

    def test_reduce_min(self):
        data = np.arange(100, dtype=np.float32) + 5.0
        self.f.create_dataset("ds", data=data)
        cached = GPUCachedDataset(self.f["ds"])
        assert float(cached.reduce(cupy.min)) == pytest.approx(5.0)

    def test_reduce_with_transform(self):
        """transform kwarg is applied before reduce, cache is not mutated."""
        data = np.array([1.0, 4.0, 9.0, 16.0], dtype=np.float64)
        self.f.create_dataset("ds", data=data)
        cached = GPUCachedDataset(self.f["ds"])
        result = float(cached.reduce(cupy.sum, transform=cupy.sqrt))
        expected = float(np.sum(np.sqrt(data)))
        assert abs(result - expected) < 1e-6
        # Cache should still contain original data (not sqrt'd)
        np.testing.assert_array_almost_equal(cupy.asnumpy(cached.array), data)

    # ------------------------------------------------------------------
    # transform (in-place cache update)
    # ------------------------------------------------------------------

    def test_transform_updates_cache(self):
        """transform() replaces the cached array with the transformed version."""
        data = np.array([1.0, 4.0, 9.0], dtype=np.float64)
        self.f.create_dataset("ds", data=data)
        cached = GPUCachedDataset(self.f["ds"])
        cached.transform(cupy.sqrt)
        np.testing.assert_array_almost_equal(
            cupy.asnumpy(cached.array), np.sqrt(data)
        )

    def test_transform_chaining(self):
        """transform() returns self, enabling chaining."""
        data = np.array([1.0, 4.0, 9.0], dtype=np.float64)
        self.f.create_dataset("ds", data=data)
        cached = GPUCachedDataset(self.f["ds"])
        result = cached.transform(cupy.sqrt).transform(lambda x: x * 2.0)
        assert result is cached
        np.testing.assert_array_almost_equal(
            cupy.asnumpy(cached.array), np.sqrt(data) * 2.0
        )

    def test_transform_then_reduce(self):
        """Chained transform + reduce computes the correct value."""
        data = np.array([1.0, 4.0, 9.0, 16.0], dtype=np.float64)
        self.f.create_dataset("ds", data=data)
        cached = GPUCachedDataset(self.f["ds"])
        result = float(cached.transform(cupy.sqrt).reduce(cupy.sum))
        expected = float(np.sum(np.sqrt(data)))
        assert abs(result - expected) < 1e-6

    # ------------------------------------------------------------------
    # free / reload
    # ------------------------------------------------------------------

    def test_free_releases_cache(self):
        data = np.arange(16, dtype=np.float32)
        self.f.create_dataset("ds", data=data)
        cached = GPUCachedDataset(self.f["ds"])
        assert cached._array is not None
        cached.free()
        assert cached._array is None

    def test_reload_refreshes_data(self):
        """reload() frees and re-reads; result matches original data."""
        data = np.arange(16, dtype=np.float32)
        self.f.create_dataset("ds", data=data)
        cached = GPUCachedDataset(self.f["ds"])
        first_ptr = cached._array.data.ptr
        returned = cached.reload()
        assert returned is cached
        assert cached._array is not None
        # May or may not be the same allocation; data must be correct
        np.testing.assert_array_almost_equal(cupy.asnumpy(cached.array), data)

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def test_context_manager_frees(self):
        """Exiting the context manager frees the cached array."""
        data = np.arange(32, dtype=np.float32)
        self.f.create_dataset("ds", data=data)
        with GPUCachedDataset(self.f["ds"]) as cached:
            assert cached._array is not None
        assert cached._array is None

    # ------------------------------------------------------------------
    # __repr__
    # ------------------------------------------------------------------

    def test_repr_loaded(self):
        data = np.zeros((4, 4), dtype=np.float32)
        self.f.create_dataset("ds", data=data, chunks=(2, 2))
        cached = GPUCachedDataset(self.f["ds"])
        r = repr(cached)
        assert "GPUCachedDataset" in r
        assert "loaded" in r
        assert "4" in r   # shape present

    def test_repr_not_loaded(self):
        data = np.zeros(8, dtype=np.float32)
        self.f.create_dataset("ds", data=data)
        cached = GPUCachedDataset(self.f["ds"], preload=False)
        r = repr(cached)
        assert "not loaded" in r

    # ------------------------------------------------------------------
    # Error handling
    # ------------------------------------------------------------------

    def test_wrong_type_raises(self):
        with pytest.raises(TypeError, match="GPUDataset or h5py.Dataset"):
            GPUCachedDataset("not a dataset")

    def test_wraps_gpu_dataset(self):
        """Accepts a GPUDataset directly (not just h5py.Dataset)."""
        data = np.arange(16, dtype=np.float32)
        self.f.create_dataset("ds", data=data)
        gpu_ds = GPUDataset(self.f["ds"])
        cached = GPUCachedDataset(gpu_ds)
        np.testing.assert_array_almost_equal(cupy.asnumpy(cached.array), data)

    def test_attribute_delegation(self):
        """Attributes not on GPUCachedDataset are delegated to the GPUDataset."""
        data = np.arange(16, dtype=np.float32)
        self.f.create_dataset("ds", data=data)
        cached = GPUCachedDataset(self.f["ds"])
        # .shape and .dtype come from the underlying h5py.Dataset via GPUDataset
        assert cached.shape == (16,)
        assert cached.dtype == np.float32


# ---------------------------------------------------------------------------
# TestReadChunksCompressed
# ---------------------------------------------------------------------------

class TestReadChunksCompressed(TestCase):
    """Tests for GPUDataset.read_chunks_compressed.

    Each test creates an HDF5 dataset with a real HDF5 compression filter,
    then reads it back via read_chunks_compressed and verifies the result
    matches the original data.

    CPU-decompressor tests (deflate, lz4+lz4lib, zstd+zstdlib) require only
    the corresponding Python library plus hdf5plugin for the write side.

    GPU-decompressor tests (lz4+nvcomp, zstd+nvcomp) additionally require
    nvidia-nvcomp.
    """

    def setUp(self):
        self.f = h5py.File(self.mktemp(), "w")

    def tearDown(self):
        if self.f:
            self.f.close()

    # ------------------------------------------------------------------
    # Fallback: uncompressed dataset uses read_chunks_to_gpu
    # ------------------------------------------------------------------

    def test_uncompressed_fallback(self):
        """read_chunks_compressed on an uncompressed dataset falls back
        to read_chunks_to_gpu and returns the correct result."""
        data = np.arange(64 * 64, dtype=np.float32).reshape(64, 64)
        self.f.create_dataset("ds", data=data, chunks=(16, 16))
        result = GPUDataset(self.f["ds"]).read_chunks_compressed()
        assert isinstance(result, cupy.ndarray)
        np.testing.assert_array_equal(cupy.asnumpy(result), data)

    def test_uncompressed_fallback_3d(self):
        """3-D uncompressed dataset also falls back correctly."""
        data = np.arange(8 * 16 * 16, dtype=np.float32).reshape(8, 16, 16)
        self.f.create_dataset("ds", data=data, chunks=(2, 8, 8))
        result = GPUDataset(self.f["ds"]).read_chunks_compressed()
        np.testing.assert_array_equal(cupy.asnumpy(result), data)

    def test_raises_on_unchunked(self):
        """Contiguous (non-chunked) dataset raises ValueError."""
        self.f.create_dataset("ds", data=np.zeros((8, 8)))
        with pytest.raises(ValueError, match="not HDF5-chunked"):
            GPUDataset(self.f["ds"]).read_chunks_compressed()

    def test_raises_on_1d(self):
        """1-D chunked dataset raises ValueError (only 2-D and 3-D supported)."""
        self.f.create_dataset("ds", data=np.zeros(32), chunks=(8,))
        with pytest.raises(ValueError, match="ndim"):
            GPUDataset(self.f["ds"]).read_chunks_compressed()

    # ------------------------------------------------------------------
    # deflate / gzip  (always CPU path — nvCOMP deflate is incompatible)
    # ------------------------------------------------------------------

    def test_deflate_2d(self):
        """2-D dataset compressed with gzip/deflate round-trips correctly."""
        data = np.arange(64 * 64, dtype=np.float32).reshape(64, 64)
        self.f.create_dataset("ds", data=data, chunks=(16, 16), compression="gzip",
                              compression_opts=4)
        result = GPUDataset(self.f["ds"]).read_chunks_compressed()
        np.testing.assert_array_equal(cupy.asnumpy(result), data)

    def test_deflate_3d(self):
        """3-D gzip-compressed dataset round-trips correctly."""
        data = np.random.rand(8, 16, 16).astype(np.float32)
        self.f.create_dataset("ds", data=data, chunks=(2, 8, 8), compression="gzip")
        result = GPUDataset(self.f["ds"]).read_chunks_compressed()
        np.testing.assert_array_almost_equal(cupy.asnumpy(result), data)

    def test_deflate_with_shuffle(self):
        """Shuffle + gzip: the GPU unshuffle kernel is applied after decompression."""
        data = np.arange(64 * 64, dtype=np.float64).reshape(64, 64)
        self.f.create_dataset("ds", data=data, chunks=(16, 16),
                              compression="gzip", shuffle=True)
        result = GPUDataset(self.f["ds"]).read_chunks_compressed()
        np.testing.assert_array_equal(cupy.asnumpy(result), data)

    def test_deflate_non_divisible(self):
        """Edge tiles (dataset dims not multiples of chunk size) are handled."""
        data = np.arange(50 * 70, dtype=np.int32).reshape(50, 70)
        self.f.create_dataset("ds", data=data, chunks=(16, 16), compression="gzip")
        result = GPUDataset(self.f["ds"]).read_chunks_compressed()
        np.testing.assert_array_equal(cupy.asnumpy(result), data)

    def test_deflate_preallocated_out(self):
        """Pre-allocated output array is filled correctly."""
        data = np.arange(32 * 32, dtype=np.float32).reshape(32, 32)
        self.f.create_dataset("ds", data=data, chunks=(8, 8), compression="gzip")
        gpu_ds = GPUDataset(self.f["ds"])
        out = cupy.empty((32, 32), dtype=np.float32)
        result = gpu_ds.read_chunks_compressed(out=out)
        assert result is out
        np.testing.assert_array_equal(cupy.asnumpy(out), data)

    def test_deflate_custom_stream(self):
        """Caller-supplied CUDA stream is accepted."""
        data = np.arange(32 * 32, dtype=np.int32).reshape(32, 32)
        self.f.create_dataset("ds", data=data, chunks=(8, 8), compression="gzip")
        stream = cupy.cuda.Stream(non_blocking=True)
        result = GPUDataset(self.f["ds"]).read_chunks_compressed(stream=stream)
        np.testing.assert_array_equal(cupy.asnumpy(result), data)

    def test_deflate_transform(self):
        """transform is applied per-tile on the GPU after decompression."""
        data = np.arange(1, 64 * 64 + 1, dtype=np.float32).reshape(64, 64)
        self.f.create_dataset("ds", data=data, chunks=(16, 16), compression="gzip")
        result = GPUDataset(self.f["ds"]).read_chunks_compressed(
            transform=lambda x: x * 2.0
        )
        np.testing.assert_array_almost_equal(cupy.asnumpy(result), data * 2.0)

    def test_deflate_matches_getitem(self):
        """read_chunks_compressed and gpu_ds[:] return the same data."""
        data = np.random.rand(64, 32).astype(np.float64)
        self.f.create_dataset("ds", data=data, chunks=(16, 16), compression="gzip")
        gpu_ds = GPUDataset(self.f["ds"])
        via_compressed = gpu_ds.read_chunks_compressed()
        via_getitem    = gpu_ds[:]
        np.testing.assert_array_almost_equal(
            cupy.asnumpy(via_compressed), cupy.asnumpy(via_getitem)
        )

    # ------------------------------------------------------------------
    # LZ4  (CPU path via `lz4` library)
    # ------------------------------------------------------------------

    @requires_lz4
    def test_lz4_cpu_2d(self):
        """2-D LZ4-compressed dataset decoded on CPU."""
        data = np.arange(64 * 64, dtype=np.float32).reshape(64, 64)
        self.f.create_dataset("ds", data=data, chunks=(16, 16),
                              **_hdf5plugin.LZ4())
        result = GPUDataset(self.f["ds"]).read_chunks_compressed()
        np.testing.assert_array_equal(cupy.asnumpy(result), data)

    @requires_lz4
    def test_lz4_cpu_with_shuffle(self):
        """Shuffle + LZ4 (CPU decompress + GPU unshuffle)."""
        data = np.arange(64 * 64, dtype=np.float32).reshape(64, 64)
        self.f.create_dataset("ds", data=data, chunks=(16, 16),
                              shuffle=True, **_hdf5plugin.LZ4())
        result = GPUDataset(self.f["ds"]).read_chunks_compressed()
        np.testing.assert_array_equal(cupy.asnumpy(result), data)

    @requires_lz4
    def test_lz4_cpu_3d(self):
        """3-D LZ4 dataset decoded on CPU."""
        data = np.random.rand(8, 16, 16).astype(np.float32)
        self.f.create_dataset("ds", data=data, chunks=(2, 8, 8),
                              **_hdf5plugin.LZ4())
        result = GPUDataset(self.f["ds"]).read_chunks_compressed()
        np.testing.assert_array_almost_equal(cupy.asnumpy(result), data)

    @requires_lz4
    def test_lz4_cpu_non_divisible(self):
        """Edge tiles with LZ4 CPU path."""
        data = np.arange(50 * 70, dtype=np.int32).reshape(50, 70)
        self.f.create_dataset("ds", data=data, chunks=(16, 16),
                              **_hdf5plugin.LZ4())
        result = GPUDataset(self.f["ds"]).read_chunks_compressed()
        np.testing.assert_array_equal(cupy.asnumpy(result), data)

    @requires_lz4
    def test_lz4_cpu_transform(self):
        """transform applied after LZ4 CPU decompression."""
        data = np.arange(1, 64 * 64 + 1, dtype=np.float32).reshape(64, 64)
        self.f.create_dataset("ds", data=data, chunks=(16, 16),
                              **_hdf5plugin.LZ4())
        result = GPUDataset(self.f["ds"]).read_chunks_compressed(
            transform=cupy.sqrt
        )
        np.testing.assert_array_almost_equal(cupy.asnumpy(result), np.sqrt(data))

    # ------------------------------------------------------------------
    # Zstd  (CPU path via `zstd` library)
    # ------------------------------------------------------------------

    @requires_zstd
    def test_zstd_cpu_2d(self):
        """2-D Zstd-compressed dataset decoded on CPU."""
        data = np.arange(64 * 64, dtype=np.float32).reshape(64, 64)
        self.f.create_dataset("ds", data=data, chunks=(16, 16),
                              **_hdf5plugin.Zstd())
        result = GPUDataset(self.f["ds"]).read_chunks_compressed()
        np.testing.assert_array_equal(cupy.asnumpy(result), data)

    @requires_zstd
    def test_zstd_cpu_with_shuffle(self):
        """Shuffle + Zstd (CPU decompress + GPU unshuffle)."""
        data = np.arange(64 * 64, dtype=np.float64).reshape(64, 64)
        self.f.create_dataset("ds", data=data, chunks=(16, 16),
                              shuffle=True, **_hdf5plugin.Zstd())
        result = GPUDataset(self.f["ds"]).read_chunks_compressed()
        np.testing.assert_array_equal(cupy.asnumpy(result), data)

    @requires_zstd
    def test_zstd_cpu_3d(self):
        """3-D Zstd dataset decoded on CPU."""
        data = np.random.rand(8, 16, 16).astype(np.float32)
        self.f.create_dataset("ds", data=data, chunks=(2, 8, 8),
                              **_hdf5plugin.Zstd())
        result = GPUDataset(self.f["ds"]).read_chunks_compressed()
        np.testing.assert_array_almost_equal(cupy.asnumpy(result), data)

    @requires_zstd
    def test_zstd_cpu_transform(self):
        """transform applied after Zstd CPU decompression."""
        data = np.arange(1, 64 * 64 + 1, dtype=np.float32).reshape(64, 64)
        self.f.create_dataset("ds", data=data, chunks=(16, 16),
                              **_hdf5plugin.Zstd())
        result = GPUDataset(self.f["ds"]).read_chunks_compressed(
            transform=lambda x: x + 1.0
        )
        np.testing.assert_array_almost_equal(cupy.asnumpy(result), data + 1.0)

    # ------------------------------------------------------------------
    # LZ4  (GPU path via nvCOMP)
    # ------------------------------------------------------------------

    @requires_nvcomp_lz4
    def test_lz4_nvcomp_2d(self):
        """2-D LZ4 dataset decoded on GPU via nvCOMP."""
        data = np.arange(64 * 64, dtype=np.float32).reshape(64, 64)
        self.f.create_dataset("ds", data=data, chunks=(16, 16),
                              **_hdf5plugin.LZ4())
        result = GPUDataset(self.f["ds"]).read_chunks_compressed()
        assert isinstance(result, cupy.ndarray)
        np.testing.assert_array_equal(cupy.asnumpy(result), data)

    @requires_nvcomp_lz4
    def test_lz4_nvcomp_with_shuffle(self):
        """Shuffle + LZ4 with nvCOMP GPU decompression + GPU unshuffle."""
        data = np.arange(64 * 64, dtype=np.float32).reshape(64, 64)
        self.f.create_dataset("ds", data=data, chunks=(16, 16),
                              shuffle=True, **_hdf5plugin.LZ4())
        result = GPUDataset(self.f["ds"]).read_chunks_compressed()
        np.testing.assert_array_equal(cupy.asnumpy(result), data)

    @requires_nvcomp_lz4
    def test_lz4_nvcomp_3d(self):
        """3-D LZ4 dataset decoded on GPU."""
        data = np.random.rand(8, 16, 16).astype(np.float32)
        self.f.create_dataset("ds", data=data, chunks=(2, 8, 8),
                              **_hdf5plugin.LZ4())
        result = GPUDataset(self.f["ds"]).read_chunks_compressed()
        np.testing.assert_array_almost_equal(cupy.asnumpy(result), data)

    @requires_nvcomp_lz4
    def test_lz4_nvcomp_non_divisible(self):
        """Edge tiles decoded via nvCOMP."""
        data = np.arange(50 * 70, dtype=np.int32).reshape(50, 70)
        self.f.create_dataset("ds", data=data, chunks=(16, 16),
                              **_hdf5plugin.LZ4())
        result = GPUDataset(self.f["ds"]).read_chunks_compressed()
        np.testing.assert_array_equal(cupy.asnumpy(result), data)

    @requires_nvcomp_lz4
    def test_lz4_nvcomp_transform(self):
        """transform applied after nvCOMP GPU decompression."""
        data = np.arange(1, 64 * 64 + 1, dtype=np.float32).reshape(64, 64)
        self.f.create_dataset("ds", data=data, chunks=(16, 16),
                              **_hdf5plugin.LZ4())
        result = GPUDataset(self.f["ds"]).read_chunks_compressed(
            transform=cupy.sqrt
        )
        np.testing.assert_array_almost_equal(cupy.asnumpy(result), np.sqrt(data))

    @requires_nvcomp_lz4
    def test_lz4_nvcomp_preallocated_out(self):
        """Pre-allocated output array filled correctly via nvCOMP path."""
        data = np.arange(32 * 32, dtype=np.float32).reshape(32, 32)
        self.f.create_dataset("ds", data=data, chunks=(8, 8),
                              **_hdf5plugin.LZ4())
        gpu_ds = GPUDataset(self.f["ds"])
        out = cupy.empty((32, 32), dtype=np.float32)
        result = gpu_ds.read_chunks_compressed(out=out)
        assert result is out
        np.testing.assert_array_equal(cupy.asnumpy(out), data)

    @requires_nvcomp_lz4
    def test_lz4_nvcomp_matches_getitem(self):
        """nvCOMP LZ4 result matches standard h5py read."""
        data = np.random.rand(64, 32).astype(np.float64)
        self.f.create_dataset("ds", data=data, chunks=(16, 16),
                              **_hdf5plugin.LZ4())
        gpu_ds = GPUDataset(self.f["ds"])
        via_nvcomp  = gpu_ds.read_chunks_compressed()
        via_getitem = gpu_ds[:]
        np.testing.assert_array_almost_equal(
            cupy.asnumpy(via_nvcomp), cupy.asnumpy(via_getitem)
        )

    # ------------------------------------------------------------------
    # Zstd  (GPU path via nvCOMP)
    # ------------------------------------------------------------------

    @requires_nvcomp_zstd
    def test_zstd_nvcomp_2d(self):
        """2-D Zstd dataset decoded on GPU via nvCOMP."""
        data = np.arange(64 * 64, dtype=np.float32).reshape(64, 64)
        self.f.create_dataset("ds", data=data, chunks=(16, 16),
                              **_hdf5plugin.Zstd())
        result = GPUDataset(self.f["ds"]).read_chunks_compressed()
        assert isinstance(result, cupy.ndarray)
        np.testing.assert_array_equal(cupy.asnumpy(result), data)

    @requires_nvcomp_zstd
    def test_zstd_nvcomp_with_shuffle(self):
        """Shuffle + Zstd with nvCOMP GPU decompression + GPU unshuffle."""
        data = np.arange(64 * 64, dtype=np.float64).reshape(64, 64)
        self.f.create_dataset("ds", data=data, chunks=(16, 16),
                              shuffle=True, **_hdf5plugin.Zstd())
        result = GPUDataset(self.f["ds"]).read_chunks_compressed()
        np.testing.assert_array_equal(cupy.asnumpy(result), data)

    @requires_nvcomp_zstd
    def test_zstd_nvcomp_3d(self):
        """3-D Zstd dataset decoded on GPU."""
        data = np.random.rand(8, 16, 16).astype(np.float32)
        self.f.create_dataset("ds", data=data, chunks=(2, 8, 8),
                              **_hdf5plugin.Zstd())
        result = GPUDataset(self.f["ds"]).read_chunks_compressed()
        np.testing.assert_array_almost_equal(cupy.asnumpy(result), data)

    @requires_nvcomp_zstd
    def test_zstd_nvcomp_transform(self):
        """transform applied after nvCOMP Zstd decompression."""
        data = np.arange(1, 64 * 64 + 1, dtype=np.float32).reshape(64, 64)
        self.f.create_dataset("ds", data=data, chunks=(16, 16),
                              **_hdf5plugin.Zstd())
        result = GPUDataset(self.f["ds"]).read_chunks_compressed(
            transform=lambda x: x * 2.0
        )
        np.testing.assert_array_almost_equal(cupy.asnumpy(result), data * 2.0)

    @requires_nvcomp_zstd
    def test_zstd_nvcomp_matches_getitem(self):
        """nvCOMP Zstd result matches standard h5py read."""
        data = np.random.rand(64, 32).astype(np.float64)
        self.f.create_dataset("ds", data=data, chunks=(16, 16),
                              **_hdf5plugin.Zstd())
        gpu_ds = GPUDataset(self.f["ds"])
        via_nvcomp  = gpu_ds.read_chunks_compressed()
        via_getitem = gpu_ds[:]
        np.testing.assert_array_almost_equal(
            cupy.asnumpy(via_nvcomp), cupy.asnumpy(via_getitem)
        )
