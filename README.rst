.. image:: https://github.com/h5py/h5py/actions/workflows/build_wheels.yml/badge.svg?branch=master
   :target: https://github.com/h5py/h5py/actions/workflows/build_wheels.yml
.. image:: https://ci.appveyor.com/api/projects/status/h3iajp4d1myotprc/branch/master?svg=true
   :target: https://ci.appveyor.com/project/h5py/h5py/branch/master
.. image:: https://dev.azure.com/h5pyappveyor/h5py/_apis/build/status/h5py.h5py?branchName=master
   :target: https://dev.azure.com/h5pyappveyor/h5py/_build/latest?definitionId=1&branchName=master

HDF5 for Python
===============
`h5py` is a thin, pythonic wrapper around `HDF5 <https://www.hdfgroup.org/solutions/hdf5/>`_,
which runs on Python 3 (3.10+).

.. note::

   **This fork extends the upstream h5py library with GPU support via CuPy.**
   See the `GPU Support`_ section below for details.

Websites
--------

* Main website: https://www.h5py.org
* Source code: https://github.com/h5py/h5py
* Discussion forum: https://forum.hdfgroup.org/c/hdf5/h5py

Installation
------------

Pre-built `h5py` can either be installed via your Python Distribution (e.g.
`Continuum Anaconda`_, `Enthought Canopy`_) or from `PyPI`_ via `pip`_.
`h5py` is also distributed in many Linux Distributions (e.g. Ubuntu, Fedora),
and in the macOS package managers `Homebrew <https://brew.sh/>`_,
`Macports <https://www.macports.org/>`_, or `Fink <http://finkproject.org/>`_.

More detailed installation instructions, including how to install `h5py` with
MPI support, can be found at: https://docs.h5py.org/en/latest/build.html.


Reporting bugs
--------------

Open a bug at https://github.com/h5py/h5py/issues.  For general questions, ask
on the HDF forum (https://forum.hdfgroup.org/c/hdf5/h5py).

.. _`Continuum Anaconda`: http://continuum.io/downloads
.. _`Enthought Canopy`: https://www.enthought.com/products/canopy/
.. _`PyPI`: https://pypi.org/project/h5py/
.. _`pip`: https://pip.pypa.io/en/stable/


GPU Support
-----------

This fork adds ``h5py.gpu``, a GPU acceleration layer built on top of
`CuPy <https://cupy.dev/>`_.  It provides a zero-copy pipeline from HDF5 files
directly into GPU device memory using CUDA pinned (page-locked) buffers and
asynchronous DMA transfers.

**Requirements:** CuPy and a CUDA-capable GPU.  All GPU classes are in
``h5py.gpu`` and are completely optional — the rest of h5py is unaffected.

Quick start
~~~~~~~~~~~

.. code-block:: python

    import h5py
    from h5py.gpu import GPUFile, GPUDataset, GPUCachedDataset
    import cupy as cp

    # Drop-in replacement for h5py.File — datasets are read to the GPU
    with GPUFile("data.h5") as f:
        arr = f["dataset"][:]        # cupy.ndarray, data lives on the GPU

    # Wrap an already-open dataset
    with h5py.File("data.h5") as f:
        gpu_ds = GPUDataset(f["dataset"])
        arr = gpu_ds[:]

Reading
~~~~~~~

``GPUDataset`` provides several read paths depending on dataset layout and
access pattern:

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Method
     - Description
   * - ``gpu_ds[...]``
     - Simple read via ``__getitem__``. Good for small datasets or
       partial reads.
   * - ``read_double_buffered(chunk_size, transform)``
     - Full-dataset read using two alternating pinned buffers.
       GPU DMA and CPU HDF5 reads run concurrently.  Works for any
       dimensionality.
   * - ``read_chunks_to_gpu(transform)``
     - Tile-by-tile read for 2-D/3-D chunked datasets.
       Uses ``memcpy2DAsync`` to place tiles directly at their strided
       position in the output array — no gather copy.
   * - ``read_chunks_parallel(n_streams, transform)``
     - Same as above but distributes tiles across *N* independent CUDA
       streams for additional H2D concurrency.
   * - ``read_selection_chunked(sel, transform)``
     - Reads only the HDF5 chunks that overlap a 2-D rectangular
       selection.  Reports ``waste %`` (bytes discarded after crop).

All read methods accept an optional ``transform`` callable.  The transform is
enqueued on the CUDA stream immediately after each H2D transfer, so compute
overlaps with the CPU reading the next chunk::

    # sqrt applied on-stream while the next band is being read from disk
    arr = gpu_ds.read_double_buffered(transform=cp.sqrt)

Reductions
~~~~~~~~~~

Streaming reductions keep GPU memory at *O(chunk size)*, regardless of dataset
size, by reducing each chunk to a scalar and combining at the end:

.. code-block:: python

    # Sum of a 2-D chunked dataset — only one chunk in GPU memory at a time
    total = gpu_ds.reduce_chunks(cp.sum)

    # Global mean (non-uniform chunks require a custom combine_fn)
    n = int(np.prod(ds.shape))
    mean = gpu_ds.reduce_chunks(cp.sum, combine_fn=lambda x: cp.sum(x) / n)

    # 1-D / contiguous datasets use the double-buffered path
    total = gpu_ds.reduce_double_buffered(cp.sum)

Both methods accept a ``transform`` applied per-chunk before the reduction.

Writing
~~~~~~~

GPU arrays can be written back to HDF5 with the same pipelined approach:

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Method
     - Description
   * - ``gpu_ds[:] = cupy_array``
     - Dispatch via ``__setitem__``.  Automatically selects the best
       write path.
   * - ``write_double_buffered(src)``
     - Full-dataset D2H write with double-buffering.  GPU DMA and
       HDF5 writes overlap.
   * - ``write_chunks_from_gpu(src)``
     - Tile-by-tile write for 2-D/3-D chunked datasets.
   * - ``write_selection_chunked(src, sel)``
     - Partial write into a selection.  Performs read-modify-write on
       each overlapping HDF5 chunk.

Keeping data on the GPU — ``GPUCachedDataset``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When the same dataset is accessed multiple times (multiple transforms,
reductions, or index lookups), ``GPUCachedDataset`` loads the data once and
keeps it resident in GPU memory:

.. code-block:: python

    from h5py.gpu import GPUCachedDataset

    # Eager load (default) — data is on GPU immediately
    with GPUCachedDataset(f["dataset"]) as cached:
        total   = cached.reduce(cp.sum)
        maximum = cached.reduce(cp.max)
        subset  = cached[10:50, 5:55]   # pure GPU slice, no I/O

    # Lazy load — deferred until first access
    cached = GPUCachedDataset(f["dataset"], preload=False)

    # Chained transform + reduce
    result = cached.transform(cp.sqrt).reduce(cp.sum)
    cached.free()

``GPUCachedDataset`` selects the most efficient read path automatically
(``read_chunks_to_gpu`` for 2-D/3-D HDF5-chunked datasets,
``read_double_buffered`` otherwise).

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - API
     - Description
   * - ``.preload()``
     - Load to GPU now (no-op if already loaded). Returns ``self``.
   * - ``.free()``
     - Release the GPU array.
   * - ``.reload()``
     - Free and re-read from disk. Returns ``self``.
   * - ``.array``
     - The ``cupy.ndarray``. Triggers load on first access.
   * - ``cached[...]``
     - Index the cached array (pure GPU, no I/O).
   * - ``.reduce(fn, transform=None)``
     - Apply a reduction. Optional ``transform`` is applied first
       without mutating the cache.
   * - ``.transform(fn)``
     - Replace the cache with ``fn(array)``. Returns ``self``.

Benchmarks
~~~~~~~~~~

Three benchmark scripts are included under ``benchmarks/``:

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Script
     - What it measures
   * - ``bench_gpu_read.py``
     - Full-dataset double-buffered read (chunk-size sweep) and
       selection read (coverage and alignment sweeps).
   * - ``bench_gpu_write.py``
     - Double-buffered write, chunk-by-chunk write, and selection
       write; baseline vs. pipelined.
   * - ``bench_gpu_transform.py``
     - Transform pipeline for 1-D and 2-D datasets across compute
       intensities (none, ``x*2``, ``sqrt``, ``exp``, ``exp(sqrt(x))``);
       multi-stream speedup sweep (1/2/4/8 streams).

Run any benchmark with::

    python benchmarks/bench_gpu_read.py --rows 4096 --cols 4096
