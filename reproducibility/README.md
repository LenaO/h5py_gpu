# Reproducibility artifact

Scripts used to produce the benchmarks, figures, and tables in the
GPU-aware h5py paper (PDSW'26). This directory is the reproducibility
artifact referenced by the paper's Artifact Description (AD) Appendix.

## Layout

- `make_benchmark_data.py` — generates the synthetic HDF5 datasets used
  by every benchmark below (reproducible via `--seed`, no external
  download needed).
- `benchmark_chunk_vs_rowband.py` — Table I (row-band vs. chunk-by-chunk
  reads/writes).
- `benchmark_selection_sweep.py` / `benchmark_write_selection_sweep.py`
  — Fig. 2 (index-addressed read/write coverage sweep).
- `benchmark_host_memory_sweep.py` — Table II (peak host memory vs.
  dataset size).
- `benchmark_fused_compute.py` — Fig. 3 (fused vs. non-fused
  transform/reduce vs. synthetic compute cost).
- `make_selection_sweep_figure.py` / `make_fused_compute_figure.py` —
  turn the CSVs the scripts above produce into the paper's actual PDF
  figures.
- `run_benchmarks_juwels_single_gpu.sh` — orchestrates all of the above
  end to end on a single GPU node (written for JUWELS Booster's SLURM +
  module environment; adapt the `module load` line and venv path for
  other systems).
- `benchmark_batch_overlap.py`, `benchmark_use_cases.py`,
  `benchmark_ml_loading.py`, `benchmark_worker_reader_scaling.py`,
  `test_stream_sharing.py`, `train_fastmri_resnet.py`,
  `prepare_fastmri.py` — additional microbenchmarks and the ML-loading
  comparison also run by the script above; not all of these are reported
  in the current paper draft, but they are part of the same suite and
  needed for the run script to complete without errors.

## Raw results

The raw CSV output of these scripts, and the exact PDF figures included
in the paper, are archived separately on Zenodo (DOI in the paper's AD
Appendix), rather than committed to this repository.

## Running it

1. Install this repository's h5py fork (`pip install -e .` from the
   repo root) into a Python environment with a CUDA-enabled CuPy.
2. `python make_benchmark_data.py --help` for the dataset generator's
   options, or run `run_benchmarks_juwels_single_gpu.sh` to generate
   data and run every benchmark end to end (expects a JUWELS-style
   module/SLURM environment; ~1 hour on a single A100 node).
3. Each `benchmark_*.py` script also runs standalone and takes a
   `--csv` flag to write its raw results; run `--help` on any of them
   for its specific options.
