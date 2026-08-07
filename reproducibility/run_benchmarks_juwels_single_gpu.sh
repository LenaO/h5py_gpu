#!/bin/sh
# run_benchmarks_juwels_single_gpu.sh - Single-GPU microbenchmarks on JUWELS
# against GPFS-backed project storage.
#
# Requests only 1 node / 1 GPU: nothing in this suite is multi-GPU-aware, so
# there is no reason to hold a larger allocation idle for it. For the
# multi-GPU train_fastmri_resnet.py comparison (baseline vs. gpu), submit
# run_benchmarks_juwels_multi_gpu.sh separately.
#
# Before running: this session's investigation found and fixed a genuine
# cross-stream memory-safety race condition and a default-buffer-size bug in
# h5py/gpu.py. Step 0 below re-syncs the checked-out copy into the venv's
# site-packages -- do not skip it, especially if the venv predates this
# session's fixes.

#SBATCH --account=exalab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --output=bench-single-out.%j
#SBATCH --error=bench-single-err.%j
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:1

## Deliberately no `set -e`: each step's stdout/stderr is redirected to its
## own log below, so one microbenchmark failing (e.g. a missing optional
## dependency) is visible in its own log without preventing the rest of the
## suite from still running.

module load Stages/2026 GCCcore/14.3.0 Python CuPy HDF5 PyTorch torchvision
module list
source /p/project1/cexalab/oden1/HDF5/juwels/bin/activate

# ---------------------------------------------------------------------------
# Paths -- adjust if your checkout or GPFS project layout differs. REPO_DIR
# is assumed to be this script's own directory, matching how
# run_train_juwels.sh invokes ./train_fastmri_resnet.py as a relative path.
# ---------------------------------------------------------------------------
REPO_DIR=$(cd "$(dirname "$0")" && pwd)
DATA_DIR="$REPO_DIR/bench_data_gpfs"
RESULTS_DIR="$REPO_DIR/results_bench_gpfs_${SLURM_JOB_ID}"
mkdir -p "$DATA_DIR" "$RESULTS_DIR"

echo "REPO_DIR    = $REPO_DIR"
echo "DATA_DIR    = $DATA_DIR   (GPFS project storage)"
echo "RESULTS_DIR = $RESULTS_DIR"

# ---------------------------------------------------------------------------
# Step 0: sync h5py/gpu.py into the venv. The installed copy in this venv is
# what an earlier run traced an OOM to; several correctness and default-size
# fixes landed in the repo's copy since. Never skip this.
# ---------------------------------------------------------------------------
SITE_GPU=$(python -c "import h5py.gpu as g; print(g.__file__)")
echo "Installed h5py.gpu : $SITE_GPU"
if ! cmp -s "$REPO_DIR/h5py/h5py/gpu.py" "$SITE_GPU"; then
    echo "  -> out of sync, copying repo version over the installed one"
    cp "$REPO_DIR/h5py/h5py/gpu.py" "$SITE_GPU"
else
    echo "  -> already in sync"
fi

# =============================================================================
# Microbenchmarks
# =============================================================================
echo "=================================================================="
echo " Single-GPU microbenchmarks"
echo "=================================================================="

# --- Generate the shared data files once, directly on GPFS, reused by the
#     benchmarks below that need a pre-existing file (CPU-only, no GPU needed)
GEN_3D="$DATA_DIR/bench_3d_chunked.h5"
GEN_2D_CONT="$DATA_DIR/bench_2d_contiguous.h5"

python "$REPO_DIR/make_benchmark_data.py" "$GEN_3D" \
    --kind 3d_chunked --n 2048 --size 512 --force \
    > "$RESULTS_DIR/gen_3d_chunked.out" 2>&1
python "$REPO_DIR/make_benchmark_data.py" "$GEN_2D_CONT" \
    --kind 2d_contiguous --rows 16384 --cols 16384 --force \
    > "$RESULTS_DIR/gen_2d_contiguous.out" 2>&1

# --- a. Selection-size sweep: naive-full vs. naive-partial vs. ours,
#     across coverage fractions, all four layouts (auto-generates its own
#     data in a temp dir -- separate from GEN_3D/GEN_2D_CONT above)
python "$REPO_DIR/benchmark_selection_sweep.py" \
    --all-layouts --size-2d 8192 --chunk-2d 256 --n-3d 512 --size-3d 512 \
    --csv "$RESULTS_DIR/selection_sweep.csv" \
    > "$RESULTS_DIR/selection_sweep.out" 2>&1

# --- a2. Write-side mirror of the selection sweep: naive-full vs.
#     naive-partial vs. ours, across coverage fractions, all four layouts.
#     write_selection_chunked now uses the same edge/interior row-band
#     split as read_selection_chunked for 2-D chunked datasets, so "ours"
#     should beat naive-partial there from ~25% coverage up (local testing:
#     10.17x vs 6.70x at 25%, 2.85x vs 2.24x at 100%). 3-D chunked still
#     falls back to one chunk at a time, and contiguous datasets have no
#     partial-selection double-buffered write path at all (see the script's
#     docstring), so expect little or no benefit from "ours" on those two.
python "$REPO_DIR/benchmark_write_selection_sweep.py" \
    --all-layouts --size-2d 8192 --chunk-2d 256 --n-3d 512 --size-3d 512 \
    --csv "$RESULTS_DIR/write_selection_sweep.csv" \
    > "$RESULTS_DIR/write_selection_sweep.out" 2>&1

# --- a3. Chunk-by-chunk vs. row-band, both directions: isolates the
#     Python/HDF5 per-call overhead that motivated batching fully-covered
#     chunks into row-bands throughout this module, by sweeping chunk size
#     (hence chunks-per-row) at fixed dataset size. Local testing (4096x4096,
#     chunk 128-2048): row-band up to 4.6x/3.5x faster (read/write) at 32
#     chunks/row, shrinking as chunk count drops, reversing to ~0.55x
#     (row-band slower) at only 2 chunks/row -- expect the same qualitative
#     crossover on GPFS, though the exact crossover point may differ.
python "$REPO_DIR/benchmark_chunk_vs_rowband.py" \
    --size 8192 --chunk-sizes 128,256,512,1024,2048,4096 \
    --csv "$RESULTS_DIR/chunk_vs_rowband.csv" \
    > "$RESULTS_DIR/chunk_vs_rowband.out" 2>&1

# --- b. Host memory footprint vs. dataset size, all four layouts
#     (spawns its own subprocess workers per measurement -- see the script's
#     docstring for why; each worker inherits this job's GPU binding)
python "$REPO_DIR/benchmark_host_memory_sweep.py" \
    --all-layouts --sizes-mb 128,512,1024,2048,4096,8192 \
    --csv "$RESULTS_DIR/host_memory_sweep.csv" \
    > "$RESULTS_DIR/host_memory_sweep.out" 2>&1

# --- c. Fused transform/reduce vs. the old read-copy-execute approach,
#     across a compute-cost sweep -- chunked and contiguous layouts.
#     Now uses read_double_buffered/reduce_double_buffered (row-band)
#     uniformly for BOTH layouts, replacing the previous chunk-wise
#     (read_chunks_to_gpu/reduce_chunks) path for the chunked case. For
#     this (1,H,W)-chunked dataset a row-band IS one chunk (chunk_size
#     defaults to chunks[0]=1), so the per-piece granularity -- and the
#     result -- should be unchanged from the old version; only the method
#     name is now the same across layouts. The piece-count/overhead
#     question itself is covered separately by benchmark_chunk_vs_rowband.py.
python "$REPO_DIR/benchmark_fused_compute.py" \
    --layout 3d_chunked --n-3d 512 --size-3d 512 \
    --n-ops 0,1,2,4,8,16,32,64,128,256,512,1024 \
    --csv "$RESULTS_DIR/fused_compute_3d_chunked.csv" \
    > "$RESULTS_DIR/fused_compute_3d_chunked.out" 2>&1

python "$REPO_DIR/benchmark_fused_compute.py" \
    --layout 2d_contiguous --size-2d 8192 \
    --n-ops 0,1,2,4,8,16,32,64,128,256,512,1024 \
    --csv "$RESULTS_DIR/fused_compute_2d_contiguous.csv" \
    > "$RESULTS_DIR/fused_compute_2d_contiguous.out" 2>&1

# --- d. Sync vs. double-buffered async 3-D whole-slice batch reads
python "$REPO_DIR/benchmark_batch_overlap.py" "$GEN_3D" \
    --batch-size 16 --batches 100 --repeats 5 \
    --csv "$RESULTS_DIR/batch_overlap.csv" \
    > "$RESULTS_DIR/batch_overlap.out" 2>&1

# --- e. The five access patterns from Table 1 (full load / index-addressed /
#     reduce-discard / reduce-keep two-pass / reduce-keep fused)
python "$REPO_DIR/benchmark_use_cases.py" "$GEN_3D" \
    --dataset images --repeats 5 \
    > "$RESULTS_DIR/use_cases_3d_chunked.out" 2>&1

python "$REPO_DIR/benchmark_use_cases.py" "$GEN_2D_CONT" \
    --dataset data --repeats 5 \
    > "$RESULTS_DIR/use_cases_2d_contiguous.out" 2>&1

# --- f. DataLoader-style throughput (baseline vs. gpu_raw vs. gpu_norm),
#     single GPU -- the multi-GPU sweep is in run_benchmarks_juwels_multi_gpu.sh
python "$REPO_DIR/benchmark_ml_loading.py" "$GEN_3D" \
    --batch-size 16 --batches 100 --repeats 5 --with-norm \
    --csv "$RESULTS_DIR/ml_loading.csv" \
    > "$RESULTS_DIR/ml_loading.out" 2>&1

# --- g. Stream-sharing / cross-stream memory-pool regression check
#     (matches the actual training batch size, since that's what the
#     original OOM investigation was calibrated against)
python "$REPO_DIR/test_stream_sharing.py" "$GEN_3D" \
    --batch-size 16 --consume-ms 170 \
    > "$RESULTS_DIR/stream_sharing.out" 2>&1

# --- h. Worker/reader scaling: does gpu mode need as many CPU
#     workers/readers as baseline to reach the same throughput? Runs
#     train_fastmri_resnet.py itself (single GPU, single process -- the
#     effect is per-process, so a multi-GPU DDP run isn't needed to see it)
#     across a sweep of --workers (baseline) / --readers (gpu), and reports
#     the crossover worker count.
python "$REPO_DIR/benchmark_worker_reader_scaling.py" "$GEN_3D" \
    --baseline-workers 1,2,4,8,16 --gpu-readers 0,1,2,4,8 \
    --batch-size 16 --steps 100 --repeats 3 \
    --csv "$RESULTS_DIR/worker_reader_scaling.csv" \
    > "$RESULTS_DIR/worker_reader_scaling.out" 2>&1

echo "=================================================================="
echo " All done. Results in $RESULTS_DIR"
echo "=================================================================="
