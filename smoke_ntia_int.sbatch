#!/bin/bash
#SBATCH --job-name=ntia-smoke-int
#SBATCH --partition=gpuq
#SBATCH --qos=gpu
#SBATCH --gres=gpu:A100.80gb:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --time=02:00:00
#SBATCH --verbose
#SBATCH --output=/scratch/%u/%x-%j.out
#SBATCH --error=/scratch/%u/%x-%j.err

set -euo pipefail

module load gnu10
module load git

export PATH="$HOME/.local/bin:$PATH"
export UV_CACHE_DIR="$SCRATCH/.uv-cache"
export HF_HOME="$SCRATCH/.hf"
export PYTHONUNBUFFERED=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8

cd /scratch/skumar43/spectralTP

RUN_TAG="smoke-int-${SLURM_JOB_ID}"
RUN_ROOT="$SCRATCH/spectralTP_runs"
RUN_DIR="$RUN_ROOT/$RUN_TAG"

echo "HOST=$(hostname)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "SCRATCH=$SCRATCH"
echo "RUN_DIR=$RUN_DIR"
git rev-parse HEAD || true
nvidia-smi || true

uv run python paper_grid_runner.py \
  --root "$RUN_ROOT" \
  --run-tag "$RUN_TAG" \
  --arch-ids 5 \
  --seed 1337 \
  --epochs 1 \
  --epoch1-snapshots 30 \
  --k95-n-sequences 64 \
  --probe-epochs 200 \
  --data int_dataset_v1_tokenized.parquet \
  --vocab tokenizer_vocab_locked_v1.json

echo "Smoke report:"
cat "$RUN_DIR/paper_grid_report.txt"

echo "Smoke summary CSV:"
cat "$RUN_DIR/paper_grid_summary.csv"

echo "Smoke outputs:"
find "$RUN_DIR" -maxdepth 2 -type f | sort
