#!/bin/bash
#SBATCH --job-name=ntia-paper-int
#SBATCH --partition=gpuq
#SBATCH --qos=gpu
#SBATCH --gres=gpu:A100.80gb:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --time=08:00:00
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

SEED="${SEED:-0}"
EPOCHS="${EPOCHS:-1}"
SNAPSHOTS_PER_EPOCH="${SNAPSHOTS_PER_EPOCH:-50}"
K95_N_SEQUENCES="${K95_N_SEQUENCES:-100}"
RUN_ROOT="$SCRATCH/spectralTP_runs"
RUN_TAG="paper-int-seed${SEED}-e${EPOCHS}-s${SNAPSHOTS_PER_EPOCH}-${SLURM_JOB_ID}"
RUN_DIR="$RUN_ROOT/$RUN_TAG"

cd /scratch/skumar43/spectralTP

echo "HOST=$(hostname)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "SCRATCH=$SCRATCH"
echo "RUN_DIR=$RUN_DIR"
echo "SEED=$SEED"
echo "EPOCHS=$EPOCHS"
echo "SNAPSHOTS_PER_EPOCH=$SNAPSHOTS_PER_EPOCH"
git rev-parse HEAD || true
nvidia-smi || true

uv run python paper_grid_runner.py \
  --root "$RUN_ROOT" \
  --run-tag "$RUN_TAG" \
  --seed "$SEED" \
  --epochs "$EPOCHS" \
  --snapshots-per-epoch "$SNAPSHOTS_PER_EPOCH" \
  --k95-n-sequences "$K95_N_SEQUENCES" \
  --probe-epochs 200 \
  --data int_dataset_v1_tokenized.parquet \
  --vocab tokenizer_vocab_locked_v1.json

echo "Paper grid report:"
cat "$RUN_DIR/paper_grid_report.txt"

echo "Paper grid summary CSV:"
cat "$RUN_DIR/paper_grid_summary.csv"

echo "Paper grid outputs:"
find "$RUN_DIR" -maxdepth 2 -type f | sort
