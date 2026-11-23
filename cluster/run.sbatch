#!/bin/bash
#SBATCH --job-name=gbhydro
#SBATCH --partition=gpu
#SBATCH --qos=freegpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=slurm/%x_%j.out
#
# Run any repo script on one GPU, e.g.
#   sbatch -J hbv_local cluster/run.sbatch scripts/training/calibrate_hbv.py --iters 3000 --restarts 5
# Same environment as train.sbatch.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export PATH=$HOME/.local/bin:$PATH
export PYTHONUNBUFFERED=1
export GBHYDRO_DEVICE=cuda:0
export GBHYDRO_RUN_DIR=$PWD/runs
mkdir -p "$GBHYDRO_RUN_DIR/slurm"
echo "host $(hostname)  job $SLURM_JOB_ID  start $(date -Is)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
uv run --no-sync python "$@"
echo "end $(date -Is)"
