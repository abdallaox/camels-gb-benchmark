#!/bin/bash
#SBATCH --job-name=gbhydro
#SBATCH --partition=gpu
#SBATCH --qos=freegpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=slurm/%x_%j.out
#
# One training run on a Slurm GPU cluster. Partition and QoS are those of the cluster used in the study;
# change them to suit yours.
# All arguments go to scripts/training/train.py, e.g.
#   sbatch cluster/train.sbatch --model lstm --seed 0
#   sbatch cluster/train.sbatch --model lstm --seed 0 --inputs full
# Runs resume from runs/<name>/last.pt if the job is requeued or resubmitted.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export PATH=$HOME/.local/bin:$PATH
export PYTHONUNBUFFERED=1            # so the Slurm log shows epochs as they happen
export GBHYDRO_DEVICE=cuda:0
export GBHYDRO_RUN_DIR=$PWD/runs
mkdir -p "$GBHYDRO_RUN_DIR/slurm"
echo "host $(hostname)  job $SLURM_JOB_ID  start $(date -Is)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
uv run --no-sync python scripts/training/train.py "$@"   # run "uv sync" once on a login node before submitting
echo "end $(date -Is)"
