#!/usr/bin/env bash
# A100 reference profile. This is NOT a verified reproduction of the paper.
# Override TRAIN_PROFILE=full explicitly to include every nonempty grounding.
# Submit from the repository root; create logs/ before sbatch.
#SBATCH --job-name=dygenc_a100_90g
#SBATCH --partition=batch
#SBATCH --nodelist=gpu03
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=90G
#SBATCH --gres=gpu:1
#SBATCH --time=2-00:00:00
#SBATCH --signal=B:USR1@180
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -Eeuo pipefail
export TRAIN_PROFILE="${TRAIN_PROFILE:-upstream}"
export PERSIST_ROOT="${PERSIST_ROOT:-/media02/lnthanh03/nnthao21}"
export RUN_DIR="${RUN_DIR:-$PERSIST_ROOT/runs/dygenc/a100_upstream_90g}"
export TARGET_EPOCHS="${TARGET_EPOCHS:-5}"
export ACCUMULATION_STEPS="${ACCUMULATION_STEPS:-32}"
export SOURCE_REPO="${SOURCE_REPO:-${SLURM_SUBMIT_DIR:?Submit with sbatch from the repo root}}"
exec bash "$SOURCE_REPO/scripts/slurm/train_agqa_full_ram.slurm"
