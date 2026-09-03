#!/usr/bin/env bash
# Submit from the DyGEnc checkout, after creating logs/ and the secret files:
#   mkdir -p logs
#   sbatch scripts/slurm/train_agqa_full_ram.slurm
# All dependencies, data, preprocessing and pretrained models live in tmpfs.
# Only checkpoints, small run metadata, credentials and Slurm logs persist.
# 96G is the school-cluster default, NOT a guarantee that full AGQA fits.

#SBATCH --job-name=dygenc_agqa_full
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --time=2-00:00:00
#SBATCH --signal=B:USR1@180
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -Eeuo pipefail
export RUN_MODE=fresh
export PERSIST_ROOT="${PERSIST_ROOT:-/media02/lnthanh03}"
if [[ -n "${SOURCE_REPO+x}" ]]; then
  [[ -f "$SOURCE_REPO/scripts/slurm/agqa_full_job.sh" ]] || {
    echo "ERROR: Explicit SOURCE_REPO is not a DyGEnc checkout: $SOURCE_REPO" >&2
    exit 1
  }
elif [[ -f "${SLURM_SUBMIT_DIR:-}/scripts/slurm/agqa_full_job.sh" ]]; then
  SOURCE_REPO="$SLURM_SUBMIT_DIR"
else
  SOURCE_REPO="$PERSIST_ROOT/code/DyGEnc"
fi
[[ -f "$SOURCE_REPO/scripts/slurm/agqa_full_job.sh" ]] || {
  echo "ERROR: Submit from the DyGEnc root, or export SOURCE_REPO=/path/to/DyGEnc" >&2
  exit 1
}
export SOURCE_REPO
exec bash "$SOURCE_REPO/scripts/slurm/agqa_full_job.sh"
