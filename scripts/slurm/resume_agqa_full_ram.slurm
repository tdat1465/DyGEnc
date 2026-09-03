#!/usr/bin/env bash
# Resume only RUN_DIR/last.pth produced by the companion fresh job.
# Every new allocation rebuilds its venv, data and model cache in tmpfs;
# the source, raw-data hashes and pretrained model revisions must match.
# TARGET_EPOCHS must match the fresh job's original total, not extra epochs.
#   mkdir -p logs
#   sbatch scripts/slurm/resume_agqa_full_ram.slurm
# For a fresh job launched with TARGET_EPOCHS=2, export the SAME value here.

#SBATCH --job-name=dygenc_agqa_resume
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
export RUN_MODE=resume
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
