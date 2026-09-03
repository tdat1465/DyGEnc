#!/usr/bin/env bash
# Shared implementation; submit a .slurm wrapper, never run on a login node.
# There is deliberately no persistent-data or persistent-venv fallback.
set -Eeuo pipefail
umask 077

die() { echo "ERROR: $*" >&2; exit 1; }
positive_integer() { [[ "$1" =~ ^[1-9][0-9]*$ ]]; }

case "${RUN_MODE:-}" in
  fresh|resume) ;;
  *) die "RUN_MODE must be fresh or resume; submit a companion .slurm wrapper" ;;
esac
[[ -n "${SLURM_JOB_ID:-}" ]] || die "This script requires an allocated Slurm job"

PERSIST_ROOT="${PERSIST_ROOT:-/media02/lnthanh03}"
SOURCE_REPO="${SOURCE_REPO:-${SLURM_SUBMIT_DIR:-$PERSIST_ROOT/code/DyGEnc}}"
RUN_DIR="${RUN_DIR:-$PERSIST_ROOT/runs/dygenc/agqa_full}"
HF_TOKEN_FILE="${HF_TOKEN_FILE:-$PERSIST_ROOT/secrets/hf_token}"
DATASET_REPO="${DATASET_REPO:-tdat1465/agqa-balanced}"
DATASET_REVISION="${DATASET_REVISION:-}"
ENG_FILE="${ENG_FILE:-}"
ENG_URL="${ENG_URL:-}"
if [[ -z "$ENG_FILE" && -z "$ENG_URL" && -f "$PERSIST_ROOT/secrets/ENG.txt" ]]; then
  ENG_FILE="$PERSIST_ROOT/secrets/ENG.txt"
fi
if [[ -z "$ENG_FILE" && -z "$ENG_URL" ]]; then
  ENG_URL='https://drive.google.com/uc?export=download&id=1d0Gx4x5qnvp13Su_sIS_nlSn47ZggY8n'
fi
TARGET_EPOCHS="${TARGET_EPOCHS:-5}"
TRAIN_PROFILE="${TRAIN_PROFILE:-full}"
ACCUMULATION_STEPS="${ACCUMULATION_STEPS:-32}"
LOSS_REDUCTION="${LOSS_REDUCTION:-token_mean}"
STOP_AFTER_UPDATES="${STOP_AFTER_UPDATES:-0}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-1000}"
SEED="${SEED:-18}"
MIN_RAM_FREE_GB="${MIN_RAM_FREE_GB:-40}"
MIN_TRAIN_HEADROOM_GB="${MIN_TRAIN_HEADROOM_GB:-24}"
DYGENC_EMBED_BATCH_SIZE="${DYGENC_EMBED_BATCH_SIZE:-64}"
DYGENC_GRAPH_CACHE_SIZE="${DYGENC_GRAPH_CACHE_SIZE:-2}"
PYTHON_BIN="${PYTHON_BIN:-/usr/bin/python3}"
RAM_BASE="${RAM_BASE:-/dev/shm}"
VIRTUALENV_PYZ_URL="${VIRTUALENV_PYZ_URL:-https://bootstrap.pypa.io/virtualenv.pyz}"
LLM_MODEL="${LLM_MODEL:-meta-llama/Llama-3.2-3B}"
MBERT_MODEL="answerdotai/ModernBERT-large"
# Empty means resolve main on fresh, but use saved immutable SHA on resume.
LLM_REVISION="${LLM_REVISION:-}"
MBERT_REVISION="${MBERT_REVISION:-}"

for name in TARGET_EPOCHS ACCUMULATION_STEPS CHECKPOINT_EVERY MIN_RAM_FREE_GB MIN_TRAIN_HEADROOM_GB DYGENC_EMBED_BATCH_SIZE DYGENC_GRAPH_CACHE_SIZE; do
  positive_integer "${!name}" || die "$name must be a positive integer"
done
[[ "$TRAIN_PROFILE" == full || "$TRAIN_PROFILE" == upstream ]] || die "TRAIN_PROFILE must be full or upstream"
[[ "$LOSS_REDUCTION" == token_mean || "$LOSS_REDUCTION" == sample_mean ]] || die "LOSS_REDUCTION must be token_mean or sample_mean"
[[ "$STOP_AFTER_UPDATES" =~ ^[0-9]+$ ]] || die "STOP_AFTER_UPDATES must be a non-negative integer"
[[ "$SEED" =~ ^[0-9]+$ ]] || die "SEED must be a non-negative integer"
[[ -z "$ENG_URL" || -z "$ENG_FILE" ]] || die "Set ENG_URL or ENG_FILE, not both"
for cmd in flock findmnt readlink stat mktemp setsid tar; do
  command -v "$cmd" >/dev/null 2>&1 || die "Required system command is missing: $cmd"
done
command -v "$PYTHON_BIN" >/dev/null 2>&1 || die "Python executable missing: $PYTHON_BIN"
for file in requirements-server.txt src/server_train.py scripts/slurm/stage_agqa.py scripts/slurm/download_agqa.py; do
  [[ -f "$SOURCE_REPO/$file" ]] || die "Current DyGEnc source file missing: $SOURCE_REPO/$file"
done
SOURCE_REPO="$(readlink -f -- "$SOURCE_REPO")"
[[ "$RUN_DIR" == /* && "$RUN_DIR" != / ]] || die "RUN_DIR must be a non-root absolute path"
[[ ! -L "$RUN_DIR" ]] || die "RUN_DIR must not be a symlink"
mkdir -p "$RUN_DIR"
RUN_DIR="$(readlink -f -- "$RUN_DIR")"
ACTUAL_UID="$(id -u)"
[[ "$(stat -c '%u' "$RUN_DIR")" == "$ACTUAL_UID" ]] || die "RUN_DIR must be owned by your UID"
RUN_FS_TYPE="$(findmnt -n -o FSTYPE --target "$RUN_DIR")"
[[ "$RUN_FS_TYPE" != tmpfs && "$RUN_FS_TYPE" != ramfs ]] || die "RUN_DIR is RAM-backed; choose persistent storage for checkpoints"
[[ ! -L "$RUN_DIR/.slurm.lock" ]] || die "Refusing symlink checkpoint lock"
exec 9>"$RUN_DIR/.slurm.lock"
flock -n 9 || die "Another job is writing $RUN_DIR"
if [[ "$RUN_MODE" == fresh ]]; then
  for file in last.pth best.pth; do
    [[ ! -e "$RUN_DIR/$file" && ! -L "$RUN_DIR/$file" ]] || die "$RUN_DIR/$file exists: submit resume or choose a NEW RUN_DIR"
  done
else
  for file in last.pth source-fingerprint.sha256 model-revisions.json dataset-source.json raw-data-manifest.json; do
    [[ -s "$RUN_DIR/$file" ]] || die "Resume artifact missing: $RUN_DIR/$file"
  done
fi
for file in source-fingerprint.sha256 model-revisions.json dataset-source.json raw-data-manifest.json "pip-freeze-${SLURM_JOB_ID}.txt"; do
  [[ ! -L "$RUN_DIR/$file" ]] || die "Refusing symlink output artifact: $RUN_DIR/$file"
done

# Credentials are the only input files allowed to persist outside the code.
# Never echo their contents, enable shell tracing, or put tokens in sbatch args.
for credential in "$HF_TOKEN_FILE"; do
  [[ -r "$credential" && -f "$credential" ]] || die "Credential file not readable: $credential"
  [[ "$(stat -c '%u' "$credential")" == "$ACTUAL_UID" ]] || die "Credential must be owned by your UID: $credential"
  mode="$(stat -c '%a' "$credential")"
  (( (8#$mode & 077) == 0 )) || die "Credential is readable by others; run chmod 600 '$credential'"
done

[[ -d "$RAM_BASE" ]] || die "RAM_BASE does not exist: $RAM_BASE"
RAM_BASE="$(readlink -f -- "$RAM_BASE")"
[[ "$RAM_BASE" != / ]] || die "RAM_BASE cannot be /"
[[ "$(findmnt -n -o FSTYPE --target "$RAM_BASE")" == tmpfs ]] || die "RAM_BASE must be tmpfs: $RAM_BASE (no disk fallback)"
RAM_OPTIONS="$(findmnt -n -o OPTIONS --target "$RAM_BASE")"
[[ ",$RAM_OPTIONS," != *,noexec,* ]] || die "$RAM_BASE is mounted noexec. Ask the administrator for an executable tmpfs, then export RAM_BASE=/that/path; do not switch to a disk directory."
[[ "$RUN_DIR/" != "$RAM_BASE/"* ]] || die "RUN_DIR is inside RAM_BASE: checkpoints would disappear"

RAM_ROOT="$(mktemp -d -- "$RAM_BASE/dygenc-${ACTUAL_UID}-${SLURM_JOB_ID}.XXXXXXXX")"
EXPECTED_RAM_ROOT="$RAM_ROOT"
chmod 700 "$RAM_ROOT"
RAM_ROOT_ID="$(stat -c '%d:%i:%u' "$RAM_ROOT")"
CHILD_PID=""
CURRENT_PHASE=""
STOP_STATUS=0

cleanup() {
  local status="$?"
  trap - EXIT
  trap '' USR1 TERM INT
  set +e
  # Do not unmap/delete libraries or model files while a child saves state.
  if [[ -n "$CHILD_PID" ]]; then
    kill -TERM -- "-$CHILD_PID" 2>/dev/null
    wait "$CHILD_PID" 2>/dev/null
  fi
  if [[ -d "$RAM_ROOT" && ! -L "$RAM_ROOT" &&
        "$RAM_ROOT" == "$EXPECTED_RAM_ROOT" &&
        "$(readlink -f -- "$RAM_ROOT")" == "$EXPECTED_RAM_ROOT" &&
        "$(stat -c '%d:%i:%u' "$RAM_ROOT")" == "$RAM_ROOT_ID" ]]; then
    rm -rf --one-file-system -- "$RAM_ROOT"
    echo "Removed this job's temporary RAM workspace: $RAM_ROOT"
  else
    echo "WARNING: RAM workspace identity changed; refusing cleanup: $RAM_ROOT" >&2
  fi
  exit "$status"
}
request_stop() {
  local signal="$1" status="$2"
  STOP_STATUS="$status"
  echo "Received $signal during ${CURRENT_PHASE:-setup}; requesting a clean stop"
  if [[ -n "$CHILD_PID" ]]; then
    if [[ "$CURRENT_PHASE" == train ]]; then
      # Trainer handles USR1/TERM as checkpoint requests, not KeyboardInterrupt.
      [[ "$signal" != INT ]] || signal=TERM
      kill -"$signal" "$CHILD_PID" 2>/dev/null || true
    else
      kill -TERM -- "-$CHILD_PID" 2>/dev/null || true
    fi
  fi
}
trap cleanup EXIT
trap 'request_stop USR1 138' USR1
trap 'request_stop TERM 143' TERM
trap 'request_stop INT 130' INT

# Bash wait is interrupted by trapped signals. Re-wait until Python actually
# exits, otherwise cleanup could erase tmpfs while a checkpoint is being saved.
run_child() {
  local phase="$1" status=0
  shift
  (( STOP_STATUS == 0 )) || return "$STOP_STATUS"
  CURRENT_PHASE="$phase"
  setsid "$@" <&0 &
  CHILD_PID=$!
  while true; do
    if wait "$CHILD_PID"; then status=0; else status=$?; fi
    if ! kill -0 "$CHILD_PID" 2>/dev/null; then break; fi
  done
  CHILD_PID=""
  CURRENT_PHASE=""
  (( STOP_STATUS == 0 )) || return "$STOP_STATUS"
  return "$status"
}

RAM_REPO="$RAM_ROOT/repo"
DOWNLOAD_DIR="$RAM_ROOT/downloads"
export AGQA_ROOT="$RAM_ROOT/agqa"
export TMPDIR="$RAM_ROOT/tmp" TMP="$RAM_ROOT/tmp" TEMP="$RAM_ROOT/tmp"
export SQLITE_TMPDIR="$TMPDIR"
export XDG_CACHE_HOME="$RAM_ROOT/xdg/cache" XDG_CONFIG_HOME="$RAM_ROOT/xdg/config"
export XDG_DATA_HOME="$RAM_ROOT/xdg/data" XDG_STATE_HOME="$RAM_ROOT/xdg/state"
export XDG_RUNTIME_DIR="$RAM_ROOT/xdg/runtime"
export HF_HOME="$RAM_ROOT/huggingface" HF_HUB_CACHE="$RAM_ROOT/huggingface/hub"
export HUGGINGFACE_HUB_CACHE="$HF_HUB_CACHE" HF_ASSETS_CACHE="$RAM_ROOT/huggingface/assets"
export HF_DATASETS_CACHE="$RAM_ROOT/huggingface/datasets" HF_XET_CACHE="$RAM_ROOT/huggingface/xet"
export HF_MODULES_CACHE="$RAM_ROOT/huggingface/modules"
export HF_TOKEN_PATH="$RAM_ROOT/huggingface/token"
export HF_HUB_DISABLE_TELEMETRY=1 HF_HUB_DISABLE_XET=1
export HF_HUB_DOWNLOAD_TIMEOUT=120
export TORCH_HOME="$RAM_ROOT/torch" TRITON_CACHE_DIR="$RAM_ROOT/triton"
export TORCH_EXTENSIONS_DIR="$RAM_ROOT/torch-extensions" TORCHINDUCTOR_CACHE_DIR="$RAM_ROOT/torchinductor"
export CUDA_CACHE_PATH="$RAM_ROOT/cuda" NUMBA_CACHE_DIR="$RAM_ROOT/numba"
export MPLCONFIGDIR="$RAM_ROOT/matplotlib" PYTHONPYCACHEPREFIX="$RAM_ROOT/pycache"
export PIP_CACHE_DIR="$RAM_ROOT/pip-cache" PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
export PIP_CONFIG_FILE=/dev/null PYTHONNOUSERSITE=1
export PYTHONUSERBASE="$RAM_ROOT/python-user"
export VIRTUALENV_OVERRIDE_APP_DATA="$RAM_ROOT/virtualenv-appdata"
export UV_CACHE_DIR="$RAM_ROOT/uv-cache"
export CARGO_HOME="$RAM_ROOT/cargo" RUSTUP_HOME="$RAM_ROOT/rustup"
export WANDB_MODE=disabled WANDB_DIR="$RAM_ROOT/wandb"
export WANDB_CACHE_DIR="$RAM_ROOT/wandb/cache" WANDB_CONFIG_DIR="$RAM_ROOT/wandb/config"
export WANDB_DATA_DIR="$RAM_ROOT/wandb/data"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export PYTHONHASHSEED="$SEED" DYGENC_PREPROCESS_SEED="$SEED" DYGENC_SAVE_NETWORKX=0
export DYGENC_LAZY_GRAPHS=1 DYGENC_GRAPH_CACHE_SIZE
export DYGENC_INDEXED_QA=1 DYGENC_GRADIENT_CHECKPOINTING=1
export DYGENC_EMBED_BATCH_SIZE OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
unset PYTHONHOME PYTHONPATH TRANSFORMERS_CACHE PYTORCH_TRANSFORMERS_CACHE PYTORCH_PRETRAINED_BERT_CACHE
unset PIP_TARGET PIP_PREFIX PIP_USER PIP_LOG PIP_BUILD_TRACKER VIRTUAL_ENV
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE
mkdir -p "$RAM_REPO" "$DOWNLOAD_DIR" "$AGQA_ROOT" "$TMPDIR" "$XDG_RUNTIME_DIR" \
  "$XDG_CACHE_HOME" "$XDG_CONFIG_HOME" "$XDG_DATA_HOME" "$XDG_STATE_HOME" \
  "$HF_HOME" "$VIRTUALENV_OVERRIDE_APP_DATA"

"$PYTHON_BIN" - <<'PY'
import sys
if not (3, 10) <= sys.version_info[:2] <= (3, 12):
    raise SystemExit(f"Python 3.10-3.12 required for the pinned CUDA wheels; found {sys.version.split()[0]}")
print("Base Python:", sys.executable, sys.version.split()[0])
PY

# tmpfs usage AND Python/process memory count against the same Slurm limit.
# Inspect every visible cgroup ancestor, not just the often-unlimited leaf.
memory_preflight() {
  "$PYTHON_BIN" - "$RAM_BASE" "$1" "$2" <<'PY'
import os
import re
import shutil
import sys
from pathlib import Path

ram, minimum_gib, phase = sys.argv[1:]
minimum = int(minimum_gib) * 1024**3
budgets = {"tmpfs free": shutil.disk_usage(ram).free}
meminfo = dict(re.findall(r"^(\w+):\s+(\d+)", Path("/proc/meminfo").read_text(), re.M))
if "MemAvailable" in meminfo:
    budgets["node MemAvailable"] = int(meminfo["MemAvailable"]) * 1024

def unescape(value):
    return re.sub(r"\\([0-7]{3})", lambda m: chr(int(m[1], 8)), value)

groups = [line.split(":", 2) for line in Path("/proc/self/cgroup").read_text().splitlines()]
found = False
for line in Path("/proc/self/mountinfo").read_text().splitlines():
    before, after = line.split(" - ", 1)
    fields, fs = before.split(), after.split()
    if fs[0] not in ("cgroup2", "cgroup"):
        continue
    for _, controllers, member in groups:
        if fs[0] == "cgroup2" and controllers:
            continue
        if fs[0] == "cgroup" and ("memory" not in controllers.split(",") or "memory" not in fs[2].split(",")):
            continue
        mount_root, mount = unescape(fields[3]), Path(unescape(fields[4]))
        if member == mount_root:
            relative = ""
        elif mount_root == "/":
            relative = member.lstrip("/")
        elif member.startswith(mount_root.rstrip("/") + "/"):
            relative = member[len(mount_root):].lstrip("/")
        else:
            relative = member.lstrip("/")  # cgroup namespace-relative path
        directory = mount / relative
        limit_name, usage_name = ("memory.max", "memory.current") if fs[0] == "cgroup2" else ("memory.limit_in_bytes", "memory.usage_in_bytes")
        while directory == mount or mount in directory.parents:
            try:
                text = (directory / limit_name).read_text().strip()
                limit = int(text) if text != "max" else None
                current = int((directory / usage_name).read_text())
                if limit is not None and limit < 2**60:
                    found = True
                    budgets[f"cgroup {directory}"] = max(0, limit - current)
            except (OSError, ValueError):
                pass
            if directory == mount:
                break
            directory = directory.parent
if not found:
    print("WARNING: no finite readable cgroup memory limit; node/Slurm checks cannot fully predict OOM", flush=True)
allocation = os.environ.get("SLURM_MEM_PER_NODE", "")
if allocation.isdecimal() and int(allocation) * 1024**2 < minimum:
    raise SystemExit(f"Slurm allocation {allocation} MiB is below {minimum_gib} GiB preflight minimum")
for name, remaining in budgets.items():
    print(f"Memory preflight ({phase}): {name} = {remaining / 1024**3:.1f} GiB", flush=True)
too_small = {key: value for key, value in budgets.items() if value < minimum}
if too_small:
    raise SystemExit(f"Less than {minimum_gib} GiB available during {phase}; request more RAM/an appropriate tmpfs. No disk fallback.")
PY
}

echo "DyGEnc mode=$RUN_MODE job=$SLURM_JOB_ID host=$(hostname)"
echo "Persistent checkpoints: $RUN_DIR"
echo "Ephemeral data/models/environment: $RAM_ROOT"
echo "RAM thresholds are fail-fast checks, not a proof that the full dataset fits."
memory_preflight "$MIN_RAM_FREE_GB" startup
command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi

# A small source snapshot prevents edits to the persistent checkout mid-job.
# Never copy data, previous checkpoints, git history, or a local environment.
tar --exclude='__pycache__' --exclude='*.pyc' -C "$SOURCE_REPO" \
  -cf - src scripts requirements-server.txt | tar -C "$RAM_REPO" -xf -
export PYTHONPATH="$RAM_REPO"
SOURCE_FINGERPRINT="$("$PYTHON_BIN" - "$RAM_REPO" <<'PY'
import hashlib
import sys
from pathlib import Path
root = Path(sys.argv[1])
files = sorted(root.glob("src/**/*.py")) + sorted(root.glob("scripts/slurm/*.py")) + [root / "requirements-server.txt"]
digest = hashlib.sha256()
for path in sorted(files):
    digest.update(path.relative_to(root).as_posix().encode() + b"\0")
    digest.update(hashlib.sha256(path.read_bytes()).digest())
print(digest.hexdigest())
PY
)"
if [[ "$RUN_MODE" == resume ]]; then
  [[ "$(tr -d '[:space:]' < "$RUN_DIR/source-fingerprint.sha256")" == "$SOURCE_FINGERPRINT" ]] || die "Training/preprocess code or pinned requirements changed; resume with the original checkout"
else
  printf '%s\n' "$SOURCE_FINGERPRINT" > "$RUN_DIR/source-fingerprint.sha256"
fi
export DYGENC_SOURCE_FINGERPRINT="$SOURCE_FINGERPRINT"

VENV_DIR="$RAM_ROOT/venv"
if ! run_child bootstrap "$PYTHON_BIN" -m venv "$VENV_DIR"; then
  (( STOP_STATUS == 0 )) || exit "$STOP_STATUS"
  echo "stdlib venv unavailable; bootstrapping virtualenv entirely inside RAM"
  run_child bootstrap "$PYTHON_BIN" - "$VIRTUALENV_PYZ_URL" "$RAM_ROOT/virtualenv.pyz" <<'PY'
import shutil
import sys
import urllib.request
url, destination = sys.argv[1:]
if not url.startswith("https://"):
    raise SystemExit("VIRTUALENV_PYZ_URL must use HTTPS")
with urllib.request.urlopen(url, timeout=60) as response, open(destination, "wb") as output:
    shutil.copyfileobj(response, output)
PY
  VENV_DIR="$RAM_ROOT/venv-fallback"
  run_child bootstrap "$PYTHON_BIN" "$RAM_ROOT/virtualenv.pyz" --app-data "$VIRTUALENV_OVERRIDE_APP_DATA" "$VENV_DIR"
fi
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
PIP_ARGS=(--no-cache-dir --retries 5 --timeout 60)
run_child install python -m pip install "${PIP_ARGS[@]}" --upgrade pip setuptools wheel
run_child install python -m pip install "${PIP_ARGS[@]}" torch==2.5.0 torchvision==0.20.0 \
  --index-url https://download.pytorch.org/whl/cu121
# Never try a slow source compilation of torch_scatter on the school node.
run_child install python -m pip install "${PIP_ARGS[@]}" --only-binary=:all: 'torch_scatter==2.1.2+pt25cu121' \
  -f https://data.pyg.org/whl/torch-2.5.0+cu121.html
run_child install python -m pip install "${PIP_ARGS[@]}" --prefer-binary -r "$RAM_REPO/requirements-server.txt"
python -m pip check
# A killed atomic save can occupy the reserved third checkpoint slot. Reclaim
# only our validated orphan temp files under the already-held run lock, before
# writing even the small per-job persistent metadata.
run_child checkpoint-cleanup python - "$RUN_DIR" <<'PY'
import sys
from src.utils.server_checkpoint import cleanup_stale_checkpoint_temps
removed = cleanup_stale_checkpoint_temps(sys.argv[1])
if removed:
    print(f"Removed {removed} orphaned checkpoint temporary file(s); final checkpoints unchanged.", flush=True)
PY
python -m pip freeze > "$RUN_DIR/pip-freeze-${SLURM_JOB_ID}.txt"
run_child gpu-check python - <<'PY'
import torch
import torch_scatter
from src.model import DyGEnc
if not torch.cuda.is_available():
    raise SystemExit("CUDA unavailable; check allocated GPU and node NVIDIA driver")
if torch.cuda.device_count() != 1:
    raise SystemExit("Expected exactly one Slurm-visible GPU; do not override CUDA_VISIBLE_DEVICES manually")
if not torch.cuda.is_bf16_supported(including_emulation=False):
    raise SystemExit("This DyGEnc configuration requires native BF16 support (T4/P100 are unsupported)")
print("CUDA:", torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))
print("CUDA probe:", torch.ones(1, device="cuda", dtype=torch.bfloat16).sum().item())
PY

# Resolve and save immutable revisions once. Resume never follows a new main.
export HF_TOKEN="$(< "$HF_TOKEN_FILE")"
HF_TOKEN="${HF_TOKEN%$'\r'}"  # Accept one Windows CRLF-terminated line.
[[ -n "$HF_TOKEN" && ! "$HF_TOKEN" =~ [[:space:]] ]] || die "HF_TOKEN_FILE must contain a single non-empty token without whitespace"
export RUN_MODE RUN_DIR LLM_MODEL MBERT_MODEL LLM_REVISION MBERT_REVISION DATASET_REPO DATASET_REVISION
run_child model-download python - <<'PY'
import json
import os
import tempfile
from pathlib import Path
from huggingface_hub import HfApi, snapshot_download

path = Path(os.environ["RUN_DIR"]) / "model-revisions.json"
requested = {
    "llm": (os.environ["LLM_MODEL"], os.environ["LLM_REVISION"]),
    "mbert": (os.environ["MBERT_MODEL"], os.environ["MBERT_REVISION"]),
}
api = HfApi()
dataset_path = Path(os.environ["RUN_DIR"]) / "dataset-source.json"
dataset_repo = os.environ["DATASET_REPO"]
dataset_revision = os.environ["DATASET_REVISION"]
if os.environ["RUN_MODE"] == "resume":
    dataset = json.loads(dataset_path.read_text())
    if dataset["repo_id"] != dataset_repo or (dataset_revision and dataset["revision"] != dataset_revision):
        raise SystemExit("Dataset repository/revision differs from saved run")
else:
    dataset = {"repo_id": dataset_repo,
               "revision": api.dataset_info(dataset_repo, revision=dataset_revision or "main").sha}
    with tempfile.NamedTemporaryFile(mode="w", dir=dataset_path.parent, prefix=".dataset-source-", delete=False) as handle:
        json.dump(dataset, handle, indent=2)
        handle.write("\n")
        dataset_temp = Path(handle.name)
    dataset_temp.replace(dataset_path)
if os.environ["RUN_MODE"] == "resume":
    models = json.loads(path.read_text())
    for name, (repo_id, revision) in requested.items():
        if models[name]["repo_id"] != repo_id:
            raise SystemExit(f"{name} model differs from saved run")
        if revision and revision != models[name]["revision"]:
            raise SystemExit(f"Resume {name} revision must be omitted or equal its saved immutable SHA")
else:
    models = {name: {"repo_id": repo, "revision": api.model_info(repo, revision=revision or "main").sha}
              for name, (repo, revision) in requested.items()}
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, prefix=".model-revisions-", delete=False) as handle:
        json.dump(models, handle, indent=2)
        handle.write("\n")
        temp = Path(handle.name)
    temp.replace(path)
for name, model in models.items():
    print(f"Downloading {name}: {model['repo_id']} @ {model['revision']}", flush=True)
    snapshot_download(repo_id=model["repo_id"], revision=model["revision"], max_workers=2,
                      allow_patterns=["*.json", "*.safetensors", "tokenizer.model", "*.txt", "*.tiktoken"])
PY
DYGENC_LLM_REVISION="$(python -c 'import json,os; print(json.load(open(os.path.join(os.environ["RUN_DIR"],"model-revisions.json")))["llm"]["revision"])')"
DYGENC_MBERT_REVISION="$(python -c 'import json,os; print(json.load(open(os.path.join(os.environ["RUN_DIR"],"model-revisions.json")))["mbert"]["revision"])')"
export DYGENC_LLM_REVISION DYGENC_MBERT_REVISION
DATASET_REVISION="$(python -c 'import json,os; print(json.load(open(os.path.join(os.environ["RUN_DIR"],"dataset-source.json")))["revision"])')"
export DATASET_REVISION
# Models are complete in RAM: prevent surprise HTTP access during computation.
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
unset HF_TOKEN

run_child data-download python "$RAM_REPO/scripts/slurm/download_agqa.py" \
  --downloads "$DOWNLOAD_DIR" --repo-id "$DATASET_REPO" --revision "$DATASET_REVISION"
STAGE_ARGS=(python "$RAM_REPO/scripts/slurm/stage_agqa.py" --downloads "$DOWNLOAD_DIR" \
  --agqa-root "$AGQA_ROOT" --manifest "$AGQA_ROOT/raw-data-manifest.json")
[[ -z "$ENG_URL" ]] || STAGE_ARGS+=(--eng-url "$ENG_URL")
[[ -z "$ENG_FILE" ]] || STAGE_ARGS+=(--eng-file "$ENG_FILE")
run_child stage "${STAGE_ARGS[@]}"
# The selective downloader removes each verified archive after extraction.
# Remove only leftover archives in this job's downloads after staging succeeds.
run_child trim-archives python - "$RAM_ROOT" "$DOWNLOAD_DIR" <<'PY'
import sys
from pathlib import Path
root, downloads = map(lambda p: Path(p).resolve(), sys.argv[1:])
if downloads.parent != root or downloads.name != "downloads":
    raise SystemExit("Refusing archive cleanup outside this job's downloads")
for path in downloads.rglob("*.zip"):
    if path.is_symlink() or downloads not in path.resolve().parents:
        raise SystemExit("Refusing a symlink/escaped archive during RAM cleanup")
    if path.is_file():
        path.unlink()
PY
# The manifest stores relative logical names and hashes, never mktemp paths.
run_child verify-data python - "$RUN_MODE" "$AGQA_ROOT/raw-data-manifest.json" "$RUN_DIR/raw-data-manifest.json" <<'PY'
import json
import os
import sys
import tempfile
from pathlib import Path
mode, candidate_path, persistent_path = sys.argv[1:]
candidate = Path(candidate_path)
persistent = Path(persistent_path)
if mode == "resume":
    if json.loads(candidate.read_text()) != json.loads(persistent.read_text()):
        raise SystemExit("Raw AGQA/ENG content changed; resume requires exactly the original dataset")
else:
    with tempfile.NamedTemporaryFile(mode="wb", dir=persistent.parent, prefix=".raw-manifest-", delete=False) as handle:
        handle.write(candidate.read_bytes())
        temp = Path(handle.name)
    temp.replace(persistent)
PY
memory_preflight "$MIN_TRAIN_HEADROOM_GB" before-preprocess
cd "$RAM_REPO"
run_child preprocess python -m src.datasets.preprocess.agqa
[[ -s "$AGQA_ROOT/preprocessed_mbert/train/qa_index.sqlite3" && -s "$AGQA_ROOT/preprocessed_mbert/test/qa_index.sqlite3" ]] || die "Preprocessing did not produce both AGQA indexed splits"
memory_preflight "$MIN_TRAIN_HEADROOM_GB" before-train

# Direct child in the Slurm batch allocation: no nested srun signal ambiguity.
# The trainer writes ONLY resumable trainable state + metadata to RUN_DIR.
cd "$RAM_ROOT"
TRAIN_CMD=(python -m src.server_train --run-dir "$RUN_DIR" \
  --data-manifest "$RUN_DIR/raw-data-manifest.json" --epochs "$TARGET_EPOCHS" \
  --checkpoint-every "$CHECKPOINT_EVERY" --seed "$SEED" --llm-model "$LLM_MODEL" \
  --profile "$TRAIN_PROFILE" --accumulation-steps "$ACCUMULATION_STEPS" \
  --loss-reduction "$LOSS_REDUCTION" --stop-after-updates "$STOP_AFTER_UPDATES")
[[ "$RUN_MODE" != resume ]] || TRAIN_CMD+=(--resume "$RUN_DIR/last.pth")
echo "Starting DyGEnc training; checkpoints survive, RAM data/models do not."
run_child train "${TRAIN_CMD[@]}"
echo "DyGEnc job completed. Persistent outputs: $RUN_DIR"
