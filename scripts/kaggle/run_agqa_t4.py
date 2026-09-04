#!/usr/bin/env python3
"""Run a bounded DyGEnc AGQA smoke test on one Kaggle Tesla T4.

The launcher intentionally imports no Torch code. It creates a small virtual
environment that reuses Kaggle's preinstalled Torch 2.10.0+cu128, pins the
remaining runtime, resolves immutable Hugging Face revisions, preprocesses a
deterministic AGQA subset, and launches the resumable trainer in subprocesses.

The four large raw inputs stay on Kaggle's read-only input mount. Only bounded
preprocessing output, model caches, the environment, logs, and checkpoints are
written below ``--work-root``/``--run-dir``.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import stat
import subprocess
import sys
import tempfile
from urllib.request import Request, urlopen


EXPECTED_PYTHON = (3, 12)
EXPECTED_TORCH = "2.10.0+cu128"
PYG_WHEEL_INDEX = "https://data.pyg.org/whl/torch-2.10.0+cu128.html"
TORCH_SCATTER = "torch_scatter==2.1.2+pt210cu128"
LLM_REPO = "meta-llama/Llama-3.2-3B"
MBERT_REPO = "answerdotai/ModernBERT-large"
ENG_URL = "https://drive.google.com/uc?export=download&id=1d0Gx4x5qnvp13Su_sIS_nlSn47ZggY8n"
ENG_SHA256 = "d35dbc5edfa1f9839e77070160db7d23d19d6c4bfe248b638fd1fce3bbd01b07"
RAW_GROUPS = {
    "AGQA_balanced": ("train_balanced.txt", "test_balanced.txt"),
    "AGQA_scene_graphs": ("AGQA_train_stsgs.pkl", "AGQA_test_stsgs.pkl"),
}
MANIFEST_NAMES = {
    "train_balanced.txt": "data/AGQA_balanced/train_balanced.txt",
    "test_balanced.txt": "data/AGQA_balanced/test_balanced.txt",
    "AGQA_train_stsgs.pkl": "data/AGQA_scene_graphs/AGQA_train_stsgs.pkl",
    "AGQA_test_stsgs.pkl": "data/AGQA_scene_graphs/AGQA_test_stsgs.pkl",
    "ENG.txt": "data/ENG.txt",
}


class KaggleSetupError(RuntimeError):
    """An actionable setup error that never contains credential values."""


def sanitized_child_environment(source: dict[str, str]) -> dict[str, str]:
    """Build a deterministic subprocess environment without forwarding secrets."""
    result = dict(source)
    for name in tuple(result):
        if name.startswith("PIP_") or name in {
            "HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACEHUB_API_TOKEN",
            "PYTHONHOME", "PYTHONPATH", "PYTHONUSERBASE",
        }:
            result.pop(name, None)
    result.update({
        "PIP_CONFIG_FILE": os.devnull,
        "PIP_DISABLE_PIP_VERSION_CHECK": "1",
        "PIP_NO_CACHE_DIR": "1",
        "PYTHONNOUSERSITE": "1",
    })
    return result


@contextmanager
def exclusive_run_lock(run_dir: Path):
    """Prevent two Kaggle cells from mutating one run/checkpoint directory."""
    if not sys.platform.startswith("linux"):
        raise KaggleSetupError("The Kaggle run lock requires Linux.")
    import fcntl

    lock_path = run_dir / ".kaggle-t4.lock"
    flags = os.O_RDWR | os.O_CREAT
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(lock_path, flags, 0o600)
    except OSError as error:
        raise KaggleSetupError(f"Cannot open the run lock: {lock_path}") from error
    handle = os.fdopen(descriptor, "r+", encoding="utf-8")
    try:
        metadata = os.fstat(handle.fileno())
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            raise KaggleSetupError("Run lock must be one regular, non-linked file.")
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise KaggleSetupError(
                "Another Kaggle launcher is already using this run directory."
            ) from error
        handle.seek(0)
        handle.truncate()
        handle.write(f"pid={os.getpid()}\n")
        handle.flush()
        yield
    finally:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()


def _resolved(path: Path | str) -> Path:
    return Path(path).expanduser().resolve()


def _within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", prefix=f".{path.name}.", suffix=".tmp",
            dir=path.parent, delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _sha256(path: Path) -> dict[str, int | str]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
            size += len(chunk)
    return {"sha256": digest.hexdigest(), "size": size}


def _find_group(mounted_root: Path, group: str, names: tuple[str, ...]) -> Path:
    """Find one payload directory while never walking the Charades tree."""
    top = mounted_root / group
    if not top.is_dir():
        raise KaggleSetupError(f"Missing mounted directory: {top}")
    resolved_top = top.resolve()
    if not _within(resolved_top, mounted_root):
        raise KaggleSetupError(f"Mounted group escapes the dataset root: {top}")
    parent_sets = []
    for name in names:
        matches = [path.resolve() for path in top.rglob(name) if path.is_file()]
        if len(matches) != 1:
            raise KaggleSetupError(
                f"Expected exactly one {name} below {top}, found {len(matches)}."
            )
        if not _within(matches[0], resolved_top):
            raise KaggleSetupError(f"Mounted input escapes its dataset group: {name}")
        parent_sets.append(matches[0].parent)
        if matches[0].stat().st_size == 0:
            raise KaggleSetupError(f"Mounted AGQA input is empty: {name}")
    if len(set(parent_sets)) != 1:
        raise KaggleSetupError(f"The required files below {top} are not in one directory.")
    return parent_sets[0]


def discover_raw_inputs(mounted_root: Path) -> tuple[Path, Path, dict[str, Path]]:
    mounted_root = _resolved(mounted_root)
    if not mounted_root.is_dir():
        raise KaggleSetupError(
            f"Kaggle input is missing: {mounted_root}. Add dataset agqa-balanced first."
        )
    balanced = _find_group(mounted_root, "AGQA_balanced", RAW_GROUPS["AGQA_balanced"])
    scene_graphs = _find_group(
        mounted_root, "AGQA_scene_graphs", RAW_GROUPS["AGQA_scene_graphs"],
    )
    files = {name: balanced / name for name in RAW_GROUPS["AGQA_balanced"]}
    files.update({name: scene_graphs / name for name in RAW_GROUPS["AGQA_scene_graphs"]})
    return balanced, scene_graphs, files


def _validate_eng(path: Path) -> Path:
    if not path.is_file() or not 0 < path.stat().st_size <= 8 * 1024 * 1024:
        raise KaggleSetupError("ENG.txt must be a nonempty regular file no larger than 8 MiB.")
    try:
        mapping = json.loads(path.read_text(encoding="utf-8-sig"))
    except (UnicodeError, ValueError) as error:
        raise KaggleSetupError("ENG.txt is not valid UTF-8 JSON.") from error
    if not isinstance(mapping, dict) or not mapping or not all(
        isinstance(key, str) and key and isinstance(value, str) and value
        for key, value in mapping.items()
    ):
        raise KaggleSetupError("ENG.txt must contain a nonempty string-to-string JSON mapping.")
    return path.resolve()


def ensure_eng(work_root: Path, supplied: Path | None) -> Path:
    if supplied is not None:
        return _validate_eng(_resolved(supplied))
    destination = work_root / "inputs" / "ENG.txt"
    if destination.is_file() and _sha256(destination)["sha256"] == ENG_SHA256:
        return _validate_eng(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.download")
    temporary.unlink(missing_ok=True)
    try:
        request = Request(ENG_URL, headers={
            "Cache-Control": "no-store", "User-Agent": "DyGEnc-Kaggle/1",
        })
        with urlopen(request, timeout=60) as response, temporary.open("xb") as output:
            if response.geturl().split(":", 1)[0].lower() != "https":
                raise KaggleSetupError("ENG.txt download redirected away from HTTPS.")
            while chunk := response.read(1024 * 1024):
                output.write(chunk)
                if output.tell() > 8 * 1024 * 1024:
                    raise KaggleSetupError("ENG.txt download exceeded 8 MiB.")
        if _sha256(temporary)["sha256"] != ENG_SHA256:
            raise KaggleSetupError("Downloaded ENG.txt does not match the verified official SHA-256.")
        os.replace(temporary, destination)
    except KaggleSetupError:
        raise
    except Exception as error:
        raise KaggleSetupError(
            "Could not download ENG.txt. Enable Internet or pass --eng-file."
        ) from error
    finally:
        temporary.unlink(missing_ok=True)
    return _validate_eng(destination)


def content_manifest(raw_files: dict[str, Path], eng_file: Path) -> dict:
    sources = {**raw_files, "ENG.txt": eng_file}
    return {
        "schema_version": 1,
        "files": {
            MANIFEST_NAMES[name]: _sha256(sources[name])
            for name in sorted(MANIFEST_NAMES)
        },
    }


def preprocessing_contract(
    repo_root: Path, manifest: dict, mbert_revision: str,
    smoke_videos: int, smoke_qa: int,
) -> dict:
    """Describe every input that can change reusable preprocessing output."""
    source_paths = (
        repo_root / "src" / "datasets" / "preprocess" / "agqa.py",
        repo_root / "src" / "datasets" / "agqa_storage.py",
        repo_root / "src" / "utils" / "lm_modeling.py",
        repo_root / "requirements-kaggle.txt",
        repo_root / "requirements-server.txt",
    )
    return {
        "schema_version": 1,
        "raw_data": manifest,
        "mbert_revision": mbert_revision,
        "smoke_videos_per_split": smoke_videos,
        "smoke_qa_per_split": smoke_qa,
        "preprocess_seed": 18,
        "embed_batch_size": 16,
        "indexed_qa": True,
        "save_networkx": False,
        "source_sha256": {
            path.relative_to(repo_root).as_posix(): _sha256(path)["sha256"]
            for path in source_paths
        },
    }


def preprocessing_root(work_root: Path, contract: dict) -> Path:
    canonical = json.dumps(contract, sort_keys=True, separators=(",", ":")).encode()
    digest = hashlib.sha256(canonical).hexdigest()
    return work_root / "agqa-preprocessed" / digest[:20]


@contextmanager
def _forward_stop_signals(process: subprocess.Popen):
    """Forward notebook interruption to the active child and then keep waiting."""
    previous_handlers = {}

    def forward(_signum, _frame):
        if process.poll() is None:
            try:
                # SIGTERM is a safe-checkpoint request for src.server_train and
                # an ordinary termination request for setup subprocesses.
                process.send_signal(signal.SIGTERM)
            except ProcessLookupError:
                pass

    for name in ("SIGINT", "SIGTERM"):
        if hasattr(signal, name):
            number = getattr(signal, name)
            previous_handlers[number] = signal.signal(number, forward)
    try:
        yield forward
    finally:
        for number, previous in previous_handlers.items():
            signal.signal(number, previous)


def _run(command: list[str], *, env: dict[str, str], cwd: Path | None = None,
         capture: bool = False) -> subprocess.CompletedProcess:
    process = subprocess.Popen(
        command, env=env, cwd=cwd, text=True,
        stdout=subprocess.PIPE if capture else None,
    )
    with _forward_stop_signals(process) as forward:
        try:
            stdout, _ = process.communicate()
        except KeyboardInterrupt:
            forward(None, None)
            stdout, _ = process.communicate()
    result = subprocess.CompletedProcess(command, process.returncode, stdout=stdout)
    if result.returncode:
        raise subprocess.CalledProcessError(
            result.returncode, command, output=result.stdout,
        )
    return result


def ensure_venv(repo_root: Path, work_root: Path, child_env: dict[str, str]) -> Path:
    # Kaggle's managed Python can expose pip through system-site-packages while
    # its bundled ensurepip bootstrap is unavailable/broken.  --without-pip
    # avoids that bootstrap; `venv/bin/python -m pip` then uses the system pip
    # module but installs packages into this venv's own prefix.
    venv = work_root / "venv-no-ensurepip-v1"
    python = venv / "bin" / "python"
    if not python.is_file():
        venv.parent.mkdir(parents=True, exist_ok=True)
        _run(
            [sys.executable, "-m", "venv", "--system-site-packages",
             "--without-pip", str(venv)],
            env=child_env,
        )
    try:
        _run([str(python), "-m", "pip", "--version"], env=child_env, capture=True)
    except subprocess.CalledProcessError as error:
        raise KaggleSetupError(
            "The no-ensurepip venv cannot import Kaggle's system pip. "
            "Start a new Kaggle session or use a new --work-root."
        ) from error
    requirements = repo_root / "requirements-kaggle.txt"
    _run(
        [str(python), "-m", "pip", "--isolated", "install", "--no-cache-dir",
         "-r", str(requirements)],
        env=child_env,
    )
    _run(
        [str(python), "-m", "pip", "--isolated", "install", "--no-cache-dir",
         "--no-index", "--find-links", PYG_WHEEL_INDEX, TORCH_SCATTER],
        env=child_env,
    )
    probe = r"""
import importlib.metadata
import platform
import torch
expected = "2.10.0+cu128"
if platform.python_version_tuple()[:2] != ("3", "12"):
    raise SystemExit(f"Expected Python 3.12, got {platform.python_version()}")
if torch.__version__ != expected or torch.version.cuda != "12.8":
    raise SystemExit(f"Expected Torch {expected}, got {torch.__version__} CUDA {torch.version.cuda}")
if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
    raise SystemExit(f"Expected exactly one child-visible GPU, got {torch.cuda.device_count()}")
name = torch.cuda.get_device_name(0)
capability = torch.cuda.get_device_capability(0)
if "T4" not in name or capability != (7, 5):
    raise SystemExit(f"Expected Tesla T4 compute capability 7.5, got {name} {capability}")
if torch.cuda.is_bf16_supported(including_emulation=False):
    raise SystemExit("This launcher is for a non-native-BF16 T4, not the A100 run")
if importlib.metadata.version("torch-scatter") != "2.1.2+pt210cu128":
    raise SystemExit("The exact Torch 2.10/cu128 torch-scatter wheel is not installed")
print(f"Runtime OK: Python {platform.python_version()} | Torch {torch.__version__} | {name}")
"""
    _run([str(python), "-c", probe], env=child_env)
    return python


def resolve_and_download_models(
    python: Path, run_dir: Path, child_env: dict[str, str], mode: str,
) -> dict:
    revisions_path = run_dir / "model-revisions.json"
    if revisions_path.is_file():
        try:
            revisions = json.loads(revisions_path.read_text(encoding="utf-8"))
        except (UnicodeError, ValueError) as error:
            raise KaggleSetupError("Existing model-revisions.json is invalid.") from error
    elif mode == "resume":
        raise KaggleSetupError("Resume requires run-dir/model-revisions.json.")
    else:
        resolver = r"""
import json
import os
from huggingface_hub import HfApi
token = os.environ.get("HF_TOKEN")
if not token:
    raise SystemExit("Missing HF_TOKEN; enable the Kaggle Secret for this notebook")
api = HfApi(token=token)
print(json.dumps({
    "llm": {"repo_id": "meta-llama/Llama-3.2-3B", "revision": api.model_info("meta-llama/Llama-3.2-3B").sha},
    "mbert": {"repo_id": "answerdotai/ModernBERT-large", "revision": api.model_info("answerdotai/ModernBERT-large").sha},
}))
"""
        result = _run([str(python), "-c", resolver], env=child_env, capture=True)
        revisions = json.loads(result.stdout)
        _atomic_json(revisions_path, revisions)
    expected_repos = {"llm": LLM_REPO, "mbert": MBERT_REPO}
    for key, repo_id in expected_repos.items():
        entry = revisions.get(key, {})
        if entry.get("repo_id") != repo_id or not re.fullmatch(
            r"[0-9a-f]{40}", entry.get("revision", ""),
        ):
            raise KaggleSetupError(f"Invalid immutable {key} model revision metadata.")
    downloader = r"""
import hashlib
import json
import os
from pathlib import Path
import sys
from huggingface_hub import hf_hub_download, snapshot_download
token = os.environ.get("HF_TOKEN")
if not token:
    raise SystemExit("Missing HF_TOKEN; enable the Kaggle Secret for this notebook")
snapshot = Path(snapshot_download(
    repo_id=sys.argv[1], revision=sys.argv[2], token=token, max_workers=2,
    allow_patterns=["*.json", "*.safetensors", "*.model", "*.txt", "*.tiktoken"],
))

# A killed/failed Hub transfer can occasionally leave a corrupt content blob
# that a later snapshot lookup regards as cached. Validate every JSON metadata
# file before going offline and force-download only the damaged object.
for metadata in sorted(snapshot.rglob("*.json")):
    try:
        json.loads(metadata.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        relative = metadata.relative_to(snapshot).as_posix()
        print(f"Repairing corrupt Hub cache metadata: {sys.argv[1]}/{relative}", flush=True)
        repaired = Path(hf_hub_download(
            repo_id=sys.argv[1], filename=relative, revision=sys.argv[2],
            token=token, force_download=True,
        ))
        try:
            json.loads(repaired.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            try:
                damaged = repaired.read_bytes()
                details = (
                    f"size={len(damaged)}, "
                    f"sha256={hashlib.sha256(damaged).hexdigest()}, "
                    f"prefix={damaged[:80]!r}"
                )
            except OSError as detail_error:
                details = f"unreadable ({detail_error.__class__.__name__})"
            raise SystemExit(
                f"Hub metadata remains invalid after HTTPS repair: {relative}; {details}"
            ) from error

root_config = snapshot / "config.json"
if not root_config.is_file():
    raise SystemExit(f"Downloaded model snapshot has no config.json: {sys.argv[1]}")
"""
    for key in ("mbert", "llm"):
        print(f"Caching {key}: {revisions[key]['repo_id']} @ {revisions[key]['revision']}", flush=True)
        _run(
            [str(python), "-c", downloader, revisions[key]["repo_id"], revisions[key]["revision"]],
            env=child_env,
        )
    return revisions


def _training_process(command: list[str], env: dict[str, str], cwd: Path) -> int:
    process = subprocess.Popen(command, env=env, cwd=cwd)
    with _forward_stop_signals(process) as forward:
        try:
            return process.wait()
        except KeyboardInterrupt:
            forward(None, None)
            return process.wait()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mounted-root", type=Path, required=True)
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument("--mode", choices=("fresh", "resume"), default="fresh")
    parser.add_argument("--eng-file", type=Path)
    parser.add_argument("--smoke-videos-per-split", type=int, default=8)
    parser.add_argument("--smoke-qa-per-split", type=int, default=128)
    parser.add_argument("--stop-after-updates", type=int, default=2)
    parser.add_argument("--checkpoint-every", type=int, default=1)
    parser.add_argument("--accumulation-steps", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--profile", choices=("upstream", "full"), default="upstream")
    return parser


def _run_locked(
    args: argparse.Namespace, repo_root: Path, work_root: Path,
    run_dir: Path, mounted_root: Path, child_env: dict[str, str],
    hf_token: str | None,
) -> int:
    if args.mode == "fresh" and any((run_dir / name).exists() for name in ("last.pth", "best.pth")):
        raise KaggleSetupError("Fresh mode found a checkpoint; use --mode resume or a new run-dir.")
    if args.mode == "resume" and not (run_dir / "last.pth").is_file():
        raise KaggleSetupError("Resume requires run-dir/last.pth from your own trusted run.")

    balanced_dir, scene_graph_dir, raw_files = discover_raw_inputs(mounted_root)
    eng_file = ensure_eng(work_root, args.eng_file)
    manifest = content_manifest(raw_files, eng_file)
    manifest_path = run_dir / "raw-data-manifest.json"
    if manifest_path.exists():
        try:
            existing_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (UnicodeError, ValueError) as error:
            raise KaggleSetupError("Existing raw-data-manifest.json is invalid.") from error
        if existing_manifest != manifest:
            raise KaggleSetupError("Mounted AGQA/ENG content differs from the existing run.")
    else:
        _atomic_json(manifest_path, manifest)

    child_env.update({
        "CUDA_VISIBLE_DEVICES": str(args.gpu_index),
        "PYTHONHASHSEED": "18",
        "TOKENIZERS_PARALLELISM": "false",
        "HF_HOME": str(work_root / "huggingface"),
        "HF_HUB_CACHE": str(work_root / "huggingface" / "hub"),
        "HF_HUB_DISABLE_TELEMETRY": "1",
        # Kaggle has intermittently returned corrupt tiny metadata through the
        # hf-xet path. This is set before each Hub subprocess imports the
        # library, so force_download repairs use its regular HTTPS backend.
        "HF_HUB_DISABLE_XET": "1",
        "TRANSFORMERS_NO_ADVISORY_WARNINGS": "1",
    })
    python = ensure_venv(repo_root, work_root, child_env)
    model_env = child_env.copy()
    if hf_token:
        model_env["HF_TOKEN"] = hf_token
    try:
        revisions = resolve_and_download_models(python, run_dir, model_env, args.mode)
    finally:
        model_env.pop("HF_TOKEN", None)

    preprocess_contract = preprocessing_contract(
        repo_root, manifest, revisions["mbert"]["revision"],
        args.smoke_videos_per_split, args.smoke_qa_per_split,
    )
    agqa_root = preprocessing_root(work_root, preprocess_contract)
    completion_path = agqa_root / "preprocessing-contract.json"
    runtime_env = child_env.copy()
    runtime_env.update({
        "AGQA_ROOT": str(agqa_root),
        "AGQA_BALANCED_DIR": str(balanced_dir),
        "AGQA_SCENE_GRAPHS_DIR": str(scene_graph_dir),
        "AGQA_ENG_FILE": str(eng_file),
        "DYGENC_INDEXED_QA": "1",
        "DYGENC_LAZY_GRAPHS": "1",
        "DYGENC_GRAPH_CACHE_SIZE": "2",
        "DYGENC_SAVE_NETWORKX": "0",
        "DYGENC_GRADIENT_CHECKPOINTING": "1",
        "DYGENC_COMPUTE_DTYPE": "fp16",
        "DYGENC_TARGET_ONLY_LOSS": "1",
        "DYGENC_EMBED_BATCH_SIZE": "16",
        "DYGENC_PREPROCESS_SEED": "18",
        "DYGENC_SMOKE_VIDEOS_PER_SPLIT": str(args.smoke_videos_per_split),
        "DYGENC_SMOKE_QA_PER_SPLIT": str(args.smoke_qa_per_split),
        "DYGENC_LLM_REVISION": revisions["llm"]["revision"],
        "DYGENC_MBERT_REVISION": revisions["mbert"]["revision"],
        "OMP_NUM_THREADS": "2",
        "MKL_NUM_THREADS": "2",
        "OPENBLAS_NUM_THREADS": "2",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
    })
    indexes = [
        agqa_root / "preprocessed_mbert" / split / "qa_index.sqlite3"
        for split in ("train", "test")
    ]
    if completion_path.is_file():
        try:
            completed_contract = json.loads(completion_path.read_text(encoding="utf-8"))
        except (UnicodeError, ValueError) as error:
            raise KaggleSetupError("Existing preprocessing completion contract is invalid.") from error
        if completed_contract != preprocess_contract or not all(
            path.is_file() and path.stat().st_size > 0 for path in indexes
        ):
            raise KaggleSetupError(
                "Existing preprocessing output is incomplete or belongs to another contract."
            )
        print("Using existing bounded preprocessing output.", flush=True)
    else:
        print("Preprocessing bounded AGQA smoke subset...", flush=True)
        _run(
            [str(python), "-m", "src.datasets.preprocess.agqa"],
            env=runtime_env, cwd=repo_root,
        )
        if not all(path.is_file() and path.stat().st_size > 0 for path in indexes):
            raise KaggleSetupError("Preprocessing did not produce both complete QA indexes.")
        _atomic_json(completion_path, preprocess_contract)

    freeze = _run(
        [str(python), "-m", "pip", "freeze", "--all"], env=runtime_env, capture=True,
    ).stdout
    (run_dir / "pip-freeze.txt").write_text(freeze, encoding="utf-8")

    command = [
        str(python), "-m", "src.server_train",
        "--run-dir", str(run_dir),
        "--data-manifest", str(manifest_path),
        "--epochs", str(args.epochs),
        "--checkpoint-every", str(args.checkpoint_every),
        "--seed", "18",
        "--llm-model", LLM_REPO,
        "--profile", args.profile,
        "--accumulation-steps", str(args.accumulation_steps),
        "--loss-reduction", "token_mean",
        "--stop-after-updates", str(args.stop_after_updates),
    ]
    if args.mode == "resume":
        command.extend(("--resume", str(run_dir / "last.pth")))
    print(
        f"Starting {args.mode} T4 smoke run: {args.smoke_videos_per_split} videos and "
        f"at most {args.smoke_qa_per_split} QA/split; FP16; GPU index {args.gpu_index}.",
        flush=True,
    )
    return _training_process(command, runtime_env, repo_root)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    for name in (
        "smoke_videos_per_split", "smoke_qa_per_split", "checkpoint_every",
        "accumulation_steps", "epochs",
    ):
        if getattr(args, name) < 1:
            raise KaggleSetupError(f"--{name.replace('_', '-')} must be positive.")
    if args.stop_after_updates < 0 or args.gpu_index < 0:
        raise KaggleSetupError("GPU index and stop-after-updates must be nonnegative.")
    if sys.version_info[:2] != EXPECTED_PYTHON:
        raise KaggleSetupError(
            f"This branch targets Kaggle Python 3.12; launcher is {sys.version.split()[0]}."
        )
    if not sys.platform.startswith("linux"):
        raise KaggleSetupError("The Kaggle T4 launcher requires Linux.")

    # Remove the notebook secret from this process before any dependency setup.
    # A short-lived copy is added only to the two Hugging Face download calls.
    hf_token = os.environ.pop("HF_TOKEN", None)
    child_env = sanitized_child_environment(os.environ)
    repo_root = Path(__file__).resolve().parents[2]
    work_root = _resolved(args.work_root)
    run_dir = _resolved(args.run_dir)
    mounted_root = _resolved(args.mounted_root)
    if _within(work_root, mounted_root) or _within(run_dir, mounted_root):
        raise KaggleSetupError("work-root and run-dir must be writable paths outside Kaggle Input.")
    work_root.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=True)
    with exclusive_run_lock(run_dir):
        return _run_locked(
            args, repo_root, work_root, run_dir, mounted_root, child_env, hf_token,
        )


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (KaggleSetupError, OSError, subprocess.CalledProcessError) as error:
        print(f"Kaggle T4 run failed: {error}", file=sys.stderr)
        raise SystemExit(2)
