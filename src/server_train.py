"""AGQA server trainer with bounded persistent output and mid-epoch resume.

Data/model caches belong in the RAM filesystem selected by the launch script.
Only last.pth and best.pth are persisted under --run-dir. A temporary third
checkpoint is required during atomic replacement. Frozen base weights are not
saved: the base revision, initialization seed, source and runtime must match.

This runner deliberately fixes batch_size=1 and workers=0. Accumulation is an
equal-weight mean of per-sample token-mean losses, NOT a global token-weighted
mean. Each sample is backpropagated immediately, so no group of GPU computation
graphs is retained. Checkpoints always follow a complete optimizer update.

CPU regression tests check exact replay. CUDA graph scatter/reduction kernels
may still be nondeterministic; resume guarantees the cursor, state and RNG
continuation, not bitwise equality across GPU operations.
"""

import argparse
from dataclasses import asdict, dataclass
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import random
import re
import signal
import sys

import numpy as np
import torch

from src.utils.server_checkpoint import (
    CHECKPOINT_VERSION, atomic_torch_save, cleanup_stale_checkpoint_temps,
    load_checkpoint, model_snapshot,
    rng_state, to_cpu,
)


PREEMPTED = 75


@dataclass
class TrainState:
    epoch: int = 0  # zero-based next/in-progress epoch
    phase: str = "train"
    train_cursor: int = 0  # next item in the deterministic epoch permutation
    optimizer_steps: int = 0
    train_loss_sum: float = 0.0
    train_samples: int = 0
    val_cursor: int = 0
    val_loss_sum: float = 0.0
    val_samples: int = 0
    best_val_loss: float = float("inf")
    best_epoch: int = -1


class StopFlag:
    """Signal handlers only flip a flag; all saves run on the normal code path."""

    def __init__(self):
        self.requested = False
        self.signum = None
        self.previous = {}

    def __call__(self):
        return self.requested

    def _handle(self, signum, _frame):
        self.requested = True
        self.signum = signum

    def __enter__(self):
        for name in ("SIGTERM", "SIGUSR1"):
            if hasattr(signal, name):
                number = getattr(signal, name)
                self.previous[number] = signal.signal(number, self._handle)
        return self

    def __exit__(self, *_exc):
        for number, previous in self.previous.items():
            signal.signal(number, previous)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def epoch_order(length, seed, epoch):
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed + epoch)
    return torch.randperm(length, generator=generator).tolist()


def learning_rate(base_lr, progress, epochs, warmup_epochs=1.0):
    if progress < warmup_epochs:
        return base_lr * progress / warmup_epochs
    denominator = max(epochs - warmup_epochs, 1e-10)
    return base_lr / 2 + base_lr / 4 * (
        1 + math.cos(math.pi * (progress - warmup_epochs) / denominator)
    )


def _loss(model, batch):
    prepared = model.prepare_train_input(batch)
    loss = model(*prepared)
    if loss.ndim != 0 or not bool(torch.isfinite(loss).item()):
        raise FloatingPointError(f"Expected a finite scalar loss, got {loss}")
    return loss


def _validate_state(state, train_size, val_size, epochs, accumulation_steps):
    if state.phase not in ("train", "validate") or not 0 <= state.epoch <= epochs:
        raise ValueError("Invalid checkpoint epoch/phase")
    if not 0 <= state.train_cursor <= train_size or not 0 <= state.val_cursor <= val_size:
        raise ValueError("Checkpoint cursor is outside dataset")
    if state.train_cursor != state.train_samples or state.val_cursor != state.val_samples:
        raise ValueError("Checkpoint loss counters disagree with sample cursors")
    if state.phase == "train" and state.train_cursor not in (0, train_size):
        if state.train_cursor % accumulation_steps:
            raise ValueError("Checkpoint is not at an optimizer boundary")
    if state.phase == "validate" and state.train_cursor != train_size:
        raise ValueError("Validation checkpoint has incomplete training epoch")


def fit(
    model, optimizer, train_dataset, val_dataset, collate, *, run_dir, signature,
    epochs, seed=18, accumulation_steps=1, checkpoint_every=100, resume=None,
    stop_requested=lambda: False, after_update=None, after_validation_sample=None,
    base_lr=None, warmup_epochs=1.0, max_grad_norm=0.1,
):
    """Return 0 on completion or 75 after a signal-requested safe checkpoint.

    Callbacks are for instrumentation/tests; they must not consume RNG when
    comparing replay. Model and datasets must be constructed before calling
    fit(), because restoring RNG precedes the next sample/model operation.
    Production callers must hold the launcher's exclusive run-directory lock
    until fit() returns, including during orphaned temporary-file cleanup.
    """
    if epochs < 1 or accumulation_steps < 1 or checkpoint_every < 1:
        raise ValueError("epochs, accumulation_steps and checkpoint_every must be positive")
    train_size, val_size = len(train_dataset), len(val_dataset)
    if train_size == 0 or val_size == 0:
        raise ValueError("Both training and validation need at least one eligible sample")
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    last_path, best_path = run_dir / "last.pth", run_dir / "best.pth"
    if last_path.is_symlink() or best_path.is_symlink():
        raise ValueError("Checkpoint destinations must not be symlinks")
    if resume is None and (last_path.exists() or best_path.exists()):
        raise FileExistsError("Run has checkpoints already; use --resume or a new --run-dir")
    removed = cleanup_stale_checkpoint_temps(run_dir)
    if removed:
        print(f"Removed {removed} incomplete checkpoint temp file(s) from a previous killed job.",
              flush=True)
    state = TrainState()
    if resume is not None:
        state = TrainState(**load_checkpoint(resume, signature, model, optimizer))
    _validate_state(state, train_size, val_size, epochs, accumulation_steps)

    def save(path=last_path):
        snapshot = {
            "checkpoint_version": CHECKPOINT_VERSION,
            "signature": signature,
            "state": asdict(state),
            **model_snapshot(model),
            "optimizer": to_cpu(optimizer.state_dict()),
            "rng": to_cpu(rng_state()),
        }
        atomic_torch_save(snapshot, path)
        print(f"checkpoint={path.name} epoch={state.epoch} phase={state.phase} "
              f"train_cursor={state.train_cursor} val_cursor={state.val_cursor} "
              f"updates={state.optimizer_steps}", flush=True)

    def stop_safely():
        if stop_requested():
            save()
            print("Stop requested; safe resume checkpoint saved (exit 75).", flush=True)
            return True
        return False

    if resume is None:
        save()  # also covers a signal before the first optimizer update
    while state.epoch < epochs:
        if stop_safely():
            return PREEMPTED
        if state.phase == "train":
            model.train()
            order = epoch_order(train_size, seed, state.epoch)
            while state.train_cursor < train_size:
                group_end = min(state.train_cursor + accumulation_steps, train_size)
                group_size = group_end - state.train_cursor
                optimizer.zero_grad(set_to_none=True)
                group_loss = 0.0
                for position in range(state.train_cursor, group_end):
                    # No DataLoader worker/iterator RNG or prefetched samples:
                    # resumed execution reads exactly the next unsaved item.
                    batch = collate([train_dataset[order[position]]])
                    loss = _loss(model, batch)
                    group_loss += loss.detach().item()
                    (loss / group_size).backward()
                    del loss, batch
                parameters = [
                    parameter for group in optimizer.param_groups
                    for parameter in group["params"] if parameter.requires_grad
                ]
                torch.nn.utils.clip_grad_norm_(
                    parameters, max_grad_norm, error_if_nonfinite=True,
                )
                if base_lr is not None:
                    lr = learning_rate(
                        base_lr, state.epoch + group_end / train_size, epochs, warmup_epochs,
                    )
                    for group in optimizer.param_groups:
                        group["lr"] = lr
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                state.train_cursor = group_end
                state.train_samples += group_size
                state.train_loss_sum += group_loss
                state.optimizer_steps += 1
                if state.optimizer_steps % checkpoint_every == 0 or group_end == train_size:
                    print(f"epoch={state.epoch + 1}/{epochs} samples={group_end}/{train_size} "
                          f"updates={state.optimizer_steps} loss={group_loss / group_size:.6f}",
                          flush=True)
                if after_update is not None:
                    after_update(state)
                if stop_safely():
                    return PREEMPTED
                if state.optimizer_steps % checkpoint_every == 0:
                    save()
            state.phase = "validate"
            save()  # training done; a restart must not train the epoch twice

        if stop_safely():
            return PREEMPTED
        model.eval()
        with torch.no_grad():
            while state.val_cursor < val_size:
                batch = collate([val_dataset[state.val_cursor]])
                loss = _loss(model, batch)
                state.val_loss_sum += loss.item()
                state.val_samples += 1
                state.val_cursor += 1
                del loss, batch
                if after_validation_sample is not None:
                    after_validation_sample(state)
                if stop_safely():
                    return PREEMPTED
                if state.val_cursor % checkpoint_every == 0:
                    save()
        val_loss = state.val_loss_sum / state.val_samples
        print(f"epoch={state.epoch + 1} train_loss={state.train_loss_sum / state.train_samples:.6f} "
              f"test_split_loss={val_loss:.6f}", flush=True)
        is_best = val_loss < state.best_val_loss
        if is_best:
            state.best_val_loss = val_loss
            state.best_epoch = state.epoch
        state.epoch += 1
        state.phase = "train"
        state.train_cursor = state.train_samples = 0
        state.train_loss_sum = 0.0
        state.val_cursor = state.val_samples = 0
        state.val_loss_sum = 0.0
        if is_best:
            save(best_path)
        save()
    return 0


def read_data_manifest(path):
    with open(path, encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("schema_version") != 1 or not isinstance(manifest.get("files"), dict):
        raise ValueError("Expected data manifest schema_version=1 with a files mapping")
    if not manifest["files"]:
        raise ValueError("Data manifest contains no files")
    for name, entry in manifest["files"].items():
        if Path(name).is_absolute() or ".." in Path(name).parts or "\\" in name:
            raise ValueError("Data manifest names must be relative POSIX paths")
        if not re.fullmatch(r"[0-9a-f]{64}", entry.get("sha256", "")):
            raise ValueError(f"Invalid SHA256 for manifest item {name}")
        if not isinstance(entry.get("size"), int) or entry["size"] < 0:
            raise ValueError(f"Invalid size for manifest item {name}")
    # Ensure signatures cannot acquire absolute RAM paths as incidental metadata.
    if set(manifest) != {"schema_version", "files"}:
        raise ValueError("Unexpected manifest metadata; only relative files may identify data")
    return manifest


def source_fingerprint(source_dir):
    result = {}
    for path in sorted(Path(source_dir).rglob("*.py")):
        result[path.relative_to(source_dir).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def runtime_contract():
    versions = {}
    for name in (
        "torch", "numpy", "transformers", "peft", "torch-geometric", "torch-scatter",
        "tokenizers", "accelerate", "safetensors", "huggingface-hub",
        "scipy", "networkx",
    ):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not-installed"
    return {
        "python": platform.python_version(),
        "packages": versions,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "gpus": [
            {"name": torch.cuda.get_device_name(i),
             "capability": list(torch.cuda.get_device_capability(i))}
            for i in range(torch.cuda.device_count())
        ],
        "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
        "numeric_environment": {
            name: os.environ.get(name) for name in (
                "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "CUBLAS_WORKSPACE_CONFIG",
            )
        },
        "preprocess_seed": os.environ.get("DYGENC_PREPROCESS_SEED", "18"),
        "embed_batch_size": os.environ.get("DYGENC_EMBED_BATCH_SIZE", "unspecified"),
        "lazy_graphs": os.environ.get("DYGENC_LAZY_GRAPHS", "0"),
    }


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--data-manifest", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=1,
                        help="Total target epochs; must remain identical on resume")
    parser.add_argument("--checkpoint-every", type=int, default=100,
                        help="Optimizer update interval (validation: sample interval)")
    parser.add_argument("--resume", type=Path, help="Trusted last.pth from the same run")
    parser.add_argument("--seed", type=int, default=18)
    parser.add_argument("--llm-model", default="meta-llama/Llama-3.2-3B")
    parser.add_argument("--accumulation-steps", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-5)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.epochs < 1 or args.checkpoint_every < 1 or args.accumulation_steps < 1:
        raise ValueError("epochs/checkpoint-every/accumulation-steps must be positive")
    if not 0 <= args.seed < 2**32 or not math.isfinite(args.lr) or args.lr <= 0:
        raise ValueError("seed must fit uint32 and lr must be finite and positive")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("This server runner needs exactly one allocated visible CUDA GPU")
    if not torch.cuda.is_bf16_supported(including_emulation=False):
        raise RuntimeError("The upstream DyGEnc model requires a BF16-capable GPU")
    revisions = {}
    for name in ("DYGENC_LLM_REVISION", "DYGENC_MBERT_REVISION"):
        revision = os.environ.get(name, "")
        if not re.fullmatch(r"[0-9a-f]{40}", revision):
            raise ValueError(f"{name} must be an immutable 40-character Hugging Face commit SHA")
        revisions[name] = revision
    manifest = read_data_manifest(args.data_manifest)
    if not os.environ.get("AGQA_ROOT"):
        raise ValueError("AGQA_ROOT must point to the prepared RAM dataset")
    # The launch script downloads model snapshots before training. These flags
    # make accidental fallback network downloads impossible in this process.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    from src.cfgs.agqa import Config
    from src.datasets.agqa import AGQADataset
    from src.model import DyGEnc
    from src.utils.collate import collate_fn

    cfg = Config()
    cfg.seed = args.seed
    cfg.num_epochs = args.epochs
    cfg.llm_model_path = args.llm_model
    cfg.train_batch_size = cfg.val_batch_size = 1
    cfg.train_num_workers = cfg.val_num_workers = 0
    cfg.accumulation_steps = args.accumulation_steps
    cfg.lr = args.lr
    # Infinity bypasses AGQADataset's empty-grounding filter and can crash it.
    # This finite upper bound retains every non-empty grounding without a cap.
    cfg.train_seq_limit = cfg.val_seq_limit = sys.float_info.max
    signature = {
        "runner_contract": 1,
        "config": asdict(cfg),
        "data": manifest,
        "source": source_fingerprint(Path(__file__).parent),
        "runtime": runtime_contract(),
        "model_revisions": revisions,
        "accumulation": "equal-mean-of-per-sample-token-mean-losses",
    }
    # No validation-driven early stopping. Upstream AGQA only provides train/test;
    # selecting best.pth using test loss is a test-set leak, not an unbiased result.
    print("WARNING: monitoring AGQA test split, as upstream does. best.pth is selected "
          "using test loss; do not report it as unbiased held-out model selection. "
          "No early stopping or sample/sequence cap is applied.", flush=True)
    with StopFlag() as stop:
        seed_everything(args.seed)
        train_data = AGQADataset("train", lm_model="mbert", seq_limit=sys.float_info.max)
        val_data = AGQADataset("test", lm_model="mbert", seq_limit=sys.float_info.max)
        # Keep random initialization of frozen added-token embeddings independent
        # of dataset deserialization work before restoring checkpoint RNG in fit.
        seed_everything(args.seed)
        model = DyGEnc(cfg)
        params = [parameter for parameter in model.parameters() if parameter.requires_grad]
        optimizer = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.wd, betas=(0.9, 0.95))
        signature["dataset_lengths"] = {"train": len(train_data), "test": len(val_data)}
        return fit(
            model, optimizer, train_data, val_data, collate_fn,
            run_dir=args.run_dir, signature=signature, epochs=args.epochs, seed=args.seed,
            accumulation_steps=args.accumulation_steps, checkpoint_every=args.checkpoint_every,
            resume=args.resume, stop_requested=stop, base_lr=cfg.lr,
            warmup_epochs=cfg.warmup_epochs,
        )


if __name__ == "__main__":
    sys.exit(main())
