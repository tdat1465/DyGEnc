"""Small, atomic resume checkpoints; never serialize the frozen LLM weights.

Only load checkpoints created by a trusted training run: torch's resume format
contains Python/NumPy RNG and optimizer objects and is not safe for untrusted files.
"""

import os
from pathlib import Path
import random
import re
import tempfile

import numpy as np
import torch


CHECKPOINT_VERSION = 1


def to_cpu(value):
    """Detach snapshots from live tensors (including CPU tensors)."""
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: to_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [to_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(to_cpu(item) for item in value)
    return value


def rng_state():
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng(state):
    if len(state["cuda"]) != torch.cuda.device_count():
        raise ValueError("CUDA device count changed; refusing an incompatible resume")
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"]:
        torch.cuda.set_rng_state_all(state["cuda"])


def model_snapshot(model):
    # remove_duplicate=False is important: frozen tied/aliased LLM parameters
    # must not accidentally escape filtering through a second state_dict key.
    parameters = {
        name: parameter for name, parameter in model.named_parameters(remove_duplicate=False)
        if parameter.requires_grad
    }
    buffers = dict(model.named_buffers(remove_duplicate=False))
    return {
        "model": to_cpu({**parameters, **buffers}),
        "trainable_names": sorted(parameters),
        "buffer_names": sorted(buffers),
    }


def restore_model(model, checkpoint):
    parameters = {
        name: parameter for name, parameter in model.named_parameters(remove_duplicate=False)
        if parameter.requires_grad
    }
    buffers = dict(model.named_buffers(remove_duplicate=False))
    if sorted(parameters) != checkpoint["trainable_names"]:
        raise ValueError("Trainable parameter set changed; cannot resume")
    if sorted(buffers) != checkpoint["buffer_names"]:
        raise ValueError("Model buffer set changed; cannot resume")
    target = {**parameters, **buffers}
    if set(target) != set(checkpoint["model"]):
        raise ValueError("Checkpoint tensors do not match its parameter/buffer metadata")
    # Validate everything before modifying any live tensor.
    for name, tensor in target.items():
        saved = checkpoint["model"][name]
        if tensor.shape != saved.shape or tensor.dtype != saved.dtype:
            raise ValueError(f"Shape/dtype mismatch for checkpoint tensor {name}")
    with torch.no_grad():
        for name, tensor in target.items():
            tensor.copy_(checkpoint["model"][name])


def atomic_torch_save(payload, destination):
    """Replace only a completed, flushed snapshot; failed writes keep old last.pth."""
    destination = Path(destination)
    if destination.is_symlink():
        raise ValueError("Checkpoint destination must not be a symlink")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", prefix=f".{destination.name}.", suffix=".tmp",
            dir=destination.parent, delete=False,
        ) as handle:
            temporary = Path(handle.name)
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
        temporary = None
        # Directory fsync provides crash durability on Linux; unsupported on Windows.
        if hasattr(os, "O_DIRECTORY"):
            try:
                directory_fd = os.open(destination.parent, os.O_RDONLY | os.O_DIRECTORY)
                try:
                    os.fsync(directory_fd)
                finally:
                    os.close(directory_fd)
            except OSError:
                # Replacement already succeeded. Some network filesystems do
                # not support directory fsync; this cannot be rolled back.
                pass
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def cleanup_stale_checkpoint_temps(run_dir):
    """Remove only our orphaned atomic-write files under the launcher's run lock.

    SIGKILL cannot run finally blocks. The caller must hold the exclusive run
    lock before invoking this routine/fit; otherwise another active writer's
    temporary checkpoint could be mistaken for an orphan. Unknown files, epoch
    checkpoints and final last/best checkpoints are never removed.
    """
    run_dir = Path(run_dir)
    if run_dir.is_symlink():
        raise ValueError("Checkpoint run directory must not be a symlink")
    recognized = re.compile(r"\.(?:last|best)\.pth\.[A-Za-z0-9_-]+\.tmp\Z")
    entries = [entry for entry in run_dir.iterdir() if recognized.fullmatch(entry.name)]
    # Check the complete set before unlinking anything; never follow symlinks.
    for entry in entries:
        if entry.is_symlink() or not entry.is_file():
            raise ValueError(f"Refusing suspicious checkpoint temporary path: {entry.name}")
    for entry in entries:
        entry.unlink()
    return len(entries)


def load_checkpoint(path, signature, model, optimizer):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("checkpoint_version") != CHECKPOINT_VERSION:
        raise ValueError("Unsupported server checkpoint format")
    if checkpoint.get("signature") != signature:
        raise ValueError(
            "Resume signature mismatch: source, data, model revisions, runtime, "
            "or training settings changed. Start a new run directory instead."
        )
    restore_model(model, checkpoint)
    # PyTorch casts per-parameter optimizer tensors to the parameter device and
    # preserves CPU step counters for non-capturable AdamW. Blindly moving every
    # tensor to CUDA would break that distinction.
    optimizer.load_state_dict(checkpoint["optimizer"])
    restore_rng(checkpoint["rng"])
    return checkpoint["state"]
