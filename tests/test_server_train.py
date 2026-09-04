"""CPU-only tests; no HF model, AGQA download or CUDA allocation is needed."""

import contextlib
import io
import json
import os
from pathlib import Path
import random
import tempfile
import unittest
from unittest import mock

try:
    import torch
except ModuleNotFoundError as error:
    if error.name == "torch":
        raise unittest.SkipTest("CPU trainer tests need PyTorch") from error
    raise

import numpy as np

from src.server_train import (
    PREEMPTED, StopFlag, epoch_order, fit, read_data_manifest, runtime_contract,
    seed_everything,
)
from src.utils.server_checkpoint import (
    atomic_torch_save, cleanup_stale_checkpoint_temps, load_checkpoint,
    model_snapshot, restore_model, to_cpu,
)


class ToyDataset:
    def __init__(self, length):
        self.length = length

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        # Exercise all three CPU RNG streams during sample access, not just
        # dropout in the model. The same resume cursor must receive the same data.
        noise = random.random() + float(np.random.random()) + torch.rand(())
        inputs = torch.arange(12, dtype=torch.float32).reshape(3, 4) / 10
        return inputs + index / 20 + noise / 100, torch.full((3, 1), index / 10)


class ToyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.frozen = torch.nn.Linear(4, 4)
        self.frozen.requires_grad_(False)
        self.frozen_alias = self.frozen
        self.norm = torch.nn.BatchNorm1d(4)
        self.dropout = torch.nn.Dropout(0.2)
        self.head = torch.nn.Linear(4, 1)
        self.register_buffer("nonpersistent", torch.tensor(0), persistent=False)

    def prepare_train_input(self, batch):
        return batch

    def forward(self, inputs, targets):
        if self.training:
            self.nonpersistent.add_(1)
        prediction = self.head(self.dropout(self.norm(self.frozen(inputs))))
        return torch.nn.functional.mse_loss(prediction, targets)


def make_model():
    seed_everything(123)
    model = ToyModel()
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=0.002,
    )
    return model, optimizer


def collate(batch):
    return batch[0]


class ServerTrainTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Tiny tensors are faster, and bitwise reproducible, with one CPU thread.
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def assert_nested_equal(self, first, second):
        if torch.is_tensor(first):
            self.assertTrue(torch.equal(first, second), "Tensor state differs after resume")
        elif isinstance(first, np.ndarray):
            np.testing.assert_array_equal(first, second)
        elif isinstance(first, dict):
            self.assertEqual(set(first), set(second))
            for key in first:
                self.assert_nested_equal(first[key], second[key])
        elif isinstance(first, (list, tuple)):
            self.assertEqual(len(first), len(second))
            for left, right in zip(first, second):
                self.assert_nested_equal(left, right)
        else:
            self.assertEqual(first, second)

    def run_toy(self, run_dir, *, resume=None, after_update=None,
                after_validation_sample=None, stop_requested=lambda: False,
                length=7, accumulation=3, signature=None, epochs=2):
        model, optimizer = make_model()
        with contextlib.redirect_stdout(io.StringIO()):
            result = fit(
                model, optimizer, ToyDataset(length), ToyDataset(3), collate,
                run_dir=run_dir, signature=signature or {"toy": 1}, epochs=epochs,
                seed=18, accumulation_steps=accumulation, checkpoint_every=2,
                resume=resume, after_update=after_update,
                after_validation_sample=after_validation_sample,
                stop_requested=stop_requested, base_lr=0.002,
            )
        return result, model, optimizer

    def read(self, path):
        return torch.load(path, map_location="cpu", weights_only=False)

    def test_mid_epoch_resume_matches_uninterrupted(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result, full_model, full_optimizer = self.run_toy(root / "full")
            self.assertEqual(result, 0)
            stop = StopFlag()

            def interrupt(state):
                if state.optimizer_steps == 1:
                    stop.requested = True

            result, _, _ = self.run_toy(
                root / "split", after_update=interrupt, stop_requested=stop,
            )
            self.assertEqual(result, PREEMPTED)
            saved = self.read(root / "split" / "last.pth")
            self.assertEqual(saved["state"]["train_cursor"], 3)
            self.assertEqual(saved["state"]["optimizer_steps"], 1)
            # Consume unrelated RNG between jobs; initialization and load must
            # reconstruct the original frozen weights and restore continuation.
            random.random()
            np.random.random(5)
            torch.rand(5)
            result, resumed_model, resumed_optimizer = self.run_toy(
                root / "split", resume=root / "split" / "last.pth",
            )
            self.assertEqual(result, 0)
            self.assert_nested_equal(full_model.state_dict(), resumed_model.state_dict())
            self.assert_nested_equal(full_optimizer.state_dict(), resumed_optimizer.state_dict())
            self.assert_nested_equal(full_model.nonpersistent, resumed_model.nonpersistent)
            self.assert_nested_equal(
                self.read(root / "full" / "last.pth"),
                self.read(root / "split" / "last.pth"),
            )
            self.assertEqual(sorted(p.name for p in (root / "split").iterdir()),
                             ["best.pth", "last.pth"])

    def test_validation_resume_does_not_repeat_training(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.run_toy(root / "full")
            stop = StopFlag()

            def interrupt(state):
                if state.epoch == 0 and state.val_cursor == 2:
                    stop.requested = True

            result, _, _ = self.run_toy(
                root / "split", after_validation_sample=interrupt, stop_requested=stop,
            )
            self.assertEqual(result, PREEMPTED)
            saved = self.read(root / "split" / "last.pth")
            self.assertEqual(saved["state"]["phase"], "validate")
            self.assertEqual(saved["state"]["train_cursor"], 7)
            self.assertEqual(saved["state"]["val_cursor"], 2)
            self.run_toy(root / "split", resume=root / "split" / "last.pth")
            self.assert_nested_equal(
                self.read(root / "full" / "last.pth"),
                self.read(root / "split" / "last.pth"),
            )

    def test_buffers_saved_and_frozen_aliases_excluded(self):
        model, _ = make_model()
        model.norm.running_mean.fill_(5)
        model.nonpersistent.fill_(8)
        snapshot = model_snapshot(model)
        self.assertIn("norm.running_mean", snapshot["model"])
        self.assertIn("norm.num_batches_tracked", snapshot["model"])
        self.assertIn("nonpersistent", snapshot["model"])
        self.assertFalse(any(name.startswith("frozen") for name in snapshot["model"]))
        model.norm.running_mean.zero_()
        model.nonpersistent.zero_()
        restore_model(model, snapshot)
        self.assertTrue(torch.equal(model.norm.running_mean, torch.full((4,), 5.0)))
        self.assertEqual(model.nonpersistent.item(), 8)

    def test_atomic_failure_keeps_previous_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "last.pth"
            atomic_torch_save({"old": torch.tensor(4)}, path)
            original = path.read_bytes()
            with mock.patch("src.utils.server_checkpoint.torch.save",
                            side_effect=OSError("disk full")):
                with self.assertRaisesRegex(OSError, "disk full"):
                    atomic_torch_save({"new": torch.tensor(6)}, path)
            self.assertEqual(path.read_bytes(), original)
            self.assertEqual([p.name for p in Path(directory).iterdir()], ["last.pth"])
            with mock.patch("src.utils.server_checkpoint.os.replace",
                            side_effect=OSError("replace failed")):
                with self.assertRaisesRegex(OSError, "replace failed"):
                    atomic_torch_save({"new": torch.tensor(6)}, path)
            self.assertEqual(path.read_bytes(), original)

    def test_signature_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.run_toy(root)
            with self.assertRaisesRegex(ValueError, "signature mismatch"):
                self.run_toy(root, resume=root / "last.pth", signature={"toy": 2})

    def test_checkpoint_records_and_enforces_scaler_mode(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.run_toy(root)
            checkpoint = self.read(root / "last.pth")
            self.assertIn("grad_scaler", checkpoint)
            self.assertIsNone(checkpoint["grad_scaler"])
            model, optimizer = make_model()
            scaler = torch.amp.GradScaler("cpu", init_scale=8.0)
            before = {name: value.clone() for name, value in model.state_dict().items()}
            with self.assertRaisesRegex(ValueError, "GradScaler state"):
                load_checkpoint(
                    root / "last.pth", checkpoint["signature"], model, optimizer,
                    grad_scaler=scaler,
                )
            for name, value in model.state_dict().items():
                self.assertTrue(torch.equal(value, before[name]))

    def test_stale_cleanup_only_removes_recognized_temps(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in (".last.pth.dead123.tmp", ".best.pth.ab_cd.tmp", "last.pth",
                         "best.pth", "user.tmp", ".last.pth.tmp", "12.pth"):
                (root / name).write_bytes(b"sentinel")
            self.assertEqual(cleanup_stale_checkpoint_temps(root), 2)
            self.assertEqual(sorted(path.name for path in root.iterdir()),
                             [".last.pth.tmp", "12.pth", "best.pth", "last.pth", "user.tmp"])

    def test_stale_cleanup_rejects_symlinks_without_deleting_target(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "user-file"
            target.write_bytes(b"keep me")
            suspicious = root / ".last.pth.symlink.tmp"
            # Mock the symlink check to exercise the safety gate on Windows,
            # where making a real symlink may require elevated privileges.
            suspicious.write_bytes(b"do not delete")
            original_is_symlink = Path.is_symlink

            def is_symlink(path):
                return path == suspicious or original_is_symlink(path)

            with mock.patch.object(Path, "is_symlink", is_symlink):
                with self.assertRaisesRegex(ValueError, "suspicious"):
                    cleanup_stale_checkpoint_temps(root)
            self.assertEqual(target.read_bytes(), b"keep me")
            self.assertTrue(suspicious.exists())

    def test_one_sample_and_short_accumulation_group(self):
        with tempfile.TemporaryDirectory() as directory:
            result, _, _ = self.run_toy(Path(directory), length=1, accumulation=8, epochs=1)
            self.assertEqual(result, 0)
            checkpoint = self.read(Path(directory) / "last.pth")
            self.assertEqual(checkpoint["state"]["optimizer_steps"], 1)

    def test_empty_dataset_rejected_without_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "at least one"):
                self.run_toy(Path(directory), length=0)
            self.assertFalse((Path(directory) / "last.pth").exists())

    def test_fresh_run_refuses_to_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.run_toy(root)
            with self.assertRaises(FileExistsError):
                self.run_toy(root)

    def test_initial_stop_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result, _, _ = self.run_toy(root, stop_requested=lambda: True)
            self.assertEqual(result, PREEMPTED)
            self.assertEqual(self.read(root / "last.pth")["state"]["optimizer_steps"], 0)
            self.assertEqual(self.run_toy(root, resume=root / "last.pth")[0], 0)

    def test_permutation_does_not_consume_global_rng(self):
        torch.manual_seed(8)
        before = torch.get_rng_state()
        self.assertEqual(epoch_order(15, 18, 2), epoch_order(15, 18, 2))
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        self.assertNotEqual(epoch_order(15, 18, 2), epoch_order(15, 18, 3))

    def test_manifest_independent_of_tmpfs_location(self):
        manifest = {"schema_version": 1, "files": {
            "data/ENG.txt": {"sha256": "a" * 64, "size": 5},
        }}
        with tempfile.TemporaryDirectory() as directory:
            first, second = Path(directory) / "job123.json", Path(directory) / "job456.json"
            first.write_text(json.dumps(manifest), encoding="utf-8")
            second.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            self.assertEqual(read_data_manifest(first), read_data_manifest(second))
            manifest["files"]["../escape"] = {"sha256": "b" * 64, "size": 0}
            second.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "relative POSIX"):
                read_data_manifest(second)

    def test_runtime_contract_fingerprints_kaggle_precision_and_smoke_limits(self):
        environment = {
            "DYGENC_COMPUTE_DTYPE": "fp16",
            "DYGENC_TARGET_ONLY_LOSS": "1",
            "DYGENC_SMOKE_VIDEOS_PER_SPLIT": "4",
            "DYGENC_SMOKE_QA_PER_SPLIT": "32",
        }
        with mock.patch.dict(os.environ, environment):
            contract = runtime_contract()
        self.assertEqual(contract["compute_dtype"], "fp16")
        self.assertIs(contract["amp_grad_scaler"], True)
        self.assertEqual(contract["target_only_loss"], "1")
        self.assertEqual(contract["smoke_limits"], {
            "DYGENC_SMOKE_VIDEOS_PER_SPLIT": "4",
            "DYGENC_SMOKE_QA_PER_SPLIT": "32",
        })

    def test_cpu_snapshot_does_not_alias_live_tensors(self):
        original = {"state": [torch.tensor(3.0)]}
        saved = to_cpu(original)
        original["state"][0].add_(5)
        self.assertEqual(saved["state"][0].item(), 3.0)


if __name__ == "__main__":
    unittest.main()
