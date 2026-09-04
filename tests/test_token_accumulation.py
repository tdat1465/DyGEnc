"""Token CE objective and checkpoint continuation; no downloads or CUDA."""
import contextlib
import io
import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F

from src.server_train import PREEMPTED, _weighted_loss, build_parser, fit


class TokenModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.logits = torch.nn.Parameter(torch.tensor([0.3, -0.2, 0.1]))

    def prepare_train_input(self, batch):
        labels = batch.unsqueeze(0)
        return torch.empty(0), torch.ones_like(labels), labels

    def forward(self, embeddings, mask, labels):
        targets = labels[:, 1:].reshape(-1)
        return F.cross_entropy(self.logits.expand(targets.numel(), -1), targets, ignore_index=-100)


class OverflowTokenModel(torch.nn.Module):
    """Finite loss whose scaled gradient overflows until the scale reaches two."""

    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))
        self.register_buffer("successful_training_forwards", torch.tensor(0))
        self.training_calls = 0

    def prepare_train_input(self, batch):
        labels = batch.unsqueeze(0)
        return torch.empty(0), torch.ones_like(labels), labels

    def forward(self, embeddings, mask, labels):
        if self.training:
            self.training_calls += 1
            self.successful_training_forwards.add_(1)
        return self.weight * 1.0e38


def collate(items):
    return items[0]


class TokenAccumulationTests(unittest.TestCase):
    def run_model(self, path, *, stop=0, resume=None, stop_after_validation_samples=0):
        model = TokenModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.2)
        data = [torch.tensor([-100, 0]), torch.tensor([-100, 1, 2, -100]),
                torch.tensor([-100, 2, 2, 1])]
        validation_stop = [False]

        def after_validation_sample(state):
            if stop_after_validation_samples and state.val_cursor >= stop_after_validation_samples:
                validation_stop[0] = True

        with contextlib.redirect_stdout(io.StringIO()):
            result = fit(model, optimizer, data, data, collate, run_dir=path,
                         signature={"tokens": 1}, epochs=2, accumulation_steps=2,
                         checkpoint_every=1, max_grad_norm=100.0,
                         loss_reduction="token_mean", stop_after_updates=stop, resume=resume,
                         stop_requested=lambda: validation_stop[0],
                         after_validation_sample=after_validation_sample)
        return result, model

    def run_scaled_model(self, path, *, stop=0, resume=None):
        model = TokenModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.2)
        scaler = torch.amp.GradScaler(
            "cpu", init_scale=8.0, growth_factor=2.0,
            backoff_factor=0.5, growth_interval=2,
        )
        data = [torch.tensor([-100, 0]), torch.tensor([-100, 1, 2, -100]),
                torch.tensor([-100, 2, 2, 1])]
        with contextlib.redirect_stdout(io.StringIO()):
            result = fit(
                model, optimizer, data, data, collate, run_dir=path,
                signature={"tokens": 1, "precision": "fp16"}, epochs=2,
                accumulation_steps=2, checkpoint_every=1, max_grad_norm=100.0,
                loss_reduction="token_mean", stop_after_updates=stop,
                resume=resume, grad_scaler=scaler,
            )
        return result, model, optimizer, scaler

    def test_unequal_lengths_equal_concatenated_ce_update(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = TokenModel()
            reference = TokenModel()
            data = [torch.tensor([-100, 0]), torch.tensor([-100, 1, -100, 2, 2])]
            target = torch.tensor([0, 1, 2, 2])
            reference_opt = torch.optim.SGD(reference.parameters(), lr=0.2)
            F.cross_entropy(reference.logits.expand(4, -1), target).backward()
            reference_opt.step()
            with contextlib.redirect_stdout(io.StringIO()):
                fit(model, torch.optim.SGD(model.parameters(), lr=0.2), data, data, collate,
                    run_dir=tmp, signature={}, epochs=1, accumulation_steps=32,
                    max_grad_norm=100, loss_reduction="token_mean")
            torch.testing.assert_close(model.logits, reference.logits, rtol=1e-6, atol=1e-7)
            checkpoint = torch.load(Path(tmp) / "last.pth", weights_only=False)
            self.assertEqual(checkpoint["state"]["optimizer_steps"], 1)

    def test_scaled_unequal_lengths_equal_concatenated_ce_update(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = TokenModel()
            reference = TokenModel()
            data = [torch.tensor([-100, 0]), torch.tensor([-100, 1, -100, 2, 2])]
            target = torch.tensor([0, 1, 2, 2])
            reference_opt = torch.optim.SGD(reference.parameters(), lr=0.2)
            F.cross_entropy(reference.logits.expand(4, -1), target).backward()
            reference_opt.step()
            scaler = torch.amp.GradScaler("cpu", init_scale=8.0)
            with contextlib.redirect_stdout(io.StringIO()):
                fit(
                    model, torch.optim.SGD(model.parameters(), lr=0.2),
                    data, data, collate, run_dir=tmp,
                    signature={"precision": "fp16"}, epochs=1,
                    accumulation_steps=32, max_grad_norm=100,
                    loss_reduction="token_mean", grad_scaler=scaler,
                )
            torch.testing.assert_close(model.logits, reference.logits, rtol=1e-6, atol=1e-7)
            checkpoint = torch.load(Path(tmp) / "last.pth", weights_only=False)
            self.assertIsNotNone(checkpoint["grad_scaler"])
            self.assertEqual(checkpoint["state"]["optimizer_steps"], 1)

    def test_probe_resume_preserves_weights_and_weighted_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, full = self.run_model(root / "full")
            status, _ = self.run_model(root / "split", stop=1)
            self.assertEqual(status, PREEMPTED)
            checkpoint = torch.load(root / "split/last.pth", weights_only=False)
            self.assertGreater(checkpoint["state"]["train_loss_weight"], 0)
            _, resumed = self.run_model(root / "split", resume=root / "split/last.pth")
            torch.testing.assert_close(full.logits, resumed.logits, rtol=0, atol=0)
            full_state = torch.load(root / "full/last.pth", weights_only=False)["state"]
            resumed_state = torch.load(root / "split/last.pth", weights_only=False)["state"]
            self.assertEqual(full_state, resumed_state)

    def test_grad_scaler_probe_resume_preserves_full_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            full_status, full, full_optimizer, full_scaler = self.run_scaled_model(
                root / "full",
            )
            self.assertEqual(full_status, 0)
            status, _, _, stopped_scaler = self.run_scaled_model(root / "split", stop=1)
            self.assertEqual(status, PREEMPTED)
            stopped = torch.load(root / "split/last.pth", weights_only=False)
            self.assertEqual(stopped["grad_scaler"], stopped_scaler.state_dict())
            self.assertEqual(stopped["grad_scaler"]["_growth_tracker"], 1)
            resumed_status, resumed, resumed_optimizer, resumed_scaler = self.run_scaled_model(
                root / "split", resume=root / "split/last.pth",
            )
            self.assertEqual(resumed_status, 0)
            torch.testing.assert_close(full.logits, resumed.logits, rtol=0, atol=0)
            self.assertEqual(full_scaler.state_dict(), resumed_scaler.state_dict())
            self.assertEqual(full_optimizer.state_dict(), resumed_optimizer.state_dict())
            full_checkpoint = torch.load(root / "full/last.pth", weights_only=False)
            resumed_checkpoint = torch.load(root / "split/last.pth", weights_only=False)
            self.assertEqual(full_checkpoint["state"], resumed_checkpoint["state"])
            self.assertEqual(full_checkpoint["grad_scaler"], resumed_checkpoint["grad_scaler"])

    def test_grad_scaler_overflow_retries_without_advancing_or_replaying_buffers(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = OverflowTokenModel()
            optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
            scaler = torch.amp.GradScaler(
                "cpu", init_scale=8.0, backoff_factor=0.5, growth_interval=100,
            )
            samples_at_update = []
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                status = fit(
                    model, optimizer, [torch.tensor([-100, 0])],
                    [torch.tensor([-100, 0])], collate, run_dir=tmp,
                    signature={"precision": "fp16-overflow"}, epochs=1,
                    accumulation_steps=1, checkpoint_every=1, max_grad_norm=1.0,
                    loss_reduction="token_mean", grad_scaler=scaler,
                    after_update=lambda state: samples_at_update.append(
                        (state.train_cursor, state.optimizer_steps)
                    ),
                )
            self.assertEqual(status, 0)
            self.assertEqual(model.training_calls, 3)
            self.assertEqual(model.successful_training_forwards.item(), 1)
            self.assertEqual(samples_at_update, [(1, 1)])
            self.assertEqual(scaler.get_scale(), 2.0)
            self.assertIn("retry 2/16", output.getvalue())

    def test_grad_scaler_overflow_retry_limit_is_bounded(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = OverflowTokenModel()
            optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
            scaler = torch.amp.GradScaler("cpu", init_scale=8.0)
            with contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(FloatingPointError, "after 0 retries"):
                    fit(
                        model, optimizer, [torch.tensor([-100, 0])],
                        [torch.tensor([-100, 0])], collate, run_dir=tmp,
                        signature={"precision": "fp16-overflow"}, epochs=1,
                        accumulation_steps=1, checkpoint_every=1,
                        loss_reduction="token_mean", grad_scaler=scaler,
                        max_amp_overflow_retries=0,
                    )
            self.assertEqual(model.successful_training_forwards.item(), 0)
            checkpoint = torch.load(Path(tmp) / "last.pth", weights_only=False)
            self.assertEqual(checkpoint["state"]["train_cursor"], 0)
            self.assertEqual(checkpoint["state"]["optimizer_steps"], 0)

    def test_validation_resume_preserves_weights_and_weighted_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            full_status, full = self.run_model(root / "full")
            self.assertEqual(full_status, 0)
            status, _ = self.run_model(root / "split", stop_after_validation_samples=2)
            self.assertEqual(status, PREEMPTED)
            checkpoint = torch.load(root / "split/last.pth", weights_only=False)
            state = checkpoint["state"]
            self.assertEqual(state["phase"], "validate")
            self.assertEqual(state["epoch"], 0)
            self.assertEqual(state["train_cursor"], 3)
            self.assertEqual(state["train_loss_weight"], 6)
            self.assertEqual(state["val_cursor"], 2)
            self.assertEqual(state["val_samples"], 2)
            # One supervised target in the first sample and two in the second:
            # preserving the sample count alone cannot restore this denominator.
            self.assertEqual(state["val_loss_weight"], 3)
            self.assertGreater(state["val_loss_sum"], 0)
            resumed_status, resumed = self.run_model(
                root / "split", resume=root / "split/last.pth",
            )
            self.assertEqual(resumed_status, 0)
            torch.testing.assert_close(full.logits, resumed.logits, rtol=0, atol=0)
            full_state = torch.load(root / "full/last.pth", weights_only=False)["state"]
            resumed_state = torch.load(root / "split/last.pth", weights_only=False)["state"]
            self.assertEqual(full_state, resumed_state)

    def test_zero_targets_rejected(self):
        with self.assertRaisesRegex(ValueError, "no supervised"):
            _weighted_loss(TokenModel(), torch.tensor([-100, -100]), "token_mean")

    def test_first_label_is_not_counted(self):
        _, weight = _weighted_loss(TokenModel(), torch.tensor([2, 1, -100]), "token_mean")
        self.assertEqual(weight, 1)

    def test_cli_defaults_and_profiles(self):
        args = build_parser().parse_args(["--run-dir", "run", "--data-manifest", "data.json"])
        self.assertEqual((args.epochs, args.accumulation_steps, args.loss_reduction),
                         (5, 32, "token_mean"))
        self.assertEqual(args.stop_after_updates, 0)


if __name__ == "__main__":
    unittest.main()
