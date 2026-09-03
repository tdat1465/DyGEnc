"""Decoder-checkpoint tests without importing the GPU-heavy DyGEnc package.

The optional tiny Llama/LoRA test constructs random weights locally: no Hub
access, tokenizer, pretrained model download, graph package, or GPU is needed.
"""

import ast
import copy
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest import mock

try:
    import torch
except ModuleNotFoundError as error:
    if error.name == "torch":
        raise unittest.SkipTest("Model memory tests need PyTorch") from error
    raise

from torch.utils.checkpoint import checkpoint


def load_checkpoint_helper():
    """Execute only the actual helper, not unrelated model module imports."""
    source = Path(__file__).resolve().parents[1] / "src/model/graph_llm.py"
    module = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
    functions = [
        node for node in module.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "configure_llama_checkpointing"
    ]
    if len(functions) != 1:
        raise AssertionError("Expected exactly one decoder checkpoint helper")
    namespace = {}
    subset = ast.Module(body=functions, type_ignores=[])
    exec(compile(subset, str(source), "exec"), namespace)
    return namespace["configure_llama_checkpointing"]


configure_llama_checkpointing = load_checkpoint_helper()


class CheckpointConfigurationTests(unittest.TestCase):
    def test_enabled_uses_nonreentrant_rng_preserving_checkpoint(self):
        model = SimpleNamespace(
            config=SimpleNamespace(use_cache=True),
            gradient_checkpointing_enable=mock.Mock(),
        )
        configure_llama_checkpointing(model, True)
        model.gradient_checkpointing_enable.assert_called_once_with(
            gradient_checkpointing_kwargs={
                "use_reentrant": False, "preserve_rng_state": True,
            },
        )
        self.assertIs(model.config.use_cache, False)

    def test_disabled_does_not_call_or_change_cache_policy(self):
        for cache_value in (True, False):
            with self.subTest(use_cache=cache_value):
                model = SimpleNamespace(
                    config=SimpleNamespace(use_cache=cache_value),
                    gradient_checkpointing_enable=mock.Mock(),
                )
                configure_llama_checkpointing(model, False)
                model.gradient_checkpointing_enable.assert_not_called()
                self.assertIs(model.config.use_cache, cache_value)


class TinyDecoder(torch.nn.Module):
    """Stateless decoder-like region; deliberately contains no BatchNorm."""

    def __init__(self):
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(5, 9), torch.nn.GELU(),
            torch.nn.Dropout(0.35), torch.nn.Linear(9, 3),
        )
        self.config = SimpleNamespace(use_cache=True)
        self.checkpoint_kwargs = None

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs):
        self.checkpoint_kwargs = dict(gradient_checkpointing_kwargs)

    def forward(self, inputs):
        if self.training and self.checkpoint_kwargs is not None:
            return checkpoint(self.layers, inputs, **self.checkpoint_kwargs)
        return self.layers(inputs)


class CheckpointPrimitiveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def test_dropout_frozen_inputs_and_bf16_replay_match(self):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(7)
            prototype = TinyDecoder()
            for autocast_enabled in (False, True):
                for input_requires_grad in (False, True):
                    with self.subTest(
                        bf16_autocast=autocast_enabled,
                        input_requires_grad=input_requires_grad,
                    ):
                        results = []
                        for enabled in (False, True):
                            model = copy.deepcopy(prototype)
                            configure_llama_checkpointing(model, enabled)
                            inputs = torch.linspace(0, 1, 20).reshape(4, 5)
                            inputs.requires_grad_(input_requires_grad)
                            torch.manual_seed(101)
                            with torch.autocast(
                                "cpu", dtype=torch.bfloat16,
                                enabled=autocast_enabled,
                            ):
                                outputs = model(inputs)
                                loss = outputs.float().square().sum()
                            loss.backward()
                            gradients = []
                            for parameter in model.parameters():
                                self.assertEqual(parameter.dtype, torch.float32)
                                self.assertIsNotNone(parameter.grad)
                                self.assertTrue(torch.isfinite(parameter.grad).all())
                                gradients.append(parameter.grad.clone())
                            input_grad = None if inputs.grad is None else inputs.grad.clone()
                            results.append((
                                outputs.detach(), gradients, input_grad,
                                torch.get_rng_state().clone(),
                            ))
                        normal, recomputed = results
                        self.assertTrue(torch.equal(normal[0], recomputed[0]))
                        for left, right in zip(normal[1], recomputed[1]):
                            self.assertTrue(torch.equal(left, right))
                        if input_requires_grad:
                            self.assertIsNotNone(normal[2])
                            self.assertTrue(torch.equal(normal[2], recomputed[2]))
                            self.assertGreater(normal[2].abs().sum().item(), 0)
                        else:
                            self.assertIsNone(normal[2])
                            self.assertIsNone(recomputed[2])
                        self.assertTrue(torch.equal(normal[3], recomputed[3]))

    def test_external_graph_batchnorm_is_not_recomputed_or_cast(self):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(9)
            graph_encoder = torch.nn.Sequential(
                torch.nn.Linear(4, 5), torch.nn.BatchNorm1d(5),
            )
            decoder = TinyDecoder()
            configure_llama_checkpointing(decoder, True)
            # Match DyGEnc's boundary: graph encoding stays outside the Llama
            # checkpoint region and outside its BF16 autocast context.
            graph_features = graph_encoder(torch.randn(4, 4))
            with torch.autocast("cpu", dtype=torch.bfloat16):
                loss = decoder(graph_features).float().square().sum()
            loss.backward()
            self.assertEqual(graph_encoder[1].num_batches_tracked.item(), 1)
            self.assertEqual(graph_features.dtype, torch.float32)
            for parameter in graph_encoder.parameters():
                self.assertEqual(parameter.dtype, torch.float32)
                self.assertIsNotNone(parameter.grad)
                self.assertTrue(torch.isfinite(parameter.grad).all())


@unittest.skipUnless(
    importlib.util.find_spec("transformers") is not None
    and importlib.util.find_spec("peft") is not None,
    "Optional tiny Llama integration needs transformers and peft",
)
class TinyLlamaIntegrationTests(unittest.TestCase):
    def test_lora_dropout_and_soft_prompt_checkpoint_replay(self):
        # Installed but broken/incompatible dependencies should fail, not be
        # silently reported as a missing-dependency skip.
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from transformers import LlamaConfig, LlamaForCausalLM

        old_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, old_threads)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(17)
            config = LlamaConfig(
                vocab_size=47, hidden_size=32, intermediate_size=64,
                num_hidden_layers=2, num_attention_heads=4,
                num_key_value_heads=2, max_position_embeddings=64,
                attention_dropout=0.1, use_cache=False,
                pad_token_id=0, bos_token_id=1, eos_token_id=2,
            )
            config._attn_implementation = "eager"
            base = LlamaForCausalLM(config).to(torch.bfloat16)
            base = prepare_model_for_kbit_training(base)
            self.assertFalse(base.is_gradient_checkpointing)
            self.assertTrue(all(p.dtype == torch.float32 for p in base.parameters()))
            self.assertTrue(all(not p.requires_grad for p in base.parameters()))
            prototype = get_peft_model(base, LoraConfig(
                r=2, lora_alpha=4, lora_dropout=0.25,
                target_modules=["q_proj", "v_proj"],
                bias="none", task_type="CAUSAL_LM",
            ))
            projector_template = torch.nn.Linear(5, config.hidden_size)
            token_ids = torch.randint(3, config.vocab_size, (1, 6))
            features = torch.randn(1, 3, 5)
            for soft_prompt in (False, True):
                with self.subTest(soft_prompt=soft_prompt):
                    results = []
                    for enabled in (False, True):
                        model = copy.deepcopy(prototype)
                        projector = copy.deepcopy(projector_template)
                        configure_llama_checkpointing(model, enabled)
                        self.assertEqual(model.is_gradient_checkpointing, enabled)
                        frozen = {
                            name: parameter.detach().clone()
                            for name, parameter in model.named_parameters()
                            if not parameter.requires_grad
                        }
                        inputs = model.get_input_embeddings()(token_ids)
                        labels = token_ids.clone()
                        labels[:, :2] = -100
                        if soft_prompt:
                            inputs = torch.cat((projector(features), inputs), dim=1)
                            labels = torch.cat((torch.full((1, 3), -100), labels), dim=1)
                        else:
                            # Some Transformers versions install an embedding
                            # output hook when checkpointing is enabled, even
                            # with frozen embedding weights. Explicitly test
                            # non-reentrant operation with a nondifferentiable
                            # input, rather than depending on that hook policy.
                            # Never detach the soft-prompt branch above.
                            inputs = inputs.detach()
                            self.assertFalse(inputs.requires_grad)
                        torch.manual_seed(113)
                        model.train()
                        with torch.autocast("cpu", dtype=torch.bfloat16):
                            output = model(
                                inputs_embeds=inputs,
                                attention_mask=torch.ones_like(labels),
                                labels=labels, use_cache=False,
                            )
                        self.assertIsNone(output.past_key_values)
                        output.loss.backward()
                        gradients = {}
                        for name, parameter in model.named_parameters():
                            self.assertEqual(parameter.dtype, torch.float32)
                            if parameter.requires_grad:
                                self.assertIsNotNone(parameter.grad, name)
                                self.assertTrue(torch.isfinite(parameter.grad).all(), name)
                                gradients[name] = parameter.grad.clone()
                            else:
                                self.assertIsNone(parameter.grad, name)
                                self.assertTrue(torch.equal(parameter, frozen[name]), name)
                        # LoRA A may have a legitimate zero gradient initially
                        # because LoRA B is initialized to zero.
                        self.assertTrue(any(
                            "lora_B" in name and gradient.abs().sum().item() > 0
                            for name, gradient in gradients.items()
                        ))
                        if soft_prompt:
                            self.assertIsNotNone(projector.weight.grad)
                            self.assertGreater(projector.weight.grad.abs().sum().item(), 0)
                            gradients["soft_prompt_projector"] = projector.weight.grad.clone()
                        results.append((
                            output.loss.detach(), gradients, torch.get_rng_state().clone(),
                        ))
                    normal, recomputed = results
                    torch.testing.assert_close(normal[0], recomputed[0], rtol=0, atol=0)
                    self.assertEqual(normal[1].keys(), recomputed[1].keys())
                    for name in normal[1]:
                        torch.testing.assert_close(
                            normal[1][name], recomputed[1][name], rtol=1e-5, atol=1e-6,
                            msg=lambda message, name=name: f"{name}: {message}",
                        )
                    self.assertTrue(torch.equal(normal[2], recomputed[2]))


if __name__ == "__main__":
    unittest.main()
