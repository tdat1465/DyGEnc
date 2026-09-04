"""Decoder-checkpoint tests without importing the GPU-heavy DyGEnc package.

The optional tiny Llama/LoRA test constructs random weights locally: no Hub
access, tokenizer, pretrained model download, graph package, or GPU is needed.
"""

import ast
import contextlib
import copy
import importlib.util
import os
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


def load_graph_llm_function(name, namespace=None):
    """Execute one helper without importing unrelated GPU-heavy modules."""
    source = Path(__file__).resolve().parents[1] / "src/model/graph_llm.py"
    module = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
    functions = [
        node for node in module.body
        if isinstance(node, ast.FunctionDef)
        and node.name == name
    ]
    if len(functions) != 1:
        raise AssertionError(f"Expected exactly one {name} helper")
    namespace = {} if namespace is None else dict(namespace)
    subset = ast.Module(body=functions, type_ignores=[])
    exec(compile(subset, str(source), "exec"), namespace)
    return namespace[name]


configure_llama_checkpointing = load_graph_llm_function("configure_llama_checkpointing")
resolve_compute_dtype = load_graph_llm_function(
    "resolve_compute_dtype", {"os": os, "torch": torch},
)
resolve_target_only_loss = load_graph_llm_function(
    "resolve_target_only_loss", {"os": os},
)
target_only_causal_loss = load_graph_llm_function(
    "target_only_causal_loss", {"torch": torch, "IGNORE_INDEX": -100},
)
prepare_kbit_spy = mock.Mock(side_effect=lambda model: model)
prepare_llama_base_for_lora = load_graph_llm_function(
    "prepare_llama_base_for_lora",
    {"torch": torch, "prepare_model_for_kbit_training": prepare_kbit_spy},
)


def load_maybe_autocast():
    source = Path(__file__).resolve().parents[1] / "src/model/graph_llm.py"
    module = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
    owner = next(node for node in module.body
                 if isinstance(node, ast.ClassDef) and node.name == "DGMap3d")
    methods = [node for node in owner.body
               if isinstance(node, ast.FunctionDef) and node.name == "maybe_autocast"]
    if len(methods) != 1:
        raise AssertionError("Expected exactly one maybe_autocast method")
    namespace = {"contextlib": contextlib, "torch": torch}
    exec(compile(ast.Module(body=methods, type_ignores=[]), str(source), "exec"), namespace)
    return namespace["maybe_autocast"]


maybe_autocast = load_maybe_autocast()


class ComputeDtypeTests(unittest.TestCase):
    def test_default_and_explicit_compute_dtypes(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertIs(resolve_compute_dtype(), torch.bfloat16)
        for name, expected in (("bf16", torch.bfloat16), ("fp16", torch.float16)):
            with self.subTest(name=name):
                self.assertIs(resolve_compute_dtype(name), expected)

    def test_invalid_compute_dtype_fails_closed(self):
        for value in ("", "BF16", "float16", "fp32", "int8"):
            with self.subTest(value=value), self.assertRaisesRegex(
                    ValueError, "DYGENC_COMPUTE_DTYPE"):
                resolve_compute_dtype(value)

    def test_fp16_freezes_unquantized_base_without_upcasting(self):
        prepare_kbit_spy.reset_mock()
        model = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.Linear(4, 2)).half()
        result = prepare_llama_base_for_lora(model, torch.float16)
        self.assertIs(result, model)
        prepare_kbit_spy.assert_not_called()
        self.assertTrue(all(parameter.dtype == torch.float16 for parameter in model.parameters()))
        self.assertTrue(all(not parameter.requires_grad for parameter in model.parameters()))

    def test_bf16_retains_existing_peft_preparation(self):
        prepare_kbit_spy.reset_mock()
        model = torch.nn.Linear(2, 2).to(torch.bfloat16)
        self.assertIs(prepare_llama_base_for_lora(model, torch.bfloat16), model)
        prepare_kbit_spy.assert_called_once_with(model)

    def test_fp16_rejects_quantized_base(self):
        for attribute, value in (
            ("is_loaded_in_4bit", True),
            ("is_loaded_in_8bit", True),
            ("quantization_method", "gptq"),
        ):
            with self.subTest(attribute=attribute):
                model = torch.nn.Linear(2, 2).half()
                setattr(model, attribute, value)
                with self.assertRaisesRegex(ValueError, "unquantized"):
                    prepare_llama_base_for_lora(model, torch.float16)

    def test_autocast_uses_selected_fp16_dtype(self):
        owner = SimpleNamespace(device=torch.device("cuda"), compute_dtype=torch.float16)
        sentinel = object()
        with mock.patch("torch.cuda.amp.autocast", return_value=sentinel) as autocast:
            self.assertIs(maybe_autocast(owner), sentinel)
        autocast.assert_called_once_with(dtype=torch.float16)

    def test_target_only_loss_flag_is_strict_and_disabled_by_default(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertIs(resolve_target_only_loss(), False)
        self.assertIs(resolve_target_only_loss("0"), False)
        self.assertIs(resolve_target_only_loss("1"), True)
        for value in ("", "true", "yes", "2"):
            with self.subTest(value=value), self.assertRaisesRegex(
                    ValueError, "DYGENC_TARGET_ONLY_LOSS"):
                resolve_target_only_loss(value)


class TargetOnlyValidationTests(unittest.TestCase):
    class Decoder:
        config = SimpleNamespace(vocab_size=11)

        def __call__(self, **_kwargs):
            raise AssertionError("Invalid target-only inputs must fail before decoder execution")

    def test_rejects_non_singleton_microbatch_and_shape_or_dtype_mismatch(self):
        valid_inputs = torch.randn(1, 4, 3)
        valid_mask = torch.ones(1, 4, dtype=torch.long)
        valid_labels = torch.tensor([[-100, -100, 3, 4]], dtype=torch.long)
        cases = (
            (valid_inputs.expand(2, -1, -1), valid_mask.expand(2, -1),
             valid_labels.expand(2, -1), "microbatch size 1"),
            (valid_inputs[:, :-1], valid_mask, valid_labels, "sequence shapes"),
            (valid_inputs, valid_mask, valid_labels.to(torch.int32), "torch.long"),
        )
        for inputs, mask, labels, message in cases:
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                target_only_causal_loss(self.Decoder(), inputs, mask, labels)

    def test_rejects_missing_or_out_of_vocabulary_supervision(self):
        inputs = torch.randn(1, 4, 3)
        mask = torch.ones(1, 4, dtype=torch.long)
        for labels, message in (
            (torch.full((1, 4), -100, dtype=torch.long), "no supervised"),
            (torch.tensor([[-100, -100, 3, 11]]), "outside the model vocabulary"),
        ):
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                target_only_causal_loss(self.Decoder(), inputs, mask, labels)


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
    def test_target_only_matches_full_causal_loss_and_gradients(self):
        from peft import LoraConfig, get_peft_model
        from transformers import LlamaConfig, LlamaForCausalLM

        old_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, old_threads)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(29)
            config = LlamaConfig(
                vocab_size=47, hidden_size=32, intermediate_size=64,
                num_hidden_layers=2, num_attention_heads=4,
                num_key_value_heads=2, max_position_embeddings=64,
                attention_dropout=0.0, use_cache=False,
                pad_token_id=0, bos_token_id=1, eos_token_id=2,
            )
            config._attn_implementation = "eager"
            base = prepare_llama_base_for_lora(
                LlamaForCausalLM(config).to(torch.float16), torch.float16,
            )
            prototype = get_peft_model(base, LoraConfig(
                r=2, lora_alpha=4, lora_dropout=0.0,
                target_modules=["q_proj", "v_proj"],
                bias="none", task_type="CAUSAL_LM",
            ))
            configure_llama_checkpointing(prototype, True)
            input_template = torch.randn(1, 9, config.hidden_size, dtype=torch.float16)
            attention_mask = torch.ones(1, 9, dtype=torch.long)
            # Include an ignored gap to verify that positions, rather than only
            # a contiguous answer suffix, drive target selection.
            labels = torch.tensor([[-100, -100, -100, 5, -100, 7, 8, 9, 10]])

            results = []
            for sparse in (False, True):
                model = copy.deepcopy(prototype)
                inputs = input_template.clone().requires_grad_(True)
                model.train()
                if sparse:
                    loss = target_only_causal_loss(model, inputs, attention_mask, labels)
                else:
                    loss = model(
                        inputs_embeds=inputs, attention_mask=attention_mask,
                        labels=labels, return_dict=True, use_cache=False,
                    ).loss
                loss.backward()
                gradients = {
                    name: parameter.grad.detach().clone()
                    for name, parameter in model.named_parameters()
                    if parameter.requires_grad
                }
                self.assertTrue(gradients)
                self.assertTrue(all(torch.isfinite(gradient).all()
                                    for gradient in gradients.values()))
                self.assertIsNotNone(inputs.grad)
                self.assertTrue(torch.isfinite(inputs.grad).all())
                results.append((loss.detach(), gradients, inputs.grad.detach().clone()))

            full, sparse = results
            torch.testing.assert_close(full[0], sparse[0], rtol=0, atol=0)
            self.assertEqual(full[1].keys(), sparse[1].keys())
            for name in full[1]:
                torch.testing.assert_close(
                    full[1][name], sparse[1][name], rtol=1e-4, atol=1e-5,
                    msg=lambda message, name=name: f"{name}: {message}",
                )
            torch.testing.assert_close(full[2], sparse[2], rtol=1e-4, atol=1e-5)

    def test_fp16_unquantized_lora_keeps_base_small_and_trainable_path_intact(self):
        from peft import LoraConfig, get_peft_model
        from transformers import LlamaConfig, LlamaForCausalLM

        old_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, old_threads)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(13)
            config = LlamaConfig(
                vocab_size=47, hidden_size=32, intermediate_size=64,
                num_hidden_layers=2, num_attention_heads=4,
                num_key_value_heads=2, max_position_embeddings=64,
                attention_dropout=0.1, use_cache=False,
                pad_token_id=0, bos_token_id=1, eos_token_id=2,
            )
            config._attn_implementation = "eager"
            base = LlamaForCausalLM(config).to(torch.float16)
            base = prepare_llama_base_for_lora(base, torch.float16)
            self.assertTrue(all(parameter.dtype == torch.float16
                                for parameter in base.parameters()))
            self.assertTrue(all(not parameter.requires_grad for parameter in base.parameters()))

            model = get_peft_model(base, LoraConfig(
                r=2, lora_alpha=4, lora_dropout=0.25,
                target_modules=["q_proj", "v_proj"],
                bias="none", task_type="CAUSAL_LM",
            ))
            configure_llama_checkpointing(model, True)
            base_parameters = {
                name: parameter for name, parameter in model.named_parameters()
                if "lora_" not in name
            }
            adapter_parameters = {
                name: parameter for name, parameter in model.named_parameters()
                if "lora_" in name
            }
            self.assertTrue(base_parameters)
            self.assertTrue(adapter_parameters)
            self.assertTrue(all(parameter.dtype == torch.float16
                                for parameter in base_parameters.values()))
            self.assertTrue(all(not parameter.requires_grad
                                for parameter in base_parameters.values()))
            self.assertTrue(all(parameter.dtype == torch.float32 and parameter.requires_grad
                                for parameter in adapter_parameters.values()))

            projector = torch.nn.Linear(5, config.hidden_size)
            features = torch.randn(1, 3, 5)
            token_ids = torch.randint(3, config.vocab_size, (1, 6))
            inputs = torch.cat((
                projector(features).to(torch.float16),
                model.get_input_embeddings()(token_ids),
            ), dim=1)
            labels = torch.randint(3, config.vocab_size, (1, inputs.shape[1]))
            labels[:, :3] = -100
            model.train()
            output = model(
                inputs_embeds=inputs,
                attention_mask=torch.ones_like(labels),
                labels=labels, use_cache=False,
            )
            self.assertTrue(torch.isfinite(output.loss))
            output.loss.backward()
            self.assertIsNotNone(projector.weight.grad)
            self.assertTrue(torch.isfinite(projector.weight.grad).all())
            self.assertTrue(all(parameter.grad is None
                                for parameter in base_parameters.values()))
            self.assertTrue(all(parameter.grad is not None and torch.isfinite(parameter.grad).all()
                                for parameter in adapter_parameters.values()))

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
