import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "kaggle" / "run_agqa_t4.py"
SPEC = importlib.util.spec_from_file_location("run_agqa_t4", SCRIPT)
kaggle = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(kaggle)


class FakeResponse:
    def __init__(self, content):
        self.content = content
        self.offset = 0

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def geturl(self):
        return "https://example.test/ENG.txt"

    def read(self, size=-1):
        if self.offset >= len(self.content):
            return b""
        end = len(self.content) if size < 0 else self.offset + size
        result = self.content[self.offset:end]
        self.offset += len(result)
        return result


class KaggleT4Tests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.mounted = self.root / "mounted"
        self.balanced = self.mounted / "AGQA_balanced" / "AGQA_balanced"
        self.graphs = self.mounted / "AGQA_scene_graphs" / "AGQA_scene_graphs"
        self.balanced.mkdir(parents=True)
        self.graphs.mkdir(parents=True)
        for name in kaggle.RAW_GROUPS["AGQA_balanced"]:
            (self.balanced / name).write_bytes(("qa-" + name).encode())
        for name in kaggle.RAW_GROUPS["AGQA_scene_graphs"]:
            (self.graphs / name).write_bytes(("sg-" + name).encode())

    def tearDown(self):
        self.temp.cleanup()

    def test_discovers_double_nested_layout_without_walking_charades(self):
        charades = self.mounted / "Charades_v1_480" / "Charades_v1_480"
        charades.mkdir(parents=True)
        (charades / "train_balanced.txt").write_bytes(b"decoy")
        balanced, graphs, files = kaggle.discover_raw_inputs(self.mounted)
        self.assertEqual(balanced, self.balanced.resolve())
        self.assertEqual(graphs, self.graphs.resolve())
        self.assertEqual(set(files), {
            "train_balanced.txt", "test_balanced.txt",
            "AGQA_train_stsgs.pkl", "AGQA_test_stsgs.pkl",
        })

    def test_duplicate_required_file_is_rejected(self):
        duplicate = self.mounted / "AGQA_balanced" / "duplicate"
        duplicate.mkdir()
        (duplicate / "train_balanced.txt").write_bytes(b"duplicate")
        with self.assertRaisesRegex(kaggle.KaggleSetupError, "exactly one"):
            kaggle.discover_raw_inputs(self.mounted)

    def test_manifest_has_server_schema_and_logical_names(self):
        _, _, files = kaggle.discover_raw_inputs(self.mounted)
        eng = self.root / "ENG.txt"
        eng.write_text('{"o1":"person"}', encoding="utf-8")
        manifest = kaggle.content_manifest(files, eng)
        self.assertEqual(set(manifest), {"schema_version", "files"})
        self.assertEqual(manifest["schema_version"], 1)
        self.assertEqual(set(manifest["files"]), set(kaggle.MANIFEST_NAMES.values()))
        for entry in manifest["files"].values():
            self.assertRegex(entry["sha256"], r"^[0-9a-f]{64}$")
            self.assertGreater(entry["size"], 0)

    def test_supplied_eng_is_validated_without_downloading(self):
        eng = self.root / "ENG.txt"
        eng.write_text('{"o1":"person"}', encoding="utf-8")
        with mock.patch.object(kaggle, "urlopen", side_effect=AssertionError("no network")):
            self.assertEqual(kaggle.ensure_eng(self.root / "work", eng), eng.resolve())

    def test_default_eng_requires_verified_digest(self):
        content = b'{"o1":"person"}'
        digest = hashlib.sha256(content).hexdigest()
        with mock.patch.object(kaggle, "ENG_SHA256", digest), mock.patch.object(
            kaggle, "urlopen", return_value=FakeResponse(content),
        ):
            result = kaggle.ensure_eng(self.root / "work", None)
        self.assertEqual(result.read_bytes(), content)

    def test_invalid_eng_mapping_is_rejected(self):
        eng = self.root / "ENG.txt"
        for content in ("{}", "[]", '{"o1":1}', '{"":"person"}'):
            with self.subTest(content=content):
                eng.write_text(content, encoding="utf-8")
                with self.assertRaises(kaggle.KaggleSetupError):
                    kaggle.ensure_eng(self.root / "work", eng)

    def test_smoke_defaults_are_bounded(self):
        args = kaggle.build_parser().parse_args([
            "--mounted-root", "input", "--work-root", "work", "--run-dir", "run",
        ])
        self.assertEqual(args.smoke_videos_per_split, 8)
        self.assertEqual(args.smoke_qa_per_split, 128)
        self.assertEqual(args.stop_after_updates, 2)
        self.assertEqual(args.accumulation_steps, 32)
        self.assertEqual(args.profile, "upstream")

    def test_child_environment_drops_token_and_python_pip_injection(self):
        environment = kaggle.sanitized_child_environment({
            "HF_TOKEN": "must-not-leak",
            "HUGGING_FACE_HUB_TOKEN": "must-not-leak",
            "PIP_INDEX_URL": "https://credentials.invalid/simple",
            "PYTHONPATH": "/untrusted/modules",
            "KEEP_ME": "yes",
        })
        self.assertEqual(environment["KEEP_ME"], "yes")
        self.assertEqual(environment["PIP_CONFIG_FILE"], os.devnull)
        self.assertEqual(environment["PYTHONNOUSERSITE"], "1")
        self.assertNotIn("HF_TOKEN", environment)
        self.assertNotIn("HUGGING_FACE_HUB_TOKEN", environment)
        self.assertNotIn("PIP_INDEX_URL", environment)
        self.assertNotIn("PYTHONPATH", environment)

    def test_venv_bypasses_kaggle_ensurepip(self):
        repo = Path(__file__).resolve().parents[1]
        work = self.root / "work"
        with mock.patch.object(kaggle, "_run", return_value=mock.Mock(stdout="")) as run:
            python = kaggle.ensure_venv(repo, work, {"SAFE": "1"})
        create_command = run.call_args_list[0].args[0]
        self.assertEqual(create_command[:3], [sys.executable, "-m", "venv"])
        self.assertIn("--system-site-packages", create_command)
        self.assertIn("--without-pip", create_command)
        self.assertEqual(python, work / "venv-no-ensurepip-v1" / "bin" / "python")

    @unittest.skipUnless(
        sys.platform.startswith("linux") and importlib.util.find_spec("fcntl"),
        "flock contention test requires Linux",
    )
    def test_run_directory_lock_rejects_concurrent_launcher(self):
        run_dir = self.root / "run"
        run_dir.mkdir()
        with kaggle.exclusive_run_lock(run_dir):
            with self.assertRaisesRegex(kaggle.KaggleSetupError, "already using"):
                with kaggle.exclusive_run_lock(run_dir):
                    self.fail("A second launcher acquired the same run lock")

    def test_preprocessing_root_changes_with_every_material_input(self):
        repo = Path(__file__).resolve().parents[1]
        manifest = {"schema_version": 1, "files": {"x": {"sha256": "a" * 64, "size": 1}}}
        base = kaggle.preprocessing_contract(repo, manifest, "b" * 40, 8, 128)
        root = kaggle.preprocessing_root(self.root, base)
        for key, replacement in (
            ("mbert_revision", "c" * 40),
            ("smoke_videos_per_split", 9),
            ("smoke_qa_per_split", 127),
            ("embed_batch_size", 8),
            ("save_networkx", True),
        ):
            changed = dict(base)
            changed[key] = replacement
            with self.subTest(key=key):
                self.assertNotEqual(kaggle.preprocessing_root(self.root, changed), root)

    def test_preprocessing_contract_hashes_sources_and_raw_manifest(self):
        repo = Path(__file__).resolve().parents[1]
        manifest = {"schema_version": 1, "files": {"x": {"sha256": "a" * 64, "size": 1}}}
        contract = kaggle.preprocessing_contract(repo, manifest, "b" * 40, 8, 128)
        self.assertIs(contract["raw_data"], manifest)
        self.assertIn("src/datasets/preprocess/agqa.py", contract["source_sha256"])
        self.assertIn("requirements-kaggle.txt", contract["source_sha256"])
        self.assertTrue(all(
            len(value) == 64 for value in contract["source_sha256"].values()
        ))

    def test_launcher_contract_does_not_copy_large_raw_inputs(self):
        source = SCRIPT.read_text(encoding="utf-8")
        self.assertNotIn("import torch", source.split("EXPECTED_PYTHON", 1)[0])
        self.assertNotIn("copyfile(", source)
        self.assertNotIn("copy2(", source)
        for name in (
            "AGQA_BALANCED_DIR", "AGQA_SCENE_GRAPHS_DIR", "AGQA_ENG_FILE",
            "DYGENC_SMOKE_VIDEOS_PER_SPLIT", "DYGENC_SMOKE_QA_PER_SPLIT",
            '"DYGENC_COMPUTE_DTYPE": "fp16"',
            '"DYGENC_TARGET_ONLY_LOSS": "1"',
            "torch_scatter==2.1.2+pt210cu128",
            "torch-2.10.0+cu128.html",
            "exclusive_run_lock(run_dir)",
            "Repairing corrupt Hub cache metadata",
            "force_download=True",
            '"HF_HUB_DISABLE_XET": "1"',
            "Hub metadata remains invalid after HTTPS repair",
        ):
            self.assertIn(name, source)


if __name__ == "__main__":
    unittest.main()
