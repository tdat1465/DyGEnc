"""Static launch contracts; never submit a job or execute shell commands."""

import ast
from pathlib import Path
import re
import shlex
import unittest


SLURM = Path(__file__).resolve().parents[1] / "scripts/slurm"
WRAPPERS = (
    "train_agqa_full_ram.slurm", "resume_agqa_full_ram.slurm",
    "train_agqa_a100_90g.slurm", "resume_agqa_a100_90g.slurm",
)


def logical_lines(text):
    """Ignore formatting-only shell continuations, not shell semantics."""
    return re.sub(r"\\\r?\n", "", text).splitlines()


def sbatch_options(text):
    options = {}
    for line in text.splitlines():
        match = re.match(r"\s*#SBATCH\s+(.+)", line)
        if not match:
            continue
        tokens = shlex.split(match.group(1), comments=True)
        position = 0
        while position < len(tokens):
            token = tokens[position]
            if "=" in token:
                name, value = token.split("=", 1)
            elif position + 1 < len(tokens) and not tokens[position + 1].startswith("-"):
                name, value = token, tokens[position + 1]
                position += 1
            else:
                name, value = token, None
            options[name] = value
            position += 1
    return options


def exported_assignments(text):
    assignments = {}
    for line in logical_lines(text):
        if not re.match(r"\s*export\s+", line):
            continue
        for token in shlex.split(line, comments=True)[1:]:
            if "=" in token:
                name, value = token.split("=", 1)
                assignments[name] = value
    return assignments


def command_tokens(text, prefix):
    for line in logical_lines(text):
        if re.match(r"\s*" + re.escape(prefix) + r"(?:\s|$)", line):
            return shlex.split(line, comments=True)
    raise AssertionError(f"Missing shell command: {prefix}")


def training_command(text):
    for line in logical_lines(text):
        match = re.match(r"\s*TRAIN_CMD=\((.*)\)\s*(?:#.*)?$", line)
        if match:
            return shlex.split(match.group(1), comments=True)
    raise AssertionError("Missing TRAIN_CMD array")


class SlurmContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.wrappers = {name: (SLURM / name).read_text(encoding="utf-8") for name in WRAPPERS}
        cls.job = (SLURM / "agqa_full_job.sh").read_text(encoding="utf-8")

    def test_all_fresh_resume_wrappers_request_under_100_decimal_gb_and_one_gpu(self):
        for name, text in self.wrappers.items():
            with self.subTest(wrapper=name):
                options = sbatch_options(text)
                self.assertEqual(options.get("--mem", "").upper(), "90G")
                self.assertEqual(options.get("--gres"), "gpu:1")
                self.assertEqual(options.get("--nodes"), "1")
                self.assertEqual(options.get("--ntasks"), "1")
        self.assertLess(90 * 1024**3, 100 * 1000**3)

    def test_a100_pair_has_same_resume_critical_defaults(self):
        fresh = exported_assignments(self.wrappers["train_agqa_a100_90g.slurm"])
        resume = exported_assignments(self.wrappers["resume_agqa_a100_90g.slurm"])
        expected_defaults = {
            "PERSIST_ROOT": "/media02/lnthanh03/nnthao21",
            "RUN_DIR": "$PERSIST_ROOT/runs/dygenc/a100_upstream_90g",
            "TRAIN_PROFILE": "upstream",
            "TARGET_EPOCHS": "5",
            "ACCUMULATION_STEPS": "32",
        }
        for variable, default in expected_defaults.items():
            with self.subTest(variable=variable):
                self.assertIn(variable, fresh)
                self.assertEqual(fresh[variable], resume[variable])
                self.assertEqual(fresh[variable], "${" + variable + ":-" + default + "}")
        self.assertEqual(fresh["SOURCE_REPO"], resume["SOURCE_REPO"])

    def test_a100_pair_dispatches_to_corresponding_fresh_resume_wrapper(self):
        for wrapper, child in (
            ("train_agqa_a100_90g.slurm", "train_agqa_full_ram.slurm"),
            ("resume_agqa_a100_90g.slurm", "resume_agqa_full_ram.slurm"),
        ):
            with self.subTest(wrapper=wrapper):
                tokens = command_tokens(self.wrappers[wrapper], "exec bash")
                self.assertEqual(tokens[2:], ["$SOURCE_REPO/scripts/slurm/" + child])

    def test_common_job_forwards_training_and_probe_controls(self):
        tokens = training_command(self.job)
        self.assertEqual(tokens[:3], ["python", "-m", "src.server_train"])
        for option, variable in (
            ("--profile", "TRAIN_PROFILE"),
            ("--epochs", "TARGET_EPOCHS"),
            ("--accumulation-steps", "ACCUMULATION_STEPS"),
            ("--loss-reduction", "LOSS_REDUCTION"),
            ("--stop-after-updates", "STOP_AFTER_UPDATES"),
        ):
            with self.subTest(option=option):
                self.assertEqual(tokens.count(option), 1)
                self.assertEqual(tokens[tokens.index(option) + 1], "$" + variable)

    def test_common_job_enables_indexed_qa_and_decoder_checkpointing(self):
        exports = exported_assignments(self.job)
        self.assertEqual(exports.get("DYGENC_INDEXED_QA"), "1")
        self.assertEqual(exports.get("DYGENC_GRADIENT_CHECKPOINTING"), "1")
        self.assertEqual(exports.get("DYGENC_LAZY_GRAPHS"), "1")
        self.assertEqual(exports.get("DYGENC_SAVE_NETWORKX"), "0")

    def test_selective_hf_downloader_replaces_kaggle_cli(self):
        tokens = command_tokens(self.job, "run_child data-download")
        self.assertEqual(tokens[:3], ["run_child", "data-download", "python"])
        self.assertEqual(tokens[3], "$RAM_REPO/scripts/slurm/download_agqa.py")
        for option, variable in (("--repo-id", "DATASET_REPO"), ("--revision", "DATASET_REVISION")):
            self.assertEqual(tokens[tokens.index(option) + 1], "$" + variable)
        self.assertNotRegex(self.job, r"(?m)^\s*(?:run_child\s+\S+\s+)?kaggle\s+datasets\s+download\b")

    def test_sqlite_temp_directory_tracks_job_ram_tmpdir(self):
        exports = exported_assignments(self.job)
        self.assertEqual(exports.get("TMPDIR"), "$RAM_ROOT/tmp")
        self.assertIn(exports.get("SQLITE_TMPDIR"), ("$TMPDIR", "${TMPDIR}"))

    def test_runtime_fingerprinted_packages_have_exact_requirement_pins(self):
        # Parse without importing the CUDA trainer: a newly fingerprinted
        # dependency must not drift when every resume rebuilds its RAM venv.
        repository = SLURM.parents[1]
        tree = ast.parse((repository / "src/server_train.py").read_text(encoding="utf-8"))
        contract = next(node for node in tree.body
                        if isinstance(node, ast.FunctionDef) and node.name == "runtime_contract")
        package_loop = next(node for node in contract.body
                            if isinstance(node, ast.For) and isinstance(node.target, ast.Name)
                            and node.target.id == "name")
        runtime_packages = set(ast.literal_eval(package_loop.iter))
        requirements = {}
        for line in (repository / "requirements-server.txt").read_text(encoding="utf-8").splitlines():
            line = line.partition("#")[0].strip()
            if not line:
                continue
            match = re.fullmatch(r"([A-Za-z0-9_.-]+)==([0-9][A-Za-z0-9.+!-]*)", line)
            self.assertIsNotNone(match, f"Requirement must use an exact, non-wildcard pin: {line}")
            package, version = match.groups()
            package = re.sub(r"[-_.]+", "-", package).lower()
            self.assertNotIn(package, requirements, f"Duplicate requirement: {package}")
            requirements[package] = version
        separately_installed = {"torch", "torch-scatter"}
        self.assertTrue(separately_installed <= runtime_packages)
        self.assertFalse(separately_installed & requirements.keys())
        for package in sorted(runtime_packages - separately_installed):
            with self.subTest(package=package):
                self.assertIn(package, requirements)
        self.assertEqual(requirements.get("safetensors"), "0.5.3")

    def test_torch_and_scatter_use_exact_matching_cuda_wheel_pins(self):
        installs = [shlex.split(line, comments=True) for line in logical_lines(self.job)
                    if re.match(r"\s*run_child install python -m pip install\s", line)]
        torch_installs = [tokens for tokens in installs
                          if any(token.startswith("torch==") for token in tokens)]
        scatter_installs = [tokens for tokens in installs
                            if any(token.startswith("torch_scatter==") for token in tokens)]
        self.assertEqual(len(torch_installs), 1)
        self.assertEqual(len(scatter_installs), 1)
        torch_tokens, scatter_tokens = torch_installs[0], scatter_installs[0]
        self.assertIn("torch==2.5.0", torch_tokens)
        self.assertIn("torchvision==0.20.0", torch_tokens)
        self.assertEqual(torch_tokens[torch_tokens.index("--index-url") + 1],
                         "https://download.pytorch.org/whl/cu121")
        self.assertIn("torch_scatter==2.1.2+pt25cu121", scatter_tokens)
        self.assertIn("--only-binary=:all:", scatter_tokens)
        self.assertEqual(scatter_tokens[scatter_tokens.index("-f") + 1],
                         "https://data.pyg.org/whl/torch-2.5.0+cu121.html")


if __name__ == "__main__":
    unittest.main()
