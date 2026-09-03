"""Small RAM-loader/preprocessing fixtures; no Hugging Face/GPU downloads."""
import ast
import json
import os
import pickle
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "PyTorch is required for the dataset fixture")
class LazyGraphTests(unittest.TestCase):
    def test_lazy_and_eager_samples_match_and_cache_is_bounded(self):
        from src.datasets.agqa import AGQADataset

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            processed = root / "preprocessed_mbert" / "train"
            (processed / "graphs").mkdir(parents=True)
            (processed / "descs").mkdir()
            (root / "data" / "AGQA_balanced").mkdir(parents=True)
            interval = ("000001", "000002")
            qa, grounding = {}, {}
            for i in range(3):
                qa[str(i)] = dict(video_id=f"video{i}", question="what?", answer="YES",
                                 ans_type="binary", global_="exists", semantic="obj", structural="query")
                qa[str(i)]["global"] = qa[str(i)].pop("global_")
                grounding[str(i)] = [interval]
                torch.save({interval: torch.tensor([i])}, processed / "graphs" / f"video{i}.pt")
                with (processed / "descs" / f"video{i}.pkl").open("wb") as out:
                    pickle.dump({interval: f"description {i}"}, out)
            with (processed / "qa2sg.pkl").open("wb") as out:
                pickle.dump(grounding, out)
            (root / "data" / "AGQA_balanced" / "train_balanced.txt").write_text(json.dumps(qa))
            with patch.dict(os.environ, {"AGQA_ROOT": tmp, "DYGENC_LAZY_GRAPHS": "0"}):
                eager = AGQADataset("train", seq_limit=100)
            with patch.dict(os.environ, {"AGQA_ROOT": tmp, "DYGENC_LAZY_GRAPHS": "1",
                                         "DYGENC_GRAPH_CACHE_SIZE": "1"}):
                lazy = AGQADataset("train", seq_limit=100)
            for i in range(3):
                a, b = eager[i], lazy[i]
                self.assertTrue(torch.equal(a.pop("graphs")[0], b.pop("graphs")[0]))
                self.assertEqual(a, b)
                self.assertEqual(len(lazy.video_cache), 1)
            self.assertNotIn("graphs", vars(lazy))


class GroundingRangeTests(unittest.TestCase):
    def test_grounding_uses_interval_keys_without_retaining_graph_tensors(self):
        # Extract this pure-data function to avoid the legacy module's eager
        # ModernBERT initialization at import time.
        source = Path(__file__).resolve().parents[1] / "src/datasets/preprocess/agqa.py"
        tree = ast.parse(source.read_text(encoding="utf-8"))
        functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                     and node.name in ("load_grounding_frames", "preprocess_qa")]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "data" / "AGQA_balanced").mkdir(parents=True)
            qa = {"q1": {"video_id": "video", "sg_grounding": {}}}
            for split in ("train", "test"):
                (root / "data" / "AGQA_balanced" / f"{split}_balanced.txt").write_text(json.dumps(qa))
            from itertools import chain
            from types import SimpleNamespace
            ranges = (("000001", "000003"), ("000003", "000003"))
            scope = dict(os=os, json=json, pickle=pickle, root_path=tmp,
                         MODEL_NAME="mbert", SG_GLOBAL={"video": ranges}, chain=chain,
                         tqdm=lambda items: items, logger=SimpleNamespace(info=lambda *args: None))
            exec(compile(ast.Module(body=functions, type_ignores=[]), str(source), "exec"), scope)
            scope["preprocess_qa"]()
            for split in ("train", "test"):
                with (root / "preprocessed_mbert" / split / "qa2sg.pkl").open("rb") as inp:
                    self.assertEqual(pickle.load(inp), {"q1": list(ranges)})


if __name__ == "__main__":
    unittest.main()
