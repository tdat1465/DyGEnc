"""Small RAM-loader/preprocessing fixtures; no Hugging Face/GPU downloads."""
import ast
import gc
import importlib.util
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
        from src.datasets.agqa_storage import INDEX_NAME, build_qa_index

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
            build_qa_index(processed / INDEX_NAME, qa.items(), lambda item: [interval])
            with patch.dict(os.environ, {"AGQA_ROOT": tmp, "DYGENC_LAZY_GRAPHS": "0", "DYGENC_INDEXED_QA": "0"}):
                eager = AGQADataset("train", seq_limit=100)
            with patch.dict(os.environ, {"AGQA_ROOT": tmp, "DYGENC_LAZY_GRAPHS": "1",
                                         "DYGENC_GRAPH_CACHE_SIZE": "1", "DYGENC_INDEXED_QA": "0"}):
                lazy = AGQADataset("train", seq_limit=100)
            with patch.dict(os.environ, {"AGQA_ROOT": tmp, "DYGENC_LAZY_GRAPHS": "1",
                                         "DYGENC_GRAPH_CACHE_SIZE": "1", "DYGENC_INDEXED_QA": "1"}):
                indexed = AGQADataset("train", seq_limit=100)
            for i in range(3):
                a, b, c = eager[i], lazy[i], indexed[i]
                graph = a.pop("graphs")[0]
                self.assertTrue(torch.equal(graph, b.pop("graphs")[0]))
                self.assertTrue(torch.equal(graph, c.pop("graphs")[0]))
                self.assertEqual(a, b)
                self.assertEqual(a, c)
                self.assertEqual(len(lazy.video_cache), 1)
                self.assertEqual(len(indexed.video_cache), 1)
            self.assertNotIn("graphs", vars(lazy))
            self.assertNotIn("qa_data", vars(indexed))
            self.assertNotIn("qa2sg", vars(indexed))
            indexed.qa_index.close()


class GroundingRangeTests(unittest.TestCase):
    def test_grounding_uses_interval_keys_without_retaining_graph_tensors(self):
        # Extract pure-data functions so these fixtures need neither PyG nor
        # Transformers; module import itself no longer initializes ModernBERT.
        source = Path(__file__).resolve().parents[1] / "src/datasets/preprocess/agqa.py"
        tree = ast.parse(source.read_text(encoding="utf-8"))
        functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                     and node.name in ("load_grounding_frames", "ground_qa_item", "preprocess_qa")]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "data" / "AGQA_balanced").mkdir(parents=True)
            qa = {"q1": {"video_id": "video", "sg_grounding": {}}}
            for split in ("train", "test"):
                (root / "data" / "AGQA_balanced" / f"{split}_balanced.txt").write_text(json.dumps(qa))
            from itertools import chain
            from types import SimpleNamespace
            ranges = (("000001", "000003"), ("000003", "000003"))
            scope = dict(os=os, json=json, pickle=pickle, gc=gc, root_path=tmp,
                         MODEL_NAME="mbert", SG_GLOBAL={(split, "video"): ranges for split in ("train", "test")}, chain=chain,
                         tqdm=lambda items: items, logger=SimpleNamespace(info=lambda *args: None))
            exec(compile(ast.Module(body=functions, type_ignores=[]), str(source), "exec"), scope)
            with patch.dict(os.environ, {"DYGENC_INDEXED_QA": "0"}):
                scope["preprocess_qa"]()
            for split in ("train", "test"):
                with (root / "preprocessed_mbert" / split / "qa2sg.pkl").open("rb") as inp:
                    self.assertEqual(pickle.load(inp), {"q1": list(ranges)})

    @unittest.skipUnless(importlib.util.find_spec("ijson"), "ijson is needed for indexed preprocessing")
    def test_indexed_preprocessing_matches_legacy_with_split_specific_grounding(self):
        from itertools import chain
        from types import SimpleNamespace
        from src.datasets.agqa_storage import INDEX_NAME, IndexedQA, build_qa_index, iter_qa_json

        source = Path(__file__).resolve().parents[1] / "src/datasets/preprocess/agqa.py"
        tree = ast.parse(source.read_text(encoding="utf-8"))
        functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                     and node.name in ("load_grounding_frames", "ground_qa_item", "preprocess_qa")]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "data" / "AGQA_balanced").mkdir(parents=True)
            qa = {"q10": {"video_id": "same_video", "sg_grounding": {}, "extra": [2**70, 1.25]},
                  "q2": {"video_id": "same_video", "sg_grounding": {"0-1": ["o1/000001", "o1/000002"]}}}
            train_ranges = (("000001", "000003"), ("000003", "000003"))
            test_ranges = (("000001", "000004"), ("000004", "000004"))
            for split in ("train", "test"):
                (root / "data" / "AGQA_balanced" / f"{split}_balanced.txt").write_text(json.dumps(qa))
            scope = dict(os=os, json=json, pickle=pickle, gc=gc, root_path=tmp, MODEL_NAME="mbert",
                         SG_GLOBAL={("train", "same_video"): train_ranges, ("test", "same_video"): test_ranges},
                         chain=chain, INDEX_NAME=INDEX_NAME, build_qa_index=build_qa_index, iter_qa_json=iter_qa_json,
                         tqdm=lambda items: items, logger=SimpleNamespace(info=lambda *args: None))
            exec(compile(ast.Module(body=functions, type_ignores=[]), str(source), "exec"), scope)
            with patch.dict(os.environ, {"DYGENC_INDEXED_QA": "0"}):
                scope["preprocess_qa"]()
            with patch.dict(os.environ, {"DYGENC_INDEXED_QA": "1"}), \
                 patch.object(json, "load", side_effect=AssertionError("indexed mode must stream")):
                scope["preprocess_qa"]()
            for split in ("train", "test"):
                directory = root / "preprocessed_mbert" / split
                with (directory / "qa2sg.pkl").open("rb") as source_pickle:
                    expected_grounding = pickle.load(source_pickle)
                indexed = IndexedQA(directory / INDEX_NAME)
                try:
                    self.assertEqual(len(indexed), len(qa))
                    for position, (qa_id, item) in enumerate(qa.items()):
                        self.assertEqual(indexed[position], (qa_id, item, expected_grounding[qa_id]))
                finally:
                    indexed.close()

    def test_grounding_preserves_duplicates_and_terminal_interval(self):
        from itertools import chain
        source = Path(__file__).resolve().parents[1] / "src/datasets/preprocess/agqa.py"
        tree = ast.parse(source.read_text(encoding="utf-8"))
        functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                     and node.name in ("load_grounding_frames", "ground_qa_item")]
        scope = dict(chain=chain)
        exec(compile(ast.Module(body=functions, type_ignores=[]), str(source), "exec"), scope)
        first, last = ("000001", "000004"), ("000004", "000004")
        item = {"video_id": "video", "sg_grounding": {"0-1": ["o1/000001", "o1/000002", "o1/000004"]}}
        self.assertEqual(scope["ground_qa_item"](item, (first, last)), [first, first, last])

    def test_preprocess_import_does_not_initialize_embedding_model(self):
        source = Path(__file__).resolve().parents[1] / "src/datasets/preprocess/agqa.py"
        tree = ast.parse(source.read_text(encoding="utf-8"))
        # Model loading and ENG access must remain in functions or __main__.
        assignments = [node for node in tree.body if isinstance(node, ast.Assign)]
        for assignment in assignments:
            calls = [node for node in ast.walk(assignment) if isinstance(node, ast.Call)]
            self.assertFalse(any(isinstance(call.func, ast.Subscript)
                                 and isinstance(call.func.value, ast.Name)
                                 and call.func.value.id in ("load_model", "load_text2embedding")
                                 for call in calls))


if __name__ == "__main__":
    unittest.main()
