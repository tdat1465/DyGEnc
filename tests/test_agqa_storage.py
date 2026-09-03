"""Streaming QA index tests require no GPU, tensors, or model downloads."""

from array import array
from contextlib import closing
import importlib.util
import json
import pickle
from pathlib import Path
import sqlite3
import sys
import tempfile
import unittest
from unittest.mock import patch


MODULE_PATH = Path(__file__).resolve().parents[1] / "src/datasets/agqa_storage.py"
SPEC = importlib.util.spec_from_file_location("_agqa_storage_test", MODULE_PATH)
storage = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = storage
SPEC.loader.exec_module(storage)


class IndexedQATests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / storage.INDEX_NAME
        self.first = ("000001", "000003")
        self.last = ("000003", "000003")

    def build(self):
        records = [
            ("q10", {"video_id": "b", "question": "đúng?", "answer": "YES", "ranges": 2,
                      "unused_fields": {"large_integer": 2**90, "real": 1.2345, "items": [None, True]}}),
            ("q2", {"video_id": "a", "question": "second?", "ranges": 1}),
            ("q1", {"video_id": "b", "question": "third?", "ranges": 3}),
        ]
        storage.build_qa_index(self.path, iter(records), lambda record: [self.first] * record["ranges"])
        return records

    def read_index(self, seq_limit=float("inf")):
        index = storage.IndexedQA(self.path, seq_limit)
        self.addCleanup(index.close)
        return index

    def test_source_order_payload_and_duplicate_grounding_preserved(self):
        records = self.build()
        index = self.read_index()
        self.assertIsInstance(index.positions, range)
        for position, (qa_id, original) in enumerate(records):
            actual_id, payload, intervals = index[position]
            self.assertEqual(actual_id, qa_id)
            self.assertEqual(payload, original)
            self.assertEqual(intervals, [self.first] * original["ranges"])
        self.assertEqual(index[-1][0], "q1")
        with self.assertRaises(IndexError):
            index[len(index)]
        with self.assertRaises(TypeError):
            index[0.5]

    def test_filter_uses_compact_positions_and_original_grounding_count(self):
        self.build()
        index = self.read_index(2)
        self.assertIsInstance(index.positions, array)
        self.assertEqual(index.positions.itemsize, 8)
        self.assertEqual([index[i][0] for i in range(len(index))], ["q10", "q2"])
        self.assertEqual(len(self.read_index(sys.float_info.max)), 3)
        self.assertEqual(len(self.read_index(0)), 0)
        self.assertEqual([self.read_index(1.9)[0][0]], ["q2"])

    def test_reader_read_only_and_pickle_reopens_connection(self):
        self.build()
        index = self.read_index()
        with self.assertRaises(sqlite3.OperationalError):
            index._connect().execute("DELETE FROM qa")
        restored = pickle.loads(pickle.dumps(index))
        self.addCleanup(restored.close)
        self.assertIsNone(restored._connection)
        self.assertEqual(restored[0], index[0])

    def test_failed_build_preserves_previous_index_and_removes_partial(self):
        self.build()
        before = self.path.read_bytes()
        with self.assertRaises(ValueError):
            storage.build_qa_index(self.path, [("other", {})], lambda record: [])
        self.assertEqual(self.path.read_bytes(), before)
        self.assertFalse(list(self.path.parent.glob("*.building-*")))

    def test_duplicate_ids_do_not_silently_shift_alignment(self):
        with self.assertRaises(sqlite3.IntegrityError):
            storage.build_qa_index(self.path, [("same", {}), ("same", {})], lambda record: [self.first])
        self.assertFalse(self.path.exists())

    def test_invalid_version_and_incomplete_indexes_rejected(self):
        self.build()
        with closing(sqlite3.connect(self.path)) as connection, connection:
            connection.execute("UPDATE metadata SET value='0' WHERE key='complete'")
        with self.assertRaisesRegex(ValueError, "not completed"):
            storage.IndexedQA(self.path)
        with closing(sqlite3.connect(self.path)) as connection, connection:
            connection.execute("PRAGMA user_version=999")
        with self.assertRaisesRegex(ValueError, "version"):
            storage.IndexedQA(self.path)

    def test_noncontiguous_positions_rejected(self):
        self.build()
        with closing(sqlite3.connect(self.path)) as connection, connection:
            connection.execute("UPDATE qa SET position=10 WHERE position=1")
        with self.assertRaisesRegex(ValueError, "noncontiguous"):
            storage.IndexedQA(self.path)

    def test_invalid_limits_rejected(self):
        self.build()
        for limit in (-1, float("nan"), -float("inf")):
            with self.subTest(limit=limit), self.assertRaises(ValueError):
                storage.IndexedQA(self.path, limit)

    def test_large_iterator_not_materialized(self):
        records = ((f"q{i}", {"position": i}) for i in range(10000))
        count = storage.build_qa_index(self.path, records, lambda record: [self.first])
        index = self.read_index()
        self.assertEqual(count, 10000)
        self.assertEqual(index[9999][1], {"position": 9999})
        self.assertIsInstance(index.positions, range)
        self.assertNotIn("records", vars(index))


@unittest.skipUnless(importlib.util.find_spec("ijson"), "ijson is needed for streaming JSON tests")
class StreamingJSONTests(unittest.TestCase):
    def test_order_all_fields_unicode_and_numeric_types_match_json_load(self):
        text = '\ufeff  {"q10":{"string":"đúng","nested":[true,null,1.25,1e-3],"large":123456789012345678901234567890},"q2":{"float":1.0,"value":-2}}'
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "qa.txt"
            path.write_text(text, encoding="utf-8")
            expected = list(json.loads(text.lstrip("\ufeff")).items())
            with patch.object(storage.json, "load", side_effect=AssertionError("must stream")):
                actual = list(storage.iter_qa_json(path))
            self.assertEqual(actual, expected)
            self.assertIsInstance(actual[0][1]["large"], int)
            self.assertIsInstance(actual[1][1]["float"], float)

    def test_top_level_array_and_nonobject_record_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "qa.txt"
            for text in ('[]', '{"q1": 42}'):
                with self.subTest(text=text):
                    path.write_text(text, encoding="utf-8")
                    with self.assertRaises(ValueError):
                        list(storage.iter_qa_json(path))


if __name__ == "__main__":
    unittest.main()
