"""Small stdlib-only staging tests; no AGQA download, GPU or dependencies."""

import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "slurm" / "stage_agqa.py"
SPEC = importlib.util.spec_from_file_location("stage_agqa", SCRIPT)
stage = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(stage)


class FakeResponse(io.BytesIO):
    def __init__(self, content, url="https://example.test/ENG.txt"):
        super().__init__(content)
        self.url = url

    def geturl(self):
        return self.url


class StageAGQATest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.downloads = self.root / "downloads"
        self.agqa = self.root / "agqa"
        self.manifest = self.agqa / "raw_manifest.json"
        self.downloads.mkdir()
        for index, name in enumerate(stage.RAW_FILES):
            nested = self.downloads / str(index)
            nested.mkdir()
            content = b'{"o1":"person","r1":"holding"}' if name == "ENG.txt" else name.encode()
            (nested / name).write_bytes(content)

    def run_stage(self, **kwargs):
        return stage.stage_agqa(self.downloads, self.agqa, self.manifest, **kwargs)

    def source(self, name):
        return next(self.downloads.rglob(name))

    def test_layout_hashes_and_idempotence(self):
        result = self.run_stage()
        self.assertEqual(result["schema_version"], 1)
        self.assertEqual(set(result["files"]), set(stage.RAW_FILES.values()))
        for name, relative in stage.RAW_FILES.items():
            content = self.source(name).read_bytes()
            target = self.agqa / relative
            self.assertEqual(target.read_bytes(), content)
            self.assertTrue(os.path.samefile(self.source(name), target))
            self.assertEqual(result["files"][relative], {
                "sha256": hashlib.sha256(content).hexdigest(), "size": len(content)
            })
        original = self.manifest.read_bytes()
        self.assertEqual(self.run_stage(), result)
        self.assertEqual(self.manifest.read_bytes(), original)

    def test_manifest_is_path_independent(self):
        result = self.run_stage()
        self.assertNotIn(str(self.root), json.dumps(result))

    def test_equivalent_existing_manifest_is_accepted_without_rewrite(self):
        result = self.run_stage()
        compact = json.dumps(result)
        self.manifest.write_text(compact, encoding="utf-8")
        self.assertEqual(self.run_stage(), result)
        self.assertEqual(self.manifest.read_text(), compact)

    def test_missing_raw_input(self):
        self.source("AGQA_train_stsgs.pkl").unlink()
        with self.assertRaisesRegex(stage.StageError, "Missing AGQA input"):
            self.run_stage()
        self.assertFalse(self.manifest.exists())

    def test_duplicate_input_fails_even_if_identical(self):
        (self.downloads / "train_balanced.txt").write_bytes(self.source("train_balanced.txt").read_bytes())
        with self.assertRaisesRegex(stage.StageError, "copies of train_balanced.txt"):
            self.run_stage()

    def test_empty_raw_input_fails(self):
        self.source("test_balanced.txt").write_bytes(b"")
        with self.assertRaisesRegex(stage.StageError, "nonempty regular file"):
            self.run_stage()

    def test_missing_eng_fails_before_canonical_files(self):
        self.source("ENG.txt").unlink()
        with self.assertRaisesRegex(stage.StageError, "Missing ENG.txt.*ENG_FILE / ENG_URL"):
            self.run_stage()
        self.assertFalse((self.agqa / "data").exists())

    def test_eng_file_is_copied_not_linked(self):
        self.source("ENG.txt").unlink()
        external = self.root / "small-mapping.txt"
        external.write_text('{"o1":"person"}', encoding="utf-8")
        self.run_stage(eng_file=external)
        target = self.agqa / "data/ENG.txt"
        self.assertEqual(target.read_bytes(), external.read_bytes())
        self.assertFalse(os.path.samefile(external, target))

    def test_downloaded_eng_takes_precedence_over_fallback(self):
        with mock.patch.object(stage, "_open_https", side_effect=AssertionError("not needed")):
            self.run_stage(eng_url="https://example.test/unused")

    def test_invalid_eng_variants(self):
        for content in ('[]', '{}', '{"o1":2}', '{"o1":"person","o1":"x"}', '<html>login</html>'):
            with self.subTest(content=content):
                self.source("ENG.txt").write_text(content, encoding="utf-8")
                with self.assertRaisesRegex(stage.StageError, "ENG.txt must"):
                    self.run_stage()
                self.assertFalse((self.agqa / "data").exists())

    def test_https_eng_download(self):
        self.source("ENG.txt").unlink()
        with mock.patch.object(stage, "_open_https", return_value=FakeResponse(b'{"o1":"person"}')):
            self.run_stage(eng_url="https://example.test/ENG.txt?token=secret")
        self.assertTrue((self.agqa / "data/ENG.txt").is_file())
        self.assertFalse(list(self.agqa.glob(".eng-download-*")))
        self.assertNotIn("secret", self.manifest.read_text())

    def test_https_error_never_reveals_url_query(self):
        self.source("ENG.txt").unlink()
        with mock.patch.object(stage, "_open_https", side_effect=OSError("https://x/?token=secret")):
            with self.assertRaises(stage.StageError) as caught:
                self.run_stage(eng_url="https://example.test/ENG.txt?token=secret")
        self.assertNotIn("secret", str(caught.exception))
        self.assertFalse(list(self.agqa.glob(".eng-download-*")))

    def test_rejects_insecure_url_and_redirect(self):
        self.source("ENG.txt").unlink()
        for url in ("http://example.test/ENG.txt", "file:///ENG.txt", "https://user:secret@example.test/x"):
            with self.subTest(url=url), self.assertRaisesRegex(stage.StageError, "HTTPS URL"):
                self.run_stage(eng_url=url)
        with mock.patch.object(stage, "_open_https", return_value=FakeResponse(b'{}', "http://example.test/x")):
            with self.assertRaisesRegex(stage.StageError, "HTTPS URL"):
                self.run_stage(eng_url="https://example.test/ENG.txt")

    def test_oversized_eng_download(self):
        self.source("ENG.txt").unlink()
        with mock.patch.object(stage, "MAX_ENG_BYTES", 16):
            with mock.patch.object(stage, "_open_https", return_value=FakeResponse(b"x" * 17)):
                with self.assertRaisesRegex(stage.StageError, "more than"):
                    self.run_stage(eng_url="https://example.test/ENG.txt")
        self.assertFalse(list(self.agqa.glob(".eng-download-*")))

    def test_conflicting_target_is_not_overwritten(self):
        target = self.agqa / "data/AGQA_balanced/test_balanced.txt"
        target.parent.mkdir(parents=True)
        target.write_bytes(b"user content")
        with self.assertRaisesRegex(stage.StageError, "Refusing to overwrite"):
            self.run_stage()
        self.assertEqual(target.read_bytes(), b"user content")
        self.assertFalse(self.manifest.exists())
        self.assertFalse((target.parent / "train_balanced.txt").exists())

    def test_existing_manifest_mismatch(self):
        self.agqa.mkdir()
        self.manifest.write_text('{"schema_version":999}', encoding="utf-8")
        with self.assertRaisesRegex(stage.StageError, "manifest differs"):
            self.run_stage()
        self.assertEqual(json.loads(self.manifest.read_text()), {"schema_version": 999})

    def test_manifest_must_stay_inside_agqa_root(self):
        self.manifest = self.root / "not-on-agqa-root.json"
        with self.assertRaisesRegex(stage.StageError, "inside AGQA_ROOT"):
            self.run_stage()

    def test_manifest_cannot_replace_raw_input(self):
        self.manifest = self.agqa / "data/ENG.txt"
        with self.assertRaisesRegex(stage.StageError, "must not replace"):
            self.run_stage()

    def test_rejects_path_traversal(self):
        self.manifest = self.agqa / ".." / "outside.json"
        with self.assertRaisesRegex(stage.StageError, "Path traversal"):
            self.run_stage()

    def test_rejects_overlapping_trees(self):
        self.agqa = self.downloads / "agqa"
        self.manifest = self.agqa / "raw_manifest.json"
        with self.assertRaisesRegex(stage.StageError, "non-overlapping"):
            self.run_stage()

    def make_symlink(self, source, destination, is_directory=False):
        try:
            destination.symlink_to(source, target_is_directory=is_directory)
        except OSError:
            self.skipTest("symlink permission is unavailable on this platform")

    def test_rejects_download_symlink(self):
        self.make_symlink(self.root, self.downloads / "escape", True)
        with self.assertRaisesRegex(stage.StageError, "Symlink"):
            self.run_stage()

    def test_rejects_target_parent_symlink(self):
        self.agqa.mkdir()
        outside = self.root / "outside"
        outside.mkdir()
        self.make_symlink(outside, self.agqa / "data", True)
        with self.assertRaisesRegex(stage.StageError, "Symlink"):
            self.run_stage()
        self.assertEqual(list(outside.iterdir()), [])

    def test_streaming_copy_fallback(self):
        real_link = os.link

        def deny_source_link(source, destination):
            if stage._within(Path(source), self.downloads):
                raise PermissionError(1, "hard links unavailable")
            return real_link(source, destination)

        with mock.patch.object(stage.os, "link", side_effect=deny_source_link):
            self.run_stage()
        self.assertEqual((self.agqa / "data/ENG.txt").read_bytes(), self.source("ENG.txt").read_bytes())
        self.assertFalse(os.path.samefile(self.agqa / "data/ENG.txt", self.source("ENG.txt")))


if __name__ == "__main__":
    unittest.main()
