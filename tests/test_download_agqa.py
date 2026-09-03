"""Tiny stdlib-only HTTP/ZIP fixtures; no network, GPU, or AGQA data download."""

import hashlib
import importlib.util
import io
import os
from pathlib import Path
import stat
import tempfile
import types
import unittest
from unittest import mock
import warnings
import zipfile


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/slurm/download_agqa.py"
SPEC = importlib.util.spec_from_file_location("download_agqa", SCRIPT)
download = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(download)
REVISION = "a" * 40


def make_zip(members):
    output = io.BytesIO()
    with warnings.catch_warnings(), zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
        warnings.simplefilter("ignore", UserWarning)
        for name, content in members:
            archive.writestr(name, content)
    return output.getvalue()


class FakeResponse(io.BytesIO):
    def __init__(self, content, url, *, headers=None):
        super().__init__(content)
        self.url = url
        self.headers = {} if headers is None else headers

    def geturl(self):
        return self.url


class DownloadAGQATest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.directory = self.root / "downloads"
        self.directory.mkdir(mode=0o700)
        self.data = {
            archive: make_zip([(f"outer/inner/{name}", name.encode()) for name in wanted])
            for archive, wanted in download.ARCHIVES.items()
        }
        self.requests = []
        self.addCleanup(mock.patch.stopall)
        mock.patch.object(download, "_validate_private_tmpfs", side_effect=lambda path: Path(path)).start()
        mock.patch.object(download, "_ensure_capacity").start()

    def checksums(self):
        lines = [f"{hashlib.sha256(raw).hexdigest()}  {name}" for name, raw in self.data.items()]
        lines.append(f"{'0' * 64}  Charades_v1_480.zip")
        return ("\n".join(lines) + "\n").encode()

    def opener(self, url):
        self.requests.append(url)
        self.assertTrue(url.startswith(f"https://huggingface.co/datasets/tdat1465/agqa-balanced/resolve/{REVISION}/"))
        name = url.rsplit("/", 1)[1]
        content = self.checksums() if name == "SHA256SUMS.txt" else self.data[name]
        return FakeResponse(content, url, headers={"Content-Length": str(len(content))})

    def run_download(self, **kwargs):
        return download.download_agqa(self.directory, REVISION, open_https=self.opener, **kwargs)

    def test_downloads_only_two_archives_and_extracts_nested_layout(self):
        result = self.run_download()
        self.assertEqual(result["revision"], REVISION)
        self.assertEqual([url.rsplit("/", 1)[1] for url in self.requests], ["SHA256SUMS.txt", *download.ARCHIVES])
        self.assertEqual(len(list(self.directory.rglob("*.*"))), 4)
        for archive_name, names in download.ARCHIVES.items():
            for name in names:
                self.assertEqual((self.directory / archive_name[:-4] / name).read_bytes(), name.encode())
        self.assertFalse(list(self.directory.rglob("*.zip")))
        self.assertFalse((self.directory / ".agqa-download.lock").exists())

    def test_first_zip_removed_before_second_download(self):
        original = self.opener

        def check_between(url):
            if url.endswith("AGQA_scene_graphs.zip"):
                # One new empty download temp belongs to the second archive only.
                self.assertEqual([p.stat().st_size for p in self.directory.glob("*.zip")], [0])
                self.assertTrue((self.directory / "AGQA_balanced/train_balanced.txt").is_file())
            return original(url)

        download.download_agqa(self.directory, REVISION, open_https=check_between)

    def test_only_wanted_files_extracted(self):
        wanted = download.ARCHIVES["AGQA_balanced.zip"]
        self.data["AGQA_balanced.zip"] = make_zip(
            [(name, name.encode()) for name in wanted] + [("video.mp4", b"irrelevant")]
        )
        self.run_download()
        self.assertFalse(list(self.directory.rglob("video.mp4")))

    def test_rejects_mutable_revision_and_repo_url(self):
        for revision in ("main", "v1", "../other", "b" * 39):
            with self.subTest(revision=revision), self.assertRaisesRegex(download.DownloadError, "immutable"):
                download.download_agqa(self.directory, revision, open_https=self.opener)
        with self.assertRaisesRegex(download.DownloadError, "owner/name"):
            self.run_download(repo_id="https://example.test/archive.zip")
        self.assertEqual(self.requests, [])

    def test_nonempty_directory_preserved(self):
        existing = self.directory / "existing.zip"
        existing.write_bytes(b"user content")
        with self.assertRaisesRegex(download.DownloadError, "empty"):
            self.run_download()
        self.assertEqual(existing.read_bytes(), b"user content")
        self.assertEqual(self.requests, [])

    def test_checksum_mismatch_removes_only_incomplete_download(self):
        with mock.patch.object(self, "checksums", return_value=(f"{'0' * 64}  AGQA_balanced.zip\n{'1' * 64}  AGQA_scene_graphs.zip\n").encode()):
            with self.assertRaisesRegex(download.DownloadError, "SHA256"):
                self.run_download()
        self.assertEqual(list(self.directory.iterdir()), [])

    def test_invalid_checksum_manifest(self):
        valid = self.checksums()
        variants = [b"<html>error</html>", valid + valid, b"", b"x" * (download.MAX_CHECKSUM_BYTES + 1)]
        for value in variants:
            with self.subTest(value=value[:40]):
                with mock.patch.object(self, "checksums", return_value=value):
                    with self.assertRaises(download.DownloadError):
                        self.run_download()
                self.assertEqual(list(self.directory.iterdir()), [])

    def test_network_error_redacts_signed_url(self):
        with self.assertRaises(download.DownloadError) as caught:
            download.download_agqa(self.directory, REVISION, open_https=mock.Mock(side_effect=OSError("https://x/?secret=token")))
        self.assertNotIn("secret", str(caught.exception))
        self.assertNotIn("token", str(caught.exception))

    def test_rejects_insecure_redirect(self):
        opener = mock.Mock(return_value=FakeResponse(self.checksums(), "http://example.test/x"))
        with self.assertRaisesRegex(download.DownloadError, "HTTPS"):
            download.download_agqa(self.directory, REVISION, open_https=opener)

    def test_archive_size_limit_without_content_length(self):
        with mock.patch.object(download, "MAX_ARCHIVE_BYTES", 10):
            opener = lambda url: FakeResponse(self.checksums() if url.endswith("txt") else b"x" * 11, url)
            with self.assertRaisesRegex(download.DownloadError, "compressed-size"):
                download.download_agqa(self.directory, REVISION, open_https=opener)
        self.assertEqual(list(self.directory.iterdir()), [])

    def test_content_length_truncation(self):
        original = self.opener

        def truncated(url):
            response = original(url)
            if url.endswith(".zip"):
                response.headers["Content-Length"] = str(len(self.data["AGQA_balanced.zip"]) + 1)
            return response

        with self.assertRaisesRegex(download.DownloadError, "Incomplete"):
            download.download_agqa(self.directory, REVISION, open_https=truncated)
        self.assertEqual(list(self.directory.iterdir()), [])

    def assert_unsafe_zip(self, extra_members, pattern):
        wanted = [(name, name.encode()) for name in download.ARCHIVES["AGQA_balanced.zip"]]
        self.data["AGQA_balanced.zip"] = make_zip(wanted + extra_members)
        with self.assertRaisesRegex(download.DownloadError, pattern):
            self.run_download()
        self.assertFalse((self.directory / "AGQA_balanced").exists())
        # A verified archive is retained on extraction failure, not any user file.
        self.assertEqual(len(list(self.directory.glob("*.zip"))), 1)

    def test_rejects_path_traversal_even_in_unselected_files(self):
        self.assert_unsafe_zip([("../outside.txt", b"bad")], "path traversal")
        self.assertFalse((self.root / "outside.txt").exists())

    def test_rejects_windows_paths(self):
        self.assert_unsafe_zip([("C:\\outside.txt", b"bad")], "Unsafe")

    def test_rejects_duplicate_member_paths(self):
        self.assert_unsafe_zip([("train_balanced.txt", b"duplicate")], "Duplicate ZIP")

    def test_rejects_duplicate_basename_nested_elsewhere(self):
        self.assert_unsafe_zip([("extra/train_balanced.txt", b"duplicate")], "Multiple source copies")

    def test_rejects_symlinks(self):
        link = zipfile.ZipInfo("bad-link")
        link.create_system = 3
        link.external_attr = (stat.S_IFLNK | 0o777) << 16
        self.assert_unsafe_zip([(link, b"/etc/passwd")], "symlinks")

    def test_rejects_missing_input(self):
        self.data["AGQA_balanced.zip"] = make_zip([("train_balanced.txt", b"train")])
        with self.assertRaisesRegex(download.DownloadError, "exactly one copy"):
            self.run_download()

    def test_zipbomb_expanded_size_limit(self):
        with mock.patch.object(download, "MAX_FILE_BYTES", 16):
            self.assert_unsafe_zip([("bomb.txt", b"x" * 17)], "expanded-size")

    def test_zipbomb_compression_ratio_limit(self):
        with mock.patch.object(download, "MAX_COMPRESSION_RATIO", 2):
            self.assert_unsafe_zip([("bomb.txt", b"x" * 1024)], "compression-ratio")


class TmpfsValidationTest(unittest.TestCase):
    def test_longest_mount_and_escaped_space(self):
        lines = (
            "1 0 1:0 / / rw - ext4 /dev/x rw\n"
            "2 1 0:1 / /dev/shm rw,nosuid,nodev - tmpfs tmpfs rw\n"
            "3 2 0:2 / /dev/shm/private\\040space rw,noexec - tmpfs tmpfs rw\n"
        )
        filesystem, options = download._mount_details(Path("/dev/shm/job/downloads"), lines)
        self.assertEqual(filesystem, "tmpfs")
        self.assertNotIn("noexec", options)
        filesystem, options = download._mount_details(Path("/dev/shm/private space/job"), lines)
        self.assertEqual(filesystem, "tmpfs")
        self.assertIn("noexec", options)

    def test_disk_non_executable_or_readonly_mount_rejected(self):
        fake_path = mock.Mock()
        fake_path.stat.return_value = types.SimpleNamespace(st_mode=stat.S_IFDIR | 0o700, st_uid=123)
        for filesystem, options in [("ext4", {"rw"}), ("tmpfs", {"rw", "noexec"}), ("tmpfs", {"ro"})]:
            with self.subTest(filesystem=filesystem, options=options):
                with mock.patch.object(download.sys, "platform", "linux"), mock.patch.object(download.os, "getuid", return_value=123, create=True), mock.patch.object(download, "_checked_path", return_value=fake_path), mock.patch.object(download.Path, "read_text", return_value=""), mock.patch.object(download, "_mount_details", return_value=(filesystem, options)):
                    with self.assertRaisesRegex(download.DownloadError, "writable executable tmpfs"):
                        download._validate_private_tmpfs("unused")

    def test_private_ownership_required(self):
        for mode, uid in [(stat.S_IFDIR | 0o755, 123), (stat.S_IFDIR | 0o700, 456)]:
            fake_path = mock.Mock()
            fake_path.stat.return_value = types.SimpleNamespace(st_mode=mode, st_uid=uid)
            with mock.patch.object(download.sys, "platform", "linux"), mock.patch.object(download.os, "getuid", return_value=123, create=True), mock.patch.object(download, "_checked_path", return_value=fake_path):
                with self.assertRaisesRegex(download.DownloadError, "caller-owned"):
                    download._validate_private_tmpfs("unused")

    def test_nonlinux_refused(self):
        with mock.patch.object(download.sys, "platform", "win32"):
            with self.assertRaisesRegex(download.DownloadError, "Linux"):
                download._validate_private_tmpfs("unused")

    def test_capacity_includes_safety_reserve(self):
        free = types.SimpleNamespace(f_bavail=100, f_frsize=1)
        with mock.patch.object(download.os, "statvfs", return_value=free, create=True), mock.patch.object(download, "FREE_SPACE_RESERVE", 50):
            download._ensure_capacity("unused", 50)
            with self.assertRaisesRegex(download.DownloadError, "Insufficient tmpfs"):
                download._ensure_capacity("unused", 51)


if __name__ == "__main__":
    unittest.main()
