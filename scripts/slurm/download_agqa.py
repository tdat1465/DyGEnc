#!/usr/bin/env python3
"""Download only AGQA QA/scene graphs from an immutable public HF revision.

The caller must create an empty, private directory on executable Linux tmpfs.
Only SHA256SUMS.txt and the two fixed archive names are requested; video files,
HF cache directories and model files are never fetched. Archives are streamed,
verified, extracted one at a time, then removed. No persistent-disk fallback.
The Slurm caller records the immutable dataset revision for fresh/resume jobs.
"""

import argparse
import hashlib
import os
from pathlib import Path, PurePosixPath
import re
import stat
import sys
import tempfile
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener
import zipfile


ARCHIVES = {
    "AGQA_balanced.zip": ("train_balanced.txt", "test_balanced.txt"),
    "AGQA_scene_graphs.zip": ("AGQA_train_stsgs.pkl", "AGQA_test_stsgs.pkl"),
}
CHUNK_BYTES = 1024 * 1024
MAX_CHECKSUM_BYTES = 64 * 1024
MAX_ARCHIVE_BYTES = 1024**3
MAX_FILE_BYTES = 12 * 1024**3
MAX_EXPANDED_BYTES = 16 * 1024**3
MAX_ZIP_MEMBERS = 10000
MAX_COMPRESSION_RATIO = 1000
FREE_SPACE_RESERVE = 512 * 1024**2


class DownloadError(RuntimeError):
    """An actionable error that never embeds signed download URLs."""


def _checked_path(value):
    path = Path(value).expanduser()
    if ".." in path.parts:
        raise DownloadError("Path traversal is not allowed in download paths.")
    path = Path(os.path.abspath(path))
    for component in (path, *path.parents):
        try:
            info = component.lstat()
        except FileNotFoundError:
            continue
        reparse = getattr(info, "st_file_attributes", 0) & getattr(
            stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0
        )
        if stat.S_ISLNK(info.st_mode) or reparse:
            raise DownloadError("Download paths cannot contain symlinks/reparse points.")
    return path


def _mount_details(path, mountinfo):
    """Find the deepest mount covering path, including mountinfo escapes."""
    matches = []
    for line in mountinfo.splitlines():
        before, separator, after = line.partition(" - ")
        fields, suffix = before.split(), after.split()
        if not separator or len(fields) < 6 or len(suffix) < 3:
            continue
        mount = Path(re.sub(r"\\([0-7]{3})", lambda m: chr(int(m[1], 8)), fields[4]))
        try:
            path.relative_to(mount)
        except ValueError:
            continue
        options = set(fields[5].split(",")) | set(suffix[2].split(","))
        matches.append((len(mount.parts), suffix[0], options))
    if not matches:
        raise DownloadError("Cannot determine the filesystem backing downloads.")
    _, filesystem, options = max(matches, key=lambda match: match[0])
    return filesystem, options


def _validate_private_tmpfs(value):
    if not sys.platform.startswith("linux"):
        raise DownloadError("AGQA RAM download requires Linux executable tmpfs; no disk fallback.")
    path = _checked_path(value)
    try:
        info = path.stat()
    except FileNotFoundError:
        raise DownloadError("Caller must first create a private downloads directory on tmpfs.") from None
    if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
        raise DownloadError("downloads must be a caller-owned directory with mode 700 or stricter.")
    filesystem, options = _mount_details(path, Path("/proc/self/mountinfo").read_text())
    if filesystem != "tmpfs" or "noexec" in options or "ro" in options or "rw" not in options:
        raise DownloadError("downloads must be on writable executable tmpfs; no disk fallback.")
    return path


def _ensure_capacity(directory, needed):
    available = os.statvfs(directory)
    if available.f_bavail * available.f_frsize < needed + FREE_SPACE_RESERVE:
        raise DownloadError("Insufficient tmpfs space for AGQA download/extraction plus safety reserve.")


def _check_https(url):
    try:
        parsed = urlsplit(url)
        valid = parsed.scheme == "https" and parsed.hostname and not parsed.username
        valid = valid and not parsed.password and not parsed.fragment
    except ValueError:
        valid = False
    if not valid:
        raise DownloadError("Download and redirect URLs must use HTTPS without embedded credentials.")


class _HTTPSOnlyRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        _check_https(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _open_https(url):
    _check_https(url)
    request = Request(url, headers={
        "Cache-Control": "no-store", "Accept-Encoding": "identity", "User-Agent": "DyGEnc-AGQA/1"
    })
    return build_opener(_HTTPSOnlyRedirect()).open(request, timeout=120)


def _content_length(response, limit):
    value = response.headers.get("Content-Length")
    if value is None:
        return None
    try:
        size = int(value)
    except (TypeError, ValueError):
        raise DownloadError("Invalid download Content-Length.") from None
    if size < 1 or size > limit:
        raise DownloadError("Download exceeds its size limit or is empty.")
    return size


def _read_checksums(url, opener):
    try:
        with opener(url) as response:
            _check_https(response.geturl())
            expected_size = _content_length(response, MAX_CHECKSUM_BYTES)
            raw = bytearray()
            while chunk := response.read(min(CHUNK_BYTES, MAX_CHECKSUM_BYTES + 1 - len(raw))):
                raw.extend(chunk)
                if len(raw) > MAX_CHECKSUM_BYTES:
                    raise DownloadError("SHA256SUMS.txt exceeds its size limit.")
            if expected_size is not None and len(raw) != expected_size:
                raise DownloadError("Incomplete SHA256SUMS.txt response.")
        content = raw.decode("utf-8-sig")
    except DownloadError:
        raise
    except Exception:
        raise DownloadError("Cannot read SHA256SUMS.txt at the pinned dataset revision.") from None
    result = {}
    for line in content.splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        match = re.fullmatch(r"([0-9a-fA-F]{64}) [ *](.+)", line)
        if not match:
            raise DownloadError("Malformed SHA256SUMS.txt; use the expected AGQA dataset layout.")
        checksum, filename = match.groups()
        # Other entries (e.g. Charades videos) are never interpreted as URLs/paths.
        if filename not in ARCHIVES:
            continue
        if filename in result:
            raise DownloadError(f"Duplicate checksum entry: {filename}")
        result[filename] = checksum.lower()
    if set(result) != set(ARCHIVES):
        raise DownloadError("SHA256SUMS.txt must contain each of the two required archives exactly once.")
    return result


def _download_archive(url, directory, checksum, opener):
    descriptor, name = tempfile.mkstemp(prefix=".agqa-download-", suffix=".zip", dir=directory)
    path = Path(name)
    digest = hashlib.sha256()
    total = 0
    try:
        with os.fdopen(descriptor, "wb") as output:
            try:
                with opener(url) as response:
                    _check_https(response.geturl())
                    expected_size = _content_length(response, MAX_ARCHIVE_BYTES)
                    if expected_size is not None:
                        _ensure_capacity(directory, expected_size)
                    while chunk := response.read(min(CHUNK_BYTES, MAX_ARCHIVE_BYTES + 1 - total)):
                        total += len(chunk)
                        if total > MAX_ARCHIVE_BYTES:
                            raise DownloadError("AGQA archive exceeds its compressed-size limit.")
                        _ensure_capacity(directory, len(chunk))
                        output.write(chunk)
                        digest.update(chunk)
                    if expected_size is not None and total != expected_size:
                        raise DownloadError("Incomplete AGQA archive response.")
            except DownloadError:
                raise
            except Exception:
                raise DownloadError("Cannot stream AGQA archive over HTTPS at the pinned revision.") from None
        if total == 0 or digest.hexdigest() != checksum:
            raise DownloadError("AGQA archive SHA256 does not match the pinned checksum manifest.")
        return path, total
    except BaseException:
        # Only this invocation's uniquely named, incomplete download is removed.
        path.unlink(missing_ok=True)
        raise


def _zip_plan(archive, wanted):
    members = archive.infolist()
    if len(members) > MAX_ZIP_MEMBERS:
        raise DownloadError("Too many ZIP members.")
    names, selected = set(), {}
    total_expanded = 0
    for member in members:
        name = member.filename
        if "\x00" in member.orig_filename or "\\" in name or ":" in name:
            raise DownloadError("Unsafe ZIP member path.")
        parts = name.rstrip("/").split("/")
        if not name or name.startswith("/") or any(part in {"", ".", ".."} for part in parts):
            raise DownloadError("ZIP member path traversal/absolute paths are not allowed.")
        normalized = PurePosixPath(*parts).as_posix()
        if normalized in names:
            raise DownloadError("Duplicate ZIP member path.")
        names.add(normalized)
        file_type = stat.S_IFMT(member.external_attr >> 16)
        allowed_type = {0, stat.S_IFDIR} if member.is_dir() else {0, stat.S_IFREG}
        if file_type not in allowed_type:
            raise DownloadError("ZIP symlinks and special files are not allowed.")
        if member.flag_bits & 1 or member.compress_type not in {zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED}:
            raise DownloadError("Encrypted or unsupported-compression ZIP member.")
        if member.file_size < 0 or member.file_size > MAX_FILE_BYTES:
            raise DownloadError("ZIP member exceeds its expanded-size limit.")
        total_expanded += member.file_size
        if total_expanded > MAX_EXPANDED_BYTES:
            raise DownloadError("ZIP archive exceeds its total expanded-size limit.")
        if member.file_size > max(1, member.compress_size) * MAX_COMPRESSION_RATIO:
            raise DownloadError("ZIP member exceeds the compression-ratio limit.")
        if parts[-1] not in wanted or member.is_dir():
            continue
        if parts[-1] in selected:
            raise DownloadError(f"Multiple source copies of {parts[-1]} inside ZIP.")
        if member.file_size == 0:
            raise DownloadError("Required AGQA input is empty.")
        selected[parts[-1]] = member
    if set(selected) != set(wanted):
        raise DownloadError("ZIP does not contain exactly one copy of each required AGQA input.")
    return selected


def _extract_archive(source, directory, archive_name):
    """Plan/validate every member before creating any canonical output."""
    created = []
    target_dir = directory / archive_name.removesuffix(".zip")
    try:
        with zipfile.ZipFile(source) as archive:
            selected = _zip_plan(archive, ARCHIVES[archive_name])
            _ensure_capacity(directory, sum(member.file_size for member in selected.values()))
            if target_dir.exists():
                raise DownloadError("Extraction target already exists; use a fresh empty downloads directory.")
            target_dir.mkdir(mode=0o700)
            for filename, member in selected.items():
                target = target_dir / filename
                flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
                descriptor = os.open(target, flags, 0o600)
                created.append(target)
                total = 0
                with os.fdopen(descriptor, "wb") as output, archive.open(member) as input_file:
                    while chunk := input_file.read(CHUNK_BYTES):
                        total += len(chunk)
                        if total > member.file_size:
                            raise DownloadError("ZIP member emitted more data than declared.")
                        output.write(chunk)
                if total != member.file_size:
                    raise DownloadError("Truncated extracted AGQA input.")
            return {name: member.file_size for name, member in selected.items()}
    except BaseException:
        for path in created:
            path.unlink(missing_ok=True)
        if created:
            target_dir.rmdir()
        raise


def download_agqa(downloads, revision, *, repo_id="tdat1465/agqa-balanced", open_https=None):
    if not re.fullmatch(r"[0-9a-fA-F]{40}", revision):
        raise DownloadError("--revision must be an immutable 40-character HF commit SHA, not main/a tag.")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*", repo_id):
        raise DownloadError("--repo-id must be a Hugging Face dataset owner/name.")
    directory = _validate_private_tmpfs(downloads)
    if any(directory.iterdir()):
        raise DownloadError("downloads must be empty; use a fresh private RAM job directory.")
    lock = directory / ".agqa-download.lock"
    try:
        lock.mkdir(mode=0o700)
    except FileExistsError:
        raise DownloadError("Another AGQA downloader is using this directory.") from None
    opener = open_https or _open_https
    root_url = f"https://huggingface.co/datasets/{repo_id}/resolve/{revision.lower()}"
    result = {"repo_id": repo_id, "revision": revision.lower(), "archives": {}}
    try:
        if set(directory.iterdir()) != {lock}:
            raise DownloadError("downloads changed while acquiring its exclusive lock.")
        checksums = _read_checksums(f"{root_url}/SHA256SUMS.txt", opener)
        for archive_name in ARCHIVES:
            path, compressed_size = _download_archive(
                f"{root_url}/{archive_name}", directory, checksums[archive_name], opener
            )
            try:
                sizes = _extract_archive(path, directory, archive_name)
            except (zipfile.BadZipFile, RuntimeError, OSError) as error:
                if isinstance(error, DownloadError):
                    raise
                raise DownloadError(f"Cannot safely extract {archive_name}; use a fresh RAM job directory.") from None
            # Extraction succeeded: remove only the unique ZIP owned by this call.
            path.unlink()
            result["archives"][archive_name] = {
                "sha256": checksums[archive_name], "compressed_bytes": compressed_size,
                "extracted_bytes": sum(sizes.values()), "files": sizes,
            }
        return result
    finally:
        lock.rmdir()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--downloads", required=True, type=Path)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--repo-id", default="tdat1465/agqa-balanced")
    args = parser.parse_args(argv)
    try:
        manifest = download_agqa(args.downloads, args.revision, repo_id=args.repo_id)
    except (DownloadError, OSError) as error:
        print(f"Selective AGQA download failed: {error}", file=sys.stderr)
        return 2
    total = sum(item["extracted_bytes"] for item in manifest["archives"].values())
    print(f"AGQA ready: four raw files, {total / 1024**3:.2f} GiB extracted; no videos or archive cache.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
