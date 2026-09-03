#!/usr/bin/env python3
"""Normalize extracted AGQA inputs without a persistent data/cache directory.

The Slurm caller must put downloads and agqa_root on its verified, private tmpfs.
This helper checks they share a filesystem, rejects symlinks/path traversal,
hard-links large files where possible, and streams fallback copies. It never
loads the large QA JSON or scene-graph pickle files. ENG_URL must be a direct
HTTPS JSON response, not a Google Drive HTML page or an archive.
"""

import argparse
import errno
import hashlib
import json
import os
from pathlib import Path
import stat
import sys
import tempfile
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener


CHUNK_BYTES = 1024 * 1024
MAX_ENG_BYTES = 8 * 1024 * 1024
RAW_FILES = {
    "train_balanced.txt": "data/AGQA_balanced/train_balanced.txt",
    "test_balanced.txt": "data/AGQA_balanced/test_balanced.txt",
    "AGQA_train_stsgs.pkl": "data/AGQA_scene_graphs/AGQA_train_stsgs.pkl",
    "AGQA_test_stsgs.pkl": "data/AGQA_scene_graphs/AGQA_test_stsgs.pkl",
    "ENG.txt": "data/ENG.txt",
}


class StageError(RuntimeError):
    """An actionable staging error safe to show without URL credentials."""


def _checked_path(value):
    path = Path(value).expanduser()
    if ".." in path.parts:
        raise StageError("Path traversal ('..') is not allowed in staging paths.")
    path = Path(os.path.abspath(path))
    for component in (path, *path.parents):
        try:
            info = component.lstat()
        except FileNotFoundError:
            continue
        is_reparse = getattr(info, "st_file_attributes", 0) & getattr(
            stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0
        )
        if stat.S_ISLNK(info.st_mode) or is_reparse:
            raise StageError(f"Symlink/reparse-point paths are not allowed: {component}")
    return path


def _within(path, root):
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _regular_file(path):
    path = _checked_path(path)
    try:
        info = path.stat()
    except FileNotFoundError:
        raise StageError(f"Missing input file: {path.name}") from None
    if not stat.S_ISREG(info.st_mode) or info.st_size == 0:
        raise StageError(f"Input must be a nonempty regular file: {path.name}")
    return path


def _signature(path):
    digest = hashlib.sha256()
    size = 0
    with _regular_file(path).open("rb") as handle:
        while chunk := handle.read(CHUNK_BYTES):
            digest.update(chunk)
            size += len(chunk)
    return {"sha256": digest.hexdigest(), "size": size}


def _discover(downloads):
    found = {name: [] for name in RAW_FILES}

    def walk_error(error):
        raise StageError("Cannot read the extracted download tree.") from error

    for directory, subdirs, files in os.walk(downloads, followlinks=False, onerror=walk_error):
        for name in subdirs + files:
            candidate = _checked_path(Path(directory) / name)
            if not _within(candidate, downloads):
                raise StageError("A download path escaped the staging directory.")
            if name in found:
                found[name].append(_regular_file(candidate))
    for name, matches in found.items():
        if len(matches) > 1:
            raise StageError(
                f"Found {len(matches)} copies of {name}; keep exactly one in downloads."
            )
    missing = [name for name in RAW_FILES if name != "ENG.txt" and not found[name]]
    if missing:
        raise StageError("Missing AGQA input(s): " + ", ".join(missing))
    return {name: matches[0] for name, matches in found.items() if matches}


def _unique_json_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def _validate_eng(path):
    path = _regular_file(path)
    if path.stat().st_size > MAX_ENG_BYTES:
        raise StageError("ENG.txt exceeds the 8 MiB mapping-file limit; supply the raw mapping.")
    try:
        with path.open("r", encoding="utf-8-sig") as handle:
            mapping = json.load(handle, object_pairs_hook=_unique_json_object)
    except (ValueError, UnicodeError):
        raise StageError("ENG.txt must be a JSON object mapping labels to English strings.") from None
    if not isinstance(mapping, dict) or not mapping or not all(
        isinstance(key, str) and key and isinstance(value, str) and value
        for key, value in mapping.items()
    ):
        raise StageError("ENG.txt must be a nonempty JSON object of string-to-string labels.")
    return path


def _check_https(url):
    try:
        parsed = urlsplit(url)
        valid = parsed.scheme == "https" and parsed.hostname and not parsed.username
        valid = valid and not parsed.password and not parsed.fragment
    except ValueError:
        valid = False
    if not valid:
        raise StageError("ENG_URL must be an HTTPS URL without embedded credentials or a fragment.")


class _HTTPSOnlyRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        _check_https(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _open_https(url):
    _check_https(url)
    request = Request(url, headers={"Cache-Control": "no-store", "User-Agent": "DyGEnc-stage/1"})
    return build_opener(_HTTPSOnlyRedirect()).open(request, timeout=60)


def _download_eng(url, directory):
    _check_https(url)
    descriptor, name = tempfile.mkstemp(prefix=".eng-download-", dir=directory)
    destination = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as output:
            try:
                with _open_https(url) as response:
                    _check_https(response.geturl())
                    total = 0
                    while chunk := response.read(min(CHUNK_BYTES, MAX_ENG_BYTES + 1 - total)):
                        total += len(chunk)
                        if total > MAX_ENG_BYTES:
                            raise StageError("ENG_URL returned more than 8 MiB; use a direct ENG.txt URL.")
                        output.write(chunk)
            except StageError:
                raise
            except Exception:
                # urllib exceptions can include a signed query string/token.
                raise StageError("Cannot download ENG.txt over HTTPS; check access and the direct URL.") from None
        _validate_eng(destination)
        return destination
    except BaseException:
        destination.unlink(missing_ok=True)
        raise


def _check_existing(destination, expected):
    _checked_path(destination)
    if destination.exists():
        if _signature(destination) != expected:
            raise StageError(f"Refusing to overwrite different existing content: {destination.name}")
        return True
    return False


def _copy_exclusive(source, destination):
    """Streaming fallback, never truncate a pre-existing destination."""
    descriptor = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as output, source.open("rb") as input_file:
            while chunk := input_file.read(CHUNK_BYTES):
                output.write(chunk)
    except BaseException:
        destination.unlink(missing_ok=True)
        raise


def _stage_file(source, destination, expected, *, external_eng=False):
    if _check_existing(destination, expected):
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    _checked_path(destination.parent)
    if external_eng:
        # ENG_FILE may be a small persistent input; never link back to that disk.
        _copy_exclusive(source, destination)
        return
    if source.stat().st_dev != destination.parent.stat().st_dev:
        raise StageError("Large AGQA inputs and destinations must share the caller's RAM filesystem.")
    try:
        os.link(source, destination)
    except FileExistsError:
        _check_existing(destination, expected)
    except OSError as error:
        if error.errno not in {
            errno.EACCES, errno.EPERM, errno.ENOTSUP, errno.ENOSYS, errno.EMLINK, errno.EXDEV
        }:
            raise
        _copy_exclusive(source, destination)


def _write_manifest(path, manifest):
    encoded = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8")
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != manifest:
            raise StageError("Existing raw manifest differs; use a fresh RAM job directory.")
        return
    descriptor, name = tempfile.mkstemp(prefix=".manifest-", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as output:
            output.write(encoded)
            output.flush()
            os.fsync(output.fileno())
        # An atomic, no-clobber publish on the caller's tmpfs.
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.read_bytes() != encoded:
                raise StageError("Refusing to overwrite a different raw manifest.") from None
    finally:
        temporary.unlink(missing_ok=True)


def stage_agqa(downloads, agqa_root, manifest_path, *, eng_file=None, eng_url=None):
    if eng_file and eng_url:
        raise StageError("Provide ENG_FILE or ENG_URL, not both.")
    downloads = _checked_path(downloads)
    agqa_root = _checked_path(agqa_root)
    manifest_path = _checked_path(manifest_path)
    if not downloads.is_dir():
        raise StageError("The downloads directory must already contain extracted AGQA inputs.")
    if _within(downloads, agqa_root) or _within(agqa_root, downloads):
        raise StageError("downloads and agqa_root must be separate, non-overlapping directories.")
    if not _within(manifest_path, agqa_root) or manifest_path == agqa_root:
        raise StageError("The raw manifest must be a file inside AGQA_ROOT (on RAM).")
    agqa_root.mkdir(parents=True, exist_ok=True)
    if downloads.stat().st_dev != agqa_root.stat().st_dev:
        raise StageError("downloads and AGQA_ROOT must share the caller's RAM filesystem.")
    sources = _discover(downloads)
    temporary_eng = None
    external_eng = False
    try:
        if "ENG.txt" not in sources:
            if eng_file:
                sources["ENG.txt"] = _validate_eng(_checked_path(eng_file))
                external_eng = True
            elif eng_url:
                temporary_eng = _download_eng(eng_url, agqa_root)
                sources["ENG.txt"] = temporary_eng
            else:
                raise StageError(
                    "Missing ENG.txt. Add the official AGQA label mapping to downloads, "
                    "or set ENG_FILE / ENG_URL (direct HTTPS JSON, not an archive or Drive page). "
                    "The AGQA supporting-data link is at "
                    "https://cs.stanford.edu/people/ranjaykrishna/agqa/."
                )
        _validate_eng(sources["ENG.txt"])
        signatures = {name: _signature(source) for name, source in sources.items()}
        manifest = {
            "schema_version": 1,
            "files": {RAW_FILES[name]: signatures[name] for name in sorted(RAW_FILES)},
        }
        # Check every target before writing any canonical data file.
        for name, relative_path in RAW_FILES.items():
            target = _checked_path(agqa_root / relative_path)
            if not _within(target, agqa_root):
                raise StageError("A canonical target escaped AGQA_ROOT.")
            if _within(target, manifest_path) or _within(manifest_path, target):
                raise StageError("The manifest path must not replace or overlap a raw data file.")
            _check_existing(target, signatures[name])
        if manifest_path.exists():
            try:
                existing = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (ValueError, UnicodeError):
                raise StageError("Existing raw manifest is invalid; use a fresh RAM job directory.") from None
            if existing != manifest:
                raise StageError("Existing raw manifest differs; use a fresh RAM job directory.")
        for name, relative_path in RAW_FILES.items():
            _stage_file(
                sources[name], agqa_root / relative_path, signatures[name],
                external_eng=external_eng and name == "ENG.txt",
            )
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        _checked_path(manifest_path)
        _write_manifest(manifest_path, manifest)
        return manifest
    finally:
        if temporary_eng is not None:
            temporary_eng.unlink(missing_ok=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--downloads", required=True, type=Path)
    parser.add_argument("--agqa-root", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    eng = parser.add_mutually_exclusive_group()
    eng.add_argument("--eng-file", type=Path)
    eng.add_argument("--eng-url")
    args = parser.parse_args(argv)
    try:
        manifest = stage_agqa(
            args.downloads, args.agqa_root, args.manifest,
            eng_file=args.eng_file, eng_url=args.eng_url,
        )
    except (StageError, OSError) as error:
        print(f"AGQA staging failed: {error}", file=sys.stderr)
        return 2
    print(f"AGQA staged: {len(manifest['files'])} verified raw inputs; manifest ready.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
