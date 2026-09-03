"""Lossless, bounded-heap storage for AGQA question/grounding records.

The SQLite file is kept under AGQA_ROOT (tmpfs in RAM-only jobs). It replaces
millions of Python dictionaries/lists, not the original graph features. Row
positions are explicitly assigned in source JSON order and the QA ID and its
grounding are written together, so training never joins unordered containers.
"""

from array import array
from decimal import Decimal
import json
import math
import operator
import os
from pathlib import Path
import sqlite3
import tempfile


INDEX_NAME = "qa_index.sqlite3"
SCHEMA_VERSION = 1


def _json_number_types(value):
    """Match json.load's int/float types without ijson's 64-bit int limit.

    ijson's default preserves large integers and parses reals as Decimal.
    Convert only the latter to Python float, as the eager JSON reader does.
    No tensor dtype or model feature representation is changed.
    """
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, list):
        return [_json_number_types(item) for item in value]
    if isinstance(value, dict):
        return {key: _json_number_types(item) for key, item in value.items()}
    return value


def iter_qa_json(path):
    """Yield one (QA ID, complete QA record) at a time in JSON source order."""
    try:
        import ijson
    except ImportError as exc:
        raise RuntimeError(
            "DYGENC_INDEXED_QA=1 requires ijson; install requirements-server.txt"
        ) from exc
    with open(path, "rb") as source:
        # Accept a UTF-8 BOM like the small metadata reader, without buffering
        # the file. Reject a top-level array rather than indexing an empty set.
        if source.read(3) != b"\xef\xbb\xbf":
            source.seek(0)
        start = source.tell()
        first = source.read(1)
        while first in (b" ", b"\t", b"\r", b"\n"):
            first = source.read(1)
        if first != b"{":
            raise ValueError("AGQA balanced JSON must be an object keyed by QA ID")
        source.seek(start)
        for qa_id, payload in ijson.kvitems(source, ""):
            if not isinstance(qa_id, str) or not isinstance(payload, dict):
                raise ValueError("Each AGQA QA ID must map to a JSON object")
            yield qa_id, _json_number_types(payload)


def build_qa_index(path, records, ground_record):
    """Atomically build an index from a streaming iterator.

    ``ground_record(payload)`` returns the exact ordered grounding sequence,
    including any duplicate ranges used by the original seq_limit filter.
    Partial/failed builds never replace a previously completed index.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=path.name + ".building-", dir=path.parent)
    os.close(descriptor)
    connection = None
    try:
        connection = sqlite3.connect(temporary)
        # The database is a disposable rebuild artifact, published by replace
        # only after completion. Avoid journal duplication on memory-backed FS.
        connection.execute("PRAGMA journal_mode=OFF")
        connection.execute("PRAGMA synchronous=OFF")
        connection.execute("PRAGMA cache_size=-2048")
        connection.execute("PRAGMA mmap_size=0")
        connection.execute(f"PRAGMA user_version={SCHEMA_VERSION}")
        connection.execute("CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
        connection.execute(
            "CREATE TABLE qa (position INTEGER PRIMARY KEY, qa_id TEXT UNIQUE NOT NULL, "
            "payload TEXT NOT NULL, grounding TEXT NOT NULL, ngraphs INTEGER NOT NULL)"
        )
        count = 0
        for qa_id, payload in records:
            if not isinstance(qa_id, str) or not isinstance(payload, dict):
                raise ValueError("Each indexed QA must have a string ID and dictionary payload")
            grounding = list(ground_record(payload))
            if not grounding:
                raise ValueError(f"QA {qa_id!r} has no grounded graph intervals")
            if any(not isinstance(interval, (tuple, list)) or len(interval) != 2
                   or any(not isinstance(frame, str) for frame in interval)
                   for interval in grounding):
                raise ValueError(f"QA {qa_id!r} has malformed graph intervals")
            connection.execute(
                "INSERT INTO qa VALUES (?, ?, ?, ?, ?)",
                (count, qa_id, json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
                 json.dumps(grounding, separators=(",", ":")), len(grounding)),
            )
            count += 1
            if count % 1000 == 0:
                connection.commit()
        connection.executemany(
            "INSERT INTO metadata VALUES (?, ?)",
            [("row_count", str(count)), ("complete", "1")],
        )
        connection.commit()
        if connection.execute("PRAGMA quick_check").fetchone()[0] != "ok":
            raise RuntimeError("AGQA QA index integrity check failed")
        connection.close()
        connection = None
        os.replace(temporary, path)
        return count
    finally:
        if connection is not None:
            connection.close()
        if os.path.exists(temporary):
            os.unlink(temporary)


class IndexedQA:
    """Read-only, per-process SQLite access with a 2 MiB page cache.

    A finite seq_limit needs only eight bytes per eligible sample for source
    positions, rather than a Python list of millions of integers/dictionaries.
    No-filter access uses a constant-space range. Dataset results retain source
    order; random shuffling remains the trainer/sampler's responsibility.
    """

    def __init__(self, path, seq_limit=float("inf")):
        self.path = str(Path(path).resolve())
        if not Path(self.path).is_file():
            raise FileNotFoundError(
                f"Missing indexed AGQA QA: {self.path}. Re-run preprocessing with DYGENC_INDEXED_QA=1."
            )
        if math.isnan(seq_limit) or seq_limit < 0:
            raise ValueError("seq_limit must be nonnegative or positive infinity")
        self._connection = None
        self._pid = None
        try:
            self._load_positions(seq_limit)
        except Exception:
            self.close()
            raise

    def _load_positions(self, seq_limit):
        connection = self._connect()
        version = connection.execute("PRAGMA user_version").fetchone()[0]
        if version != SCHEMA_VERSION:
            raise ValueError(f"Unsupported AGQA QA index version: {version}")
        metadata = dict(connection.execute("SELECT key, value FROM metadata"))
        if metadata.get("complete") != "1":
            raise ValueError("AGQA QA index was not completed")
        self.total_count = int(metadata["row_count"])
        count, first, last = connection.execute(
            "SELECT COUNT(*), MIN(position), MAX(position) FROM qa"
        ).fetchone()
        if count != self.total_count or (count and (first != 0 or last != count - 1)):
            raise ValueError("AGQA QA index has missing or noncontiguous positions")
        if seq_limit == float("inf"):
            self.positions = range(count)
        else:
            # Clamp huge finite limits (full-data mode) before binding to
            # SQLite's signed integer type; graph counts cannot exceed this.
            limit = min(math.floor(seq_limit), 2**63 - 1)
            self.positions = array("Q", (row[0] for row in connection.execute(
                "SELECT position FROM qa WHERE ngraphs > 0 AND ngraphs <= ? ORDER BY position", (limit,)
            )))

    def _connect(self):
        if self._connection is None or self._pid != os.getpid():
            self.close()
            uri = Path(self.path).as_uri() + "?mode=ro"
            self._connection = sqlite3.connect(uri, uri=True)
            self._connection.execute("PRAGMA query_only=ON")
            self._connection.execute("PRAGMA cache_size=-2048")
            self._connection.execute("PRAGMA mmap_size=0")
            self._pid = os.getpid()
        return self._connection

    def __len__(self):
        return len(self.positions)

    def __getitem__(self, index):
        position = self.positions[operator.index(index)]
        row = self._connect().execute(
            "SELECT qa_id, payload, grounding FROM qa WHERE position=?", (position,)
        ).fetchone()
        if row is None:
            raise RuntimeError(f"AGQA QA index is missing position {position}")
        qa_id, payload, grounding = row
        return qa_id, json.loads(payload), [tuple(interval) for interval in json.loads(grounding)]

    def close(self):
        if self._connection is not None:
            self._connection.close()
            self._connection = None
        self._pid = None

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_connection"] = None
        state["_pid"] = None
        return state

    def __del__(self):
        if getattr(self, "_connection", None) is not None:
            self.close()
