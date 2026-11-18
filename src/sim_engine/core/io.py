"""Content identity and durable JSON records."""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import tempfile
from pathlib import Path


def file_sha256(path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def fingerprint(value) -> str:
    """SHA-256 of canonical JSON; the identity of a request, result or configuration."""
    canonical = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(canonical.encode()).hexdigest()


def read_json(path):
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as stream:
        return json.load(stream)


def create_json(path, value):
    """Durable create-only record: an existing file is never overwritten."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def replace_json(path, value):
    """Atomically replace a mutable document such as a status view or run manifest."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as stream:
        json.dump(value, stream, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
        temporary = Path(stream.name)
    temporary.replace(path)


SOURCE_SUFFIXES = {".py", ".cpp", ".h", ".so", ".pyd"}


def implementation_sources(root=None) -> dict[str, str]:
    """Hashes of every Python, native source and binary file of the installed package."""
    root = Path(root) if root is not None else Path(__file__).parents[1]
    return {str(p.relative_to(root)): file_sha256(p) for p in sorted(root.rglob("*"))
            if p.is_file() and p.suffix in SOURCE_SUFFIXES}
