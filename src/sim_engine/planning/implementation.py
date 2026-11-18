"""Freeze Python, native source and binary identity together for both endpoints."""

from pathlib import Path

from sim_engine.core.io import file_sha256


def implementation_sources(root=None):
    root = Path(root) if root is not None else Path(__file__).parents[1]
    return {
        str(p.relative_to(root)): file_sha256(p)
        for p in sorted(root.rglob("*"))
        if p.is_file() and p.suffix in {".py", ".cpp", ".h", ".so", ".pyd"}
    }
