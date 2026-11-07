"""Versioned, append-only experiment outputs. Resume refuses changed inputs or settings."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def json_hash(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{i}: incomplete/corrupt record; recover this file before resuming") from exc
    return rows


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(path)


class RunStore:
    def __init__(self, folder: Path, identity: dict):
        self.folder = Path(folder)
        self.folder.mkdir(parents=True, exist_ok=True)
        manifest = self.folder / "manifest.json"
        expected = {"fingerprint": json_hash(identity), "identity": identity}
        if manifest.exists():
            if json.loads(manifest.read_text(encoding="utf-8")) != expected:
                raise ValueError(f"{folder}: settings, code, data or adapter changed; use a new run directory")
        elif any(self.folder.iterdir()):
            raise ValueError(f"{folder}: existing files without a manifest; refusing to mix runs")
        else:
            write_json(manifest, expected)
        self.rows = read_jsonl(self.folder / "traces.jsonl")
        ids = [r["episode"] for r in self.rows]
        if len(set(ids)) != len(ids):
            raise ValueError(f"{folder}: duplicate episode records")
        self.completed = set(ids)

    def append(self, records: list[dict]) -> None:
        ids = [r["episode"] for r in records]
        if len(set(ids)) != len(ids) or self.completed.intersection(ids):
            raise ValueError("An episode may only be written once")
        with (self.folder / "traces.jsonl").open("a", encoding="utf-8") as f:
            for row in records:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())
        self.rows.extend(records)
        self.completed.update(ids)
