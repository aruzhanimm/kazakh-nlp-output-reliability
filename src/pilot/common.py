from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys

ROOT = Path(__file__).resolve().parents[1]


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def digest(value):
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line]


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(canonical(row) + "\n")
    temp.replace(path)


def environment():
    return {"python": sys.version, "platform": platform.platform(),
            "packages": {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()}}


def code_hash():
    paths = sorted((ROOT / "pilot").glob("*.py")) + sorted((ROOT / "prompts").glob("*"))
    return digest({str(p.relative_to(ROOT)): file_hash(p) for p in paths if p.is_file()})


class RunStore:
    """Immutable identity + atomic per-item records; never skip a missing prediction."""
    def __init__(self, path, identity):
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        self.identity = {**identity, "code_sha256": code_hash(), "environment": environment()}
        self.hash = digest(self.identity)
        meta = self.path / "identity.json"
        guard = self.path / "IDENTITY.lock"
        # Serialize initial metadata creation before the longer model-run lock.
        with guard.open("x", encoding="utf-8") as f:
            f.write(str(os.getpid()))
        try:
            if (self.path / "RUNNING.lock").exists():
                raise ValueError("A writer already owns this run")
            if meta.exists():
                if read_json(meta) != self.identity:
                    raise ValueError("Incompatible run identity; choose a new run directory")
            else:
                write_json(meta, self.identity)
                for source in sorted((ROOT / "pilot").glob("*.py")) + sorted((ROOT / "prompts").glob("*")):
                    if source.is_file():
                        target = self.path / "source" / source.relative_to(ROOT)
                        target.parent.mkdir(parents=True, exist_ok=True)
                        target.write_bytes(source.read_bytes())
        finally:
            guard.unlink()
        (self.path / "items").mkdir(exist_ok=True)

    def item_path(self, item_id):
        return self.path / "items" / (digest(item_id) + ".json")

    def get(self, item_id):
        path = self.item_path(item_id)
        if not path.exists():
            return None
        item = read_json(path)
        if item["run_hash"] != self.hash or item["id"] != item_id:
            raise ValueError("Corrupt or incompatible item")
        return item

    def put(self, item):
        if self.get(item["id"]) is not None:
            raise ValueError("Refusing to overwrite a completed prediction")
        write_json(self.item_path(item["id"]), {**item, "run_hash": self.hash})


class RunLock:
    """Prevent two writers. Remove stale lock manually only after verifying PID is dead."""
    def __init__(self, path):
        self.path = Path(path) / "RUNNING.lock"

    def __enter__(self):
        with self.path.open("x", encoding="utf-8") as f:
            f.write(str(os.getpid()))
        return self

    def __exit__(self, *exc):
        self.path.unlink()
