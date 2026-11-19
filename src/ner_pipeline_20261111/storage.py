"""Immutable, hash-checked evidence files and process locks for isolated NER work.

Files are fsynced before no-overwrite publication by a same-directory Windows
rename (which fails if destination exists), or a POSIX hard link.
This is tested against process interruption; power-loss durability on Windows is
not certified. No destructive recovery or automatic stale-attempt repair exists.
"""
from contextlib import contextmanager
from pathlib import Path
import errno
import hashlib
import json
import os
import re
import uuid


class StoreConflict(ValueError):
    pass


class StoreCorruption(ValueError):
    pass


class StoreBusy(RuntimeError):
    pass


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def _sync_directory(path):
    # POSIX can additionally flush directory metadata. Windows standard-library
    # file handles do not provide a portable equivalent for a directory.
    if os.name != "nt":
        fd = os.open(path, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)


def read_evidence(path):
    path = Path(path)
    try:
        envelope = json.loads(path.read_text(encoding="utf-8"))
        if set(envelope) != {"payload", "payload_sha256"}:
            raise ValueError("Invalid envelope keys")
        if digest(envelope["payload"]) != envelope["payload_sha256"]:
            raise ValueError("Payload digest differs")
        return envelope["payload"]
    except (ValueError, UnicodeError, TypeError, KeyError) as exc:
        raise StoreCorruption(f"Invalid evidence: {path.name}") from exc


def write_evidence(path, payload, fault_hook=None):
    """Publish a complete JSON envelope exactly once, never replace an artifact.

    Existing byte-equivalent payloads are reused after verification. An orphaned
    .partial file after interrupted publication is retained for inspection.
    """
    path = Path(path)
    encoded = canonical({"payload": payload, "payload_sha256": digest(payload)})
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if canonical(read_evidence(path)) != canonical(payload):
            raise StoreConflict(f"Refusing changed evidence: {path.name}")
        return False
    # A short unrelated basename avoids adding the full artifact name twice on
    # Windows. Its exclusive creation still refuses even a UUID collision.
    partial = path.with_name(".partial_" + uuid.uuid4().hex[:16])
    with partial.open("xb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    if fault_hook:
        fault_hook("after_fsync_before_publish", path)
    try:
        # Windows rename() fails if destination exists. POSIX rename can replace
        # an existing file, so POSIX must instead use exclusive hard-link publish.
        # No os.replace() or write-in-place fallback exists on either platform.
        if os.name == "nt":
            os.rename(partial, path)
        else:
            os.link(partial, path)
        _sync_directory(path.parent)
    except FileExistsError:
        if canonical(read_evidence(path)) != canonical(payload):
            raise StoreConflict(f"Concurrent conflicting evidence: {path.name}")
    if fault_hook:
        fault_hook("after_publish_before_cleanup", path)
    if partial.exists():
        partial.unlink()  # only this function's uniquely created, nonrecursive temp
    return True


@contextmanager
def process_lock(path):
    """Nonblocking OS lock; a crashed process cannot leave a live lock behind.

    The small lock file stays in place; no process may replace or remove it.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    stream = path.open("a+b")
    acquired = False
    try:
        if path.stat().st_size == 0:
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            if exc.errno in (errno.EACCES, errno.EAGAIN, errno.EDEADLK, 13, 36):
                raise StoreBusy("Another process owns this evidence lock") from exc
            raise
        acquired = True
        yield
    finally:
        if acquired:
            stream.seek(0)
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
        stream.close()


class EvidenceStore:
    """One immutable run/condition binding per explicitly selected directory."""
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def bind(self, binding):
        with process_lock(self.root / ".binding.lock"):
            write_evidence(self.root / "RUN_BINDING.json", binding)

    def request_dir(self, identity):
        if not isinstance(identity, str) or not re.fullmatch(r"[0-9a-f]{64}", identity):
            raise ValueError("Request identity must be a SHA256 hex digest")
        path = self.root / "requests" / identity
        path.mkdir(parents=True, exist_ok=True)
        return path
