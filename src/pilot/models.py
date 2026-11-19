"""Resolve model IDs once, then always load by immutable commit."""
import argparse
import os
from pathlib import Path
from .common import ROOT, file_hash, read_json, write_json

MODELS = {"encoder": "FacebookAI/xlm-roberta-base", "llm": "Qwen/Qwen2.5-7B-Instruct"}


def cache_dir():
    path = ROOT / ".cache/huggingface"
    path.mkdir(parents=True, exist_ok=True)
    return str(path)


def load_tokenizer(slot):
    from transformers import AutoTokenizer
    spec = read_json(ROOT / "configs/models.lock.json")[slot]
    folder = ROOT / "artifacts/tokenizers" / slot
    if not spec.get("tokenizer_sha256"):
        raise ValueError("Run python -m pilot.models to pin tokenizer files")
    for name, sha in spec["tokenizer_sha256"].items():
        if file_hash(folder / name) != sha:
            raise ValueError("Tokenizer file checksum mismatch")
    return AutoTokenizer.from_pretrained(folder, use_fast=True, trust_remote_code=False, local_files_only=True)


def resolve(weights=False, slot=None):
    from huggingface_hub import HfApi, snapshot_download
    from transformers import AutoTokenizer
    path = ROOT / "configs/models.lock.json"
    specs = read_json(path) if path.exists() else {}
    for name, model_id in MODELS.items():
        if slot and name != slot:
            continue
        if name not in specs:
            specs[name] = {"model_id": model_id, "revision": HfApi().model_info(model_id).sha}
            write_json(path, specs)
        spec = specs[name]
        tok = AutoTokenizer.from_pretrained(model_id, revision=spec["revision"], cache_dir=cache_dir(), use_fast=True)
        tok.save_pretrained(ROOT / "artifacts/tokenizers" / name)
        specs[name]["tokenizer_sha256"] = {p.name: file_hash(p) for p in sorted((ROOT / "artifacts/tokenizers" / name).iterdir()) if p.is_file()}
        write_json(path, specs)
        if weights:
            snapshot = Path(snapshot_download(model_id, revision=spec["revision"], cache_dir=cache_dir(),
                            allow_patterns=["*.safetensors", "*.json", "tokenizer*", "sentencepiece*", "*.model"], max_workers=3))
            hashes = {p.name: file_hash(p) for p in sorted(snapshot.iterdir()) if p.is_file()}
            write_json(ROOT / "artifacts" / f"{name}_snapshot_hashes.json", {"revision": spec["revision"], "files": hashes})
        print(name, spec, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--weights", action="store_true")
    p.add_argument("--slot", choices=MODELS)
    a = p.parse_args()
    resolve(a.weights, a.slot)
