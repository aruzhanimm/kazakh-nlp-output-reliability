"""Immutable source audit, derived training/dev views, deterministic samples."""
import argparse
from collections import Counter, defaultdict
import math
from pathlib import Path
import random

from audit_data import download_and_verify, invalid_iob2_count, key, load_all, split_summary
from .common import ROOT, digest, file_hash, read_json, write_json, write_jsonl


def retained_views(splits):
    test = {key(r["text"]) for r in splits["test"]}
    dev = {key(r["text"]) for r in splits["valid"]}
    views, exclusions = {}, []
    for split in ("train", "valid"):
        views[split] = []
        for row in splits[split]:
            k, reasons = key(row["text"]), []
            if k in test:
                reasons.append("text_in_official_test")
            if split == "train" and k in dev:
                reasons.append("text_in_original_dev")
            if split == "train" and invalid_iob2_count(row.get("labels", [])):
                reasons.append("invalid_iob2")
            if reasons:
                exclusions.append({"id": row["id"], "text_key": k, "reasons": reasons})
            else:
                views[split].append({**row, "text_key": k})
    return views, exclusions


def stratified_sample(rows, n, seed):
    """Proportional SA label x domain; largest remainders, deterministic ties."""
    groups = defaultdict(list)
    for r in rows:
        groups[(r["label"], r["domain"])].append(r)
    quotas = {k: n * len(v) / len(rows) for k, v in groups.items()}
    counts = {k: math.floor(v) for k, v in quotas.items()}
    order = sorted(groups, key=lambda k: (-(quotas[k] - counts[k]), k))
    for k in order[:n - sum(counts.values())]:
        counts[k] += 1
    rng, selected = random.Random(seed), []
    for k in sorted(groups):
        selected.extend(rng.sample(groups[k], counts[k]))
    rng.shuffle(selected)
    return selected


def ner_train_sample(rows, n, seed, types):
    """Coverage-constrained pilot train only; dev remains simple random sampling."""
    ordered = sorted(rows, key=lambda r: digest([seed, r["id"]]))
    selected, covered = [], set()
    for kind in sorted(types, key=lambda t: sum(any(x == "B-" + t for x in r["labels"]) for r in rows)):
        if kind not in covered:
            row = next(r for r in ordered if "B-" + kind in r["labels"])
            selected.append(row)
            covered.update(t[2:] for t in row["labels"] if t != "O")
    selected_ids = {r["id"] for r in selected}
    selected += random.Random(seed).sample([r for r in rows if r["id"] not in selected_ids], n - len(selected))
    random.Random(seed).shuffle(selected)
    return selected


def choose_demos(task, rows, seed):
    ordered = sorted(rows, key=lambda r: digest([seed, r["id"]]))
    if task == "sa":
        chosen = []
        for label in ("0", "1"):
            chosen.extend([r for r in ordered if r["label"] == label and 3 <= len(r["text"].split()) <= 48][:2])
        if len(chosen) != 4:
            raise ValueError("Insufficient short class-balanced training demonstrations")
        return chosen
    short = [r for r in ordered if len(r["tokens"]) <= 32]
    chosen = [next(r for r in short if all(t == "O" for t in r["labels"]))]
    covered = set()
    for _ in range(3):
        candidates = [r for r in short if r not in chosen]
        best = max(candidates, key=lambda r: len({t[2:] for t in r["labels"] if t != "O"} - covered))
        chosen.append(best)
        covered.update(t[2:] for t in best["labels"] if t != "O")
    return chosen


def prepare(root=ROOT, train_n=4000, dev_n=200, seed=13):
    manifest = read_json(root / "data_manifest.json")
    for rec in manifest["files"]:
        download_and_verify(rec, root / "data/raw", False)
    data = load_all(root / "data/raw", manifest["files"])
    out = root / "data/derived"
    meta = {"source_manifest_sha256": file_hash(root / "data_manifest.json"), "seed": seed,
            "train_n": train_n, "dev_n": dev_n, "tasks": {}, "stage": "development_only"}
    for task, splits in data.items():
        for split, rows in splits.items():
            if len({r["id"] for r in rows}) != len(rows):
                raise ValueError(f"Nonunique source IDs: {task}/{split}")
        views, exclusions = retained_views(splits)
        write_jsonl(out / f"{task}_exclusions.jsonl", exclusions)
        for split, rows in views.items():
            write_jsonl(out / f"{task}_{split}.jsonl", rows)
        types = sorted({t[2:] for r in splits["train"] for t in r["labels"] if t != "O"}) if task == "ner" else []
        if task == "ner":
            write_json(out / "ner_labels.json", {"types": types, "bio": ["O"] + [p + t for t in types for p in ("B-", "I-")]})
        samples = {}
        for split, n in (("train", train_n), ("valid", dev_n)):
            rows = views[split]
            if n > len(rows):
                raise ValueError("Sample exceeds retained view")
            if task == "sa":
                selected = stratified_sample(rows, n, seed + (split == "valid"))
            elif split == "train":
                selected = ner_train_sample(rows, n, seed, types)
            else:
                selected = random.Random(seed + 1).sample(rows, n)
            path = out / f"{task}_pilot_{split}.jsonl"
            write_jsonl(path, selected)
            samples[split] = {"ids": [r["id"] for r in selected], "sha256": file_hash(path),
                              "summary": split_summary(selected, task)}
        # Demos from the same 4,000-row retained pilot training view.
        from .common import read_jsonl
        demos = choose_demos(task, read_jsonl(out / f"{task}_pilot_train.jsonl"), seed)
        write_jsonl(out / f"{task}_demos.jsonl", demos)
        meta["tasks"][task] = {"retained": {s: split_summary(r, task) for s, r in views.items()},
                               "exclusions": len(exclusions), "samples": samples,
                               "demo_ids": [r["id"] for r in demos]}
    write_json(out / "manifest.json", meta)
    print({t: {s: r["rows"] for s, r in m["retained"].items()} for t, m in meta["tasks"].items()})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()
    prepare(seed=args.seed)
