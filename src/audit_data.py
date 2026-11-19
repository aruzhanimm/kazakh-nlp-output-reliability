"""Reproduce the stage-0 audit using pinned official files; no model inference.

Python 3.10+; standard library only. No automatic relabelling or data cleaning.
Run: python audit_data.py --download --data-dir data --output outputs/data_audit.json
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import io
import json
from pathlib import Path
import unicodedata
import urllib.request
import zipfile


def canonical_text(text):
    """NFC and whitespace collapse only: case, punctuation and script are retained."""
    return " ".join(unicodedata.normalize("NFC", text).split())


def key(text):
    return hashlib.sha256(canonical_text(text).encode("utf-8")).hexdigest()


def download_and_verify(record, root, download):
    path = root / record["dataset"] / record["file_name"]
    if not path.exists():
        if not download:
            raise FileNotFoundError(f"{path}: use --download for missing official files")
        path.parent.mkdir(parents=True, exist_ok=True)
        request = urllib.request.Request(record["download_url"], headers={"User-Agent": "Kazakh-NLP-research-audit"})
        with urllib.request.urlopen(request, timeout=45) as response:
            payload = response.read()
        if hashlib.sha256(payload).hexdigest() != record["sha256"]:
            raise ValueError(f"Checksum mismatch for downloaded {path.name}")
        temp = path.with_suffix(path.suffix + ".part")
        temp.write_bytes(payload)
        temp.replace(path)
    if hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256"]:
        raise ValueError(f"Existing file checksum mismatch: {path}; no overwrite performed")
    return path


def read_ner(path, split):
    rows, tokens, tags = [], [], []
    for line in path.read_text(encoding="utf-8-sig").splitlines() + [""]:
        if not line.strip():
            if tokens:
                rows.append({"id": f"ner:{split}:{len(rows)}", "tokens": tokens,
                             "labels": tags, "text": " ".join(tokens)})
                tokens, tags = [], []
        else:
            fields = line.split()
            if len(fields) != 2:
                raise ValueError(f"Expected exactly two CoNLL columns in {path.name}")
            token, tag = fields
            if tag != "O" and not (tag.startswith("B-") or tag.startswith("I-")):
                raise ValueError(f"Invalid BIO tag: {tag!r}")
            tokens.append(token)
            tags.append(tag)
    return rows


def read_sa(path, split):
    with zipfile.ZipFile(path) as archive:
        members = [x for x in archive.namelist() if x.endswith(".csv")]
        if len(members) != 1:
            raise ValueError("Expected one CSV per official archive")
        with archive.open(members[0]) as stream:
            reader = csv.DictReader(io.TextIOWrapper(stream, encoding="utf-8-sig"))
            required = {"custom_id", "text", "text_cleaned", "label", "domain"}
            if not required.issubset(reader.fieldnames or []):
                raise ValueError(f"Unexpected SA schema: {reader.fieldnames}")
            rows = []
            for row in reader:
                if row["label"] not in {"0", "1"}:
                    raise ValueError(f"Unexpected polarity label: {row['label']!r}")
                rows.append({"id": f"sa:{split}:{row['custom_id']}", **row})
            return rows


def text_groups(rows, column="text"):
    groups = defaultdict(list)
    for row in rows:
        groups[key(row[column])].append(row)
    return groups


def invalid_iob2_count(tags):
    prev, bad = "O", 0
    for tag in tags:
        if tag.startswith("I-") and prev not in {"B-" + tag[2:], tag}:
            bad += 1
        prev = tag
    return bad


def split_summary(rows, task):
    groups = text_groups(rows)
    label_field = "labels" if task == "ner" else "label"
    signatures = lambda row: tuple(row[label_field]) if task == "ner" else row[label_field]
    result = {
        "rows": len(rows), "unique_normalized_texts": len(groups),
        "duplicate_text_rows": len(rows) - len(groups),
        "conflicting_label_text_groups": sum(len({signatures(row) for row in group}) > 1 for group in groups.values()),
        "empty_text_rows": sum(not canonical_text(row["text"]) for row in rows),
    }
    if task == "ner":
        counts = Counter(tag[2:] for row in rows for tag in row["labels"] if tag.startswith("B-"))
        bad_counts = [invalid_iob2_count(row["labels"]) for row in rows]
        result.update(tokens=sum(len(row["tokens"]) for row in rows), entity_types=len(counts),
                      entity_counts=dict(sorted(counts.items())), invalid_iob2_transitions=sum(bad_counts),
                      sentences_with_invalid_iob2=sum(x > 0 for x in bad_counts))
    else:
        cleaned = text_groups(rows, "text_cleaned")
        result.update(label_counts=dict(sorted(Counter(r["label"] for r in rows).items())),
                      domains=dict(sorted(Counter(r["domain"] for r in rows).items())),
                      duplicate_cleaned_text_rows=len(rows) - len(cleaned),
                      duplicate_custom_id_rows=len(rows) - len({r["custom_id"] for r in rows}))
    return result


def overlap(a, b, task, column="text"):
    ga, gb = text_groups(a, column), text_groups(b, column)
    common = ga.keys() & gb.keys()
    label_field = "labels" if task == "ner" else "label"
    conflicting = 0
    for group in common:
        signatures = {tuple(row[label_field]) if task == "ner" else row[label_field]
                      for row in ga[group] + gb[group]}
        conflicting += len(signatures) > 1
    return {"shared_unique_texts": len(common), "left_rows": sum(len(ga[k]) for k in common),
            "right_rows": sum(len(gb[k]) for k in common), "shared_groups_with_label_disagreement": conflicting}


def proposed_hygiene_counts(splits):
    """Count a text-only holdout-priority rule without changing the source files."""
    test_keys = {key(r["text"]) for r in splits["test"]}
    valid_keys = {key(r["text"]) for r in splits["valid"]}
    held_out_keys = test_keys | valid_keys
    train_kept = [r for r in splits["train"] if key(r["text"]) not in held_out_keys]
    valid_kept = [r for r in splits["valid"] if key(r["text"]) not in test_keys]
    strict_train = [r for r in train_kept if not invalid_iob2_count(r.get("labels", []))]
    return {"rule": "Remove train rows matching any official dev/test text; remove dev rows matching test. Keep test unchanged.",
            "train_rows_retained": len(train_kept), "valid_rows_retained": len(valid_kept),
            "test_rows_retained": len(splits["test"]),
            "train_rows_after_excluding_invalid_iob2_sequences": len(strict_train),
            "additional_invalid_iob2_train_rows_excluded": len(train_kept) - len(strict_train),
            "note": "Counts only; no source files modified. Within-split duplicates and conflicting labels retained and reported."}


def load_all(root, records):
    data = {"ner": {}, "sa": {}}
    for record in records:
        path = root / record["dataset"] / record["file_name"]
        parser = read_ner if record["task"] == "ner" else read_sa
        data[record["task"]][record["split"]] = parser(path, record["split"])
    return data


def audit(data):
    result = {"normalization": "Unicode NFC + whitespace collapse; case/punctuation/script retained",
              "scope": "Exact normalized-text audit, not near-duplicate detection and not model evaluation",
              "tasks": {}}
    for task, splits in data.items():
        entry = {"splits": {s: split_summary(rows, task) for s, rows in splits.items()}, "overlaps": {}}
        for a, b in [("train", "valid"), ("train", "test"), ("valid", "test")]:
            pair = {"text": overlap(splits[a], splits[b], task)}
            if task == "sa":
                pair["text_cleaned"] = overlap(splits[a], splits[b], task, "text_cleaned")
                pair["shared_custom_ids"] = len({r["custom_id"] for r in splits[a]} & {r["custom_id"] for r in splits[b]})
            entry["overlaps"][a + "__" + b] = pair
        entry["proposed_hygiene"] = proposed_hygiene_counts(splits)
        result["tasks"][task] = entry
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path(__file__).with_name("data_manifest.json"))
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, default=Path("outputs/data_audit.json"))
    parser.add_argument("--download", action="store_true", help="Fetch missing official files; verify every SHA-256")
    args = parser.parse_args()
    records = json.loads(args.manifest.read_text(encoding="utf-8"))["files"]
    with ThreadPoolExecutor(max_workers=3) as pool:
        list(pool.map(lambda rec: download_and_verify(rec, args.data_dir, args.download), records))
    report = audit(load_all(args.data_dir, records))
    report["source_manifest_sha256"] = hashlib.sha256(args.manifest.read_bytes()).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "counts": {
        task: {split: r["rows"] for split, r in item["splits"].items()}
        for task, item in report["tasks"].items()}}, indent=2))


if __name__ == "__main__":
    main()
