"""Restore the exact 500 SA inputs from a separately obtained pinned source ZIP.

This script does not download data. See DATA_ATTRIBUTION.md for source identity.
"""
from pathlib import Path
from zipfile import ZipFile
import argparse
import csv
import hashlib
import io
import json

SOURCE_SHA256 = "cec729d4fd5dda6e1d73c3ce9d6e5feed01b24be79b8135bfb0a204e3f1e6451"


def restore(source, output):
    if hashlib.sha256(source.read_bytes()).hexdigest() != SOURCE_SHA256:
        raise ValueError("The source ZIP is not the pinned KazSAnDRA test release.")
    items = [json.loads(line) for line in Path(__file__).with_name("sa_items.jsonl").read_text(encoding="utf-8").splitlines()]
    with ZipFile(source) as archive:
        rows = {"sa:test:" + row["custom_id"]: row for row in csv.DictReader(io.StringIO(archive.read("05_pc_test.csv").decode("utf-8-sig")))}
    restored = []
    for item in items:
        row = rows[item["id"]]
        text = row["text"]
        visible = text[:item["visible_characters"]]
        for value, key in ((text, "original_text_sha256"), (visible, "visible_text_sha256")):
            if hashlib.sha256(value.encode("utf-8")).hexdigest() != item[key]:
                raise ValueError("Text hash mismatch: " + item["id"])
        if int(row["label"]) != item["gold_label"]:
            raise ValueError("Gold label mismatch: " + item["id"])
        restored.append({"id": item["id"], "original_text": text, "visible_text": visible, "gold_label": item["gold_label"]})
    if output.exists():
        raise FileExistsError("Refusing to overwrite " + str(output))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in restored), encoding="utf-8")
    print(json.dumps({"status": "PASS_PINNED_SOURCE_AND_ALL_TEXT_HASHES", "items": len(restored)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-zip", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    restore(args.source_zip, args.output)
