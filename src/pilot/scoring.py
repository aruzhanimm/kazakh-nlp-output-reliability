"""Explicit failure-aware metrics. Gold BIO is strict; predicted illegal I -> B."""
from __future__ import annotations

import json


def bio_spans(tags, repair=False):
    spans, active, start, repairs = [], None, None, 0
    for i, tag in enumerate(list(tags) + ["O"]):
        if tag != "O" and (len(tag) < 3 or tag[:2] not in {"B-", "I-"}):
            raise ValueError(f"Invalid tag {tag}")
        prefix, kind = ("O", None) if tag == "O" else (tag[0], tag[2:])
        if prefix == "I" and active != kind:
            if not repair:
                raise ValueError(f"Illegal IOB2 at word {i}: {tag}")
            prefix = "B"
            repairs += 1
        if active is not None and (prefix != "I" or kind != active):
            spans.append((start, i, active))
            active = None
        if prefix == "B":
            start, active = i, kind
    return spans, repairs


def spans_bio(spans, n):
    tags = ["O"] * n
    for start, end, kind in spans:
        for i in range(start, end):
            if tags[i] != "O":
                raise ValueError("Overlapping spans")
            tags[i] = ("B-" if i == start else "I-") + kind
    return tags


def parse_sa(raw):
    try:
        obj = json.loads(raw)
        if not isinstance(obj, dict) or set(obj) != {"label"} or obj["label"] not in ("negative", "positive"):
            raise ValueError("Expected only an allowed label")
        return {"label": 1 if obj["label"] == "positive" else 0, "failed": False, "errors": []}
    except (ValueError, TypeError) as e:
        return {"label": None, "failed": True, "errors": [str(e)]}


def parse_ner(raw, n_words, allowed):
    result = {"spans": [], "invalid_candidates": 0, "failed": False, "errors": []}
    try:
        obj = json.loads(raw)
        if not isinstance(obj, list):
            raise ValueError("Expected a JSON list")
    except (ValueError, TypeError) as e:
        return {**result, "failed": True, "errors": [str(e)]}
    seen = set()
    for record in obj:
        signature = json.dumps(record, sort_keys=True, ensure_ascii=False)
        if signature in seen:
            continue
        seen.add(signature)
        valid = isinstance(record, dict) and set(record) == {"start", "end", "type"}
        valid = valid and type(record["start"]) is int and type(record["end"]) is int
        valid = valid and 0 <= record["start"] < record["end"] <= n_words
        valid = valid and isinstance(record["type"], str) and record["type"] in allowed
        if valid:
            start, end, kind = record["start"], record["end"], record["type"]
            # First valid candidate in response order wins any overlap.
            valid = not any(start < b and a < end for a, b, _ in result["spans"])
        if not valid:
            result["invalid_candidates"] += 1
            result["errors"].append({"invalid_record": record})
        else:
            result["spans"].append((start, end, kind))
    result["failed"] = bool(result["errors"])
    return result


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return {"precision": p, "recall": r, "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
            "tp": tp, "fp": fp, "fn": fn, "support": tp + fn}


def score_sa(gold, predictions):
    if not gold or len(gold) != len(predictions):
        raise ValueError("One prediction (including failure) required per example")
    if any(y not in (0, 1) for y in gold):
        raise ValueError("Invalid gold class")
    per = {}
    confusion = [[0, 0, 0], [0, 0, 0]]  # negative, positive, INVALID
    for y, p in zip(gold, predictions):
        confusion[y][p if type(p) is int and p in (0, 1) else 2] += 1
    for c in (0, 1):
        tp = confusion[c][c]
        fp = confusion[1 - c][c]
        fn = sum(confusion[c]) - tp
        per[str(c)] = prf(tp, fp, fn)
    return {"n": len(gold), "macro_f1": sum(v["f1"] for v in per.values()) / 2,
            "accuracy": sum(confusion[c][c] for c in (0, 1)) / len(gold),
            "invalid": sum(row[2] for row in confusion), "per_class": per,
            "confusion_columns": ["negative", "positive", "INVALID"], "confusion": confusion}


def score_ner(gold, predictions, allowed):
    if not gold or len(gold) != len(predictions):
        raise ValueError("One prediction (including failure) required per example")
    totals = [0, 0, 0]
    by_type = {t: [0, 0, 0] for t in allowed}
    exact, failures, invalid = 0, 0, 0
    for true_spans, pred in zip(gold, predictions):
        g, p = set(map(tuple, true_spans)), set(map(tuple, pred["spans"]))
        extra = pred.get("invalid_candidates", 0)
        counts = (len(g & p), len(p - g) + extra, len(g - p))
        totals = [a + b for a, b in zip(totals, counts)]
        for t in allowed:
            gt, pt = {s for s in g if s[2] == t}, {s for s in p if s[2] == t}
            by_type[t] = [a + b for a, b in zip(by_type[t], (len(gt & pt), len(pt - gt), len(gt - pt)))]
        exact += g == p and not pred.get("failed", False) and extra == 0
        failures += bool(pred.get("failed", False))
        invalid += extra
    per = {t: prf(*v) for t, v in by_type.items()}
    supported = [v["f1"] for v in per.values() if v["support"]]
    return {"n": len(gold), **prf(*totals), "per_type": per,
            "macro_f1_supported_types": sum(supported) / len(supported) if supported else 0.0,
            "exact_set_accuracy": exact / len(gold), "format_failures": failures,
            "format_failure_rate": failures / len(gold), "invalid_candidate_fp": invalid,
            "note": "Invalid candidate FPs enter micro-F1; cannot be assigned a valid entity type."}
