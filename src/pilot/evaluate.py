import argparse
import math
import statistics
from pathlib import Path
from .common import ROOT, digest, read_json, read_jsonl, write_json, write_jsonl
from .inputs import load_panel
from .scoring import bio_spans, score_ner, score_sa


def summarize(task, rows, predictions):
    if len({p["id"] for p in predictions}) != len(predictions):
        raise ValueError("Duplicate prediction IDs")
    by_id = {p["id"]: p for p in predictions}
    if set(by_id) != {r["id"] for r in rows}:
        raise ValueError("Predictions must cover exactly the shared development panel")
    ordered = [by_id[r["id"]] for r in rows]
    if task == "sa":
        result = score_sa([int(r["label"]) for r in rows], [p["parsed"]["label"] for p in ordered])
        result["format_failures"] = sum(p["parsed"]["failed"] for p in ordered)
        result["format_failure_rate"] = result["format_failures"] / len(rows)
    else:
        allowed = read_json(ROOT / "data/derived/ner_labels.json")["types"]
        result = score_ner([bio_spans(r["labels"])[0] for r in rows], [p["parsed"] for p in ordered], allowed)
        result["bio_repairs"] = sum(p["parsed"].get("repairs", 0) for p in ordered)
        result["bio_repair_rate_per_word"] = result["bio_repairs"] / sum(len(r["tokens"]) for r in rows)
    times = sorted(p["seconds"] for p in ordered)
    result["timing"] = {"total_prediction_seconds": sum(times), "p50_seconds": statistics.median(times),
                        "p95_seconds": times[max(0, math.ceil(.95 * len(times)) - 1)],
                        "examples_per_second": len(rows) / sum(times) if sum(times) else None,
                        "scope": "batch 1, warmed, synchronized; model loading excluded"}
    result["peak_gpu_allocated_bytes"] = max(p.get("peak_gpu_allocated_bytes", 0) for p in ordered)
    result["peak_gpu_reserved_bytes"] = max(p.get("peak_gpu_reserved_bytes", 0) for p in ordered)
    result["generated_tokens"] = sum(p.get("generated_tokens", 0) for p in ordered)
    result["runtime_failures"] = sum(bool(p.get("runtime_error")) for p in ordered)
    result["truncated_outputs"] = sum(p.get("truncated_outputs", 0) for p in ordered)
    result["stage"] = "development_pilot_not_final_test"
    return result


def finish(store, task, rows):
    predictions = [store.get(r["id"]) for r in rows]
    if any(p is None for p in predictions):
        raise ValueError("Run incomplete: metrics are not a full-panel result")
    metrics = summarize(task, rows, predictions)
    write_jsonl(store.path / "predictions.jsonl", predictions)
    write_json(store.path / "metrics.json", metrics)
    print({k: v for k, v in metrics.items() if k not in ("per_type", "per_class")}, flush=True)
    return metrics


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Re-score a complete development run")
    p.add_argument("--run", type=Path, required=True)
    a = p.parse_args()
    identity = read_json(a.run / "identity.json")
    task = identity["task"]
    rows = load_panel(task, "valid")
    if identity["panel_ids"] != [r["id"] for r in rows] or identity["panel_sha256"] != digest(rows):
        raise ValueError("Panel differs from the recorded run")
    predictions = read_jsonl(a.run / "predictions.jsonl")
    run_hash = digest(identity)
    if any(p["run_hash"] != run_hash for p in predictions):
        raise ValueError("Prediction identity differs from the recorded run")
    by_id = {r["id"]: r for r in rows}
    if any(p["id"] not in by_id or p["input_sha256"] != digest(by_id[p["id"]]) for p in predictions):
        raise ValueError("Prediction input differs from the saved panel")
    result = summarize(task, rows, predictions)
    write_json(a.run / "metrics_rescored.json", result)
    print(result)
