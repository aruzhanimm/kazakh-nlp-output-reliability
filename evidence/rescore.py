"""Reparse preserved task answers and rescore all fixed common-panel conditions.

No model, credential, network access or training data is needed. The established
parsers and metric helpers are loaded unchanged from the accompanying project.
"""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import math
import statistics

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent / "project" if (HERE.parent / "project" / "src").is_dir() else HERE.parent
PROJECT = PROJECT_ROOT / "src"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def same_spans(a, b):
    return sorted(map(tuple, a)) == sorted(map(tuple, b))


def check_numbers(actual, expected, name):
    for key, value in expected.items():
        if isinstance(value, (int, float)):
            if key not in actual or not math.isclose(actual[key], value, rel_tol=0, abs_tol=1e-12):
                raise ValueError(f"Metric mismatch: {name}.{key}")
        elif isinstance(value, list) and actual.get(key) != value:
            raise ValueError(f"Count/matrix mismatch: {name}.{key}")


def run(bootstrap=False):
    manifest = read(HERE / "MANIFEST.json")
    for relative, digest in manifest["files"].items():
        if sha(HERE / relative) != digest:
            raise ValueError("Evidence file hash mismatch: " + relative)
    for relative, digest in manifest["source_modules"].items():
        if sha(PROJECT / relative) != digest:
            raise ValueError("Scoring source hash mismatch: " + relative)
    adapter = load("supplement_adapter", PROJECT / "research_execution_20261112/final_evaluation/output_adapter.py")
    fence = load("supplement_fence", PROJECT / "research_execution_20261112/final_analysis/fence_only_sensitivity.py")
    expected = read(HERE / "EXPECTED_METRICS.json")
    panels = {task: jsonl(HERE / f"{task}_items.jsonl") for task in ("ner", "sa")}
    allowed = manifest["ner_types"]
    conditions, seedmeans, secondary, diagnostics, aligned = {}, {}, {}, {}, {"ner": {}, "sa": {}}
    parsed_answers = 0
    for condition, definition in manifest["conditions"].items():
        task, kind = definition["task"], definition["kind"]
        items = panels[task]
        rows = jsonl(HERE / definition["file"])
        if len(items) != 500 or [r["id"] for r in rows] != [r["id"] for r in items]:
            raise ValueError("Coverage/order mismatch: " + condition)
        predictions, secondary_rows, alignment_rows = [], [], []
        failure_counts = {"syntax_or_root_items": 0, "contract_items": 0, "alignment_items": 0,
                          "gold_empty_items": 0, "gold_empty_items_with_fp": 0, "gold_empty_fp": 0}
        for item, row in zip(items, rows):
            if task == "ner":
                gold, repairs = adapter.strict.bio_to_spans(item["gold_bio"], repair_prediction=False)
                if repairs or not same_spans(gold, item["gold_spans"]):
                    raise ValueError("Gold BIO/span mismatch: " + item["id"])
                if kind == "encoder":
                    spans, repairs = adapter.strict.bio_to_spans(row["word_tags"], repair_prediction=True)
                    prediction = {"id": row["id"], "spans": spans, "failed": row["failed"]}
                    if repairs != row["bio_repairs"]:
                        raise ValueError("BIO repair count mismatch")
                else:
                    record = row["record"]
                    if hashlib.sha256(record["raw"].encode("utf-8")).hexdigest() != row["raw_task_answer_sha256"]:
                        raise ValueError("Raw answer hash mismatch")
                    windows = [{"start": 0, "end": len(item["tokens"]), "tokens": item["tokens"]}]
                    prediction = adapter.assemble_final_ner(row["id"], item["tokens"], windows, [record], allowed)
                    parsed_answers += 1
                    notes = [d for w in prediction["window_results"] if w["parsed"] is not None
                             for d in w["parsed"]["diagnostics"]]
                    failure_counts["syntax_or_root_items"] += any(d.get("reason") == "json_syntax_or_root" for d in notes)
                    failure_counts["contract_items"] += any(d.get("category") == "format" and d.get("reason") != "json_syntax_or_root" for d in notes)
                    failure_counts["alignment_items"] += any(d.get("category") == "alignment" for d in notes) or prediction["merge_alignment_failure"]
                    if condition in fence.CONDITIONS:
                        secondary_rows.append(fence.item(row["id"], item["tokens"], windows, [record], allowed, prediction, adapter))
                if not same_spans(prediction["spans"], row["spans"]) or prediction["failed"] != row["failed"]:
                    raise ValueError("Prediction reconstruction differs: " + condition + "/" + row["id"])
                if not gold:
                    failure_counts["gold_empty_items"] += 1
                    failure_counts["gold_empty_items_with_fp"] += bool(prediction["spans"])
                    failure_counts["gold_empty_fp"] += len(prediction["spans"])
                alignment_rows.append({"id": row["id"], "gold_spans": gold, "pred_spans": prediction["spans"]})
                predictions.append(prediction)
            else:
                if kind == "encoder":
                    label = max(range(2), key=lambda k: row["logits"][k])
                    prediction = {"label": label, "failed": False}
                else:
                    record = row["record"]
                    if hashlib.sha256(record["raw"].encode("utf-8")).hexdigest() != row["raw_task_answer_sha256"]:
                        raise ValueError("Raw answer hash mismatch")
                    prediction = adapter.parse_final_sa(record["raw"], truncated=record["truncated"],
                        refusal=record["refusal"], runtime_error=record["runtime_error"],
                        terminal_failed=record["status"] == "terminal_failure")
                    parsed_answers += 1
                if prediction["label"] != row["label"] or prediction["failed"] != row["failed"]:
                    raise ValueError("Prediction reconstruction differs: " + condition + "/" + row["id"])
                predictions.append(prediction["label"])
                alignment_rows.append({"id": row["id"], "gold_label": item["gold_label"], "pred_label": prediction["label"]})
        if task == "ner":
            counts = [adapter.strict.exact_counts(item["gold_spans"], prediction["spans"])
                      for item, prediction in zip(items, predictions)]
            metrics = adapter.strict.prf(*(sum(c[i] for c in counts) for i in range(3)))
            metrics["items"] = len(items)
            if secondary_rows:
                summary = fence.summarize(secondary_rows, {r["id"]: r["gold_spans"] for r in items}, metrics, adapter)
                secondary[condition] = {"metrics": summary["fence_only_secondary"], "counts": summary["counts"]}
                check_numbers(summary["fence_only_secondary"], expected["secondary"][condition]["metrics"], condition + "/fence")
                check_numbers(summary["counts"], expected["secondary"][condition]["counts"], condition + "/fence-counts")
            diagnostics[condition] = failure_counts
            check_numbers(failure_counts, expected["diagnostics"][condition], condition + "/diagnostics")
        else:
            score = adapter.sa.score_sa([r["gold_label"] for r in items], predictions)
            metrics = {"items": score["n"], "f1": score["macro_f1"], "accuracy": score["accuracy"],
                "macro_precision": statistics.mean(v["precision"] for v in score["per_class"].values()),
                "macro_recall": statistics.mean(v["recall"] for v in score["per_class"].values()),
                "invalid": score["invalid"], "confusion": score["confusion"]}
        check_numbers(metrics, expected["conditions"][condition], condition)
        conditions[condition] = metrics
        aligned[task][condition] = alignment_rows
    for task in ("ner", "sa"):
        for model in ("xlmr", "mbert"):
            name = f"{task}_{model}_seedmean"
            members = [conditions[f"{task}_{model}_seed{seed}"] for seed in (13, 42, 2026)]
            keys = ("precision", "recall", "f1") if task == "ner" else ("macro_precision", "macro_recall", "f1", "accuracy")
            summary = {key: statistics.mean(v[key] for v in members) for key in keys}
            summary["seed_sample_sd"] = statistics.stdev(v["f1"] for v in members)
            check_numbers(summary, expected["seedmeans"][name], name)
            seedmeans[name] = summary
    output = {"status": "PASS_ALL_PRIMARY_METRICS_RECOMPUTED", "items_per_task": 500,
        "condition_count": len(conditions), "prediction_count": sum(len(v) for t in aligned.values() for v in t.values()),
        "task_answers_reparsed": parsed_answers, "conditions": conditions, "seedmeans": seedmeans,
        "fence_only_secondary": secondary, "ner_diagnostics": diagnostics,
        "new_model_calls": 0, "bootstrap_recomputed": False}
    if bootstrap:
        kernel = load("supplement_kernel", PROJECT / "research_execution_20261112/final_analysis/offline_kernel.py")
        output["bootstrap"] = {}
        for task in ("ner", "sa"):
            items = panels[task]
            ids = [r["id"] for r in items]
            counts = kernel.align_prediction_rows(task, ids, aligned[task])
            groups = {f"{task}_{model}_seedmean": {seed: f"{task}_{model}_seed{seed}" for seed in (13, 42, 2026)} for model in ("xlmr", "mbert")}
            contrasts = [(r["left"], r["right"]) for r in expected["statistics"][task]["paired_differences"]]
            stats = kernel.paired_analysis(task, ids, [r["cluster_key"] for r in items], counts,
                seed_groups=groups, contrasts=contrasts, replicates=50000, random_seed=20261111)
            for name, value in stats["conditions"].items():
                ref = expected["statistics"][task]["conditions"][name]
                for a, b in zip(value["ci95_pointwise"], ref["ci95_pointwise"]):
                    if not math.isclose(a, b, abs_tol=1e-12):
                        raise ValueError("Bootstrap condition interval mismatch: " + name)
            for got, ref in zip(stats["paired_differences"], expected["statistics"][task]["paired_differences"]):
                for key in ("ci95_pointwise", "ci_bonferroni_K6"):
                    if key in ref and any(not math.isclose(a, b, abs_tol=1e-12) for a, b in zip(got[key], ref[key])):
                        raise ValueError("Bootstrap contrast interval mismatch")
                # Only the prespecified primary family is assigned adjusted intervals.
                if "ci_bonferroni_K6" not in ref:
                    got.pop("ci_bonferroni_K6", None)
                got["family"] = ref["family"]
            output["bootstrap"][task] = {"conditions": stats["conditions"], "paired_differences": stats["paired_differences"], "settings": stats["bootstrap"]}
        output["bootstrap_recomputed"] = True
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bootstrap", action="store_true", help="Also regenerate 50,000 paired draws with the established kernel (NumPy required).")
    parser.add_argument("--output", type=Path, help="Write the full recalculated report to a new JSON file.")
    args = parser.parse_args()
    result = run(args.bootstrap)
    if args.output:
        if args.output.exists():
            raise FileExistsError("Refusing to overwrite output: " + str(args.output))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("status", "condition_count", "prediction_count", "task_answers_reparsed", "bootstrap_recomputed")}))
