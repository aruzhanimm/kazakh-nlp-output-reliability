"""Pure in-memory preparation for the existing protocol, with no corpus/file I/O.

This is NOT a final-test entry point or a protocol-freeze authorizer. Callers must
first verify prediction/input/gold provenance and explicit terminal records.
It preserves the established dev analysis: common multinomial text-cluster
weights; counts -> metric for each seed -> mean of seed metrics. SA adds the
existing fixed-two-class confusion convention with an explicit INVALID column.
"""
import hashlib
import json
import unicodedata
import numpy as np

NAMESPACES = {"main": "kazakh-final-v2", "reference": "kazakh-reference-v2",
              "sensitivity": "kazakh-demo-sensitivity-v2"}
EXPECTED_SEEDS = (13, 42, 2026)


def normalized_text(text):
    if not isinstance(text, str):
        raise ValueError("Original text must be a string")
    return " ".join(unicodedata.normalize("NFC", text).split())


def ranked_panels(row_ids, task, *, seed=20261110, main_n=2000,
                  reference_n=500, sensitivity_n=200):
    """Rank supplied IDs only; this function never obtains a test ID population."""
    ids = list(row_ids)
    if task not in ("ner", "sa") or not ids or any(not isinstance(x, str) or not x for x in ids):
        raise ValueError("Explicit nonempty IDs and supported task required")
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate IDs")
    if not 0 < reference_n <= main_n <= len(ids) or not 0 < sensitivity_n <= main_n:
        raise ValueError("Invalid panel cardinalities")

    def rank(population, namespace):
        def key(rid):
            value = json.dumps([namespace, seed, task, rid], ensure_ascii=False,
                               separators=(",", ":")).encode("utf-8")
            return hashlib.sha256(value).hexdigest(), rid
        return sorted(population, key=key)

    main = rank(ids, NAMESPACES["main"])[:main_n]
    return {"main": main,
            "reference": rank(main, NAMESPACES["reference"])[:reference_n],
            "sensitivity": rank(main, NAMESPACES["sensitivity"])[:sensitivity_n]}


def align_prediction_rows(task, expected_ids, records_by_condition):
    """Require complete conditions; make count arrays from already parsed records.

    NER records: id, gold_spans, pred_spans. SA records: id, gold_label,
    pred_label (0, 1 or None for an explicitly accounted INVALID result).
    This does not parse raw generation or infer that an absent row is a failure.
    """
    expected = list(expected_ids)
    if not expected or len(set(expected)) != len(expected) or not records_by_condition:
        raise ValueError("Unique nonempty panel and conditions required")
    if task not in ("ner", "sa"):
        raise ValueError("Unknown task")
    gold_identity = None
    arrays = {}
    for condition, records in records_by_condition.items():
        rows = list(records)
        by_id = {row["id"]: row for row in rows}
        if len(by_id) != len(rows) or set(by_id) != set(expected):
            raise ValueError("Missing, duplicate or extra prediction IDs: " + condition)
        counts, gold = [], []
        for rid in expected:
            row = by_id[rid]
            if task == "ner":
                def spans(values):
                    result = set()
                    for value in values:
                        if len(value) != 3:
                            raise ValueError("Expected a typed span")
                        a, b, typ = value
                        if type(a) is not int or type(b) is not int or not 0 <= a < b or not isinstance(typ, str):
                            raise ValueError("Invalid typed span")
                        result.add((a, b, typ))
                    if len(result) != len(values):
                        raise ValueError("Duplicate spans require upstream reconciliation")
                    return result
                g, p = spans(row["gold_spans"]), spans(row["pred_spans"])
                gold.append(sorted(g))
                counts.append((len(g & p), len(p - g), len(g - p)))
            else:
                g, p = row["gold_label"], row["pred_label"]
                if type(g) is not int or g not in (0, 1) or not (p is None or type(p) is int and p in (0, 1)):
                    raise ValueError("Expected fixed SA labels or explicit None INVALID")
                matrix = [[0, 0, 0], [0, 0, 0]]
                matrix[g][2 if p is None else p] = 1
                counts.append(matrix)
                gold.append(g)
        if gold_identity is not None and gold != gold_identity:
            raise ValueError("Gold differs across systems")
        gold_identity = gold
        arrays[condition] = np.asarray(counts, dtype=np.int64)
    return arrays


def f1_from_counts(task, counts):
    """Vectorized corpus F1; never average per-item F1."""
    counts = np.asarray(counts)
    if task == "ner":
        tp, fp, fn = counts[..., 0], counts[..., 1], counts[..., 2]
        den = 2 * tp + fp + fn
        return np.divide(2 * tp, den, out=np.zeros_like(den, dtype=float), where=den != 0)
    if task != "sa":
        raise ValueError("Unknown task")
    scores = []
    for cls in (0, 1):
        tp = counts[..., cls, cls]
        fp = counts[..., 1 - cls, cls]
        fn = counts[..., cls, :].sum(axis=-1) - tp
        den = 2 * tp + fp + fn
        scores.append(np.divide(2 * tp, den, out=np.zeros_like(den, dtype=float), where=den != 0))
    return (scores[0] + scores[1]) / 2


def paired_analysis(task, item_ids, cluster_keys, counts_by_condition, *,
                    seed_groups=None, contrasts=(), replicates=50000,
                    random_seed=20261111, batch_size=500):
    """Conditional paired percentile intervals, including protocol K=6 adjustment.

    seed_groups maps a summary name to {13: condition, 42: condition,
    2026: condition}. Contrasts are (left, right); all receive the prespecified
    K=6 endpoints, even if this task contains fewer than six contrasts. The final
    binding must enforce the exact six comparison identities across both tasks.
    """
    ids, clusters = list(item_ids), list(cluster_keys)
    if not ids or len(set(ids)) != len(ids) or len(clusters) != len(ids):
        raise ValueError("Complete unique IDs and one cluster key per item required")
    if any(not isinstance(c, str) for c in clusters) or not counts_by_condition:
        raise ValueError("Explicit original-text clusters and condition arrays required")
    if type(replicates) is not int or replicates < 1 or type(batch_size) is not int or batch_size < 1:
        raise ValueError("Positive replicate and batch counts required")
    if len(contrasts) > 6 or len(set(contrasts)) != len(contrasts):
        raise ValueError("Duplicate contrasts or more than the planned family")
    shape = (3,) if task == "ner" else (2, 3) if task == "sa" else None
    if shape is None:
        raise ValueError("Unknown task")
    unique = list(dict.fromkeys(clusters))
    group_id = {key: i for i, key in enumerate(unique)}
    indexes = np.asarray([group_id[c] for c in clusters])
    grouped = {}
    gold_support = None
    for condition, value in counts_by_condition.items():
        a = np.asarray(value)
        if a.shape != (len(ids),) + shape or not np.issubdtype(a.dtype, np.integer) or np.any(a < 0):
            raise ValueError("Nonnegative integer count tensor required: " + condition)
        support = a[:, 0] + a[:, 2] if task == "ner" else a.sum(axis=2)
        if task == "sa" and np.any(a.sum(axis=(1, 2)) != 1):
            raise ValueError("Every SA item must remain represented, including INVALID")
        if gold_support is not None and not np.array_equal(support, gold_support):
            raise ValueError("Per-item gold support differs")
        gold_support = support
        g = np.zeros((len(unique),) + shape, dtype=np.int64)
        np.add.at(g, indexes, a)
        grouped[condition] = g
    seed_groups = seed_groups or {}
    for name, seeds in seed_groups.items():
        if name in grouped or set(seeds) != set(EXPECTED_SEEDS) or len(set(seeds.values())) != 3:
            raise ValueError("Summary must contain exactly the three distinct accepted seeds")
        if any(c not in grouped for c in seeds.values()):
            raise ValueError("Seed condition missing")
    known = set(grouped) | set(seed_groups)
    if any(a not in known or b not in known or a == b for a, b in contrasts):
        raise ValueError("Unknown or identical comparison systems")
    rng = np.random.default_rng(random_seed)
    boot = {c: np.empty(replicates) for c in grouped}
    for start in range(0, replicates, batch_size):
        n = min(batch_size, replicates - start)
        weights = rng.multinomial(len(unique), [1 / len(unique)] * len(unique), size=n)
        for condition, g in grouped.items():
            total = (weights @ g.reshape(len(unique), -1)).reshape((n,) + shape)
            boot[condition][start:start+n] = f1_from_counts(task, total)
    point = {c: float(f1_from_counts(task, g.sum(axis=0))) for c, g in grouped.items()}
    summaries = {}
    for name, seeds in seed_groups.items():
        names = [seeds[s] for s in EXPECTED_SEEDS]
        point[name] = float(np.mean([point[c] for c in names]))
        boot[name] = np.mean([boot[c] for c in names], axis=0)
        summaries[name] = {"seeds": seeds, "seed_sample_sd": float(np.std([point[c] for c in names], ddof=1)),
                           "interpretation": "mean of separate seed metrics; not pooled counts or ensemble"}
    def quantiles(values, probabilities):
        return np.quantile(values, probabilities, method="linear").tolist()
    comparisons = [{"left": a, "right": b, "delta_f1": point[a] - point[b],
        "ci95_pointwise": quantiles(boot[a] - boot[b], [.025, .975]),
        "ci_bonferroni_K6": quantiles(boot[a] - boot[b], [.05 / 12, 1 - .05 / 12])}
        for a, b in contrasts]
    return {"status": "unbound_in_memory_statistical_computation", "publication_ready": False,
        "task": task, "items": len(ids), "normalized_text_clusters": len(unique),
        "conditions": {c: {"f1": value, "ci95_pointwise": quantiles(boot[c], [.025, .975])}
                       for c, value in point.items()},
        "seed_summaries": summaries, "paired_differences": comparisons,
        "bootstrap": {"replicates": replicates, "seed": random_seed, "numpy": np.__version__,
            "rng": "numpy.default_rng/PCG64", "batch_size": batch_size,
            "unit": "normalized original-text group", "shared_weights_all_conditions": True,
            "quantile_method": "linear", "primary_family_K": 6,
            "training_uncertainty_included": False},
        "provenance_binding_verified": False, "protocol_freeze_verified": False,
        "caution": "Conditional fixed-system uncertainty; no p-values or equivalence claim"}
