"""Synthetic fixtures only: no corpus, checkpoints, server or model access."""
from pathlib import Path
import importlib.util
import json
import unittest
import numpy as np
from offline_kernel import ranked_panels, normalized_text, align_prediction_rows, paired_analysis, f1_from_counts

ROOT = Path(__file__).resolve().parents[2]


def source_module(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


strict = source_module("reference_strict", "ner_evidence_20261111/analysis/strict_ner.py")
scoring = source_module("reference_scoring", "pilot/scoring.py")


class KernelChecks(unittest.TestCase):
    def test_ranked_panels_nested_deterministic(self):
        ids = [f"SYNTHETIC_ONLY_{i}" for i in range(2200)]
        a, b = ranked_panels(ids, "ner"), ranked_panels(list(reversed(ids)), "ner")
        self.assertEqual(a, b)
        self.assertEqual([len(a[k]) for k in ("main", "reference", "sensitivity")], [2000, 500, 200])
        self.assertTrue(set(a["reference"]) <= set(a["main"]))
        self.assertTrue(set(a["sensitivity"]) <= set(a["main"]))
        self.assertNotEqual(a, ranked_panels(ids, "sa"))

    def test_panel_rejects_invalid_populations(self):
        with self.assertRaises(ValueError): ranked_panels(["a", "a"], "ner", main_n=2, reference_n=1, sensitivity_n=1)
        with self.assertRaises(ValueError): ranked_panels(["a"], "ner")

    def test_nfc_original_text_normalization(self):
        self.assertEqual(normalized_text("  e\u0301\t қаласы\n"), "é қаласы")

    def test_exact_typed_spans_reuse_reference(self):
        g = [(0, 1, "PERSON"), (2, 4, "GPE")]
        p = [(0, 1, "PERSON"), (2, 3, "GPE"), (5, 6, "PERSON")]
        rows = {"a": [{"id": "SYNTHETIC", "gold_spans": g, "pred_spans": p}]}
        counts = align_prediction_rows("ner", ["SYNTHETIC"], rows)["a"][0]
        self.assertEqual(tuple(counts), strict.exact_counts(g, p))
        self.assertAlmostEqual(float(f1_from_counts("ner", counts)), strict.prf(*counts)["f1"])

    def test_sa_invalid_matches_existing_scorer(self):
        gold, pred = [0, 0, 1, 1], [None, 1, 1, None]
        rows = [{"id": str(i), "gold_label": g, "pred_label": p} for i, (g, p) in enumerate(zip(gold, pred))]
        array = align_prediction_rows("sa", list(map(str, range(4))), {"a": rows})["a"]
        reference = scoring.score_sa(gold, pred)
        self.assertEqual(array.sum(axis=0).tolist(), reference["confusion"])
        self.assertAlmostEqual(float(f1_from_counts("sa", array.sum(axis=0))), reference["macro_f1"])
        self.assertEqual(int(array.sum()), 4)

    def test_missing_duplicate_extra_and_disagreeing_gold_rejected(self):
        r = {"id": "a", "gold_label": 0, "pred_label": 0}
        for rows in ([], [r, r], [r, dict(r, id="extra")]):
            with self.assertRaises(ValueError): align_prediction_rows("sa", ["a"], {"x": rows})
        with self.assertRaises(ValueError):
            align_prediction_rows("sa", ["a"], {"x": [r], "y": [dict(r, gold_label=1)]})

    def test_reordered_input_rows_align_by_id(self):
        a = {"id": "a", "gold_label": 0, "pred_label": 1}
        b = {"id": "b", "gold_label": 1, "pred_label": 0}
        arrays = align_prediction_rows("sa", ["a", "b"], {"x": [a, b], "y": [b, a]})
        np.testing.assert_array_equal(arrays["x"], arrays["y"])

    def test_pairing_and_duplicate_cluster_weights(self):
        a = np.asarray([[1, 0, 0], [0, 0, 1], [1, 2, 0]])
        result = paired_analysis("ner", ["a", "b", "c"], ["same", "same", "different"],
                                 {"x": a, "y": a.copy()}, contrasts=(("x", "y"),), replicates=127)
        self.assertEqual(result["normalized_text_clusters"], 2)
        self.assertEqual(result["paired_differences"][0]["ci_bonferroni_K6"], [0.0, 0.0])
        # Independent explicit reference draws whole groups, including both rows.
        rng = np.random.default_rng(20261111)
        weights = rng.multinomial(2, [.5, .5], size=127)
        vals = [strict.prf(*(w[0] * (a[0]+a[1]) + w[1]*a[2]))["f1"] for w in weights]
        np.testing.assert_allclose(result["conditions"]["x"]["ci95_pointwise"], np.quantile(vals, [.025,.975]))

    def test_seed_mean_is_not_pooled_counts(self):
        counts = {"s13": np.array([[1,0,0], [1,0,0]]),
                  "s42": np.array([[0,0,1], [0,0,1]]),
                  "s2026": np.array([[1,8,0], [1,8,0]])}
        result = paired_analysis("ner", ["a", "b"], ["a", "b"], counts,
            seed_groups={"encoder": {13:"s13",42:"s42",2026:"s2026"}}, replicates=51)
        value = result["conditions"]["encoder"]["f1"]
        self.assertAlmostEqual(value, (1+0+.2)/3)
        pooled = float(f1_from_counts("ner", sum(counts.values()).sum(axis=0)))
        self.assertNotAlmostEqual(value, pooled)
        self.assertAlmostEqual(result["seed_summaries"]["encoder"]["seed_sample_sd"], np.std([1,0,.2], ddof=1))

    def test_missing_seed_and_gold_support_rejected(self):
        a = np.array([[1,0,0]])
        with self.assertRaises(ValueError):
            paired_analysis("ner", ["a"], ["a"], {"x":a}, seed_groups={"m":{13:"x"}}, replicates=1)
        with self.assertRaises(ValueError):
            paired_analysis("ner", ["a"], ["a"], {"x":a, "y":np.array([[2,0,0]])}, replicates=1)

    def test_bonferroni_family_stays_six(self):
        a, b = np.array([[1,0,0],[0,0,1]]), np.array([[0,0,1],[1,1,0]])
        result = paired_analysis("ner", ["a","b"], ["a","b"], {"x":a,"y":b},
            contrasts=(("x","y"),), replicates=1001)
        self.assertEqual(result["bootstrap"]["primary_family_K"], 6)
        rng = np.random.default_rng(20261111)
        weights = rng.multinomial(2,[.5,.5],size=1001)
        delta = f1_from_counts("ner", weights@a)-f1_from_counts("ner", weights@b)
        np.testing.assert_allclose(result["paired_differences"][0]["ci_bonferroni_K6"],
                                   np.quantile(delta,[.05/12,1-.05/12]))

    def test_fixed_sa_labels_even_if_one_class_absent(self):
        a = np.array([[[1,0,0],[0,0,0]], [[1,0,0],[0,0,0]]])
        result = paired_analysis("sa", ["a","b"], ["a","b"], {"x":a}, replicates=13)
        self.assertEqual(result["conditions"]["x"]["f1"], .5)
        self.assertFalse(result["publication_ready"])
        self.assertFalse(result["protocol_freeze_verified"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
