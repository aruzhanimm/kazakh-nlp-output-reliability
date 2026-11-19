"""Synthetic diagnostics: no final corpus or generated experimental answers."""
import unittest,importlib.util
from pathlib import Path
import ner_boundary_diagnostics as d

class BoundaryDiagnosticsChecks(unittest.TestCase):
    def test_wrong_type_same_boundary_is_detection_hit(self):
        result=d.analyze([{'id':'x','spans':[(0,2,'ORG')]}],{'x':[(0,2,'PERSON')]},['PERSON','ORG'])
        self.assertEqual(result['untyped_exact_boundary']['f1'],1.)
        self.assertEqual(result['conditional_type_assignment']['accuracy'],0.)
        self.assertEqual(result['type_confusion']['matrix'],[[0,1],[0,0]])

    def test_multi_type_boundary_deduplicates_detection_without_arbitrary_type(self):
        result=d.analyze([{'id':'x','spans':[(0,2,'ORG'),(0,2,'PERSON'),(4,5,'ORG'),(4,5,'PERSON')]}],{'x':[(0,2,'PERSON')]},['PERSON','ORG'])
        self.assertEqual(result['untyped_exact_boundary'],{'tp':1,'fp':1,'fn':0,'support':1,'precision':.5,'recall':1.,'f1':2/3})
        self.assertEqual(result['counts']['multi_type_predicted_boundaries'],2)
        self.assertEqual(result['counts']['multi_type_predicted_typed_spans'],4)
        self.assertEqual(result['counts']['multi_type_on_gold_boundaries'],1)
        self.assertEqual(result['counts']['multi_type_on_spurious_boundaries'],1)
        self.assertIsNone(result['conditional_type_assignment']['accuracy'])
        self.assertEqual(result['type_confusion']['matrix'],[[0,0],[0,0]])

    def test_partial_overlap_never_forced_to_confusion_match(self):
        result=d.analyze([{'id':'x','spans':[(1,3,'ORG'),(6,7,'PERSON')]}],{'x':[(0,3,'PERSON'),(8,9,'ORG')]},['PERSON','ORG'])
        self.assertEqual((result['untyped_exact_boundary']['tp'],result['untyped_exact_boundary']['fp'],result['untyped_exact_boundary']['fn']),(0,2,2))
        self.assertEqual(result['conditional_type_assignment']['eligible_boundaries'],0)

    def test_micro_counts_and_empty_gold(self):
        result=d.analyze([{'id':'a','spans':[(0,1,'PERSON')]},{'id':'b','spans':[]},{'id':'c','spans':[(0,1,'ORG')]}],
                         {'a':[(0,1,'PERSON')],'b':[(0,1,'ORG'),(2,3,'PERSON')],'c':[]},['PERSON','ORG'])
        self.assertEqual(result['untyped_exact_boundary']['f1'],.4)
        self.assertEqual(result['counts']['missing_boundaries'],2)
        self.assertEqual(result['counts']['spurious_boundaries'],1)

    def test_conflicting_gold_and_duplicate_prediction_rejected(self):
        with self.assertRaises(ValueError):d.analyze([{'id':'a','spans':[]}],{'a':[(0,1,'PERSON'),(0,1,'ORG')]},['PERSON','ORG'])
        with self.assertRaises(ValueError):d.analyze([{'id':'a','spans':[(0,1,'PERSON'),(0,1,'PERSON')]}],{'a':[]},['PERSON'])

    def test_primary_scores_failures_and_per_type_unchanged_from_preserved_v1(self):
        import assemble_results as current
        path=Path(__file__).resolve().parent/'source_history/assemble_results.v1.py'
        spec=importlib.util.spec_from_file_location('preserved_assembler_v1',path);old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
        rows=[{'id':'a','spans':[(0,2,'ORG')],'failed':False},
              {'id':'b','spans':[],'failed':True,'truncated':True},
              {'id':'c','spans':[(0,1,'PERSON')],'failed':False}]
        gold={'a':[(0,2,'PERSON')],'b':[(1,2,'ORG')],'c':[]}
        before=old.score_rows('ner',rows,gold,['PERSON','ORG']);after=current.score_rows('ner',rows,gold,['PERSON','ORG'])
        self.assertEqual(before,{k:after[k] for k in before})
        self.assertIn('ner_boundary_type_diagnostics',after)

if __name__=='__main__':unittest.main(verbosity=2)
