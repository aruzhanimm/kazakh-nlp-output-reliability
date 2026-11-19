"""Synthetic wrapper fixtures only; no experimental claims or final inputs."""
import copy,unittest,importlib.util
from pathlib import Path
import assemble_results as a
import fence_only_sensitivity as f

class FenceOnlyChecks(unittest.TestCase):
    def test_only_whole_fence_valid_list_with_explicit_tag_policy(self):
        for raw in ['```json\n[]\n```',' \n```JSON\r\n[]\r\n```\t','```\n[]\n```']:
            body,note=f.unwrap(raw,a.evidence.adapter.parser)
            self.assertTrue(note['unwrapped']);self.assertEqual(body,'[]')

    def test_prose_multifence_inline_unknown_tag_and_nonJSON_rejected(self):
        bad=['Here is the answer:\n```json\n[]\n```','```json\n[]\n```\n```json\n[]\n```',
             '```json\n[]\n``` after','```python\n[]\n```','```json []```',
             '```json\n[bad]\n```','```json\n{}\n```','```json\n[NaN]\n```',
             '```json\n[{"x":1,"x":2}]\n```','```json\n["```"]\n```']
        for raw in bad:
            body,note=f.unwrap(raw,a.evidence.adapter.parser)
            self.assertFalse(note['unwrapped'],raw);self.assertEqual(body,raw)

    def test_secondary_changes_only_fixed_wrapper_primary_records_untouched(self):
        adapter=a.evidence.adapter;tokens=['Alice'];windows=[{'start':0,'end':1,'tokens':tokens}]
        raw='```json\n[{"mention":"Alice","type":"PERSON","occurrence":1}]\n```'
        records=[{'window_index':0,'raw':raw,'status':'completed','runtime_error':False,'refusal':False,'truncated':False}]
        before=copy.deepcopy(records);primary=adapter.assemble_final_ner('x',tokens,windows,records,['PERSON']);snapshot=copy.deepcopy(primary)
        secondary=f.item('x',tokens,windows,records,['PERSON'],primary,adapter)
        self.assertEqual(records,before);self.assertEqual(primary,snapshot)
        self.assertEqual(primary['spans'],[]);self.assertTrue(primary['failed'])
        self.assertEqual(secondary['spans'],[(0,1,'PERSON')]);self.assertFalse(secondary['failed'])
        summary=f.summarize([secondary],{'x':[(0,1,'PERSON')]},adapter.strict.prf(0,0,1),adapter)
        self.assertEqual(summary['strict_primary']['f1'],0.)
        self.assertEqual(summary['fence_only_secondary']['f1'],1.)
        self.assertEqual(summary['counts']['items_unwrapped'],1)

    def test_truncation_refusal_and_terminal_flags_preserved(self):
        adapter=a.evidence.adapter;tokens=['Alice'];windows=[{'start':0,'end':1,'tokens':tokens}]
        base={'window_index':0,'raw':'```json\n[]\n```','status':'completed','runtime_error':False,'refusal':False,'truncated':True}
        for record in [base,{**base,'refusal':True},{**base,'status':'terminal_failure','runtime_error':True}]:
            primary=adapter.assemble_final_ner('x',tokens,windows,[record],['PERSON'])
            secondary=f.item('x',tokens,windows,[record],['PERSON'],primary,adapter)
            self.assertTrue(secondary['failed']);self.assertEqual(secondary['spans'],[])
            if record['runtime_error'] or record['refusal']:self.assertFalse(secondary['windows'][0]['unwrapped'])

    def test_primary_report_unchanged_from_preserved_V2(self):
        path=Path(__file__).resolve().parent/'source_history/assemble_results.v2.py'
        spec=importlib.util.spec_from_file_location('preserved_assembler_v2',path);old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
        rows=[{'id':'a','spans':[(0,1,'PERSON')],'failed':False},{'id':'b','spans':[],'failed':True}]
        gold={'a':[(0,1,'PERSON')],'b':[(1,2,'ORG')]}
        self.assertEqual(a.score_rows('ner',rows,gold,['PERSON','ORG']),old.score_rows('ner',rows,gold,['PERSON','ORG']))
        self.assertEqual(a.PRIMARY,old.PRIMARY);self.assertEqual(a.DESCRIPTIVE,old.DESCRIPTIVE)

if __name__=='__main__':unittest.main(verbosity=2)
