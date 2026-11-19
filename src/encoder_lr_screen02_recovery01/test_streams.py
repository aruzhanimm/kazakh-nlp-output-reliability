import copy,hashlib,json,unittest
import numpy as np
import torch
from pilot.common import read_json
from stage_preflight.state import fingerprint
from checkpoint_integrity01.independent_verifier import independent_hash
from .common import D
from .streams import fingerprint_stream,rehash_stream,compare_streams

class StreamTests(unittest.TestCase):
    def setUp(self):
        self.folder=D/'test_fixtures'/self._testMethodName;self.folder.mkdir(parents=True,exist_ok=False)
    def test_original_format_noncontiguous_mixed_keys_and_exact_buffers(self):
        value={'mixed':{3:('Қазақ',None,True,-0.0),'3':[1,2.5]},'tensor':torch.arange(12,dtype=torch.float32).reshape(3,4).T,
            'rng':np.arange(9,dtype=np.uint32)[::2],'empty':torch.empty(0),'scalar':torch.tensor(7.)}
        buffers=[]
        class Spy:
            def __init__(self):self.h=hashlib.sha256()
            def update(self,b):
                assert type(b) is bytes;buffers.append(b);self.h.update(b)
            def hexdigest(self):return self.h.hexdigest()
        base=self.folder/'fixture';actual=fingerprint_stream(value,base,'root',Spy)
        self.assertEqual(actual,fingerprint(value));self.assertEqual(actual,independent_hash(value))
        self.assertEqual(base.with_suffix('.bin').read_bytes(),b''.join(buffers))
        index=[json.loads(x) for x in base.with_suffix('.index.jsonl').read_text(encoding='utf-8').splitlines()]
        self.assertEqual(len(index),len(buffers))
        for row,b in zip(index,buffers):self.assertEqual(row['buffer_sha256'],hashlib.sha256(b).hexdigest());self.assertEqual(row['length'],len(b))
        self.assertTrue(rehash_stream(base,self.folder/'rehash.json')['passed'])
    def test_wire_format_bfloat16_and_tuple_list_difference(self):
        value=torch.arange(8,dtype=torch.bfloat16)
        self.assertEqual(fingerprint_stream(value,self.folder/'bf16'),fingerprint(value))
        a=fingerprint_stream([1,2],self.folder/'list');b=fingerprint_stream((1,2),self.folder/'tuple');self.assertNotEqual(a,b)
    def test_same_bytes_and_changed_tensor_offsets(self):
        a={'w':torch.arange(4,dtype=torch.float32)};b=copy.deepcopy(a)
        fingerprint_stream(a,self.folder/'a','model');fingerprint_stream(b,self.folder/'b','model')
        self.assertTrue(compare_streams(self.folder/'a',self.folder/'b',self.folder/'same.json')['input_bytes_identical'])
        b['w'][2]+=1;fingerprint_stream(b,self.folder/'c','model')
        with self.assertRaisesRegex(AssertionError,'divergence'):compare_streams(self.folder/'a',self.folder/'c',self.folder/'different.json')
        evidence=read_json(self.folder/'different.json');self.assertEqual(evidence['first_difference']['left_hash_update']['path'],"model['w']")
        self.assertEqual(evidence['first_difference']['left_hash_update']['part'],'tensor_data')
    def test_different_digest_on_identical_bytes_is_distinguished(self):
        fingerprint_stream([1],self.folder/'a');fingerprint_stream([1],self.folder/'b')
        p=(self.folder/'b').with_suffix('.stream.json');value=read_json(p);value['digest_observed_during_hash_updates']='0'*64
        # Fault injection confined to a small new synthetic fixture, never an original artifact.
        p.write_text(json.dumps(value),encoding='utf-8')
        with self.assertRaisesRegex(AssertionError,'divergence'):compare_streams(self.folder/'a',self.folder/'b',self.folder/'digest_error.json')
        evidence=read_json(self.folder/'digest_error.json');self.assertTrue(evidence['input_bytes_identical']);self.assertTrue(evidence['identical_bytes_but_different_recorded_digests'])
        with self.assertRaisesRegex(AssertionError,'rehash discrepancy'):rehash_stream(self.folder/'b',self.folder/'rejected.json')

if __name__=='__main__':unittest.main()
