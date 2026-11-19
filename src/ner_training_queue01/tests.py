import copy,unittest,random
from unittest.mock import patch
import torch,numpy as np
from pilot.common import read_json,file_hash,write_json
from encoder_main01.control import new_progress,select
from stage_preflight.state import fingerprint
import encoder_lr_screen02.checkpoints as original
from .common import D
from . import checkpoints as cp
from .core import order

class Tests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2);self.enterContext(patch.object(torch.cuda,'is_available',return_value=False))
        self.folder=D/'tests'/self._testMethodName;self.folder.mkdir(parents=True,exist_ok=False)
        torch.manual_seed(17);random.seed(17);np.random.seed(17)
        self.model=torch.nn.Linear(3,2);self.opt=torch.optim.AdamW(self.model.parameters(),lr=1e-5);self.sched=torch.optim.lr_scheduler.LambdaLR(self.opt,lambda s:1.)
        self.model(torch.ones(1,3)).sum().backward();self.opt.step();self.sched.step();self.opt.zero_grad(set_to_none=True)
        self.p=new_progress();self.p.update(updates=1,cursor=16,checkpoint_sequence=1,rolling_sequence=1,trace=[{'update':1}])
        self.rng={'random':random.getstate(),'numpy':np.random.get_state(),'torch':torch.get_rng_state(),'cuda':[torch.get_rng_state().clone()]}
        self.enterContext(patch.object(original,'rng',return_value=self.rng))
    def snapshot(self,g=1):
        self.p.update(checkpoint_sequence=g,rolling_sequence=g)
        return cp.capture(self.model,self.opt,self.sched,self.p,'SYNTHETIC','rolling_'+str((g-1)%2))
    def test_capture_independence_and_rejected_bad_metadata(self):
        s=self.snapshot();expected=cp.signatures(s)
        with torch.no_grad():self.model.weight.add_(1);next(iter(self.opt.state.values()))['exp_avg'].add_(3)
        self.p['cursor']=32;self.rng['torch'][0]^=1
        self.assertEqual(expected,cp.signatures(s));meta=cp.commit(self.folder,s)
        bad=copy.deepcopy(meta);bad['state_digests']['optimizer']='0'*64
        with self.assertRaises(AssertionError):cp.verify_payload(self.folder/meta['file'],bad,self.folder/'bad.json')
        bad=copy.deepcopy(meta);bad['checkpoint_identity']['generation']=99
        with self.assertRaises(AssertionError):cp.verify_payload(self.folder/meta['file'],bad,self.folder/'bad_generation.json')
    def test_interrupted_commit_preserves_latest(self):
        cp.commit(self.folder,self.snapshot());before=file_hash(self.folder/'latest.json')
        with self.assertRaises(RuntimeError):cp.commit(self.folder,self.snapshot(2),fail_at='before_publish')
        self.assertEqual(before,file_hash(self.folder/'latest.json'));self.assertFalse((self.folder/'generation_000002.committed.json').exists())
    def test_retention_interruption_and_survivors(self):
        first=cp.commit(self.folder,self.snapshot());cp.commit(self.folder,self.snapshot(2));cp.commit(self.folder,self.snapshot(3))
        with self.assertRaises(RuntimeError):cp.retain(self.folder,fail_at='after_retire_delete')
        self.assertFalse((self.folder/first['file']).exists());self.assertTrue((self.folder/'generation_000001.retirement_intent.json').exists())
        cp.retain(self.folder)
        self.assertEqual(len(list(self.folder.glob('*.pt'))),2);self.assertTrue((self.folder/'generation_000001.retired.json').exists())
        self.assertEqual(cp.committed_meta(self.folder)[1]['sequence'],3)
    def test_seed_coverage_and_order(self):
        from transformers import set_seed
        results=[]
        for seed in [13,42,2026]:
            pair=[]
            for repeat in range(2):
                set_seed(seed);m=torch.nn.Linear(5,2);drop=torch.nn.Dropout(.5)(torch.ones(128))
                pair.append((fingerprint(m.state_dict()),fingerprint(drop),random.random(),float(np.random.random()),order(0,100,seed)))
            self.assertEqual(pair[0],pair[1]);results.append(pair[0])
        for i in range(3):
            for j in range(i):
                for k in range(5):self.assertNotEqual(results[i][k],results[j][k])
    def test_selection_ties_and_early_stop(self):
        p=new_progress();select(p,.8,1);select(p,.8,2);self.assertFalse(p['stop']);select(p,.8,3)
        self.assertTrue(p['stop']);self.assertEqual(p['best_epoch'],1)

if __name__=='__main__':unittest.main()
