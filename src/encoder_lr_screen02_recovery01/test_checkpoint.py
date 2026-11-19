import copy,unittest
from unittest.mock import patch
import torch,numpy as np,random
from pilot.common import read_json,file_hash
from stage_preflight.state import fingerprint
from encoder_main01.control import new_progress
import encoder_lr_screen02.checkpoints as original
from .common import D
from . import checkpoints as cp

class CheckpointTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2);torch.manual_seed(13);random.seed(13);np.random.seed(13)
        self.folder=D/'test_fixtures'/self._testMethodName;self.folder.mkdir(parents=True,exist_ok=False)
        self.model=torch.nn.Linear(3,2);self.opt=torch.optim.AdamW(self.model.parameters(),lr=3e-5)
        self.sched=torch.optim.lr_scheduler.LambdaLR(self.opt,lambda s:1.)
        self.model(torch.ones(1,3)).sum().backward();self.opt.step();self.sched.step();self.opt.zero_grad(set_to_none=True)
        self.p=new_progress();self.p.update(updates=1,cursor=16,checkpoint_sequence=1,rolling_sequence=1,trace=[{'update':1}])
        self.random={'random':random.getstate(),'numpy':np.random.get_state(),'torch':torch.get_rng_state(),'cuda':[torch.get_rng_state().clone()]}
        self.enterContext(patch.object(torch.cuda,'is_available',return_value=False));self.enterContext(patch.object(original,'rng',return_value=self.random))
    def capture(self):return cp.capture(self.model,self.opt,self.sched,self.p,'SYNTHETIC_CHECK_ONLY','rolling_0')
    def test_owned_snapshot_and_fresh_reader_before_publication(self):
        s=self.capture();expected=cp.signatures(s)
        with torch.no_grad():self.model.weight.add_(1);next(iter(self.opt.state.values()))['exp_avg'].add_(2)
        self.p['active_training_seconds']=7;self.random['torch'][0]^=1
        self.assertEqual(cp.signatures(s),expected)
        meta=cp.commit(self.folder/'commits',s)
        proof=read_json(self.folder/'commits/generation_000001.independent.json');self.assertTrue(proof['accepted'])
        self.assertEqual(proof['independent_signatures'],expected)
        bad=copy.deepcopy(meta);bad['state_digests']['optimizer']='0'*64
        with self.assertRaisesRegex(AssertionError,'verification failed'):cp.verify_payload(self.folder/'commits'/meta['file'],bad,self.folder/'invalid.json')
        # Independent production check keeps the original old verifier usable/unchanged.
        import encoder_main01.finalize as old
        from pilot.common import ROOT
        old.D=ROOT;old.check_checkpoint(self.folder/'commits/generation_000001.json',meta['run_hash'])
    def test_interruption_does_not_publish_candidate(self):
        first=cp.commit(self.folder,self.capture());pointer=file_hash(self.folder/'latest.json')
        self.p.update(checkpoint_sequence=2,rolling_sequence=2)
        with self.assertRaisesRegex(RuntimeError,'Synthetic interruption'):cp.commit(self.folder,self.capture(),fail_at='before_publish')
        self.assertEqual(file_hash(self.folder/'latest.json'),pointer)
        self.assertEqual(cp.committed_meta(self.folder)[1]['sequence'],1)
        self.assertFalse((self.folder/'generation_000002.committed.json').exists())

if __name__=='__main__':unittest.main()
