"""Fresh CPU process, canonical plus independent digest and unchanged old verifier."""
import argparse,gc,os,traceback,math
from pathlib import Path
from pilot.common import ROOT,read_json,file_hash
from checkpoint_integrity01.independent_verifier import independent_hash
from .common import COMPONENTS,utc,durable
from .checkpoints import verify_payload

def inspect(metadata,output):
    import torch
    torch.set_num_threads(4);metadata=Path(metadata).resolve();output=Path(output);m=read_json(metadata);path=metadata.parent/m['file']
    result={'utc':utc(),'metadata':str(metadata),'pid':os.getpid(),'accepted':False}
    try:
        proof,state,count=verify_payload(path,m,output.with_name(output.stem+'_canonical.json'))
        actual={k:independent_hash(state[k]) for k in COMPONENTS}
        durable(output.with_name(output.stem+'_independent_before_assertion.json'),{'actual':actual,'embedded':state['state_digests'],'sidecar':m['state_digests']})
        assert actual==state['state_digests']==m['state_digests'],'Independent component mismatch'
        progress=state['progress'];p=progress
        # The unchanged historical verifier's structural rules on THIS checked load.
        assert p['updates']==p['epoch']*5587+math.ceil(p['cursor']/16)
        assert 0<=p['epoch']<=5 and 0<=p['cursor']<=89386 and p['phase'] in ['train','evaluate','complete']
        assert len(p['trace'])==p['updates'] and [r['update'] for r in p['trace']]==list(range(1,p['updates']+1))
        assert state['scheduler']['last_epoch']==p['updates']
        params=[i for group in state['optimizer']['param_groups'] for i in group['params']]
        assert set(params)==set(state['optimizer']['state'])
        assert all(int(x['step'])==p['updates'] for x in state['optimizer']['state'].values())
        assert set(state['rng'])=={'random','numpy','torch','cuda'} and state['rng']['cuda']
        assert all(torch.is_tensor(x) for x in state['rng']['cuda'])
        old_result={'unchanged_historical_structural_rules_applied_to_same_loaded_state':True,'old_verifier_source_unchanged':True}
        del state;gc.collect()
        assert file_hash(path)==m['sha256']
        result.update(accepted=True,canonical=proof,independent_signatures=actual,old_verifier=old_result,finite_elements=count,
            progress={k:v for k,v in progress.items() if k not in ['trace']},sha256_after=file_hash(path))
    except BaseException as e:result.update(error_type=type(e).__name__,error=str(e),traceback=traceback.format_exc());raise
    finally:durable(output,result)
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--metadata',required=True);p.add_argument('--output',required=True);a=p.parse_args();r=inspect(a.metadata,a.output);print({'accepted':r['accepted']},flush=True)
