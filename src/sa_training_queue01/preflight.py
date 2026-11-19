"""One fixed train-only microbatch benchmark per encoder; no development-score choice."""
import time
from pilot.common import read_json
from encoder_cont01.awake import Awake
from .common import Q,utc,durable,verify_plan
from .prepare import make_config
from .queue import stage
from .disk import audit

def main():
    # Reuse subprocess/PID handling while keeping the preflight ledger separate.
    from . import queue as queue_module
    from .common import append
    from encoder_lr_screen02.checkpoints import durable_json
    def preflight_save(q,event=None):
        durable_json(Q/'PREFLIGHT_STATE.json',q)
        if event:append(Q/'preflight.jsonl',{'utc':utc(),**event});print(event,flush=True)
    queue_module.save=preflight_save
    verify_plan();budget=read_json(Q/'BUDGET.json');end=budget['preparation_deadline_unix'];assert time.time()<end
    assert not (Q/'PREFLIGHT.json').exists(),'Preflight is one predetermined sequence, not repeatable until pass'
    audit();q={'status':'preflight','conditions':[],'deadline_unix':end};durable(Q/'PREFLIGHT.json',q)
    try:
        with Awake(Q/'preflight_awake.json') as awake:
            for enc in ['xlmr','mbert']:
                entry={'id':enc+'_train_only_preflight','encoder':enc,'status':'running'};q['conditions'].append(entry)
                model_end=min(end,time.time()+3600);out=[]
                for mb in [2,4,8]:
                    d=Q/'preflight'/enc/f'mb{mb}';make_config(d,enc,13,1e-5,mb)
                    stage(q,entry,'benchmark_'+str(mb),'sa_training_queue01.worker',['benchmark','--microbatch',str(mb)],d,model_end,awake)
                    out.append(read_json(d/f'BENCHMARK_{mb}.json'))
                assert out[0]['status']=='passed' and out[0]['headroom_fraction']>=.2,'Baseline execution/memory preflight failed'
                eligible=[r for r in out[1:] if r['status']=='passed' and r['cv']<=.2 and r['headroom_fraction']>=.2 and r['mean_seconds']<=out[0]['mean_seconds']*.95]
                chosen=min(eligible,key=lambda r:r['mean_seconds']) if eligible else out[0]
                durable(Q/(enc.upper()+'_EXECUTION_SELECTION.json'),{'utc':utc(),'microbatch':chosen['microbatch'],'accumulation':16//chosen['microbatch'],'score_based':False,'candidates':out,
                    'rule':'fixed2x8,4x4,8x2; at least5% faster than2x8, CV<=.20, reserved VRAM headroom>=.20; otherwisebaseline',
                    'projected_training_update_seconds_per_five_epochs':chosen['mean_seconds']*41845,'projection_limit':'25 early batches and longest stress; main epochs and checkpoint/eval overhead not yet measured'})
                entry['status']='passed'
        assert time.time()<=end
        durable(Q/'PREPARATION.json',{'utc':utc(),'passed':True,'cpu_tests':True,'tokenizer_reencoding':True,'train_only_preflights_passed':True,'elapsed_seconds':time.time()-budget['start_unix'],'historical_cause':'unresolved'})
        q['status']='passed'
    except BaseException as exc:
        q.update(status='blocked',error=repr(exc));raise
    finally:
        # stage() uses its own durable ledger; keep preflight distinct from main queue.
        from encoder_lr_screen02.checkpoints import durable_json
        durable_json(Q/'PREFLIGHT_FINAL.json',q)

if __name__=='__main__':main()
