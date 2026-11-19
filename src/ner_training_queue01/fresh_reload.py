import argparse,time
from unittest.mock import patch
from pilot.common import read_json,read_jsonl,file_hash
from stage_preflight.state import fingerprint
from .core import load_data,initialize
from checkpoint_recovery01.postwrite import compare_predictions,scientific_metrics
from .common import D,RUN,utc,durable,verify_plan
from .restore import restore

def main(deadline):
    verify_plan();post=read_json(D/'audit/POSTWRITE.json');assert post['all_verified'] and post['writer_complete']
    epoch=post['selected_epoch'];metadata=RUN/'checkpoints'/f'epoch_{epoch:02d}.json';model,tok,opt,sched=initialize()
    p,source,meta=restore(metadata,model,opt,sched,D/'fresh_reload/restore')
    expected=source['state_digests']['model'];del source
    def forbidden(*a,**kw):raise AssertionError('No optimizer/scheduler updates in fresh dev verification')
    from . import evaluation
    evaluation.RUN=D/'fresh_reload';p=dict(p);p.update(epoch=epoch-1,phase='evaluate',evaluation_model_sha256=expected)
    with patch.object(opt,'step',forbidden),patch.object(type(sched),'step',forbidden):
        metrics=evaluation.evaluate(model,load_data('valid'),evaluation.dev_rows(),p,meta['run_hash'],deadline)
    assert metrics is not None,'Fresh dev reproduction incomplete at deadline'
    actual=fingerprint(model.state_dict());durable(D/'fresh_reload/model_after.json',{'actual':actual,'expected':expected})
    assert actual==expected and file_hash(metadata.parent/meta['file'])==meta['sha256']
    new=read_jsonl(D/'fresh_reload/evaluations'/f'epoch_{epoch:02d}'/'predictions.jsonl');old=read_jsonl(RUN/'evaluations'/f'epoch_{epoch:02d}'/'predictions.jsonl')
    comparison=compare_predictions(new,old);comparison['scientific_metrics_match']=scientific_metrics(metrics)==scientific_metrics(read_json(RUN/'evaluations'/f'epoch_{epoch:02d}'/'metrics.json'))
    comparison.update(utc=utc(),selected_epoch=epoch,fresh_process=True,optimizer_updates=0,exact_confidence_required=True)
    durable(D/'audit/FRESH_RELOAD.json',comparison)
    assert comparison['all_agree'] and comparison['scientific_metrics_match'],'Fresh full-dev reproduction mismatch'

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--deadline',type=float,required=True);main(p.parse_args().deadline)
