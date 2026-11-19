import argparse
from unittest.mock import patch
from pilot.common import read_json,read_jsonl,file_hash
from stage_preflight.state import fingerprint
from .core import load_data,initialize
from .common import D,RUN,utc,durable,verify_plan
from .restore import restore
from .evaluation import compare_predictions,scientific_metrics

def main(deadline):
    verify_plan();post=read_json(D/'audit/POSTWRITE.json');assert post['all_verified'] and post['writer_complete']
    epoch=post['selected_epoch'];metadata=RUN/'checkpoints'/f'epoch_{epoch:02d}.json';model,tok,opt,sched=initialize()
    p,source,meta=restore(metadata,model,opt,sched,D/'fresh_reload/restore');expected=source['state_digests']['model'];del source
    def forbidden(*a,**kw):raise AssertionError('No optimizer/scheduler updates during fresh dev reproduction')
    from . import evaluation
    evaluation.RUN=D/'fresh_reload';p=dict(p);p.update(epoch=epoch-1,phase='evaluate',evaluation_model_sha256=expected)
    with patch.object(opt,'step',forbidden),patch.object(type(sched),'step',forbidden):metrics=evaluation.evaluate(model,load_data('valid'),evaluation.dev_rows(),p,meta['run_hash'],deadline)
    assert metrics is not None,'Fresh dev verification incomplete at ceiling'
    actual=fingerprint(model.state_dict());durable(D/'fresh_reload/model_after.json',{'actual':actual,'expected':expected})
    assert actual==expected and file_hash(metadata.parent/meta['file'])==meta['sha256']
    new=read_jsonl(D/'fresh_reload/evaluations'/f'epoch_{epoch:02d}'/'predictions.jsonl');old=read_jsonl(RUN/'evaluations'/f'epoch_{epoch:02d}'/'predictions.jsonl')
    result=compare_predictions(new,old);result['scientific_metrics_match']=scientific_metrics(metrics)==scientific_metrics(read_json(RUN/'evaluations'/f'epoch_{epoch:02d}'/'metrics.json'))
    result.update(utc=utc(),selected_epoch=epoch,fresh_process=True,optimizer_updates=0,exact_logits_probabilities_required=True)
    durable(D/'audit/FRESH_RELOAD.json',result);assert result['all_agree'] and result['scientific_metrics_match'],'Fresh full-dev mismatch'

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--deadline',type=float,required=True);main(p.parse_args().deadline)
