"""Read-only compact progress view. No process control, model loading or inference."""
import json
from pathlib import Path
from pilot.common import read_json
from .common import Q

def optional(path):return read_json(path) if path.exists() else {}
def current():
    q=read_json(Q/'QUEUE.json');result={'queue_status':q['status'],'verified_conditions':sum(e['status']=='verified_complete' for e in q['conditions'])}
    e=next((e for e in q['conditions'] if e['status'] in ['running','preparing','attached_existing']),None)
    if e is None:
        result['failure']=q.get('failure',{}).get('error');return result
    stages=e.get('stages',{});stage=next((k for k,v in reversed(list(stages.items())) if v['status']=='running'),None)
    result.update(condition=e['id'],learning_rate=e.get('learning_rate'),seed=e['seed'],stage=stage)
    if e.get('directory'):
        d=Path(e['directory']);s=optional(d/'run/status.json');fresh=optional(d/'fresh_reload/status.json')
        result.update(training_updates=s.get('updates'),phase=s.get('phase'),completed_epochs=len(s.get('history',[])),
            best_dev_f1=s.get('best_f1'),best_epoch_candidate=s.get('best_epoch'),dev_predictions=s.get('dev_predictions'),
            fresh_dev_predictions=fresh.get('dev_predictions'))
        pointer=optional(d/'run/checkpoints/latest.json')
        if pointer:
            m=read_json(d/'run/checkpoints'/pointer['metadata']);result['last_committed_update']=m['updates'];result['last_committed_generation']=m['sequence']
    return result

if __name__=='__main__':print(json.dumps(current(),ensure_ascii=False,sort_keys=True),flush=True)
