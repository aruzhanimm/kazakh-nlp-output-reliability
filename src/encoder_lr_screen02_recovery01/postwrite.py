"""Independent checkpoint audit after writer exit; never resumes training."""
from pilot.common import read_json,write_json,write_jsonl
from .common import D,RUN,utc,durable
from .reader import inspect

def main():
    outcome=read_json(RUN/'execution/RESULT.json');assert outcome['status'] in ['complete','incomplete_budget']
    checks=[]
    for receipt_path in sorted((RUN/'checkpoints').glob('generation_*.committed.json')):
        receipt=read_json(receipt_path);metadata=RUN/'checkpoints'/receipt['metadata'];meta=read_json(metadata)
        assert receipt['sha256']==meta['sha256'] and receipt['independent_precommit_verified']
        checks.append(inspect(metadata,D/'audit/postwrite'/f'generation_{meta["sequence"]:06d}.json'))
    assert checks
    p=outcome['progress'];write_json(RUN/'learning_curve.json',p['history'])
    durable(D/'audit/POSTWRITE.json',{'utc':utc(),'all_verified':True,'retained_checkpoints':len(checks),'writer_complete':outcome['status']=='complete',
        'selected_epoch':p['best_epoch'],'updates':p['updates'],'latest':outcome['latest'],'writer_exited':True})

if __name__=='__main__':main()
