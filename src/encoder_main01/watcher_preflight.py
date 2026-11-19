"""Read-only validation against an immutable completed epoch; no new inference."""
import csv
import subprocess
from pilot.common import ROOT,read_json,read_jsonl,write_json,digest
from pilot.scoring import spans_bio
from seqeval.metrics import f1_score
from seqeval.scheme import IOB2
from .common import D,verify
from .main import RUN,dev_rows
from .finalize import WATCH,check_checkpoint

def main():
    verify('main_plan.lock.json')
    result,progress=check_checkpoint(RUN/'checkpoints/epoch_01.json',digest(read_json(RUN/'identity.json')))
    rows=dev_rows();pred=read_jsonl(RUN/'evaluations/epoch_01/predictions.jsonl');m=read_json(RUN/'evaluations/epoch_01/metrics.json')
    independent=f1_score([r['labels'] for r in rows],[spans_bio(p['parsed']['spans'],len(r['tokens'])) for p,r in zip(pred,rows)],mode='strict',scheme=IOB2,zero_division=0)
    assert abs(independent-m['f1'])<1e-12
    preview=WATCH/'preflight';(preview/'run_xlmr_ner_seed13_lr1e-5').mkdir(parents=True,exist_ok=True)
    write_json(preview/'run_xlmr_ner_seed13_lr1e-5/learning_curve.json',progress['history'])
    from .postprocess import csv_rows
    csv_rows(preview/'learning_curve.csv',progress['history'])
    with (preview/'training_curve_100_updates.csv').open('w',encoding='utf-8',newline='') as f:
        w=csv.writer(f);w.writerow(['update','epoch_fraction','token_weighted_training_loss'])
        for i in range(0,len(progress['trace']),100):
            b=progress['trace'][i:i+100];w.writerow([b[-1]['update'],b[-1]['update']/5587,sum(s['loss_sum'] for s in b)/sum(s['valid_labels'] for s in b)])
    p=subprocess.run([str(D/'plot_curves.exe'),str(preview)],cwd=ROOT,capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=120)
    write_json(WATCH/'preflight_result_native.json',{'checkpoint':result,'independent_seqeval_f1':independent,'scorer_agreement':True,
        'plot_returncode':p.returncode,'plot_stdout':p.stdout,'plot_stderr':p.stderr,'scope':'Existing epoch only; no new training or model inference'})
    assert p.returncode==0 and (preview/'learning_curves.png').stat().st_size>1000
    print({'checkpoint_integrity':True,'seqeval_f1':independent,'plot':str(preview/'learning_curves.png')})

if __name__=='__main__':main()
