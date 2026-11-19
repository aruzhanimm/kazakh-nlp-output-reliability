"""Read saved outputs only; no model loading, external API or final-test access."""
import csv,json,statistics,time,zipfile,hashlib,sys
from pathlib import Path
from pilot.common import ROOT,read_json,read_jsonl,file_hash,digest
from pilot.evaluate import summarize
from .common import Q,D,RUN,utc,durable,config,verify_plan

def optional(path):return read_json(path) if Path(path).exists() else {}
def lines(path):return read_jsonl(path) if Path(path).exists() else []
def csv_out(path,rows):
    if not rows:return
    with Path(path).open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)));w.writeheader();w.writerows(rows)

def condition():
    verify_plan();c=config();outcome=read_json(D/'WRITER_RESULT.json');p=outcome['progress'];post=read_json(D/'audit/POSTWRITE.json');fresh=optional(D/'audit/FRESH_RELOAD.json')
    complete=outcome['status']=='complete' and post['all_verified'] and fresh.get('all_agree') and fresh.get('scientific_metrics_match')
    panel=read_jsonl(ROOT/'data/pilot/sa_valid.jsonl');curves=[];classes=[]
    for h in p['history']:
        path=RUN/'evaluations'/f'epoch_{h["epoch"]:02d}';m=read_json(path/'metrics.json');pred=read_jsonl(path/'predictions.jsonl');by_id={r['id']:r for r in pred}
        assert len(pred)==len(by_id)==16766
        subset=[by_id[r['id']] for r in panel];assert all(x['visible_text_sha256']==digest(r['visible_text']) for x,r in zip(subset,panel))
        pm=summarize('sa',panel,subset);durable(D/'panel'/f'epoch_{h["epoch"]:02d}.json',{'source_sha256':file_hash(path/'predictions.jsonl'),'metrics':pm,'selection_used':False,'ids':[r['id'] for r in panel]})
        curves.append({**h,'dev_cross_entropy':m['dev_cross_entropy'],'panel200_macro_f1_descriptive':pm['macro_f1'],'runtime_failures':m['runtime_failures'],'format_failures':m['format_failures']})
        classes += [{'epoch':h['epoch'],'class':k,**v} for k,v in m['per_class'].items()]
    csv_out(D/'learning_curve.csv',curves);csv_out(D/'per_class_all_epochs.csv',classes)
    telemetry=[];evals=[]
    for path in (RUN/'attempts').glob('*/telemetry.jsonl'):telemetry+=lines(path)
    for path in (RUN/'attempts').glob('*/evaluation_telemetry.jsonl'):evals+=lines(path)
    resources={'training_synchronized_update_wall_seconds':sum(x['seconds'] for x in telemetry),'new_updates':len(telemetry),
        'evaluation_wall_seconds':sum(x['seconds'] for x in evals),'checkpoint_capture_verification_commit_retention_seconds':sum(x['seconds'] for x in lines(RUN/'checkpoints/save_events.jsonl')),
        'peak_allocated_bytes':max((x['peak_gpu_allocated_bytes'] for x in telemetry),default=0),'peak_reserved_bytes':max((x['peak_gpu_reserved_bytes'] for x in telemetry),default=0),
        'updates_above_10_seconds':[x for x in telemetry if x['seconds']>10],'scope':'instrumented synchronized update wall; not pure GPU compute; startup/fresh checks separately in QUEUE.json'}
    durable(D/'RESOURCES.json',resources);epoch=p['best_epoch'];meta=read_json(RUN/'checkpoints'/f'epoch_{epoch:02d}.json') if epoch else None
    final={'utc':utc(),'status':'verified_complete' if complete else 'incomplete_budget','encoder':c['encoder'],'learning_rate':c['learning_rate'],'seed':c['seed'],
        'completed_epochs':len(p['history']),'completed_updates':p['updates'],'selected_epoch':epoch,'full_dev_f1':p['best_f1'] if epoch else None,'primary_metric':'macro_f1',
        'selected_checkpoint':str(RUN/'checkpoints'/meta['file']) if meta else None,'selected_checkpoint_sha256':meta['sha256'] if meta else None,
        'all_retained_checkpoints_verified':post['all_verified'],'fresh_exact_labels_logits_probabilities_metrics':bool(fresh.get('all_agree') and fresh.get('scientific_metrics_match')),
        'resources':resources,'historical_cause':'unresolved','final_test_locked':True,'shutdown_enabled':False}
    if epoch:final['selected_metrics']=read_json(RUN/'evaluations'/f'epoch_{epoch:02d}'/'metrics.json')
    durable(D/'FINAL_STATUS.json',final)
    table='\n'.join(f'| {r["epoch"]} | {r["training_loss"]:.6f} | {r["dev_macro_f1"]:.9f} | {r["dev_accuracy"]:.9f} | {r["dev_negative_recall"]:.6f} | {r["dev_positive_recall"]:.6f} | {r["panel200_macro_f1_descriptive"]:.9f} |' for r in curves)
    (D/'REPORT.md').write_text(f'''# {D.name}

Status: **{final['status']}**. Fresh pinned pretrained, seed{c['seed']}, LR{c['learning_rate']}.
Selected epoch{epoch}; full-dev SA macro-F1 {final['full_dev_f1']}.
Checkpoint `{final['selected_checkpoint']}`, SHA256 `{final['selected_checkpoint_sha256']}`.

| Epoch | Train loss | Dev macro-F1 | Accuracy | Negative recall | Positive recall | Panel200 descriptive |
|---|---:|---:|---:|---:|---:|---:|
{table}

Independent postwriter verification: {post['all_verified']}.
Fresh16766 labels/logits/probabilities/scientific metrics exact: {final['fresh_exact_labels_logits_probabilities_metrics']}.
Per-class P/R/F1/support, confusion including INVALID, majority baseline and distributions
are retained for every epoch. Failures remain accounted for. Argmax two logits only.

Update wall seconds {resources['training_synchronized_update_wall_seconds']:.3f}; evaluation
{resources['evaluation_wall_seconds']:.3f}; checkpoint capture/check/readers/commit/retention
{resources['checkpoint_capture_verification_commit_retention_seconds']:.3f}. Peak allocated/reserved
{resources['peak_allocated_bytes']}/{resources['peak_reserved_bytes']} bytes. These include
instrumentation; residual wall time is not labeled active GPU compute. Loading, planned restart,
postwriter readers and fresh selected dev have separate stage records.

SA summed CE is normalized by actual reviews in the full optimizer batch; partial6 uses6.
Selection/stopping and five-epoch horizon are frozen. All five state components checked,
no exclusions; CPU source remains alive through synchronized restore verification.
Historical failure cause unresolved. Dev exploratory; panel only derived from saved outputs.
Final test locked, shutdown disabled. Prior NER/data/environment artifacts unchanged.
''',encoding='utf-8')

def summary(q):
    rows=[];curves=[];classes=[]
    for e in q['conditions']:
        d=Path(e.get('directory',str(Q/'conditions'/e['id'])));f=optional(d/'FINAL_STATUS.json');res=f.get('resources',{});m=f.get('selected_metrics',{})
        r={'condition':e['id'],'model':e['encoder'],'lr':e.get('learning_rate'),'seed':e['seed'],'selected_epoch':f.get('selected_epoch'),'full_dev_macro_f1':f.get('full_dev_f1'),
            'accuracy':m.get('accuracy'),'checkpoint':f.get('selected_checkpoint'),'checkpoint_sha256':f.get('selected_checkpoint_sha256'),'status':e['status'],
            'fresh_exact_verification':f.get('fresh_exact_labels_logits_probabilities_metrics'),'stage_wall_seconds':sum(s.get('seconds',0) for s in e.get('stages',{}).values()),**res};rows.append(r)
        for path,target in [(d/'learning_curve.csv',curves),(d/'per_class_all_epochs.csv',classes)]:
            if path.exists():
                with path.open(encoding='utf-8',newline='') as h:target.extend({'condition':e['id'],**x} for x in csv.DictReader(h))
    csv_out(Q/'CONDITIONS.csv',rows);csv_out(Q/'ALL_LEARNING_CURVES.csv',curves);csv_out(Q/'ALL_PER_CLASS.csv',classes)
    aggregates=[]
    for enc in ['xlmr','mbert']:
        lr=q['selected_lr'].get(enc);values={r['seed']:r['full_dev_macro_f1'] for r in rows if r['model']==enc and r['lr']==lr and r['status']=='verified_complete'};complete=set(values)=={13,42,2026}
        aggregates.append({'model':enc,'selected_lr':lr,'seed13':values.get(13),'seed42':values.get(42),'seed2026':values.get(2026),
            'mean':statistics.mean(values.values()) if complete else None,'sample_sd':statistics.stdev(values.values()) if complete else None,'complete_three_seeds':complete})
    csv_out(Q/'SEED_AGGREGATES.csv',aggregates)
    from .llm_reference import comparison
    panel_rows=comparison(q);csv_out(Q/'PANEL200_COMPARISON.csv',panel_rows)
    preservation={'utc':utc(),'changed':[]}
    try:verify_plan();preservation['protected_source_data_context_checks_passed']=True
    except Exception as exc:preservation.update(protected_source_data_context_checks_passed=False,error=str(exc))
    for n,h in read_json(Q/'PRIOR_EVIDENCE.json')['files'].items():
        if not (ROOT/n).exists() or file_hash(ROOT/n)!=h:preservation['changed'].append(n)
    if preservation['changed'] or not preservation['protected_source_data_context_checks_passed']:q['status']='blocked'
    durable(Q/'PRESERVATION_AFTER.json',preservation)
    final={'utc':utc(),'status':q['status'],'conditions':rows,'seed_aggregates':aggregates,'historical_cause':'unresolved','failure':q.get('failure'),
        'shutdown_enabled':False,'final_test_locked':True,'server_task_accessed':False,'elapsed_seconds':time.time()-q['start_unix'],'preservation':preservation}
    durable(Q/'FINAL_STATUS.json',final)
    table='\n'.join('| '+' | '.join(str(r.get(k)) for k in ['model','lr','seed','selected_epoch','full_dev_macro_f1','accuracy','stage_wall_seconds','peak_allocated_bytes','status'])+' |' for r in rows)
    agg='\n'.join('| '+' | '.join(str(r[k]) for k in ['model','selected_lr','seed13','seed42','seed2026','mean','sample_sd'])+' |' for r in aggregates)
    (Q/'REPORT.md').write_text(f'''# SA queue

**{q['status']}**. Historical checkpoint cause remains unresolved.
Failure if any: {q.get('failure')}. Eight fixed sequential conditions; no NER reruns.

| Model | LR | Seed | Epoch | Full-dev macro-F1 | Accuracy | Stage wall seconds | Peak allocated bytes | Status |
|---|---:|---:|---:|---:|---:|---:|---:|---|
{table}

| Model | Selected LR | Seed13 | Seed42 | Seed2026 | Mean | Sample SD (n-1) |
|---|---:|---:|---:|---:|---:|---:|
{agg}

Exact checkpoint paths/hashes are in CONDITIONS.csv. All losing LR/epoch results retained.
Means/SD require the complete selected-LR triplet, without best-seed selection. One seed13
LR screen does not establish LR significance. SD describes training-seed variability,
not uncertainty over the population of reviews.

16766 full-dev reviews, macro-F1 primary, original class map and visible text. Full train
133894 with natural imbalance. No weights/oversampling/threshold search. Majority baseline,
per-class support/P/R/F1/confusions/distributions and all failures are saved. Fixed200
panel scores derive only from saved full-dev predictions and never determine selection.
PANEL200_COMPARISON.csv includes all completed encoder conditions and the preserved
Qwen BF16 / Sherkala BF16 zero/four-shot outputs. IDs, full row hashes and literal text
were verified from completed local archives; no server task was accessed or changed.
These are descriptive comparisons with unequal adaptation budgets, not causal rankings.
Dev remains exploratory after repeated use, not final-test or SOTA evidence. Unequal
label budgets relative to zero/few-shot LLMs and pretraining/novelty limitations remain.

Every new checkpoint independently read before publication; all epochs and two rolling
generations retained, with retirement metadata. Initial roundtrip and planned update1000
restart verify all five components. Final selected dev labels/logits/probabilities and
metrics require exact fresh-process reproduction. No acceptance by retry or changed hashes.

RESOURCE/QUEUE records distinguish train, evaluation, checkpoint verification, initialization,
planned restart, fresh reproduction and residual wall overhead. Instrumented wall is not
pure GPU compute. Preparation/resource benchmark ceilings are not measured runtime forecasts.
Source/config/package locks, child commands/PIDs/create times, failures and preservation
evidence are retained. No previous data/weights/code/package/driver changes; no final test,
server operations, LLM/API calls or shutdown. Master Plan unchanged.
''',encoding='utf-8')
    (Q/'COMMANDS.md').write_text('''# Exact entry commands (historical; terminal queues must not be relaunched)

```powershell
Set-Location -LiteralPath 'C:\\Users\\Aru\\Desktop\\рисерч ниц ласт\\kazakh_nlp_starter'
.\\.venv\\Scripts\\python.exe -B -X utf8 -m sa_training_queue01.prepare views
.\\.venv\\Scripts\\python.exe -B -X utf8 -m unittest sa_training_queue01.tests -v
.\\.venv\\Scripts\\python.exe -B -X utf8 -m sa_training_queue01.prepare freeze
.\\.venv\\Scripts\\python.exe -B -X utf8 -m sa_training_queue01.preflight
.\\.venv\\Scripts\\python.exe -B -X utf8 -m sa_training_queue01.queue
```

Sources/configs are frozen before execution; do not regenerate them mid-run.
QUEUE.json and PREFLIGHT_STATE.json retain every exact child command, PID/create time,
deadline and result. common.env specifies environment settings and condition directories.
The owner lock excludes duplicate controllers; terminal blocked/partial/completed queues
never restart automatically. An ordinary interruption may resume only after auditing
accepted checkpoint/explicit progress and preserving original deadlines; process exit
alone is insufficient. Never manually retry an integrity gate until pass.
''',encoding='utf-8')
    names=[]
    for p in Q.rglob('*'):
        if not p.is_file() or any(s in p.parts for s in ['runtime','items','orders','tests','__pycache__']):continue
        if p.suffix not in ['.py','.md','.json','.csv','.patch','.sha256','.log'] and p.name not in ['queue.jsonl','preflight.jsonl']:continue
        if p.name in ['BUNDLE_MANIFEST.json','BUNDLE_VERIFICATION.json'] or p.stat().st_size>8*2**20:continue
        names.append(p)
    manifest={p.relative_to(Q).as_posix():file_hash(p) for p in names};durable(Q/'BUNDLE_MANIFEST.json',manifest)
    archive=Q/'review_bundle.zip'
    with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED) as z:
        for p in names+[Q/'BUNDLE_MANIFEST.json']:z.write(p,p.relative_to(Q))
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        for n,h in manifest.items():assert hashlib.sha256(z.read(n)).hexdigest()==h
    durable(Q/'BUNDLE_VERIFICATION.json',{'utc':utc(),'sha256':file_hash(archive),'member_hashes_verified':True,'weights_or_secrets_included':False,'bytes':archive.stat().st_size})

if __name__=='__main__':condition() if sys.argv[1]=='condition' else summary(read_json(Q/'QUEUE.json'))
