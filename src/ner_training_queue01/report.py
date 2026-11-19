"""Saved-prediction reporting only: no model loading, no test access."""
import csv,json,statistics,time,zipfile,hashlib,sys
from pathlib import Path
from pilot.common import ROOT,read_json,read_jsonl,file_hash,write_json
from pilot.evaluate import summarize
from .common import Q,D,RUN,utc,durable,config,verify_plan

def optional(path,default=None):return read_json(path) if Path(path).exists() else ({} if default is None else default)
def csv_out(path,rows):
    if not rows:return
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with Path(path).open('w',encoding='utf-8',newline='') as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
def lines(path):return read_jsonl(path) if Path(path).exists() else []

def condition():
    verify_plan();c=config();outcome=read_json(D/'WRITER_RESULT.json');p=outcome['progress'];post=read_json(D/'audit/POSTWRITE.json');fresh=optional(D/'audit/FRESH_RELOAD.json')
    complete=outcome['status']=='complete' and post['all_verified'] and fresh.get('all_agree') and fresh.get('scientific_metrics_match')
    panel=read_jsonl(ROOT/'data/pilot/ner_valid.jsonl');curves=[];types=[]
    for h in p['history']:
        path=RUN/'evaluations'/f'epoch_{h["epoch"]:02d}';m=read_json(path/'metrics.json');pred=read_jsonl(path/'predictions.jsonl');by_id={r['id']:r for r in pred}
        assert len(pred)==len(by_id)==11100
        panelmetrics=summarize('ner',panel,[by_id[r['id']] for r in panel]);durable(D/'panel'/f'epoch_{h["epoch"]:02d}.json',{'source_sha256':file_hash(path/'predictions.jsonl'),'metrics':panelmetrics,'selection_used':False})
        curves.append({**h,'panel200_f1_descriptive':panelmetrics['f1'],'runtime_failures':m['runtime_failures'],'format_failures':m['format_failures']})
        types += [{'epoch':h['epoch'],'type':typ,**value} for typ,value in m['per_type'].items()]
    csv_out(D/'learning_curve.csv',curves);csv_out(D/'per_type_all_epochs.csv',types)
    telemetry=[];evals=[]
    for path in (RUN/'attempts').glob('*/telemetry.jsonl'):telemetry+=lines(path)
    for path in (RUN/'attempts').glob('*/evaluation_telemetry.jsonl'):evals+=lines(path)
    resources={'training_synchronized_update_wall_seconds':sum(x['seconds'] for x in telemetry),'new_updates':len(telemetry),
        'evaluation_wall_seconds':sum(x['seconds'] for x in evals),'checkpoint_capture_verification_commit_retention_seconds':sum(x['seconds'] for x in lines(RUN/'checkpoints/save_events.jsonl')),
        'peak_allocated_bytes':max((x['peak_gpu_allocated_bytes'] for x in telemetry),default=0),'peak_reserved_bytes':max((x['peak_gpu_reserved_bytes'] for x in telemetry),default=0),
        'updates_above_10_seconds':[x for x in telemetry if x['seconds']>10],'scope':'instrumented synchronized wall time, not GPU kernel time; loading/restoration/fresh checks remain separate in queue stages'}
    durable(D/'RESOURCES.json',resources)
    epoch=p['best_epoch'];meta=read_json(RUN/'checkpoints'/f'epoch_{epoch:02d}.json') if epoch else None
    final={'utc':utc(),'status':'verified_complete' if complete else 'incomplete_budget','encoder':c['encoder'],'learning_rate':c['learning_rate'],'seed':c['seed'],
        'completed_epochs':len(p['history']),'completed_updates':p['updates'],'selected_epoch':epoch,'full_dev_f1':p['best_f1'] if epoch else None,
        'selected_checkpoint':str(RUN/'checkpoints'/meta['file']) if meta else None,'selected_checkpoint_sha256':meta['sha256'] if meta else None,
        'all_retained_checkpoints_verified':post['all_verified'],'fresh_exact_tags_spans_confidences_metrics':bool(fresh.get('all_agree') and fresh.get('scientific_metrics_match')),
        'resources':resources,'historical_cause':'unresolved','final_test_locked':True,'shutdown_enabled':False}
    durable(D/'FINAL_STATUS.json',final)
    curve='\n'.join(f'| {r["epoch"]} | {r["training_loss"]:.6f} | {r["dev_micro_f1"]:.9f} | {r["dev_precision"]:.9f} | {r["dev_recall"]:.9f} | {r["panel200_f1_descriptive"]:.9f} |' for r in curves)
    (D/'REPORT.md').write_text(f'''# {D.name}

Status: **{final['status']}**. Fresh pinned pretrained initialization, seed {c['seed']}, LR {c['learning_rate']}.
Selected epoch {epoch}; full-dev exact-span/type micro-F1 {final['full_dev_f1']}.
Checkpoint: `{final['selected_checkpoint']}`; SHA-256 `{final['selected_checkpoint_sha256']}`.

| Epoch | Train loss | Full-dev F1 | Precision | Recall | Panel200, descriptive only |
|---|---:|---:|---:|---:|---:|
{curve}

All retained checkpoints independently verified after writer exit: {post['all_verified']}.
Fresh-process selected-checkpoint tags/spans/confidences/scientific metrics exact: {final['fresh_exact_tags_spans_confidences_metrics']}.
All five restored state components are checked; no digest exclusions. Generation receipts,
independent precommit audits, retirement intentions and receipts remain in run/checkpoints.
Source model retained until synchronized restoration verification; no historical checkpoint repaired.

Training update wall seconds {resources['training_synchronized_update_wall_seconds']:.3f}; evaluation
wall seconds {resources['evaluation_wall_seconds']:.3f}; checkpoint/checking/retention seconds
{resources['checkpoint_capture_verification_commit_retention_seconds']:.3f}. Peak allocated/reserved
{resources['peak_allocated_bytes']}/{resources['peak_reserved_bytes']} bytes. These are instrumented
wall measurements, not pure GPU compute. Startup, planned restart, fresh reproduction and other
overhead appear separately in QUEUE.json. No interruption is labeled compute without evidence.

The main NER objective and scorer are unchanged: summed CE divided by all supervised first
subwords in the complete optimizer batch; exact original-token entity span and type micro-F1;
existing deterministic BIO repair, with failures retained. Full25 entity types/51 BIO labels.
Full-dev is exploratory and repeatedly used. Panel results come only from saved predictions.
Per-type results and distributions for every epoch remain saved. Historical cause unresolved;
successful checks are not a proven explanation of earlier failures. Final test locked, shutdown disabled.
''',encoding='utf-8')

def summary(q):
    rows=[];allcurves=[];alltypes=[]
    for e in q['conditions']:
        if e['id']=='xlmr_lr1e-5_seed13':
            f=read_json(ROOT/'checkpoint_recovery01/FINAL_STATUS.json');r={'model':'xlmr','lr':1e-5,'seed':13,'selected_epoch':f['selected_epoch_by_frozen_rule'],
                'full_dev_f1':f['selected_full_dev_f1'],'checkpoint':f['selected_checkpoint'],'status':e['status'],'verification':'verified historical completed condition; not repeated',
                'elapsed_seconds':'historical multi-milestone; see original timing audits','peak_allocated_bytes':'see preserved original resource evidence'}
        elif e['id']=='xlmr_lr3e-5_seed13_recovery':
            f=optional(ROOT/'encoder_lr_screen02_recovery01/FINAL_STATUS.json');res=optional(ROOT/'encoder_lr_screen02_recovery01/resource_results.json')
            r={'model':'xlmr','lr':3e-5,'seed':13,'selected_epoch':f.get('selected_epoch'),'full_dev_f1':f.get('selected_full_dev_f1'),
                'checkpoint':f.get('selected_checkpoint'),'status':e['status'],'verification':f.get('fresh_full_dev_exact_reproduction'),
                'elapsed_seconds':res.get('elapsed_at_report_seconds'),'peak_allocated_bytes':res.get('peak_gpu_allocated_bytes')}
        else:
            f=optional(Path(e.get('directory',str(Q/'conditions'/e['id'])))/'FINAL_STATUS.json');res=f.get('resources',{})
            r={'model':e['encoder'],'lr':e.get('learning_rate'),'seed':e['seed'],'selected_epoch':f.get('selected_epoch'),'full_dev_f1':f.get('full_dev_f1'),
                'checkpoint':f.get('selected_checkpoint'),'status':e['status'],'verification':f.get('fresh_exact_tags_spans_confidences_metrics'),
                'elapsed_seconds':sum(s.get('seconds',0) for s in e.get('stages',{}).values()),'peak_allocated_bytes':res.get('peak_allocated_bytes')}
            if e.get('directory'):
                d=Path(e['directory'])
                for path,target in [(d/'learning_curve.csv',allcurves),(d/'per_type_all_epochs.csv',alltypes)]:
                    if path.exists():
                        with path.open(encoding='utf-8',newline='') as handle:target.extend({'condition':e['id'],**v} for v in csv.DictReader(handle))
        r['condition']=e['id'];rows.append(r)
    csv_out(Q/'CONDITIONS.csv',rows);csv_out(Q/'NEW_CONDITIONS_LEARNING_CURVES.csv',allcurves);csv_out(Q/'NEW_CONDITIONS_PER_TYPE.csv',alltypes)
    aggregates=[]
    for enc in ['xlmr','mbert']:
        lr=q['selected_lr'].get(enc);selected=[r for r in rows if r['model']==enc and r['lr']==lr and r['status']=='verified_complete']
        values={r['seed']:r['full_dev_f1'] for r in selected if r['full_dev_f1'] is not None};complete=set(values)=={13,42,2026}
        aggregates.append({'model':enc,'selected_lr':lr,'seed13':values.get(13),'seed42':values.get(42),'seed2026':values.get(2026),
            'mean':statistics.mean(values.values()) if complete else None,'sample_sd':statistics.stdev(values.values()) if complete else None,'complete_three_seeds':complete})
    csv_out(Q/'SEED_AGGREGATES.csv',aggregates)
    table='\n'.join(f'| {r["model"]} | {r["lr"]} | {r["seed"]} | {r["selected_epoch"]} | {r["full_dev_f1"]} | {r["elapsed_seconds"]} | {r["peak_allocated_bytes"]} | {r["status"]} | {r["verification"]} |' for r in rows)
    agg='\n'.join(f'| {a["model"]} | {a["selected_lr"]} | {a["seed13"]} | {a["seed42"]} | {a["seed2026"]} | {a["mean"]} | {a["sample_sd"]} |' for a in aggregates)
    preservation={'utc':utc(),'changed':[]}
    try:verify_plan();preservation['protected_source_data_context_checks_passed']=True
    except Exception as e:preservation.update(protected_source_data_context_checks_passed=False,error=str(e))
    for n,h in read_json(Q/'PRIOR_EVIDENCE.json')['files'].items():
        if not (ROOT/n).exists() or file_hash(ROOT/n)!=h:preservation['changed'].append(n)
    durable(Q/'PRESERVATION_AFTER.json',preservation)
    if preservation['changed'] or not preservation['protected_source_data_context_checks_passed']:q['status']='blocked';q['preservation_failure']=preservation
    durable(Q/'FINAL_STATUS.json',{'utc':utc(),'status':q['status'],'conditions':rows,'seed_aggregates':aggregates,'historical_cause':'unresolved','failure':q.get('failure'),
        'shutdown_enabled':False,'final_test_locked':True,'server_task_accessed':False,'elapsed_seconds':time.time()-q['start_unix'],'preservation':preservation})
    (Q/'REPORT.md').write_text(f'''# Sequential NER queue

**Status: {q['status']}. Historical checkpoint cause: unresolved.**
The original recovery retained its own stricter deadlines. New conditions were sequential on one local GPU.
No previous condition was repeated, no failed stage replaced and no final-test data loaded.
Failure/blocker: `{q.get('failure',{}).get('error','none')}`.

| Model | LR | Seed | Selected epoch | Full-dev F1 | Wall seconds | Peak allocated bytes | Status | Fresh verification |
|---|---:|---:|---:|---:|---|---|---|---|
{table}

Exact checkpoint paths are in CONDITIONS.csv and each FINAL_STATUS.json. Pending/unstarted entries
have no measured score. A completed epoch does not make a condition verified complete.

| Model | Selected LR | Seed13 | Seed42 | Seed2026 | Mean | Sample SD (n-1) |
|---|---:|---:|---:|---:|---:|---:|
{agg}

Mean/SD require all three verified selected-LR seeds; incomplete triplets are not summarized as final.
Losing LR results remain retained. LR uses each seed13 condition's best full-dev F1; exact tie
retains1e-5, and selection is recorded before seeds42/2026. No best-seed reporting.

All new conditions initialize from pinned pretrained weights with fresh task head and optimizer.
Python, NumPy, CPU/CUDA RNG are seeded before initialization; epoch order uses seed+epoch.
Dropout uses restored RNG. Five-epoch scheduler horizons are independently derived from each
model's actual retained features; maximum length, ownership and tokenizer-specific features
are checked without changing shared text. mBERT execution preflight is score-free and bounded.
No package, driver, attention, precision, objective or scoring change was made for a passing result.

Every new checkpoint is independently read before commit; owned capture avoids aliasing live
training values. Two latest accepted rolling generations and all epochs are retained; deletion
is confined to this queue and metadata/audit trails remain. Selected full-dev tags, spans,
confidences and scientific metrics require exact fresh-process reproduction. The checked
execution policy is not proof of the historical failure's cause.

Runtime accounting: QUEUE.json contains start/end/PID/create-time/command for every stage.
Per-condition RESOURCES.json separates synchronized update wall time, evaluation and checkpoint
overhead. Initialization, planned restart and fresh verification are separate stage costs;
residual wall time is not labeled GPU compute. Old multi-milestone timing is not replaced by
the small recovery replay duration. Budgets are ceilings, not speed forecasts.

Full-dev remains exploratory. Fine-tuned encoders and zero/few-shot systems have unequal
task-label budgets; this queue does not resolve contribution/novelty or pretraining contamination.
Per-type results matter, especially rare classes. No final paper result or test conclusion is claimed.
The supplied Master Plan was read as context and not changed; current user instructions govern.
Source/configuration snapshots, exact commands, failures and checkpoint lineage remain local.
''',encoding='utf-8')
    commands='''# Commands and re-entry

```powershell
Set-Location -LiteralPath 'C:\\Users\\Aru\\Desktop\\рисерч ниц ласт\\kazakh_nlp_starter'
$env:PYTHONDONTWRITEBYTECODE='1'
.\\.venv\\Scripts\\python.exe -B -X utf8 -m ner_training_queue01.queue
```

The owner lock and persisted stage statuses prevent duplicate launches. Terminal blocked/partial
queues do not automatically restart. Exact child commands, condition environment directories,
PIDs/create times and immutable deadlines are in QUEUE.json; environment in common.env.
Do not execute child stages manually to retry a failed integrity gate. A normal interruption
requires auditing the explicit saved result/accepted checkpoint before resumption, never accepting
process disappearance as success. A running worker is not replaced by a second worker.

CPU-only preparation: `-m ner_training_queue01.prepare views`, then the unittest command in
PREPARATION_TESTS.json, then `-m ner_training_queue01.prepare freeze`. These are historical
commands; frozen sources must not be regenerated during execution.
'''
    (Q/'COMMANDS.md').write_text(commands,encoding='utf-8')
    # Compact evidence: source, configurations, metrics, audit summaries; no weights, streams or secrets.
    names=[]
    for path in Q.rglob('*'):
        if not path.is_file() or any(s in path.parts for s in ['runtime','items','orders','tests','__pycache__']):continue
        if path.suffix not in ['.py','.md','.json','.csv','.patch','.sha256']:continue
        if path.name in ['BUNDLE_MANIFEST.json','BUNDLE_VERIFICATION.json'] or path.stat().st_size>8*2**20:continue
        names.append(path)
    manifest={p.relative_to(Q).as_posix():file_hash(p) for p in names};durable(Q/'BUNDLE_MANIFEST.json',manifest)
    archive=Q/'review_bundle.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for path in names+[Q/'BUNDLE_MANIFEST.json']:z.write(path,path.relative_to(Q))
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        for n,h in manifest.items():assert hashlib.sha256(z.read(n)).hexdigest()==h
    durable(Q/'BUNDLE_VERIFICATION.json',{'utc':utc(),'sha256':file_hash(archive),'member_hashes_verified':True,'weights_or_secrets_included':False,'bytes':archive.stat().st_size})

if __name__=='__main__':
    if sys.argv[1]=='condition':condition()
    else:summary(read_json(Q/'QUEUE.json'))
