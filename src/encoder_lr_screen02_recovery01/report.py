"""Read-only scientific reporting and preservation audit; no model loading/inference."""
import csv,json,time,zipfile,hashlib
from pathlib import Path
from pilot.common import ROOT,read_json,read_jsonl,file_hash,write_json
from pilot.evaluate import summarize
from .common import D,RUN,utc,durable,config,verify_plan

def optional(path,default=None):return read_json(path) if Path(path).exists() else ({} if default is None else default)
def csv_out(name,rows):
    if not rows:return
    with (D/name).open('x',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def main():
    c=config();budget=read_json(D/'BUDGET.json');assert budget['status'] in ['blocked','verified_complete','verified_incomplete_budget']
    before=read_json(D/'audit/preservation_before.json');changed=[]
    for n,v in before.items():
        p=ROOT/n
        if not p.is_file() or file_hash(p)!=v['sha256']:changed.append(n)
    protection=None
    try:verify_plan()
    except Exception as e:protection=str(e)
    durable(D/'audit/preservation_after.json',{'utc':utc(),'files_checked':len(before),'changed_or_missing':changed,'protected_source_error':protection})
    outcome=optional(RUN/'execution/RESULT.json');failure=optional(D/'FAILURE.json');p=outcome.get('progress',{})
    history=optional(RUN/'learning_curve.json',p.get('history',[]));fresh=optional(D/'audit/FRESH_RELOAD.json');post=optional(D/'audit/POSTWRITE.json')
    equivalent=optional(D/'validation/EQUIVALENCE.json');roundtrip=optional(D/'validation/ROUNDTRIP.json')
    complete=budget['status']=='verified_complete' and post.get('all_verified') and fresh.get('all_agree') and fresh.get('scientific_metrics_match') and not changed and not protection
    selected=p.get('best_epoch');selected_meta=optional(RUN/'checkpoints'/f'epoch_{selected:02d}.json') if selected else {}
    baseline=read_json(ROOT/'encoder_lr_screen02/COMPARATOR.json');score=p.get('best_f1') if selected else None
    winner=(3e-5 if score>baseline['full_dev_f1'] else 1e-5) if complete else None
    curves=[];types=[];panel=read_jsonl(ROOT/'data/pilot/ner_valid.jsonl');panel_ids=[r['id'] for r in panel];assert len(panel_ids)==200
    for h in history:
        epoch=h['epoch'];folder=RUN/'evaluations'/f'epoch_{epoch:02d}';m=read_json(folder/'metrics.json');predictions=read_jsonl(folder/'predictions.jsonl')
        assert len(predictions)==11100;by_id={x['id']:x for x in predictions};subset=[by_id[i] for i in panel_ids]
        panel_score=summarize('ner',panel,subset)
        durable(D/'panel'/f'epoch_{epoch:02d}.json',{'derivation':'saved full-dev predictions only; not used for selection','source_sha256':file_hash(folder/'predictions.jsonl'),
            'panel_sha256':file_hash(ROOT/'data/pilot/ner_valid.jsonl'),'ids':panel_ids,'metrics':panel_score})
        curves.append({'epoch':epoch,'updates':h['updates'],'training_loss':h['training_loss'],'full_dev_f1':m['f1'],'precision':m['precision'],'recall':m['recall'],
            'bio_repairs':m['bio_repairs'],'runtime_failures':m['runtime_failures'],'format_failures':m['format_failures'],'invalid_candidate_fp':m['invalid_candidate_fp'],'panel200_f1_descriptive':panel_score['f1']})
        for typ,v in m['per_type'].items():types.append({'epoch':epoch,'type':typ,**v})
    csv_out('learning_curve.csv',curves);csv_out('per_type_all_epochs.csv',types)
    lr_rows=[{'lr':1e-5,'status':'verified_completed','best_epoch':baseline['selected_epoch'],'best_full_dev_f1':baseline['full_dev_f1'],'checkpoint':baseline['selected_checkpoint'],'sha256':baseline['sha256']},
        {'lr':3e-5,'status':'verified_complete' if complete else budget['status'],'best_epoch':selected,'best_full_dev_f1':score,'checkpoint':str((RUN/'checkpoints'/selected_meta['file']).relative_to(ROOT)) if selected_meta else None,'sha256':selected_meta.get('sha256')}]
    csv_out('two_lr_comparison.csv',lr_rows)
    lineage=[]
    for path in D.rglob('generation_*.committed.json'):
        if 'test_fixtures' in path.parts:continue
        receipt=read_json(path);meta=read_json(path.parent/receipt['metadata'])
        lineage.append({'metadata':str((path.parent/receipt['metadata']).relative_to(D)),'payload':str((path.parent/meta['file']).relative_to(D)),
            'sha256':meta['sha256'],'updates':meta['updates'],'sequence':meta['sequence'],'logical_name':meta['checkpoint_identity']['logical_name'],
            'lineage':meta.get('lineage'),'independent_precommit_verified':receipt['independent_precommit_verified']})
    durable(D/'CHECKPOINT_LINEAGE.json',{'original_input':c['input'],'original_sha256':c['input_sha256'],'entries':lineage,'components_mixed':False,'old_orphan_used':False})
    def jsonl(path):return read_jsonl(path) if path.exists() else []
    scientific_compute=jsonl(D/'trajectories/C/telemetry.jsonl')+jsonl(RUN/'execution/telemetry.jsonl')
    validation_compute=jsonl(D/'trajectories/R1/telemetry.jsonl')+jsonl(D/'trajectories/R2/telemetry.jsonl')
    compute=sum(x['seconds'] for x in scientific_compute);extra=sum(x['seconds'] for x in validation_compute)
    evaltime=sum(x['seconds'] for x in jsonl(RUN/'execution/evaluation_telemetry.jsonl'))
    save_events=[]
    for path in D.rglob('save_events.jsonl'):
        if 'test_fixtures' not in path.parts:save_events+=read_jsonl(path)
    checkpoint_seconds=sum(x['seconds'] for x in save_events)
    stages=budget['stages'];remaining=0 if complete else max(0,27935-p.get('updates',1000))
    resource={'utc':utc(),'stages':stages,'C_and_continuation_synchronized_update_seconds':compute,'R_validation_synchronized_update_seconds':extra,
        'C_and_continuation_new_updates':len(scientific_compute),'R_validation_updates':len(validation_compute),'full_dev_evaluation_wall_seconds':evaltime,
        'save_capture_verify_independent_reader_commit_seconds_all_training_branches':checkpoint_seconds,
        'peak_gpu_allocated_bytes':max((x['peak_gpu_allocated_bytes'] for x in scientific_compute+validation_compute),default=None),
        'peak_gpu_reserved_bytes':max((x['peak_gpu_reserved_bytes'] for x in scientific_compute+validation_compute),default=None),
        'updates_over_10_seconds':[x for x in scientific_compute+validation_compute if x['seconds']>10],
        'timing_scope':'synchronized update wall time, not pure GPU compute; checkpoint/reader/evaluation/startup stages separate; residual time not labeled compute',
        'remaining_updates_max_horizon':remaining,'projected_remaining_update_seconds':remaining*compute/len(scientific_compute) if scientific_compute else None,
        'preparation_and_validation_seconds':budget.get('preparation_validation_seconds'),'elapsed_at_report_seconds':time.time()-c['start_unix']}
    durable(D/'resource_results.json',resource)
    status={'utc':utc(),'historical_cause_status':'unresolved','observed_recovery_status':'verified_complete' if complete else budget['status'],
        'historical_fix_proven':False,'one_recovery_attempt':True,'roundtrip_passed':roundtrip.get('passed',False),'training_restart_equivalence_passed':equivalent.get('passed',False),
        'continuation_started':'continuation_C' in stages,'completed_updates':p.get('updates'),'completed_epochs':len(history),'selected_epoch':selected,
        'selected_checkpoint':lr_rows[1]['checkpoint'],'selected_checkpoint_sha256':selected_meta.get('sha256'),'selected_full_dev_f1':score,
        'lr_selected_by_frozen_rule':winner,'lr_selection_pending':not complete,'fresh_full_dev_exact_reproduction':fresh.get('all_agree',False),
        'scientific_metrics_reproduce':fresh.get('scientific_metrics_match',False),'all_retained_checkpoints_postwriter_verified':post.get('all_verified',False),
        'last_continuation_checkpoint':outcome.get('latest'),'preservation_files_checked':len(before),'preservation_changed_or_missing':changed,
        'shutdown_enabled':False,'final_test_locked':True,'server_task_accessed':False,'other_conditions_started':0,'failure':failure}
    durable(D/'FINAL_STATUS.json',status)
    table='\n'.join(f"| {n} | {v.get('status')} | {v.get('seconds',0):.3f} |" for n,v in stages.items())
    curve='\n'.join(f"| {x['epoch']} | {x['updates']} | {x['training_loss']:.6f} | {x['full_dev_f1']:.9f} | {x['precision']:.6f} | {x['recall']:.6f} | {x['panel200_f1_descriptive']:.9f} |" for x in curves)
    report=f'''# One gated recovery of XLM-R NER / seed 13 / LR 3e-5

**Observed outcome: {status['observed_recovery_status']}. Historical cause: unresolved.** No successful check is presented as proof of the historical cause. This single recovery is authorized independently of root-cause resolution; historical failures, old expected signatures and blocked reports remain intact.

## Gates and execution

| Stage | Recorded result | Wall seconds |
|---|---|---:|
{table}

Zero-update actual capture/save/readback and fresh GPU restore passed: **{roundtrip.get('passed',False)}**. Full C/R training-restart equivalence passed: **{equivalent.get('passed',False)}**. Automatic continuation started: **{status['continuation_started']}**. All unlisted later stages were not executed. Failure, if present: `{failure.get('error','none')}`; complete failure traceback and any per-stage failures remain in the milestone. No stage was retried or replaced.

The checked execution policy uses the original initializer before CPU deserialization, retains the CPU source through synchronized all-five-state restoration, and compares unchanged source model bytes. Actual/embedded/sidecar signatures are persisted before assertions. Validation captures model hash-input streams for independent rehash. It performs zero updates in the roundtrip processes and uses a separate process for eventual training.

Capture uses the unchanged original owned CPU snapshot function. Signatures and serialization come from that same snapshot. Every new payload passes writer readback and an independent fresh-process canonical/independent signature reader **before** atomic commit; live state is compared around saving. Generation metadata and commit receipts bind the payload. The unchanged historical verifier's structural rules are applied to the same checked read in the new reader; the old verifier source is untouched and exercised on a synthetic valid fixture. All new generations are retained for audit instead of deleting superseded rolling payloads; this changes storage retention only. `CHECKPOINT_LINEAGE.json` links every new commit back to its parent. No components are combined and the old update-3000 orphan is never promoted.

Six meaningful synthetic tests passed before real execution: four fingerprint/immutable-byte fixtures and two checkpoint tests covering owned mutation isolation, independent precommit readback, bad sidecar rejection, unchanged old-verifier compatibility, and interruption leaving the previous latest pointer intact. Historical hardening test evidence is preserved; no failed real state check was bypassed.

## Exact C/R comparison and canonical bookkeeping

C was predefined as updates1001–1100 in one process. R was predefined as1001–1050, save/exit, fresh restore, then1051–1100. Both save at1050 and1100 with identical checkpoint and rolling generation counters. C is the sole predefined continuation branch; scores never choose between branches.

Comparison requires exact fingerprints of model, optimizer, scheduler, RNG and **entire progress**, plus identical ordered batch IDs and complete scientific update traces including LR/loss/gradient norm. No progress field is excluded and no tolerance is used. Per-point actual signatures, differences and traces are retained under `validation/equivalence_1050` and `validation/equivalence_1100` if reached.

All inherited progress fields and the first1000 trace entries remain intact. Inherited cumulative timing/peak fields keep their historical values; new time/memory/process/branch telemetry is written outside canonical training state. New trace/history entries retain scientific values. This implements the requested metadata separation without changing optimization, objective, data order or selection; it does not manufacture equality by excluding fields at comparison time.

## Frozen scientific condition and development results

Only the existing3e-5 condition is continued. Same pinned XLM-R, seed13, 89,386 retained training sentences, 25 types/51 BIO labels,8×2 batches, summed FP32 cross-entropy divided by all nonignored first-subword labels per optimizer batch, BF16 forward, SDPA, AdamW/clip settings,27,935-step horizon and1,676 warmup. Original earliest-best selection and patience2/minimum3/min_delta0.0001 early stopping are reused. Full11,100-dev evaluation uses the existing scorer/BIO repair/failure accounting and visible-text views.

| Epoch | Updates | Training loss | Full-dev exact F1 | Precision | Recall | Panel200 F1, descriptive |
|---|---:|---:|---:|---:|---:|---:|
{curve or '| — | — | — | No completed development evaluation | — | — | — |'}

Selected checkpoint: `{lr_rows[1]['checkpoint']}`; selected full-dev F1: **{score}**. Fresh selected-checkpoint reproduction of tags, spans, confidences and failure accounting: **{fresh.get('all_agree',False)}**; scientific metrics: **{fresh.get('scientific_metrics_match',False)}**. Timing/memory telemetry is excluded from prediction equality; confidence values are exact. Per-type metrics, prediction distributions and full raw predictions remain with each completed epoch. Panel scores are derived solely from saved full-dev predictions and never used for selection.

Frozen LR comparison: verified completed1e-5 condition has best full-dev F1 **{baseline['full_dev_f1']:.9f}**, selected epoch{baseline['selected_epoch']}, checkpoint `{baseline['selected_checkpoint']}`. Recovered3e-5 best: **{score}**. Selected LR: **{winner if winner is not None else 'pending verified completion'}**. Rule: compare each condition's best full-dev exact-span/type micro-F1; an exact tie retains1e-5. Both conditions and all epoch results remain preserved. Scores remain exploratory development evidence, not final-test or paper-final results.

## Runtime, preservation and limits

Total budget4 hours: preparation/validation up to45 minutes, continuation up to180 minutes, final reporting reserve15 minutes. Elapsed through report preparation: **{(time.time()-c['start_unix'])/60:.2f} minutes**. C plus continuation synchronized update wall time: **{compute/60:.2f} min**; additional R validation updates: **{extra/60:.2f} min**; completed/partial full-dev evaluation wall time: **{evaltime/60:.2f} min**; training-branch checkpoint capture/readers/commit: **{checkpoint_seconds/60:.2f} min**. These instrumented wall measurements are not GPU kernel time. Parent stages separate initialization, validation/restoration, independent post-writer checks and fresh evaluation. Residual time is not assigned to compute.

Completed continuation updates: {p.get('updates')}; completed epochs: {len(history)}. Remaining maximum-horizon updates: {remaining}; projected remaining update-only seconds: {resource['projected_remaining_update_seconds']}. This projection excludes future evaluation/checkpoint/initialization overhead and is not an authorization to continue. If incomplete, the verified latest checkpoint is recorded in FINAL_STATUS.json; no automatic new budget exists.

Preservation: {len(before)} previously inventoried files checked; changed/missing: {changed}; protected source/input error: {protection}. Scope includes blocked LR run, diag01–diag04, frozen original dependencies/model/tokenizer/features and metadata; no OS-wide write claim is made. Every write is confined to the new milestone. Existing environments, datasets, payloads and reports were not changed. Update2000 and the old orphan were only included in preservation hashing, never deserialized. Server/Qwen task was not accessed. Shutdown remains disabled, final test locked, other conditions unstarted. Awake acquisition/verification/release is logged.

## Deliverables and next action

`COMMANDS.md`, `SOURCE_DIFF.patch`, `CONFIG.lock.json`, `SOURCE.lock.json`, `source/`, `CHECKPOINT_LINEAGE.json`, `validation/`, `resource_results.json`, `FINAL_STATUS.json` and the compact review bundle contain the implementation and evidence. Model weights/full raw streams and duplicate per-item prediction caches stay local and are excluded from the review ZIP.

{'This condition is complete and verified; do not repeat it. The next condition or final-test stage needs a separate instruction.' if complete else 'Stop here: preserve the saved partial/blocked state and review the specific gate/failure evidence. No retry, replacement recovery, diagnostic series or further condition is launched.'} Historical cause remains unresolved independently of this outcome.
'''
    (D/'REPORT.md').write_text(report,encoding='utf-8')
    commands='''# Exact execution commands

The milestone is single-use. Existing locks/evidence prevent repeated launch; these commands document execution, not a new budget or permission to rerun.

```powershell
Set-Location -LiteralPath 'C:\\Users\\Aru\\Desktop\\рисерч ниц ласт\\kazakh_nlp_starter'
$env:PYTHONDONTWRITEBYTECODE='1'
$env:TEMP=(Join-Path (Get-Location) 'encoder_lr_screen02_recovery01/runtime/tmp')
$env:TMP=$env:TEMP
.\\.venv\\Scripts\\python.exe -B -X utf8 -m encoder_lr_screen02_recovery01.prepare
.\\.venv\\Scripts\\python.exe -B -X utf8 -m encoder_lr_screen02_recovery01.launch
.\\.venv\\Scripts\\python.exe -B -X utf8 -m encoder_lr_screen02_recovery01.report
```

The frozen supervisor runs roundtrip capture, fresh roundtrip restore, independent stream gate, C1001–1100, R1001–1050, exact comparison1050, fresh R1051–1100, exact comparison1100. Only after all gates pass does it fresh-restore C1100 and continue with the frozen condition. Each new checkpoint invokes the independent reader before commit. Post-writer audit and selected full-dev fresh reproduction follow completion. Exact per-stage commands, deadlines, process IDs, return codes and durations are in BUDGET.json; unreached stages have no execution record. There are no failure retries.

Scientific metadata is frozen in CONFIG.lock.json. Worker environment is in common.env: offline pinned models,16 intra/inter-op threads,OMP/MKL/OpenBLAS16,CUBLAS_WORKSPACE_CONFIG=:4096:8,bytecode disabled,local temp/cache. Existing packages and drivers are not modified. No shutdown or server-task command exists.
'''
    (D/'COMMANDS.md').write_text(commands,encoding='utf-8')
    files={}
    for path in D.rglob('*'):
        if not path.is_file() or any(x in path.parts for x in ['items','runtime','__pycache__']):continue
        if path.name in ['ARTIFACT_MANIFEST.json','review_bundle.zip','BUNDLE_VERIFICATION.json']:continue
        files[path.relative_to(D).as_posix()]={'bytes':path.stat().st_size,'sha256':file_hash(path)}
    durable(D/'ARTIFACT_MANIFEST.json',{'utc':utc(),'files':files,'excluded':'duplicate prediction item caches, runtime caches/bytecode and self-referential package records; authoritative aggregate predictions/checkpoints/streams are hashed'})
    archive=D/'review_bundle.zip';members={}
    with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED) as z:
        for name in list(files)+['ARTIFACT_MANIFEST.json']:
            path=D/name
            if any(x in path.parts for x in ['test_fixtures','orders','items']):continue
            if path.suffix in ['.pt','.pending','.bin'] or path.name in ['predictions.jsonl','training_committed.jsonl','scientific_trace.jsonl','telemetry.jsonl','batch_order.jsonl']:continue
            z.write(path,name);members[name]=file_hash(path)
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        for n,h in members.items():assert hashlib.sha256(z.read(n)).hexdigest()==h
    durable(D/'BUNDLE_VERIFICATION.json',{'utc':utc(),'sha256':file_hash(archive),'bytes':archive.stat().st_size,'member_count':len(members),'crc_and_member_hashes_verified':True,
        'model_weights_included':False,'elapsed_total_seconds':time.time()-c['start_unix'],'within_four_hours':time.time()<=c['total_deadline_unix']})
    print(status,flush=True)

if __name__=='__main__':main()
