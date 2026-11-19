"""CPU-only finalization. Never trains, generates predictions or shuts down Windows."""
import json
import math
import os
from pathlib import Path
import subprocess
from datetime import datetime,timezone
from pilot.common import ROOT,read_json,write_json,write_jsonl,file_hash,digest
from stage_preflight.state import fingerprint
from .common import D,verify,preserve
from .main import RUN
from .control import status
from .shutdown_policy import terminal_kind

WATCH=D/'shutdown_watch'

def durable_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_suffix(path.suffix+'.tmp')
    with temp.open('w',encoding='utf-8',newline='\n') as f:
        json.dump(value,f,ensure_ascii=False,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
    os.replace(temp,path)

def check_checkpoint(metadata,run_hash,expected_progress=None):
    import torch
    torch.set_num_threads(min(4,max(1,os.cpu_count() or 1)))
    meta=read_json(metadata);path=Path(metadata).parent/meta['file']
    assert meta['run_hash']==run_hash and file_hash(path)==meta['sha256'],'Checkpoint hash/identity mismatch'
    state=torch.load(path,map_location='cpu',weights_only=False)
    assert state['run_hash']==run_hash
    p=state['progress'];assert p['updates']==meta['updates'] and p['checkpoint_sequence']==meta['sequence']
    assert p['updates']==p['epoch']*5587+math.ceil(p['cursor']/16)
    assert 0<=p['epoch']<=5 and 0<=p['cursor']<=89386 and p['phase'] in ['train','evaluate','complete']
    assert len(p['trace'])==p['updates'] and [r['update'] for r in p['trace']]==list(range(1,p['updates']+1))
    assert state['scheduler']['last_epoch']==p['updates']
    params=[i for group in state['optimizer']['param_groups'] for i in group['params']]
    assert set(params)==set(state['optimizer']['state'])
    assert all(int(x['step'])==p['updates'] for x in state['optimizer']['state'].values())
    assert set(state['rng'])=={'random','numpy','torch','cuda'} and state['rng']['cuda']
    assert all(torch.is_tensor(x) for x in state['rng']['cuda'])
    checked=0
    def finite(x):
        nonlocal checked
        if torch.is_tensor(x) and (x.is_floating_point() or x.is_complex()):
            # Chunked checks keep temporary allocations small, entirely on CPU.
            for part in x.reshape(-1).split(1000000):assert bool(torch.isfinite(part).all()),'Nonfinite checkpoint tensor'
            checked+=x.numel()
        elif isinstance(x,dict):
            for v in x.values():finite(v)
        elif isinstance(x,(list,tuple)):
            for v in x:finite(v)
    finite(state['model']);finite(state['optimizer'])
    expected=state.get('state_digests')
    if expected:
        actual={k:fingerprint(state[k]) for k in ['model','optimizer','scheduler','rng','progress']}
        assert actual==expected,'Deserialized state signatures differ'
    if expected_progress is not None:assert status(p)==expected_progress,'Terminal outcome and committed progress differ'
    result={'checkpoint':str(path.relative_to(D)),'sha256':meta['sha256'],'updates':p['updates'],'epoch':p['epoch'],
        'cursor':p['cursor'],'phase':p['phase'],'finite_floating_elements_checked':checked,'state_signatures_checked':bool(expected),
        'optimizer_scheduler_rng_progress_valid':True}
    return result,p

def verify_manifest(path):
    m=read_json(path)
    for name,entry in m['files'].items():
        target=D/name
        assert target.resolve().is_relative_to(D.resolve())
        assert target.stat().st_size==entry['bytes'] and file_hash(target)==entry['sha256'],name
    return {'files':len(m['files']),'sha256':file_hash(path),'bytes':sum(v['bytes'] for v in m['files'].values())}

def finalize(binding):
    verify('main_plan.lock.json');verify('benchmark_plan.lock.json')
    assert digest(read_json(RUN/'identity.json'))==binding['run_hash']
    budget=read_json(D/'main_budget.json');last=budget['attempts'][-1]
    outcome=read_json(RUN/'attempts'/str(last['attempt'])/'outcome.json')
    kind=terminal_kind(budget,outcome,(RUN/'completed.json').exists());assert kind is not None
    assert not (D/'RUNNING.lock').exists() and not (RUN/'RUNNING.lock').exists()
    metas=[p for p in (RUN/'checkpoints').glob('*.json') if p.name!='latest.json']
    newest=max(metas,key=lambda p:read_json(p)['sequence'])
    checkpoint,progress=check_checkpoint(newest,binding['run_hash'],outcome['progress'])
    selected=RUN/'checkpoints'/f'epoch_{progress["best_epoch"]:02d}.json' if progress['best_epoch'] is not None else None
    selected_check=None
    if selected and selected!=newest:selected_check,_=check_checkpoint(selected,binding['run_hash'])
    elif selected:selected_check=checkpoint
    if kind=='incomplete_budget':
        assert progress['phase'] in ['train','evaluate']
        write_jsonl(RUN/'training_committed.jsonl',progress['trace'])
        write_json(RUN/'incomplete.json',{'run_hash':binding['run_hash'],'status':'incomplete_eight_hour_ceiling',
            'resumable_checkpoint':checkpoint,**status(progress)})
        if selected:
            import shutil
            folder=RUN/'evaluations'/f'epoch_{progress["best_epoch"]:02d}'
            for name in ['predictions.jsonl','metrics.json','distributions.json']:
                target=RUN/name
                if target.exists():assert file_hash(target)==file_hash(folder/name)
                else:shutil.copy2(folder/name,target)
    from .postprocess import report
    report()
    if progress['history']:
        command=[str(D/'plot_curves.exe'),str(D)]
        rendered=subprocess.run(command,cwd=ROOT,capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=120)
        durable_json(WATCH/'plot_result.json',{'returncode':rendered.returncode,'stdout':rendered.stdout,'stderr':rendered.stderr})
        assert rendered.returncode==0 and (D/'learning_curves.png').stat().st_size>1000,'Learning-curve rendering failed'
    preserve(check=True)
    assert not read_json(D/'audit/preservation_verified.json')['changed_or_missing']
    assert read_json(D/'audit/main_verified.json')['run_hash']==binding['run_hash']
    completion={'run_hash':binding['run_hash'],'status':kind,'completed_epochs':len(progress['history']),
        'selected_checkpoint':selected_check,'resumable_checkpoint':checkpoint,'updates':progress['updates'],
        'next_update':progress['updates']+1 if kind!='complete' else None,'final_test_locked':True,
        'verified_utc':datetime.now(timezone.utc).isoformat()}
    durable_json(D/'audit/finalization_verified.json',completion)
    status_text='COMPLETE' if kind=='complete' else 'INCOMPLETE — graceful eight-hour ceiling; safe recovery verified'
    selected_text=selected_check['checkpoint'] if selected_check else 'None: no full epoch evaluation completed.'
    lines=['# Tomorrow: exact saved experiment status','',f'**{status_text}**',
        f'- Run identity: `{binding["run_hash"]}`',f'- Condition: XLM-R base / NER / seed 13 / LR 1e-5.',
        f'- Completed epochs: {len(progress["history"])}; committed optimizer updates: {progress["updates"]:,}.',
        f'- Selected earliest-best checkpoint: `{selected_text}`.',f'- Latest verified recovery: `{checkpoint["checkpoint"]}`.',
        f'- Recovery SHA-256: `{checkpoint["sha256"]}`.',
        f'- Exact progress: phase `{progress["phase"]}`, zero-based epoch index {progress["epoch"]}, next feature cursor {progress["cursor"]}.',
        '- Final test inference remains locked. No other encoder condition or experiment was started.','',
        '## Continue tomorrow','',
        'Open `REPORT.md`, `METHOD_DECISIONS.md`, `resource_results.json`, and `FINAL_STATUS.json` in this milestone. `FINAL_STATUS.json` records whether shutdown was requested, cancelled, or blocked; normal Windows shutdown may be delayed by an application, because forced closure is not used.','',
        '```powershell',"Set-Location -LiteralPath 'C:\\Users\\Aru\\Desktop\\рисерч ниц ласт\\kazakh_nlp_starter'",
        '.venv\\Scripts\\python.exe -X utf8 -m encoder_main01.completion_watcher --status',
        '.venv\\Scripts\\python.exe -X utf8 -m encoder_main01.completion_watcher --verify-manifest','```','']
    if kind=='complete':
        lines+=['This condition is complete: do not restart it. The selected checkpoint and every epoch checkpoint are preserved. The next proposed work is the remaining three seed-13 NER LR-screen conditions, followed later by SA and reseeds, as described in the report. These are unstarted and require a later instruction; the watcher cannot launch them.']
    else:
        lines += [f'Resume would start at optimizer update {progress["updates"]+1}, preserving the original five-epoch scheduler horizon and saved optimizer/RNG state. The original eight-hour allowance has ended: do not delete the budget ledger, alter its deadline or restart from pretrained weights.',
            'Before continuation, explicitly authorize a new bounded continuation budget. Create a separate continuation ledger/source snapshot bound to the verified recovery hash above; keep the original incomplete run and all frozen settings. The current launcher deliberately refuses an expired/finished budget. No continuation is authorized or launched by this watcher.']
    lines+=['','Sherkala remains a separate HTTP-401 access blocker. Its earlier record is unchanged. Qwen prompts and the NER interface remain frozen.','',
        'All development scores remain exploratory; this single condition is not the paper’s final experiment.']
    (D/'TOMORROW.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    report_path=D/'REPORT.md'
    with report_path.open('a',encoding='utf-8') as f:
        f.write('\n## Separately authorized shutdown watcher\n\nThe user subsequently authorized normal Windows shutdown after verified finalization. This supersedes the earlier no-shutdown instruction for this run only; the frozen training source/configuration was not edited. The separate watcher checks explicit terminal outcomes, checkpoint integrity, reports, preservation and an artifact manifest before a cancellable 60-second grace period. It invokes only `shutdown.exe /s /t 0`, without `/f`. See `TOMORROW.md`, `FINAL_STATUS.json` and `shutdown_watch/` for the saved outcome.\n')
        f.flush();os.fsync(f.fileno())
    files={}
    for folder,dirs,names in os.walk(D):
        dirs[:]=sorted(x for x in dirs if x not in ['__pycache__','tmp','shutdown_watch'])
        for name in sorted(names):
            p=Path(folder)/name
            if name in ['ARTIFACT_MANIFEST.json','FINAL_STATUS.json','FINALIZING.lock'] or name.endswith(('.tmp','.pyc')):continue
            files[p.relative_to(D).as_posix()]={'bytes':p.stat().st_size,'sha256':file_hash(p)}
    # Explicit immutable watcher evidence; lifecycle status/logs remain outside this manifest.
    volatile={'status.json','heartbeat.json','stdout.log','stderr.log','WATCHER.lock','CANCEL','finalization_receipt.json'}
    for p in sorted(WATCH.rglob('*')):
        if p.is_file() and p.name not in volatile and not p.name.endswith('.tmp'):
            files[p.relative_to(D).as_posix()]={'bytes':p.stat().st_size,'sha256':file_hash(p)}
    manifest={'run_hash':binding['run_hash'],'status':kind,'created_utc':datetime.now(timezone.utc).isoformat(),'files':files,
        'excluded':'manifest itself; mutable FINAL_STATUS/watcher lifecycle logs and locks; temporary/cache files; previous milestones covered by preservation.json and its verified audit'}
    durable_json(D/'ARTIFACT_MANIFEST.json',manifest)
    verification=verify_manifest(D/'ARTIFACT_MANIFEST.json')
    return {**completion,'manifest':verification,'report_sha256':file_hash(report_path),'tomorrow_sha256':file_hash(D/'TOMORROW.md')}
