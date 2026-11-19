"""Separate fail-closed watcher for one explicitly authorized Windows shutdown."""
import argparse
import ctypes
from datetime import datetime,timezone
import os
from pathlib import Path
import subprocess
import time
import traceback
import psutil
from pilot.common import ROOT,read_json,digest,file_hash
from .common import D,verify
from .main import RUN
from .finalize import WATCH,durable_json,finalize,verify_manifest
from .shutdown_policy import terminal_kind,grace_ready

def utc():return datetime.now(timezone.utc).isoformat()

def process_record(pid):
    try:
        p=psutil.Process(int(pid));return {'pid':p.pid,'create_time':p.create_time()}
    except psutil.NoSuchProcess:return None

def alive(record):
    now=process_record(record['pid']);return now is not None and now['create_time']==record['create_time']

def get_outcome(budget):
    attempts=budget.get('attempts',[])
    if not attempts:return {}
    p=RUN/'attempts'/str(attempts[-1]['attempt'])/'outcome.json'
    return read_json(p) if p.exists() else {}

def arm():
    assert os.name=='nt','This watcher is for the authorized Windows host only'
    WATCH.mkdir(parents=True,exist_ok=True)
    assert not (WATCH/'binding.json').exists(),'Existing watcher binding is immutable; do not rearm automatically'
    budget=read_json(D/'main_budget.json');assert budget['status']=='running'
    assert (D/'RUNNING.lock').exists() and (RUN/'RUNNING.lock').exists()
    processes=[]
    for pid in [int((D/'RUNNING.lock').read_text()),int((RUN/'RUNNING.lock').read_text())]+[a['pid'] for a in budget['attempts'] if a.get('status')=='started']:
        r=process_record(pid)
        if r and r not in processes:processes.append(r)
    assert len(processes)>=2
    source=['completion_watcher.py','shutdown_policy.py','finalize.py','postprocess.py','plot_curves.cs','plot_curves.exe']
    binding={'authorization':'User explicitly authorized normal shutdown for this run after verified completion or graceful ceiling stop; supersedes earlier prohibition.',
        'armed_utc':utc(),'run_path':str(RUN.resolve()),'run_hash':digest(read_json(RUN/'identity.json')),
        'main_plan_sha256':verify('main_plan.lock.json'),'budget_start_unix':budget['start_unix'],'budget_deadline_unix':budget['deadline_unix'],
        'main_config_sha256':file_hash(D/'main_config.json'),'processes':processes,'grace_seconds':60,
        'shutdown_argv':['shutdown.exe','/s','/t','0'],'force_apps':False,'cancel_path':str((WATCH/'CANCEL').resolve()),
        'finalizer_source_sha256':{name:file_hash(D/name) for name in source}}
    durable_json(WATCH/'binding.json',binding)
    durable_json(WATCH/'status.json',{'state':'prepared_not_running','utc':utc(),'run_hash':binding['run_hash']})
    print({'prepared':True,'run_hash':binding['run_hash'],'cancel_path':binding['cancel_path']})

def binding_valid(binding):
    assert str(RUN.resolve())==binding['run_path']
    assert verify('main_plan.lock.json')==binding['main_plan_sha256']
    assert file_hash(D/'main_config.json')==binding['main_config_sha256']
    assert digest(read_json(RUN/'identity.json'))==binding['run_hash']
    for name,sha in binding['finalizer_source_sha256'].items():assert file_hash(D/name)==sha,'Finalizer source changed: '+name
    budget=read_json(D/'main_budget.json')
    assert budget['start_unix']==binding['budget_start_unix'] and budget['deadline_unix']==binding['budget_deadline_unix']
    return budget

def state(name,binding,**values):
    value={'state':name,'utc':utc(),'run_hash':binding['run_hash'],'watcher_pid':os.getpid(),'final_test_locked':True,**values}
    durable_json(WATCH/'status.json',value);durable_json(D/'FINAL_STATUS.json',value)
    print(value,flush=True)

def run():
    binding=read_json(WATCH/'binding.json');binding_valid(binding)
    lock=WATCH/'WATCHER.lock'
    with lock.open('x',encoding='utf-8') as f:f.write(str(os.getpid()))
    records=list(binding['processes']);receipt=None;cancel_logged=False
    if os.name=='nt':ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    try:
        state('armed_waiting_for_explicit_terminal_outcome',binding,processes=records,cancel_path=binding['cancel_path'])
        while True:
            if (WATCH/'CANCEL').exists() and not cancel_logged:
                state('shutdown_cancelled_waiting_for_run_finalization',binding,note='Training and artifact finalization continue; shutdown is disabled.')
                cancel_logged=True
            budget=binding_valid(binding)
            for pid in [a['pid'] for a in budget['attempts'] if a.get('status')=='started']:
                record=process_record(pid)
                if record and record not in records:records.append(record)
            child_lock=RUN/'RUNNING.lock'
            if child_lock.exists():
                record=process_record(int(child_lock.read_text()))
                if record and record not in records:records.append(record)
            kind=terminal_kind(budget,get_outcome(budget),(RUN/'completed.json').exists())
            running=[r for r in records if alive(r)]
            if kind is not None:
                if not running and not (D/'RUNNING.lock').exists() and not child_lock.exists():break
                if time.time()>binding['budget_deadline_unix']+300:raise RuntimeError('Terminal records exist but processes/locks remain; no shutdown')
            elif not running:raise RuntimeError('Processes disappeared without explicit safe completion; no shutdown')
            durable_json(WATCH/'heartbeat.json',{'utc':utc(),'run_status':budget['status'],'alive_processes':running,'run_hash':binding['run_hash']})
            time.sleep(5)
        state('finalizing_artifacts_no_shutdown_yet',binding,terminal_status=kind)
        final_lock=D/'FINALIZING.lock'
        with final_lock.open('x',encoding='utf-8') as f:f.write(str(os.getpid()))
        try:receipt=finalize(binding)
        finally:final_lock.unlink()
        durable_json(WATCH/'finalization_receipt.json',receipt)
        if (WATCH/'CANCEL').exists():raise InterruptedError('Cancellation requested after finalization')
        binding_valid(binding)
        assert not any(alive(r) for r in records) and not (D/'RUNNING.lock').exists() and not (RUN/'RUNNING.lock').exists()
        began=time.monotonic();grace_end=time.time()+60
        state('verified_grace_period',binding,grace_ends_unix=grace_end,grace_seconds=60,finalization=receipt,cancel_path=binding['cancel_path'])
        while not grace_ready(time.monotonic()-began,(WATCH/'CANCEL').exists(),verified=True):
            if (WATCH/'CANCEL').exists():raise InterruptedError('Cancelled during 60-second grace period')
            time.sleep(.5)
        if (WATCH/'CANCEL').exists():raise InterruptedError('Cancelled before shutdown request')
        binding_valid(binding)
        assert file_hash(D/'ARTIFACT_MANIFEST.json')==receipt['manifest']['sha256']
        assert file_hash(D/'REPORT.md')==receipt['report_sha256'] and file_hash(D/'TOMORROW.md')==receipt['tomorrow_sha256']
        assert not any(alive(r) for r in records) and not (D/'RUNNING.lock').exists() and not (RUN/'RUNNING.lock').exists()
        # Persist this exact terminal status before requesting normal immediate shutdown.
        state('requesting_normal_windows_shutdown',binding,command=binding['shutdown_argv'],finalization=receipt,grace_seconds_completed=60)
        if (WATCH/'CANCEL').exists():raise InterruptedError('Cancelled at final pre-request check')
        exe=Path(os.environ.get('SystemRoot','C:\\Windows'))/'System32/shutdown.exe'
        result=subprocess.run([str(exe),'/s','/t','0'],capture_output=True,text=True,timeout=15)
        if result.returncode!=0:raise RuntimeError('Windows shutdown request failed: '+str(result.returncode)+' '+result.stderr)
        state('normal_shutdown_request_accepted',binding,command=binding['shutdown_argv'],finalization=receipt,
            note='Normal Windows shutdown requested; applications may delay it. No forced closure or positive Windows timeout was used.')
    except InterruptedError as error:
        state('cancelled_computer_left_running',binding,reason=str(error),finalization=receipt)
    except BaseException as error:
        failure={'utc':utc(),'type':type(error).__name__,'message':str(error),'traceback':traceback.format_exc(),'run_hash':binding['run_hash']}
        durable_json(WATCH/f'error_{time.time_ns()}.json',failure)
        state('blocked_computer_left_running',binding,error=failure,finalization=receipt)
    finally:
        if os.name=='nt':ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
        lock.unlink(missing_ok=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--arm',action='store_true');p.add_argument('--status',action='store_true');p.add_argument('--verify-manifest',action='store_true');a=p.parse_args()
    if a.arm:arm()
    elif a.status:
        print(read_json(WATCH/'status.json'))
        if (WATCH/'heartbeat.json').exists():print(read_json(WATCH/'heartbeat.json'))
    elif a.verify_manifest:print(verify_manifest(D/'ARTIFACT_MANIFEST.json'))
    else:run()
