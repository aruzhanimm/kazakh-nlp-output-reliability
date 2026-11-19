"""One condition, one cumulative eight-hour ceiling, one planned recovery restart."""
import ctypes
import os
import subprocess
import sys
import time
from datetime import datetime,timezone
from pilot.common import ROOT,read_json,write_json,RunLock
from .common import D,verify
from .main import RUN

def main():
    verify('main_plan.lock.json');ledger_path=D/'main_budget.json'
    if ledger_path.exists():
        ledger=read_json(ledger_path)
        if ledger.get('status') in ['complete','incomplete_budget','blocked']:raise RuntimeError('Preserve finished/blocked budget; no automatic extension or replay')
    else:
        now=time.time();ledger={'start_unix':now,'deadline_unix':now+28800,'limit_seconds':28800,'attempts':[],'status':'running'};write_json(ledger_path,ledger)
    with RunLock(D):
        # Temporary system-awake request, released at exit. No shutdown or persistent power-setting changes.
        if os.name=='nt':ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
        try:
            while time.time()<ledger['deadline_unix']-60:
                attempt=len(ledger['attempts'])+1;path=RUN/'process_logs';path.mkdir(parents=True,exist_ok=True)
                record={'attempt':attempt,'utc':datetime.now(timezone.utc).isoformat(),'status':'started'};ledger['attempts'].append(record);write_json(ledger_path,ledger)
                env=dict(os.environ,HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONUTF8='1',PYTHONUNBUFFERED='1')
                started=time.perf_counter()
                with (path/f'attempt_{attempt:02d}.log').open('x',encoding='utf-8') as log:
                    proc=subprocess.Popen([sys.executable,'-X','utf8','-m','encoder_main01.main','--deadline',str(ledger['deadline_unix']),'--attempt',str(attempt)],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
                    record['pid']=proc.pid;write_json(ledger_path,ledger)
                    try:code=proc.wait(timeout=max(1,ledger['deadline_unix']-time.time()))
                    except subprocess.TimeoutExpired:
                        proc.kill();proc.wait();code='hard_ceiling_process_unresponsive'
                outcome_path=RUN/'attempts'/str(attempt)/'outcome.json'
                outcome=read_json(outcome_path) if outcome_path.exists() else {'status':'blocked','error':'Process ended without safe outcome'}
                record.update(seconds=time.perf_counter()-started,returncode=code,status=outcome['status']);write_json(ledger_path,ledger);print(record,flush=True)
                if code==0 and outcome['status']=='planned_restart_ready':continue
                ledger['status']=outcome['status'];break
            else:ledger['status']='incomplete_budget'
        finally:
            ledger['elapsed_wall_seconds']=time.time()-ledger['start_unix'];ledger['ended_utc']=datetime.now(timezone.utc).isoformat();write_json(ledger_path,ledger)
            if os.name=='nt':ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)

if __name__=='__main__':main()
