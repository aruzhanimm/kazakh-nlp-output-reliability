import os
import subprocess
import sys
import time
from datetime import datetime,timezone
from pilot.common import ROOT,read_json,write_json,RunLock
from .common import D,verify
from .benchmark import SCHEDULE,choose

def main():
    verify('benchmark_plan.lock.json');ledger_path=D/'benchmark_budget.json'
    if ledger_path.exists():raise ValueError('Preserve completed/partial benchmark; no automatic replay')
    ledger={'limit_seconds':1200,'attempts':[],'spent_seconds':0};write_json(ledger_path,ledger)
    with RunLock(D):
        for micro,repeat in SCHEDULE:
            remaining=1200-ledger['spent_seconds']
            if remaining<=0:raise RuntimeError('Benchmark budget exhausted; main blocked')
            path=D/'benchmark'/f'micro_{micro}_repeat_{repeat}';path.mkdir(parents=True,exist_ok=True)
            attempt={'microbatch':micro,'repeat':repeat,'utc':datetime.now(timezone.utc).isoformat(),'status':'started'};ledger['attempts'].append(attempt);write_json(ledger_path,ledger)
            env=dict(os.environ,HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONUTF8='1',PYTHONUNBUFFERED='1');started=time.perf_counter()
            with (path/'process.log').open('x',encoding='utf-8') as log:
                proc=subprocess.Popen([sys.executable,'-X','utf8','-m','encoder_main01.benchmark','--microbatch',str(micro),'--repeat',str(repeat)],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
                attempt['pid']=proc.pid;write_json(ledger_path,ledger)
                try:code=proc.wait(timeout=remaining)
                except subprocess.TimeoutExpired:proc.kill();proc.wait();code='budget_exhausted'
            attempt.update(status='complete' if code==0 else 'failed',returncode=code,seconds=time.perf_counter()-started);ledger['spent_seconds']+=attempt['seconds'];write_json(ledger_path,ledger);print(attempt,flush=True)
            if code!=0 and not (path/'result.json').exists():
                write_json(path/'result.json',{'status':'failed','microbatch':micro,'repeat':repeat,'returncode':code,'error':'Process terminated without complete result'})
            if code=='budget_exhausted':raise RuntimeError('Benchmark time budget exhausted')
        choose()

if __name__=='__main__':main()
