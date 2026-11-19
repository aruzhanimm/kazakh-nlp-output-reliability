"""CPU-only simulated owner failures; no real child/model/GPU commands."""
import subprocess,time
from unittest.mock import Mock,patch
from . import queue
from .common import Q,utc,durable

def main():
    evidence=[]
    for scenario in ['normal_exit','awake_failure']:
        d=Q/'preparation'/'controller_checks'/scenario;d.mkdir(parents=True,exist_ok=False)
        entry={'id':scenario};p=Mock(pid=999999,returncode=0);awake=Mock()
        if scenario=='normal_exit':p.wait.return_value=0
        else:
            p.wait.side_effect=subprocess.TimeoutExpired('simulated',30)
            awake.verify.side_effect=RuntimeError('simulated awake failure')
        with patch.object(queue.subprocess,'Popen',return_value=p),patch.object(queue.psutil,'Process',return_value=Mock(create_time=lambda:1.)),patch.object(queue,'env',return_value={}),patch.object(queue,'save'),patch.object(queue,'stop_child') as stop:
            error=None
            try:queue.stage({},entry,scenario,'DO_NOT_EXECUTE',[],d,time.time()+60,awake)
            except RuntimeError as exc:error=str(exc)
            if scenario=='normal_exit':assert entry['stages'][scenario]['status']=='passed' and not stop.called
            else:assert error=='simulated awake failure' and entry['stages'][scenario]['status']=='owner_failure' and stop.call_count==1
        evidence.append({'scenario':scenario,'passed':True,'record':entry['stages'][scenario],'actual_processes_launched':0})
    durable(Q/'ORCHESTRATION_CHECK.json',{'utc':utc(),'passed':True,'checks':evidence,'scientific_code_changed':False})

if __name__=='__main__':main()
