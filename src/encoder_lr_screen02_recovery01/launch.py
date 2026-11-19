"""One gate-bound recovery invocation; stages are never retried or replaced."""
import os,sys,time,subprocess,traceback
from pilot.common import ROOT,read_json,write_json
from encoder_cont01.awake import Awake
from .common import D,RUN,utc,durable,config,verify_plan,env

def stop_owned(p):
    import psutil
    if p.poll() is not None:return
    owned=psutil.Process(p.pid)
    for child in reversed(owned.children(recursive=True)):
        try:child.terminate()
        except psutil.NoSuchProcess:pass
    p.terminate();p.wait(timeout=20)

def main():
    c=config();verify_plan();assert read_json(D/'PREPARATION_GATE.json')['passed']
    durable(D/'ATTEMPT_STARTED.json',{'utc':utc(),'pid':os.getpid(),'one_recovery_attempt':True})
    b={'start_unix':c['start_unix'],'total_deadline_unix':c['total_deadline_unix'],'preparation_deadline_unix':c['preparation_deadline_unix'],
        'status':'validation_running','stages':{},'historical_cause':'unresolved','shutdown_enabled':False};write_json(D/'BUDGET.json',b);process=None
    try:
        with Awake(D/'awake.json') as awake:
            def stage(name,args,deadline):
                nonlocal process
                assert name not in b['stages'],'No repeated stage'
                assert time.time()<deadline-30,'Stage allowance exhausted'
                tick=time.perf_counter();entry={'started_utc':utc(),'status':'running','command':[sys.executable,'-B','-X','utf8','-m',*args]};b['stages'][name]=entry;write_json(D/'BUDGET.json',b)
                with (D/(name+'.log')).open('x',encoding='utf-8') as f:
                    process=subprocess.Popen(entry['command'],cwd=ROOT,env=env(),stdout=f,stderr=subprocess.STDOUT);entry['launcher_child_pid']=process.pid;write_json(D/'BUDGET.json',b)
                    while True:
                        try:code=process.wait(timeout=30);break
                        except subprocess.TimeoutExpired:
                            awake.verify('held_'+name)
                            if time.time()>=deadline:
                                stop_owned(process);entry.update(status='time_limit',seconds=time.perf_counter()-tick);raise TimeoutError(name+' deadline')
                entry.update(returncode=code,seconds=time.perf_counter()-tick,ended_utc=utc(),status='passed' if code==0 else 'failed');write_json(D/'BUDGET.json',b);process=None
                assert code==0,'Required stage failed; no retry: '+name
                awake.verify('after_'+name)
            deadline=c['preparation_deadline_unix']
            stage('roundtrip_capture',['encoder_lr_screen02_recovery01.roundtrip','capture'],deadline)
            stage('roundtrip_restore',['encoder_lr_screen02_recovery01.roundtrip','restore'],deadline)
            stage('roundtrip_gate',['encoder_lr_screen02_recovery01.gates','roundtrip'],deadline)
            stage('trajectory_C',['encoder_lr_screen02_recovery01.train','--mode','C','--deadline',str(deadline)],deadline)
            assert read_json(D/'trajectories/C/RESULT.json')['status']=='trajectory_complete'
            stage('trajectory_R1',['encoder_lr_screen02_recovery01.train','--mode','R1','--deadline',str(deadline)],deadline)
            assert read_json(D/'trajectories/R1/RESULT.json')['status']=='trajectory_complete'
            stage('equivalence_1050',['encoder_lr_screen02_recovery01.gates','1050'],deadline)
            stage('trajectory_R2',['encoder_lr_screen02_recovery01.train','--mode','R2','--deadline',str(deadline)],deadline)
            assert read_json(D/'trajectories/R2/RESULT.json')['status']=='trajectory_complete'
            stage('equivalence_1100',['encoder_lr_screen02_recovery01.gates','1100'],deadline)
            assert read_json(D/'validation/EQUIVALENCE.json')['passed'];verify_plan()
            continuation_start=time.time();end=min(continuation_start+c['continuation_limit_seconds'],c['total_deadline_unix']-c['report_reserve_seconds'])
            writer_end=end-c['postwriter_reserve_seconds']
            b.update(status='continuing_C',preparation_validation_seconds=continuation_start-c['start_unix'],continuation_start_unix=continuation_start,
                continuation_deadline_unix=end,writer_deadline_unix=writer_end);write_json(D/'BUDGET.json',b)
            stage('continuation_C',['encoder_lr_screen02_recovery01.train','--mode','continue','--deadline',str(writer_end)],writer_end)
            outcome=read_json(RUN/'execution/RESULT.json');assert outcome['status'] in ['complete','incomplete_budget']
            stage('postwrite',['encoder_lr_screen02_recovery01.postwrite'],end)
            if outcome['status']=='complete':
                stage('fresh_reload',['encoder_lr_screen02_recovery01.fresh_reload','--deadline',str(end)],end)
                b['status']='verified_complete'
            else:b['status']='verified_incomplete_budget'
    except BaseException as e:
        if process is not None:stop_owned(process)
        b['status']='blocked';durable(D/'FAILURE.json',{'utc':utc(),'type':type(e).__name__,'error':str(e),'traceback':traceback.format_exc(),'no_retry':True})
    finally:
        b.update(execution_ended_utc=utc(),elapsed_through_execution_seconds=time.time()-c['start_unix']);write_json(D/'BUDGET.json',b)
        print({'status':b['status'],'stages':{k:v['status'] for k,v in b['stages'].items()}},flush=True)

if __name__=='__main__':main()
