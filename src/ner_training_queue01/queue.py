"""Durable single-GPU owner; no automatic replacement following a failed gate."""
import os,sys,time,subprocess,traceback,shutil,statistics,msvcrt
from pathlib import Path
import psutil
from pilot.common import ROOT,read_json,write_json,file_hash
from encoder_cont01.awake import Awake
from .common import Q,utc,durable,append,env,verify_plan
from .prepare import make_config
START=1794216500.0  # 2026-11-09 09:28:20 UTC, conservative authorization observation.
REC=ROOT/'encoder_lr_screen02_recovery01'
REC_PID=8596;REC_CTIME=1794214116.1742678

def alive(pid,created):
    try:p=psutil.Process(pid);return abs(p.create_time()-created)<.01 and p.is_running()
    except psutil.NoSuchProcess:return False

def save(q,event=None):
    write_json(Q/'QUEUE.json',q)
    if event:append(Q/'queue.jsonl',{'utc':utc(),**event});print(event,flush=True)

def stop_child(p):
    if p.poll() is not None:return
    parent=psutil.Process(p.pid)
    for c in reversed(parent.children(recursive=True)):
        try:c.terminate()
        except psutil.NoSuchProcess:pass
    p.terminate();p.wait(timeout=20)

def stage(q,entry,name,module,args,directory,deadline,awake):
    stages=entry.setdefault('stages',{})
    if name in stages:
        s=stages[name]
        if s['status']=='passed':return
        # Do not infer success from disappearance or silently retry a failed stage.
        if s['status']=='running' and s.get('pid') and alive(s['pid'],s['process_create_time']):
            while alive(s['pid'],s['process_create_time']):
                if time.time()>=deadline:raise TimeoutError('Existing worker exceeded unchanged budget: '+name)
                awake.verify('reattached_'+name);time.sleep(20)
            raise RuntimeError('Controller interruption: worker ended without collected exit status; accepted checkpoint may be resumed only after explicit artifact audit')
        raise RuntimeError('Existing unfinished/failed stage is not automatically repeated: '+name)
    assert time.time()<deadline-30,'Stage budget exhausted'
    command=[sys.executable,'-B','-X','utf8','-m',module,*args]
    s={'status':'starting','command':command,'started_unix':time.time(),'started_utc':utc(),'deadline_unix':deadline,'condition_directory':str(directory)};stages[name]=s;save(q)
    log=Path(directory)/'logs'/f'{name}.log';log.parent.mkdir(parents=True,exist_ok=True)
    with log.open('x',encoding='utf-8') as f:
        p=subprocess.Popen(command,cwd=ROOT,env=env(directory),stdout=f,stderr=subprocess.STDOUT)
        s.update(status='running',pid=p.pid,process_create_time=psutil.Process(p.pid).create_time(),log=str(log));save(q,{'event':'stage_started','condition':entry['id'],'stage':name,'pid':p.pid})
        while True:
            try:code=p.wait(timeout=30);break
            except subprocess.TimeoutExpired:
                awake.verify(entry['id']+'_'+name)
                write_json(Q/'HEARTBEAT.json',{'utc':utc(),'condition':entry['id'],'stage':name,'pid':p.pid,'deadline':deadline})
                if time.time()>=deadline:
                    stop_child(p);s.update(status='time_limit',ended_utc=utc(),seconds=time.time()-s['started_unix']);save(q);raise TimeoutError(name)
    s.update(status='passed' if code==0 else 'failed',returncode=code,ended_utc=utc(),seconds=time.time()-s['started_unix']);save(q,{'event':'stage_ended','condition':entry['id'],'stage':name,'status':s['status']})
    assert code==0,'Required stage failed; no replacement: '+name

def existing(q,awake):
    entry=q['conditions'][1]
    if entry['status']=='verified_complete':return
    entry.update(status='attached_existing',pid=REC_PID,process_create_time=REC_CTIME);save(q)
    while alive(REC_PID,REC_CTIME):
        b=read_json(REC/'BUDGET.json');entry['existing_budget_status']=b['status'];save(q)
        awake.verify('observing_original_recovery_without_changes');time.sleep(30)
    b=read_json(REC/'BUDGET.json');assert b['status'] in ['verified_complete','verified_incomplete_budget','blocked'],'Process exit without explicit finalization'
    if not (REC/'FINAL_STATUS.json').exists():
        stage(q,entry,'original_report','encoder_lr_screen02_recovery01.report',[],REC,b['total_deadline_unix'],awake)
    r=read_json(REC/'FINAL_STATUS.json');entry['result_path']=str(REC/'FINAL_STATUS.json');entry['result']=r
    entry['status']=r['observed_recovery_status'];save(q)
    if entry['status']=='verified_incomplete_budget':raise TimeoutError('Current recovery reached its original ceiling; no extra continuation budget')
    assert entry['status']=='verified_complete','Current recovery did not finish successfully within its existing budget'
    assert r['fresh_full_dev_exact_reproduction'] and r['scientific_metrics_reproduce'] and r['all_retained_checkpoints_postwriter_verified']
    assert not r['preservation_changed_or_missing'] and file_hash(ROOT/r['selected_checkpoint'])==r['selected_checkpoint_sha256']
    base=read_json(ROOT/'encoder_lr_screen02/COMPARATOR.json');assert file_hash(ROOT/base['selected_checkpoint'])==base['sha256']
    selection={'utc':utc(),'rule':'best full-dev exact-span/type micro-F1; exact tie retains1e-5','scores':{'1e-5':base['full_dev_f1'],'3e-5':r['selected_full_dev_f1']},
        'selected_lr':3e-5 if r['selected_full_dev_f1']>base['full_dev_f1'] else 1e-5,'panel_used':False,'conditions_retained':True}
    durable(Q/'XLMR_LR_SELECTION.json',selection);q['selected_lr']['xlmr']=selection['selected_lr'];save(q,{'event':'xlmr_lr_frozen',**selection})

def condition(q,entry,awake):
    if entry['status']=='verified_complete':return
    if entry['status'] in ['blocked','incomplete_budget']:raise RuntimeError('Condition cannot be silently replaced: '+entry['id'])
    d=Q/'conditions'/entry['id'];d.mkdir(parents=True,exist_ok=True)
    if 'start_unix' not in entry:
        entry.update(start_unix=time.time(),deadline_unix=min(time.time()+14400,q['deadline_unix']-900),status='preparing',directory=str(d));save(q)
    end=entry['deadline_unix'];assert end-time.time()>900,'Insufficient remaining condition budget'
    enc=entry['encoder'];lr=entry.get('learning_rate') or q['selected_lr'][enc];micro=8 if enc=='xlmr' else q.get('mbert_microbatch')
    if enc=='mbert' and micro is None:
        benchmark_end=min(end-900,time.time()+600);out=[]
        for mb in [2,4,8]:
            bd=Q/'mbert_preflight'/f'mb{mb}'
            if not (bd/'CONFIG.lock.json').exists():make_config(bd,'mbert',13,1e-5,mb)
            stage(q,entry,'benchmark_'+str(mb),'ner_training_queue01.worker',['benchmark','--microbatch',str(mb)],bd,benchmark_end,awake)
            out.append(read_json(bd/f'BENCHMARK_{mb}.json'))
        assert out[0]['status']=='passed' and out[0]['headroom_fraction']>=.2,'Baseline mBERT memory preflight failed'
        eligible=[x for x in out[1:] if x['status']=='passed' and x['cv']<=.2 and x['headroom_fraction']>=.2 and x['mean_seconds']<out[0]['mean_seconds']*.95]
        chosen=min(eligible,key=lambda x:x['mean_seconds']) if eligible else out[0];micro=chosen['microbatch'];q['mbert_microbatch']=micro
        durable(Q/'MBERT_EXECUTION_SELECTION.json',{'utc':utc(),'microbatch':micro,'accumulation':16//micro,'score_based':False,'candidates':out,'fixed_rule':read_json(Q/'QUEUE_PLAN.json')['mbert_rule']});save(q)
    if not (d/'CONFIG.lock.json').exists():make_config(d,enc,entry['seed'],lr,micro)
    entry.update(learning_rate=lr,microbatch=micro,configuration=str(d/'CONFIG.lock.json'),status='running');save(q)
    usage=shutil.disk_usage(Q);required=40*2**30;durable(d/'DISK_BEFORE.json',{'utc':utc(),'free_bytes':usage.free,'required_bytes':required,'passed':usage.free>=required})
    assert usage.free>=required,'Insufficient disk for five epochs, two rolling generations and in-flight snapshot'
    c=read_json(d/'CONFIG.lock.json');assert c['seed']==entry['seed'] and c['learning_rate']==lr
    stage(q,entry,'zero_capture','ner_training_queue01.worker',['capture'],d,end-900,awake)
    stage(q,entry,'zero_fresh_restore','ner_training_queue01.worker',['restore'],d,end-900,awake)
    initial=read_json(d/'validation/INITIAL.json');relations=[]
    for previous in q['conditions']:
        if previous['id']==entry['id'] or previous['encoder']!=enc or not previous.get('directory'):continue
        ip=Path(previous['directory'])/'validation/INITIAL.json'
        if not ip.exists():continue
        before=read_json(ip);same_seed=previous['seed']==entry['seed']
        relation={'condition':previous['id'],'same_pretrained_encoder':before['pretrained_encoder_signature']==initial['pretrained_encoder_signature'],
            'same_seed':same_seed,'same_head':before['head_signature']==initial['head_signature'],'same_rng':before['all_components']['rng']==initial['all_components']['rng']}
        relations.append(relation)
    if not (d/'validation/SEED_RELATIONS.json').exists():durable(d/'validation/SEED_RELATIONS.json',{'utc':utc(),'relations':relations})
    assert all(x['same_pretrained_encoder'] and x['same_head']==x['same_seed'] and x['same_rng']==x['same_seed'] for x in relations),'Seed/head initialization relation failed'
    writer_end=end-900
    stage(q,entry,'training_initial','ner_training_queue01.worker',['train','--stage','initial','--deadline',str(writer_end)],d,writer_end,awake)
    result=read_json(d/'run/attempts/initial/RESULT.json')
    if result['status']=='planned_restart_ready':
        latest=read_json(d/'run/checkpoints/latest.json');meta=d/'run/checkpoints'/latest['metadata']
        assert read_json(meta)['updates']==1000
        stage(q,entry,'training_resumed','ner_training_queue01.worker',['train','--stage','resumed','--resume',str(meta),'--deadline',str(writer_end)],d,writer_end,awake)
        result=read_json(d/'run/attempts/resumed/RESULT.json')
    assert result['status'] in ['complete','incomplete_budget']
    durable(d/'WRITER_RESULT.json',result)
    stage(q,entry,'postwrite','ner_training_queue01.worker',['postwrite'],d,end-120,awake)
    if result['status']=='complete':stage(q,entry,'fresh_selected_dev','ner_training_queue01.fresh_reload',['--deadline',str(end-120)],d,end-120,awake)
    stage(q,entry,'condition_report','ner_training_queue01.report',['condition'],d,end,awake)
    final=read_json(d/'FINAL_STATUS.json');entry.update(status=final['status'],result=final,result_path=str(d/'FINAL_STATUS.json'));save(q,{'event':'condition_finalized','condition':entry['id'],'status':entry['status']})
    if entry['status']=='incomplete_budget':raise TimeoutError('Condition reached its four-hour ceiling; verified partial state retained')
    assert entry['status']=='verified_complete','Condition failed verification; queue stopped'

def main():
    Q.mkdir(exist_ok=True);handle=(Q/'queue.owner.lock').open('a+b');handle.seek(0);handle.write(b'0');handle.flush();handle.seek(0)
    try:msvcrt.locking(handle.fileno(),msvcrt.LK_NBLCK,1)
    except OSError:print('Existing queue owner active; no duplicate launched.',flush=True);return
    verify_plan();assert read_json(Q/'PREPARATION.json')['passed']
    path=Q/'QUEUE.json'
    if path.exists():
        q=read_json(path)
        if q['status'] in ['verified_complete','blocked','partial']:print('Terminal queue retained; not restarted.',flush=True);return
    else:
        conditions=[{'id':'xlmr_lr1e-5_seed13','encoder':'xlmr','seed':13,'learning_rate':1e-5,'status':'verified_complete','result_path':str(ROOT/'checkpoint_recovery01/FINAL_STATUS.json')},
            {'id':'xlmr_lr3e-5_seed13_recovery','encoder':'xlmr','seed':13,'learning_rate':3e-5,'status':'pending_existing'},
            {'id':'xlmr_selected_seed42','encoder':'xlmr','seed':42,'status':'pending'}, {'id':'xlmr_selected_seed2026','encoder':'xlmr','seed':2026,'status':'pending'},
            {'id':'mbert_lr1e-5_seed13','encoder':'mbert','seed':13,'learning_rate':1e-5,'status':'pending'}, {'id':'mbert_lr3e-5_seed13','encoder':'mbert','seed':13,'learning_rate':3e-5,'status':'pending'},
            {'id':'mbert_selected_seed42','encoder':'mbert','seed':42,'status':'pending'}, {'id':'mbert_selected_seed2026','encoder':'mbert','seed':2026,'status':'pending'}]
        q={'start_unix':START,'deadline_unix':START+86400,'status':'running','conditions':conditions,'selected_lr':{},'historical_cause':'unresolved','shutdown_enabled':False,'final_test_locked':True}
    q.update(owner_pid=os.getpid(),owner_create_time=psutil.Process().create_time());save(q,{'event':'queue_owner_started','pid':os.getpid()})
    try:
        with Awake(Q/'awake.json') as awake:
            existing(q,awake)
            for entry in q['conditions'][2:]:
                if time.time()>=q['deadline_unix']-1800:raise TimeoutError('Global reporting reserve reached')
                condition(q,entry,awake)
                if entry['id']=='mbert_lr3e-5_seed13':
                    a,b=q['conditions'][4:6];scorea=a['result']['full_dev_f1'];scoreb=b['result']['full_dev_f1']
                    selection={'utc':utc(),'selected_lr':3e-5 if scoreb>scorea else 1e-5,'scores':{'1e-5':scorea,'3e-5':scoreb},'rule':'best full-dev F1; exact tie retains1e-5','panel_used':False}
                    durable(Q/'MBERT_LR_SELECTION.json',selection);q['selected_lr']['mbert']=selection['selected_lr'];save(q,{'event':'mbert_lr_frozen',**selection})
            q['status']='verified_complete'
    except BaseException as e:
        q['status']='partial' if isinstance(e,TimeoutError) else 'blocked';q['failure']={'utc':utc(),'type':type(e).__name__,'error':str(e),'traceback':traceback.format_exc()}
        for entry in q['conditions']:
            if entry['status'] in ['running','preparing','attached_existing']:entry['status']='blocked' if q['status']=='blocked' else 'incomplete_budget'
        durable(Q/'FAILURE.json',q['failure'])
    finally:
        q['ended_utc']=utc();q['elapsed_seconds']=time.time()-START;save(q,{'event':'queue_stopped','status':q['status']})
        from .report import summary
        summary(q)
        handle.seek(0);msvcrt.locking(handle.fileno(),msvcrt.LK_UNLCK,1);handle.close()

if __name__=='__main__':main()
