"""Original owned capture; added auditable verification and independent precommit reader."""
import copy,gc,os,time,threading
from pathlib import Path
from pilot.common import read_json,file_hash
from stage_preflight.state import digests
from encoder_lr_screen02.checkpoints import capture,owned,signatures
from .common import D,COMPONENTS,utc,durable,append,reader_process
LOCK=threading.RLock()

def verify_payload(path,meta,audit):
    import torch,numpy as np
    path=Path(path);before=file_hash(path)
    durable(Path(audit).with_name(Path(audit).stem+'_file_before.json'),{'utc':utc(),'actual':before,'expected':meta['sha256']})
    assert before==meta['sha256'],'Payload SHA mismatch'
    state=torch.load(path,map_location='cpu',weights_only=False)
    actual=signatures(state);after=file_hash(path)
    p=state['progress'];ident=state['checkpoint_identity']
    errors=[]
    if actual!=state['state_digests'] or actual!=meta['state_digests']:errors.append('component signatures')
    if before!=after:errors.append('changed payload bytes')
    if ident!=meta['checkpoint_identity']:errors.append('identity')
    if not (ident['generation']==p['checkpoint_sequence']==meta['sequence']):errors.append('generation')
    if ident['rolling_sequence']!=p['rolling_sequence']:errors.append('rolling generation')
    if not (ident['run_hash']==state['run_hash']==meta['run_hash']):errors.append('run identity')
    if any(p[k]!=meta[k] for k in ['updates','epoch','phase']):errors.append('progress metadata')
    report={'utc':utc(),'sha256_before':before,'sha256_after':after,'actual':actual,'embedded':state['state_digests'],'sidecar':meta['state_digests'],
        'differing_components':[k for k in COMPONENTS if actual[k]!=state['state_digests'][k] or actual[k]!=meta['state_digests'][k]],
        'errors':errors,'recorded_before_assertion':True}
    durable(audit,report);assert not errors,'Checkpoint verification failed: '+repr(errors)
    count=0
    def finite(x):
        nonlocal count
        if torch.is_tensor(x) and (x.is_floating_point() or x.is_complex()):
            for part in x.reshape(-1).split(1000000):assert bool(np.isfinite(part.numpy()).all()),'Nonfinite saved state'
            count+=x.numel()
        elif isinstance(x,dict):
            for v in x.values():finite(v)
        elif isinstance(x,(list,tuple)):
            for v in x:finite(v)
    finite(state['model']);finite(state['optimizer'])
    return report,state,count

def commit(directory,state,lineage=None,fail_at=None):
    import torch
    directory=Path(directory).resolve();assert directory.is_relative_to(D.resolve());directory.mkdir(parents=True,exist_ok=True)
    with LOCK:
        ident=state['checkpoint_identity'];gen=ident['generation'];logical=ident['logical_name'];filename=f'{logical}_g{gen:06d}.pt'
        target=directory/filename;pending=directory/(filename+'.pending');assert not target.exists() and not pending.exists()
        before=signatures(state)
        durable(directory/f'generation_{gen:06d}.capture.json',{'utc':utc(),'actual':before,'embedded':state['state_digests'],'recorded_before_assertion':True})
        assert before==state['state_digests'],'Owned snapshot mutated before serialization'
        with pending.open('xb') as f:torch.save(state,f);f.flush();os.fsync(f.fileno())
        p=state['progress'];meta={'file':pending.name,'run_hash':state['run_hash'],'updates':p['updates'],'epoch':p['epoch'],'phase':p['phase'],
            'sequence':gen,'bytes':pending.stat().st_size,'sha256':file_hash(pending),'state_digests':copy.deepcopy(state['state_digests']),
            'checkpoint_identity':copy.deepcopy(ident),'lineage':lineage}
        candidate=directory/f'generation_{gen:06d}.candidate.json';durable(candidate,meta)
        proof,loaded,_=verify_payload(pending,meta,directory/f'generation_{gen:06d}.writer_readback.json');del loaded;gc.collect()
        independent_seconds=reader_process(candidate,directory/f'generation_{gen:06d}.independent.json')
        assert file_hash(pending)==meta['sha256'] and signatures(state)==before
        if fail_at=='before_publish':raise RuntimeError('Synthetic interruption before publication')
        os.replace(pending,target);meta['file']=filename
        generation=directory/f'generation_{gen:06d}.json';durable(generation,meta)
        durable(directory/f'generation_{gen:06d}.committed.json',{'utc':utc(),'metadata':generation.name,'sequence':gen,'sha256':meta['sha256'],
            'file':filename,'independent_precommit_verified':True,'independent_reader_seconds':independent_seconds})
        from encoder_lr_screen02.checkpoints import durable_json
        durable_json(directory/(logical+'.json'),meta)
        durable_json(directory/'latest.json',{'metadata':generation.name,'sequence':gen,'sha256':meta['sha256']})
        return meta

def save(directory,model,opt,sched,progress,run_hash,epoch=None,lineage=None):
    start=time.perf_counter()
    lineage=copy.deepcopy(lineage) if lineage else {}
    if (Path(directory)/'latest.json').exists():
        predecessor_path,predecessor=committed_meta(directory)
        lineage['previous_committed']={'metadata':str(predecessor_path),'sha256':predecessor['sha256'],'updates':predecessor['updates'],'sequence':predecessor['sequence']}
    progress['checkpoint_sequence']+=1
    if epoch is None:progress['rolling_sequence']+=1
    logical=f'epoch_{epoch:02d}' if epoch is not None else f'rolling_{(progress["rolling_sequence"]-1)%2}'
    state=capture(model,opt,sched,progress,run_hash,logical)
    before_commit=digests(model,opt,sched,progress)
    durable(Path(directory)/f'generation_{progress["checkpoint_sequence"]:06d}.live_before.json',{'utc':utc(),'actual':before_commit,'capture':state['state_digests'],'recorded_before_assertion':True})
    assert before_commit==state['state_digests'],'Live state differs from owned capture'
    meta=commit(directory,state,lineage=lineage)
    # Compare LIVE values with the capture, without changing snapshot-derived signatures.
    live=digests(model,opt,sched,progress)
    durable(Path(directory)/f'generation_{meta["sequence"]:06d}.live_after.json',{'utc':utc(),'actual':live,'capture':state['state_digests'],'recorded_before_assertion':True})
    assert live==state['state_digests'],'Live state changed during checkpoint saving'
    retain(directory)
    append(Path(directory)/'save_events.jsonl',{'utc':utc(),'updates':progress['updates'],'sequence':meta['sequence'],'seconds':time.perf_counter()-start,'file':meta['file']})
    return meta

def committed_meta(directory):
    directory=Path(directory);pointer=read_json(directory/'latest.json');meta=read_json(directory/pointer['metadata'])
    receipt=read_json(directory/f'generation_{meta["sequence"]:06d}.committed.json')
    assert pointer['sha256']==meta['sha256']==receipt['sha256'] and pointer['sequence']==meta['sequence']==receipt['sequence']
    return directory/pointer['metadata'],meta

from encoder_lr_screen02.checkpoints import durable_json
def retain(directory,fail_at=None):
    """Delete only superseded committed rolling payloads owned by this milestone."""
    import re
    directory=Path(directory).resolve()
    assert directory.is_relative_to(D.resolve()) and directory!=D.resolve()
    receipts=[]
    for path in directory.glob('generation_*.committed.json'):
        receipt=read_json(path);meta=read_json(directory/receipt['metadata'])
        assert receipt['sequence']==meta['sequence'] and receipt['sha256']==meta['sha256'] and receipt['file']==meta['file']
        assert meta['checkpoint_identity']['generation']==meta['sequence']
        if meta['checkpoint_identity']['logical_name'].startswith('rolling_'):receipts.append(meta)
    receipts.sort(key=lambda m:m['sequence'])
    if len(receipts)<=2:return
    keep=receipts[-2:]
    # Both surviving recovery payloads must verify before any deletion.
    for meta in keep:
        # Previously independent-verified survivors, stable bytes required now.
        proof=read_json(directory/f'generation_{meta["sequence"]:06d}.independent.json')
        assert proof['accepted'] and proof['sha256_after']==meta['sha256']
        assert file_hash(directory/meta['file'])==meta['sha256']
    pointer=read_json(directory/'latest.json');latest_meta=read_json(directory/pointer['metadata'])
    assert pointer['sha256']==latest_meta['sha256'] and pointer['sequence']==latest_meta['sequence']
    for meta in receipts[:-2]:
        gen=meta['sequence'];name=meta['file'];target=(directory/name).resolve()
        assert target.parent==directory and re.fullmatch(r'rolling_[01]_g\d{6}\.pt',name)
        assert name==f'{meta["checkpoint_identity"]["logical_name"]}_g{gen:06d}.pt'
        assert latest_meta['file']!=name and all(m['file']!=name for m in keep)
        intent=directory/f'generation_{gen:06d}.retirement_intent.json';done=directory/f'generation_{gen:06d}.retired.json'
        if done.exists():
            assert not target.exists(),'Retired payload unexpectedly reappeared'
            assert read_json(done)['sha256']==meta['sha256'];continue
        if target.exists():assert file_hash(target)==meta['sha256'],'Superseded payload bytes changed'
        else:assert intent.exists(),'Missing payload without authorized retirement intent'
        if not intent.exists():durable_json(intent,{'utc':utc(),'file':name,'sequence':gen,'sha256':meta['sha256'],
            'retained_rolling_generations':[m['sequence'] for m in keep],'latest_sequence':pointer['sequence']})
        if fail_at=='before_retire_delete':raise RuntimeError('Injected interruption before retirement deletion')
        if target.exists():target.unlink()
        if fail_at=='after_retire_delete':raise RuntimeError('Injected interruption after retirement deletion')
        durable_json(done,{**read_json(intent),'deletion_verified_utc':utc(),'payload_absent':not target.exists()})
