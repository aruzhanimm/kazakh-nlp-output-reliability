"""Owned snapshot capture and generation-bound durable commits. No training semantics."""
import copy
import os
import threading
import time
from pathlib import Path
from pilot.common import read_json,file_hash
from stage_preflight.state import fingerprint,rng
from .common import D,COMPONENTS,utc

_save_lock=threading.RLock()

def owned(value):
    import numpy as np
    import torch
    if torch.is_tensor(value):return value.detach().to(device='cpu',copy=True)
    if isinstance(value,np.ndarray):return value.copy()
    if isinstance(value,dict):return {key:owned(v) for key,v in value.items()}
    if isinstance(value,tuple):return tuple(owned(v) for v in value)
    if isinstance(value,list):return [owned(v) for v in value]
    return copy.deepcopy(value)

def signatures(state):return {name:fingerprint(state[name]) for name in COMPONENTS}

def capture(model,optimizer,scheduler,progress,run_hash,logical_name):
    """Caller is at a quiescent update boundary; no live state re-fetch for signatures."""
    import torch
    with _save_lock:
        if torch.cuda.is_available():torch.cuda.synchronize()
        state={'run_hash':run_hash,'model':owned(model.state_dict()),'optimizer':owned(optimizer.state_dict()),
            'scheduler':owned(scheduler.state_dict()),'rng':owned(rng()),'progress':owned(progress)}
        state['checkpoint_identity']={'version':1,'generation':state['progress']['checkpoint_sequence'],
            'rolling_sequence':state['progress']['rolling_sequence'],'logical_name':logical_name,'run_hash':run_hash}
        state['state_digests']=signatures(state)
        return state

def durable_json(target,value):
    import json
    target=Path(target);temp=target.with_name(target.name+'.pending')
    with temp.open('x',encoding='utf-8',newline='\n') as f:
        json.dump(value,f,ensure_ascii=False,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
    os.replace(temp,target)

def verify_payload(path,meta,finite=True):
    import torch
    path=Path(path);before=file_hash(path);assert before==meta['sha256'],'Payload SHA mismatch'
    state=torch.load(path,map_location='cpu',weights_only=False)
    identity=state['checkpoint_identity'];p=state['progress']
    assert identity==meta['checkpoint_identity'],'Payload/metadata identity mismatch'
    assert identity['generation']==p['checkpoint_sequence']==meta['sequence'],'Generation mismatch'
    assert identity['rolling_sequence']==p['rolling_sequence']
    assert identity['run_hash']==state['run_hash']==meta['run_hash']
    assert p['updates']==meta['updates'] and p['epoch']==meta['epoch'] and p['phase']==meta['phase']
    actual=signatures(state)
    assert actual==state['state_digests']==meta['state_digests'],'Deserialized component signatures differ'
    count=0
    def check(x):
        nonlocal count
        if torch.is_tensor(x) and (x.is_floating_point() or x.is_complex()):
            import numpy as np
            for chunk in x.reshape(-1).split(1000000):assert bool(np.isfinite(chunk.numpy()).all()),'Nonfinite saved state'
            count+=x.numel()
        elif isinstance(x,dict):
            for v in x.values():check(v)
        elif isinstance(x,(list,tuple)):
            for v in x:check(v)
    if finite:check(state['model']);check(state['optimizer'])
    after=file_hash(path);assert before==after,'Payload changed during deserialization'
    return {'utc':utc(),'sha256_before':before,'sha256_after':after,'capture_signatures':meta['state_digests'],
        'deserialized_signatures':actual,'identity':identity,'finite_floating_elements':count},state

def commit(directory,state,fail_at=None):
    """Never overwrite generation payloads; publish latest only after verified readback."""
    import torch
    directory=Path(directory);assert directory.resolve().is_relative_to(D.resolve())
    directory.mkdir(parents=True,exist_ok=True)
    with _save_lock:
        identity=state['checkpoint_identity'];generation=identity['generation'];logical=identity['logical_name']
        filename=f'{logical}.pt' if logical.startswith('epoch_') else f'{logical}_g{generation:06d}.pt'
        target=directory/filename;pending=directory/(filename+'.pending')
        assert not target.exists(),'Refusing to overwrite immutable generation'
        assert signatures(state)==state['state_digests'],'Snapshot changed before serialization'
        with pending.open('xb') as f:
            torch.save(state,f);f.flush();os.fsync(f.fileno())
        if fail_at=='before_payload_publish':raise RuntimeError('Injected interruption before payload publication')
        os.replace(pending,target)
        if fail_at=='after_payload_publish':raise RuntimeError('Injected interruption after payload publication')
        p=state['progress'];meta={'file':filename,'run_hash':state['run_hash'],'updates':p['updates'],
            'sequence':generation,'phase':p['phase'],'epoch':p['epoch'],'bytes':target.stat().st_size,
            'sha256':file_hash(target),'state_digests':copy.deepcopy(state['state_digests']),
            'checkpoint_identity':copy.deepcopy(identity)}
        checked,_=verify_payload(target,meta)
        durable_json(directory/f'generation_{generation:06d}.readback.json',checked)
        generation_name=f'generation_{generation:06d}.json'
        assert not (directory/generation_name).exists()
        durable_json(directory/generation_name,meta)
        if fail_at=='after_generation_metadata':raise RuntimeError('Injected interruption after generation metadata')
        durable_json(directory/(logical+'.json'),meta)
        if fail_at=='before_latest_pointer':raise RuntimeError('Injected interruption before latest pointer')
        durable_json(directory/'latest.json',{'metadata':generation_name,'sequence':generation,'sha256':meta['sha256']})
        if fail_at=='after_latest_pointer':raise RuntimeError('Injected interruption after latest pointer')
        durable_json(directory/f'generation_{generation:06d}.committed.json',{'sequence':generation,'metadata':generation_name,
            'sha256':meta['sha256'],'file':filename,'utc':utc()})
        if fail_at=='before_retention':raise RuntimeError('Injected interruption before retention')
        retain(directory,fail_at=fail_at)
        return meta

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
    for meta in keep:verify_payload(directory/meta['file'],meta)
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

def latest(directory):
    directory=Path(directory);pointer=read_json(directory/'latest.json');meta=read_json(directory/pointer['metadata'])
    assert meta['sequence']==pointer['sequence'] and meta['sha256']==pointer['sha256'],'Latest pointer generation mismatch'
    receipt=read_json(directory/f'generation_{meta["sequence"]:06d}.committed.json')
    assert receipt['sha256']==meta['sha256'] and receipt['file']==meta['file'] and receipt['sequence']==meta['sequence'] and receipt['metadata']==pointer['metadata'],'Missing or mismatched commit receipt'
    proof,state=verify_payload(directory/meta['file'],meta)
    return meta,proof,state

def save(directory,model,optimizer,scheduler,progress,run_hash,epoch=None):
    started=time.perf_counter()
    with _save_lock:
        # Identical bookkeeping to the original saver, performed before pure capture.
        progress['checkpoint_sequence']+=1
        if epoch is None:progress['rolling_sequence']+=1
        logical=f'epoch_{epoch:02d}' if epoch is not None else f'rolling_{(progress["rolling_sequence"]-1)%2}'
        state=capture(model,optimizer,scheduler,progress,run_hash,logical)
        meta=commit(directory,state)
        import json
        with (Path(directory)/'save_events.jsonl').open('a',encoding='utf-8') as f:
            f.write(json.dumps({'utc':utc(),'sequence':meta['sequence'],'updates':meta['updates'],'file':meta['file'],
                'seconds':time.perf_counter()-started,'scope':'capture, signatures, serialization, fsync, readback, publication, retention'})+'\n')
        return meta
