"""Owned snapshot capture and generation-bound durable commits. No training semantics."""
import copy
import os
import threading
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
        return meta

def latest(directory):
    directory=Path(directory);pointer=read_json(directory/'latest.json');meta=read_json(directory/pointer['metadata'])
    assert meta['sequence']==pointer['sequence'] and meta['sha256']==pointer['sha256'],'Latest pointer generation mismatch'
    proof,state=verify_payload(directory/meta['file'],meta)
    return meta,proof,state

def save(directory,model,optimizer,scheduler,progress,run_hash,epoch=None):
    with _save_lock:
        # Identical bookkeeping to the original saver, performed before pure capture.
        progress['checkpoint_sequence']+=1
        if epoch is None:progress['rolling_sequence']+=1
        logical=f'epoch_{epoch:02d}' if epoch is not None else f'rolling_{(progress["rolling_sequence"]-1)%2}'
        state=capture(model,optimizer,scheduler,progress,run_hash,logical)
        return commit(directory,state)
