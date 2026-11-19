"""Two-slot atomic commits with checksum validation and immutable epoch snapshots."""
from pathlib import Path
import os
import time
from pilot.common import read_json,write_json,file_hash
from stage_preflight.state import rng,restore_rng,digests

def atomic_payload(target,state,fail_at=None):
    import torch
    target=Path(target);target.parent.mkdir(parents=True,exist_ok=True)
    temporary=target.with_suffix('.pt.tmp')
    with temporary.open('wb') as f:
        torch.save(state,f);f.flush();os.fsync(f.fileno())
    if fail_at=='before_replace':raise RuntimeError('Injected pre-replace failure')
    os.replace(temporary,target)
    if fail_at=='after_replace':raise RuntimeError('Injected pre-metadata failure')

def commit(path,name,state,immutable=False,fail_at=None):
    target=Path(path)/(name+'.pt');meta=target.with_suffix('.json');started=time.perf_counter()
    if immutable and target.exists():raise ValueError('Refusing to overwrite epoch checkpoint')
    atomic_payload(target,state,fail_at)
    sha=file_hash(target)
    value={'file':target.name,'run_hash':state['run_hash'],'updates':state['progress']['updates'],
        'sequence':state['progress']['checkpoint_sequence'],'phase':state['progress']['phase'],
        'epoch':state['progress']['epoch'],'bytes':target.stat().st_size,'sha256':sha,
        'seconds_including_checksum':time.perf_counter()-started,'state_digests':state.get('state_digests')}
    write_json(meta,value);write_json(Path(path)/'latest.json',{'metadata':meta.name,'sequence':value['sequence'],'sha256':sha})
    return value

def candidates(path,run_hash):
    valid=[];rejected=[]
    for meta in Path(path).glob('*.json'):
        if meta.name=='latest.json':continue
        try:
            m=read_json(meta)
            if 'file' not in m:continue
            if m['run_hash']!=run_hash:raise ValueError('Different run identity')
            target=Path(path)/m['file']
            if file_hash(target)!=m['sha256']:raise ValueError('Uncommitted/corrupt payload')
            valid.append((m,target))
        except (OSError,ValueError,KeyError) as e:rejected.append({'metadata':meta.name,'reason':str(e)})
    return sorted(valid,key=lambda x:(x[0]['sequence'],x[0]['updates']),reverse=True),rejected

def save(path,model,optimizer,scheduler,progress,run_hash,epoch=None):
    import torch
    torch.cuda.synchronize();progress['checkpoint_sequence']+=1
    if epoch is None:progress['rolling_sequence']=progress.get('rolling_sequence',0)+1
    state={'run_hash':run_hash,'model':model.state_dict(),'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),
        'progress':progress,'rng':rng()}
    # Full state signatures at planned restart and epoch boundaries; payload SHA for every commit.
    if progress['updates']==1000 or epoch is not None:
        state['state_digests']=digests(model,optimizer,scheduler,progress)
    name=f'epoch_{epoch:02d}' if epoch is not None else f'rolling_{(progress["rolling_sequence"]-1)%2}'
    return commit(path,name,state,immutable=epoch is not None)

def restore(path,model,optimizer,scheduler,run_hash):
    import torch
    valid,rejected=candidates(path,run_hash)
    if not valid:return None,{'rejected':rejected,'restored':False}
    meta,target=valid[0];started=time.perf_counter()
    state=torch.load(target,map_location='cpu',weights_only=False)
    assert state['run_hash']==run_hash
    model.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer']);scheduler.load_state_dict(state['scheduler'])
    progress=state['progress'];restore_rng(state['rng']);expected=state.get('state_digests');del state
    actual=digests(model,optimizer,scheduler,progress) if expected is not None else None
    assert expected is None or expected==actual,'Restored model/optimizer/scheduler/RNG/progress mismatch'
    return progress,{'restored':True,'checkpoint':target.name,'updates':progress['updates'],'sequence':meta['sequence'],
        'seconds':time.perf_counter()-started,'exact_state_signatures_verified':expected is not None,
        'expected':expected,'actual':actual,'rejected':rejected}
