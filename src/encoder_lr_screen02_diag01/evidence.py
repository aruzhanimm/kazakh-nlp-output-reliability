"""Owned post-detection capture and compact, exact byte/key comparisons."""
import copy,hashlib,json,os,time
from pathlib import Path
from pilot.common import file_hash
from stage_preflight.state import fingerprint
from checkpoint_integrity01.independent_verifier import independent_hash
from .common import COMPONENTS,utc

def durable(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+'.pending')
    assert not path.exists(),'Evidence records are append-only'
    with temp.open('x',encoding='utf-8',newline='\n') as f:
        json.dump(value,f,ensure_ascii=False,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
    os.replace(temp,path)

def owned(value):
    import torch,numpy as np
    if torch.is_tensor(value):return value.detach().to(device='cpu',copy=True)
    if isinstance(value,np.ndarray):return value.copy()
    if isinstance(value,dict):return {k:owned(v) for k,v in value.items()}
    if isinstance(value,list):return [owned(v) for v in value]
    if isinstance(value,tuple):return tuple(owned(v) for v in value)
    return copy.deepcopy(value)

def walk(value,path=''):
    import torch,numpy as np
    if torch.is_tensor(value) or isinstance(value,np.ndarray):yield path,value
    elif isinstance(value,dict):
        for k in sorted(value,key=lambda k:(type(k).__name__,str(k))):yield from walk(value[k],path+'['+repr(k)+']')
    elif isinstance(value,(list,tuple)):
        for i,v in enumerate(value):yield from walk(v,path+f'[{i}]')

def array(value):
    import torch
    return value.detach().cpu().contiguous().numpy() if torch.is_tensor(value) else value

def tensor_inventory(state):
    import torch
    records={}
    for component in COMPONENTS:
        for path,v in walk(state[component],component):
            a=array(v);raw=a.tobytes(order='C')
            item={'kind':'tensor' if torch.is_tensor(v) else 'ndarray','dtype':str(v.dtype),'shape':list(v.shape),
                'bytes':len(raw),'byte_sha256':hashlib.sha256(raw).hexdigest()}
            if torch.is_tensor(v):item.update(stride=list(v.stride()),storage_offset=v.storage_offset(),data_ptr=v.data_ptr(),storage_data_ptr=v.untyped_storage().data_ptr())
            records[path]=item
    return records

def inventory_changes(a,b):
    fields=('kind','dtype','shape','bytes','byte_sha256')
    return [p for p in sorted(set(a)|set(b)) if p not in a or p not in b or any(a[p][k]!=b[p][k] for k in fields)]

def signatures(state,algorithm):
    fn=fingerprint if algorithm=='canonical' else independent_hash
    actual={};errors={}
    for k in COMPONENTS:
        try:actual[k]=fn(state[k])
        except Exception as e:errors[k]={'type':type(e).__name__,'message':str(e)}
    return actual,errors

def capture_failure(state,folder,reason):
    """Capture after detection; never claim this reconstructs capture-time values."""
    import torch
    folder=Path(folder);folder.mkdir(exist_ok=False)
    clone=owned(state);path=folder/'post_detection_state.pt'
    with path.open('xb') as f:torch.save(clone,f);f.flush();os.fsync(f.fileno())
    canonical,ce=signatures(clone,'canonical');independent,ie=signatures(clone,'independent')
    inventory=tensor_inventory(clone);durable(folder/'tensor_inventory.json',inventory)
    durable(folder/'capture.json',{'utc':utc(),'reason':reason,'scope':'owned copy of live state after discrepancy detection; not an independent copy of capture-time expected tensors',
        'payload':path.name,'bytes':path.stat().st_size,'sha256':file_hash(path),'canonical':canonical,'independent':independent,
        'canonical_errors':ce,'independent_errors':ie,'embedded_original_expected':copy.deepcopy(clone['state_digests']),
        'no_evidence_reload_performed':True})
    return clone

def small(value):
    s=repr(value);return s if len(s)<=300 else s[:300]+' [truncated scalar representation]'

def tensor_difference(a,b,ia,ib):
    import numpy as np
    out={'left':{k:ia[k] for k in ('dtype','shape','bytes','byte_sha256')},'right':{k:ib[k] for k in ('dtype','shape','bytes','byte_sha256')}}
    if ia['shape']!=ib['shape'] or ia['dtype']!=ib['dtype']:return out
    av=array(a).reshape(-1);bv=array(b).reshape(-1);changed_bytes=0;changed_elements=0;samples=[];maximum=0.
    for begin in range(0,av.size,1000000):
        x=av[begin:begin+1000000];y=bv[begin:begin+1000000]
        xb=x.view(np.uint8).reshape(-1,x.dtype.itemsize);yb=y.view(np.uint8).reshape(-1,y.dtype.itemsize)
        different=xb!=yb;changed_bytes+=int(np.count_nonzero(different));indices=np.flatnonzero(different.any(axis=1));changed_elements+=len(indices)
        if len(indices):
            if np.issubdtype(x.dtype,np.number):
                xx=x.astype(np.float64);yy=y.astype(np.float64);valid=np.isfinite(xx)&np.isfinite(yy)
                if valid.any():maximum=max(maximum,float(np.max(np.abs(xx[valid]-yy[valid]))))
            for i in indices[:max(0,8-len(samples))]:samples.append({'flat_index':begin+int(i),'left':small(x[i].item()),'right':small(y[i].item()),
                'left_bytes':xb[i].tobytes().hex(),'right_bytes':yb[i].tobytes().hex()})
    out.update(changed_bytes=changed_bytes,changed_elements_bitwise=changed_elements,max_finite_absolute_difference=maximum,first_differences=samples)
    return out

def compare_states(left,right,folder,label):
    """Every changed tensor/key is enumerated; tensor values are only small samples."""
    import torch,numpy as np
    li=tensor_inventory(left);ri=tensor_inventory(right);differences=[]
    def visit(a,b,path):
        ta=torch.is_tensor(a) or isinstance(a,np.ndarray);tb=torch.is_tensor(b) or isinstance(b,np.ndarray)
        if ta or tb:
            if not (ta and tb):differences.append({'path':path,'kind':'type','left_type':type(a).__name__,'right_type':type(b).__name__});return
            if path in inventory_changes({path:li[path]},{path:ri[path]}):differences.append({'path':path,'kind':'tensor_or_array',**tensor_difference(a,b,li[path],ri[path])})
        elif type(a)!=type(b):differences.append({'path':path,'kind':'type','left_type':type(a).__name__,'right_type':type(b).__name__})
        elif isinstance(a,dict):
            for k in sorted(set(a)|set(b),key=lambda k:(type(k).__name__,str(k))):
                if k not in a or k not in b:differences.append({'path':path+'['+repr(k)+']','kind':'missing_key','left_present':k in a,'right_present':k in b})
                else:visit(a[k],b[k],path+'['+repr(k)+']')
        elif isinstance(a,(list,tuple)):
            if len(a)!=len(b):differences.append({'path':path,'kind':'length','left':len(a),'right':len(b)})
            for i,(x,y) in enumerate(zip(a,b)):visit(x,y,path+f'[{i}]')
        elif fingerprint(a)!=fingerprint(b):differences.append({'path':path,'kind':'scalar','left':small(a),'right':small(b)})
    for k in COMPONENTS:visit(left[k],right[k],k)
    result={'utc':utc(),'label':label,'all_five_component_values_equal':not differences,'different_keys':len(differences),
        'differing_components':sorted({x['path'].split('[')[0] for x in differences}),'differences':differences,
        'left_tensor_inventory':li,'right_tensor_inventory':ri,
        'scope':'comparison of actual loaded/post-detection values; no inferred capture-time expected tensor values'}
    durable(Path(folder)/(label+'.json'),result)
    return {k:v for k,v in result.items() if k not in ['differences','left_tensor_inventory','right_tensor_inventory']}
