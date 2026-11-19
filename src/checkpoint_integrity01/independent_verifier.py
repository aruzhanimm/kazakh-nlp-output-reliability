"""Independent implementation of the frozen digest wire format; preserves old verifier."""
import argparse
import hashlib
import json
import os
from pilot.common import ROOT,read_json,write_json,file_hash
from .common import D,CK,COMPONENTS,guard,utc

def independent_hash(value):
    import numpy as np
    import torch
    result=hashlib.sha256();pending=[value]
    while pending:
        x=pending.pop()
        if isinstance(x,torch.Tensor):
            t=x.detach().to(device='cpu').contiguous()
            result.update(b'tensor');result.update(str((t.dtype,tuple(t.shape))).encode())
            result.update(t.numpy().tobytes(order='C'))
        elif isinstance(x,np.ndarray):
            result.update(b'ndarray');result.update(str((x.dtype,x.shape)).encode());result.update(x.tobytes())
        elif isinstance(x,dict):
            result.update(b'dict')
            ordered=[]
            for key in sorted(x,key=lambda k:(type(k).__name__,str(k))):ordered.extend([key,x[key]])
            pending.extend(reversed(ordered))
        elif isinstance(x,(tuple,list)):
            result.update(type(x).__name__.encode());pending.extend(reversed(x))
        else:result.update(json.dumps([type(x).__name__,x],sort_keys=True).encode())
    return result.hexdigest()

def main():
    import gc
    import torch
    import encoder_main01.finalize as old
    guard();torch.set_num_threads(4);rows=[];old.D=ROOT
    for name in ['epoch_05','rolling_0','rolling_1']:
        path=CK/(name+'.pt');m=read_json(path.with_suffix('.json'))
        with path.open('rb') as f:sha=hashlib.file_digest(f,'sha256').hexdigest()
        s=torch.load(path,map_location='cpu',weights_only=False)
        independent={k:independent_hash(s[k]) for k in COMPONENTS}
        canonical=next(r for r in read_json(D/'audit/components_host_2.json') if r['checkpoint']==path.name)['actual']
        assert independent==canonical
        row={'utc':utc(),'process_id':os.getpid(),'checkpoint':path.name,'independent_file_sha256':sha,
            'component_digests':independent,'agrees_with_original_digest_implementation':True}
        del s;gc.collect()
        try:
            verdict,_=old.check_checkpoint(path.with_suffix('.json'),m['run_hash'])
            row['unchanged_old_verifier']={'accepted':True,'result':verdict}
        except Exception as e:row['unchanged_old_verifier']={'accepted':False,'error_type':type(e).__name__,'message':str(e)}
        row['file_unchanged_after_old_verifier']=file_hash(path)==sha
        assert row['file_unchanged_after_old_verifier']
        rows.append(row);write_json(D/'audit/independent_verification.json',rows)
        print({'checkpoint':name,'independent_agrees':True,'old_verifier':row['unchanged_old_verifier']['accepted']},flush=True)

if __name__=='__main__':main()
