"""Canonical digests for trusted local checkpoint state, including RNG."""
import hashlib
import json
import random

def fingerprint(value):
    import numpy as np
    import torch
    h=hashlib.sha256()
    def visit(x):
        if torch.is_tensor(x):
            a=x.detach().cpu().contiguous(); h.update(b'tensor'); h.update(str((a.dtype,tuple(a.shape))).encode()); h.update(a.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(x,np.ndarray):
            h.update(b'ndarray');h.update(str((x.dtype,x.shape)).encode());h.update(x.tobytes())
        elif isinstance(x,dict):
            h.update(b'dict')
            for k in sorted(x,key=lambda k:(type(k).__name__,str(k))): visit(k);visit(x[k])
        elif isinstance(x,(tuple,list)):
            h.update(type(x).__name__.encode())
            for item in x: visit(item)
        else: h.update(json.dumps([type(x).__name__,x],sort_keys=True).encode())
    visit(value);return h.hexdigest()

def rng():
    import numpy as np
    import torch
    return {'random':random.getstate(),'numpy':np.random.get_state(),'torch':torch.get_rng_state(),
            'cuda':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}

def restore_rng(state):
    import numpy as np
    import torch
    random.setstate(state['random']);np.random.set_state(state['numpy']);torch.set_rng_state(state['torch'])
    if state['cuda'] is not None: torch.cuda.set_rng_state_all(state['cuda'])

def digests(model,optimizer,scheduler,progress):
    return {'model':fingerprint(model.state_dict()),'optimizer':fingerprint(optimizer.state_dict()),
        'scheduler':fingerprint(scheduler.state_dict()),'rng':fingerprint(rng()),'progress':fingerprint(progress)}
