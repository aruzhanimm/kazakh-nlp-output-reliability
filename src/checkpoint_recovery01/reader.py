"""Independent-process checkpoint readback, including the unchanged old verifier."""
import argparse
import gc
from pathlib import Path
from pilot.common import ROOT,read_json,write_json,file_hash
from checkpoint_integrity01.independent_verifier import independent_hash
from .checkpoints import verify_payload
from .common import COMPONENTS,utc

def inspect(metadata):
    import torch
    import encoder_main01.finalize as old
    torch.set_num_threads(4);old.D=ROOT
    metadata=Path(metadata).resolve();m=read_json(metadata);p=metadata.parent/m['file']
    proof,state=verify_payload(p,m)
    independently={k:independent_hash(state[k]) for k in COMPONENTS}
    assert independently==proof['capture_signatures']
    progress=state['progress'];del state;gc.collect()
    old_check,_=old.check_checkpoint(metadata,m['run_hash'])
    assert file_hash(p)==proof['sha256_before']
    return {'utc':utc(),'metadata':str(metadata.relative_to(ROOT)),'payload':str(p.relative_to(ROOT)),
        'readback':proof,'independent_component_signatures':independently,
        'old_verifier':old_check,'accepted':True,'progress':{k:v for k,v in progress.items() if k not in ['trace','history']}},progress

def main():
    p=argparse.ArgumentParser();p.add_argument('--metadata',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    try:r,_=inspect(a.metadata)
    except Exception as error:
        write_json(a.output,{'utc':utc(),'accepted':False,'error_type':type(error).__name__,'message':str(error)});raise
    write_json(a.output,r);print({'accepted':True,'payload':r['payload']},flush=True)

if __name__=='__main__':main()
