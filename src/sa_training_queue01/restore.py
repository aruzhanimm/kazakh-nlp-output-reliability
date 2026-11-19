"""Initialize-before-load is enforced by callers; retain CPU source through verification."""
from pathlib import Path
from pilot.common import read_json,file_hash
from stage_preflight.state import digests,restore_rng,fingerprint
from .common import D,COMPONENTS,utc,durable,config
from .checkpoints import verify_payload
from encoder_lr_screen02_recovery01.streams import fingerprint_stream,compare_streams

def restore(metadata,model,opt,sched,audit,streams=False):
    import torch
    metadata=Path(metadata);audit=Path(audit);audit.mkdir(parents=True,exist_ok=False);meta=read_json(metadata);path=metadata.parent/meta['file']
    assert path.resolve().is_relative_to(D.resolve()),'Unapproved recovery source'
    assert meta['run_hash']==config()['run_hash'],'Incompatible condition'
    proof,state,_=verify_payload(path,meta,audit/'cpu_readback.json')
    expected=state['state_digests'];original_source_id=id(state);original_model_id=id(state['model'])
    if streams:before=fingerprint_stream(state['model'],audit/'streams/source_before/model','model')
    else:before=fingerprint(state['model'])
    durable(audit/'source_before.json',{'actual':before,'expected':expected['model'],'embedded':state['state_digests'],'sidecar':meta['state_digests']})
    assert before==expected['model']
    model.load_state_dict(state['model'],strict=True);torch.cuda.synchronize()
    opt.load_state_dict(state['optimizer']);torch.cuda.synchronize()
    sched.load_state_dict(state['scheduler']);restore_rng(state['rng']);progress=state['progress'];torch.cuda.synchronize()
    actual=digests(model,opt,sched,progress)
    if streams:
        source_after=fingerprint_stream(state['model'],audit/'streams/source_after/model','model')
        dest=fingerprint_stream(model.state_dict(),audit/'streams/destination/model','model')
    else:source_after=fingerprint(state['model']);dest=actual['model']
    differences=[k for k in COMPONENTS if actual[k]!=expected[k] or actual[k]!=meta['state_digests'][k]]
    record={'utc':utc(),'actual':actual,'embedded':expected,'sidecar':meta['state_digests'],'differing_components':differences,
        'cpu_source_before':before,'cpu_source_after':source_after,'destination_model':dest,'cpu_source_alive':id(state)==original_source_id and id(state['model'])==original_model_id,
        'payload_sha256_after':file_hash(path),'progress':{k:v for k,v in progress.items() if k not in ['trace']},'recorded_before_assertion':True}
    durable(audit/'restored_before_assertion.json',record)
    if streams:
        compare_streams(audit/'streams/source_before/model',audit/'streams/source_after/model',audit/'source_bytes_comparison.json')
        compare_streams(audit/'streams/source_after/model',audit/'streams/destination/model',audit/'destination_bytes_comparison.json')
    assert not differences and before==source_after==dest==expected['model'] and record['payload_sha256_after']==meta['sha256'],'Restored state mismatch'
    assert sched.last_epoch==progress['updates'] and sched.base_lrs==[config()['learning_rate']]
    assert all(int(v['step'])==progress['updates'] for v in opt.state.values())
    return progress,state,meta
