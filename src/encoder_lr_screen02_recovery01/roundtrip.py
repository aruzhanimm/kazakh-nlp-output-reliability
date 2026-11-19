import argparse,time,traceback
from contextlib import ExitStack
from unittest.mock import patch
from pilot.common import read_json,file_hash
from stage_preflight.state import digests,fingerprint
from .common import D,META,COMPONENTS,utc,durable,initialize,verify_plan
from .restore import restore
from .checkpoints import capture,commit,committed_meta
from .streams import fingerprint_stream,compare_streams

def main(phase):
    import torch
    verify_plan();folder=D/'validation'/phase;folder.mkdir(parents=True,exist_ok=False);start=time.perf_counter()
    try:
        model,tok,opt,sched=initialize()  # Always before CPU checkpoint loading.
        calls=[]
        def forbidden(*a,**kw):calls.append('forbidden');raise AssertionError('No training/inference in roundtrip')
        with ExitStack() as guards:
            for obj,name in [(torch.nn.Module,'_call_impl'),(torch.Tensor,'backward'),(torch.autograd,'backward'),(opt,'step'),(type(sched),'step')]:guards.enter_context(patch.object(obj,name,forbidden))
            metadata=META if phase=='capture' else committed_meta(D/'validation/checkpoints')[0]
            progress,source,input_meta=restore(metadata,model,opt,sched,folder/'restore',streams=True)
            assert progress['updates']==1000 and progress['cursor']==16000 and progress['epoch']==0
            before=digests(model,opt,sched,progress);source_before=fingerprint(source['model'])
            output_meta=None
            if phase=='capture':
                snapshot=capture(model,opt,sched,progress,input_meta['run_hash'],'validation')
                after_capture=digests(model,opt,sched,progress)
                durable(folder/'capture_before_assertion.json',{'actual_before':before,'actual_after':after_capture,'snapshot':snapshot['state_digests'],'expected':source['state_digests']})
                assert before==after_capture==snapshot['state_digests']==source['state_digests']
                output_meta=commit(D/'validation/checkpoints',snapshot,lineage={'source_metadata':str(metadata),'source_sha256':input_meta['sha256'],'zero_updates':True})
            after=digests(model,opt,sched,progress)
            source_after=fingerprint_stream(source['model'],folder/'restore/streams/source_end/model','model')
            durable(folder/'after_before_assertion.json',{'actual_before':before,'actual_after':after,'embedded':source['state_digests'],'sidecar':input_meta['state_digests'],
                'source_model_before':source_before,'source_model_after':source_after,'checkpoint_bookkeeping_changed':False})
            compare_streams(folder/'restore/streams/source_before/model',folder/'restore/streams/source_end/model',folder/'source_end_comparison.json')
            assert before==after==source['state_digests'] and source_before==source_after and not calls
            durable(folder/'RESULT.json',{'utc':utc(),'passed':True,'seconds':time.perf_counter()-start,'updates_performed':0,'all_five_equal':True,'unchanged_cpu_source_model_bytes':True,
                'input_meta':input_meta,'saved_meta':output_meta,'guard_violations':calls,'policy':'checked execution policy; not a proven historical fix'})
    except BaseException as e:durable(folder/'FAILURE.json',{'utc':utc(),'type':type(e).__name__,'error':str(e),'traceback':traceback.format_exc()});raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['capture','restore']);main(p.parse_args().phase)
