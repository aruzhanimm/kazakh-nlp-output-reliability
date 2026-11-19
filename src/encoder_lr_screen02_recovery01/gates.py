import argparse,gc
from pathlib import Path
from pilot.common import read_json,read_jsonl,file_hash
from stage_preflight.state import fingerprint
from .common import D,COMPONENTS,utc,durable
from .checkpoints import verify_payload
from .streams import rehash_stream

def roundtrip():
    checks=[read_json(D/'validation'/x/'RESULT.json') for x in ['capture','restore']]
    results=[]
    for p in sorted((D/'validation').rglob('*.stream.json')):
        base=Path(str(p)[:-len('.stream.json')]);target=D/'validation/independent_streams'/p.relative_to(D/'validation')
        results.append(rehash_stream(base,target))
    passed=all(x['passed'] and x['all_five_equal'] and x['unchanged_cpu_source_model_bytes'] for x in checks) and all(x['passed'] for x in results)
    durable(D/'validation/ROUNDTRIP.json',{'utc':utc(),'passed':passed,'streams_independently_rehashed':len(results),'zero_training_updates':True,
        'policy':'checked initialize-before-load, retained CPU source, synchronized restore, owned capture and independent precommit reader; historical cause unresolved'})
    assert passed

def compare(point):
    cm=read_json(D/'trajectories/C'/f'checkpoint_{point}.json');rm=read_json(D/'trajectories'/('R1' if point==1050 else 'R2')/f'checkpoint_{point}.json')
    folder=D/'validation'/f'equivalence_{point}';folder.mkdir(parents=True,exist_ok=False)
    cp=D/'trajectories/C/checkpoints'/cm['file'];rp=D/'trajectories/R/checkpoints'/rm['file']
    ca,a,_=verify_payload(cp,cm,folder/'C_actual.json');ra,b,_=verify_payload(rp,rm,folder/'R_actual.json')
    differing=[k for k in COMPONENTS if ca['actual'][k]!=ra['actual'][k]]
    trace_a=a['progress']['trace'][1000:point];trace_b=b['progress']['trace'][1000:point]
    batch_a=[r for r in read_jsonl(D/'trajectories/C/batch_order.jsonl') if r['update']<=point]
    batch_b=read_jsonl(D/'trajectories/R1/batch_order.jsonl')
    if point==1100:batch_b+=read_jsonl(D/'trajectories/R2/batch_order.jsonl')
    trace_diff=[{'update':x['update'],'fields':[k for k in set(x)|set(y) if x.get(k)!=y.get(k)],'C':x,'R':y} for x,y in zip(trace_a,trace_b) if x!=y]
    details={}
    if 'model' in differing:details['model_keys']=[k for k in set(a['model'])|set(b['model']) if fingerprint(a['model'].get(k))!=fingerprint(b['model'].get(k))]
    if 'optimizer' in differing:
        details['optimizer_keys']=[]
        for k in set(a['optimizer']['state'])|set(b['optimizer']['state']):
            x=a['optimizer']['state'].get(k,{});y=b['optimizer']['state'].get(k,{})
            for key in set(x)|set(y):
                if fingerprint(x.get(key))!=fingerprint(y.get(key)):details['optimizer_keys'].append(f'state.{k}.{key}')
    if 'progress' in differing:details['progress_keys']=[k for k in set(a['progress'])|set(b['progress']) if fingerprint(a['progress'].get(k))!=fingerprint(b['progress'].get(k))]
    passed=not differing and not trace_diff and len(trace_a)==len(trace_b)==point-1000 and batch_a==batch_b and len(batch_a)==point-1000
    result={'utc':utc(),'point':point,'passed':passed,'C_actual':ca['actual'],'R_actual':ra['actual'],'differing_components':differing,'details':details,
        'full_progress_compared_without_exclusions':True,'identical_batch_order':batch_a==batch_b,'trace_differences':trace_diff,
        'trace_rows':len(trace_a),'all_trace_fields_exact_including_lr_loss_gradient_norm':not trace_diff,'C_checkpoint':cm,'R_checkpoint':rm}
    durable(folder/'RESULT.json',result);assert passed,'Predefined training/restart equivalence failed'
    if point==1100:
        first=read_json(D/'validation/equivalence_1050/RESULT.json');assert first['passed']
        durable(D/'validation/EQUIVALENCE.json',{'utc':utc(),'passed':True,'comparison_points':[1050,1100],'all_five_exact':True,'no_progress_exclusions':True,
            'identical_full_batch_order':True,'identical_update_traces':True,'predefined_continuation_branch':'C','selected_on_development_score':False})
    print({'equivalence_point':point,'passed':passed},flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['roundtrip','1050','1100']);a=p.parse_args()
    if a.mode=='roundtrip':roundtrip()
    else:compare(int(a.mode))
