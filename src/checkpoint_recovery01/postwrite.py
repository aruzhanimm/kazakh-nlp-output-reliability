"""Separate-process integrity and historical comparison after the writer exits."""
import gc
from pilot.common import ROOT,read_json,read_jsonl,write_json,file_hash
from stage_preflight.state import fingerprint
from .common import D,RUN,PREVIOUS,COMPONENTS,verify_plan,utc
from .reader import inspect

OPERATIONAL_METRICS={'timing','peak_gpu_allocated_bytes','peak_gpu_reserved_bytes'}
def scientific_metrics(m):return {k:v for k,v in m.items() if k not in OPERATIONAL_METRICS}

def compare_predictions(new,old):
    assert len(new)==len(old)==11100
    categories={'ids':[],'input_hashes':[],'tags':[],'spans_and_failure_accounting':[],'confidence':[]}
    max_confidence_delta=0.
    for a,b in zip(new,old):
        checks={'ids':a['id']==b['id'],'input_hashes':a['input_sha256']==b['input_sha256'],
            'tags':a['raw'].get('word_tags')==b['raw'].get('word_tags'),
            'spans_and_failure_accounting':a['parsed']==b['parsed'],
            'confidence':a['raw'].get('word_confidence')==b['raw'].get('word_confidence')}
        for key,ok in checks.items():
            if not ok:categories[key].append(a['id'])
        av=a['raw'].get('word_confidence',[]);bv=b['raw'].get('word_confidence',[])
        if len(av)==len(bv) and av:max_confidence_delta=max(max_confidence_delta,max(abs(x-y) for x,y in zip(av,bv)))
    return {'n':len(new),'all_agree':not any(categories.values()),'mismatch_counts':{k:len(v) for k,v in categories.items()},
        'mismatch_ids':categories,'maximum_confidence_absolute_difference':max_confidence_delta,
        'excluded_fields':['seconds','peak GPU bytes','runtime telemetry; execution identities are linked separately']}

def compare_states(a,b):
    import torch
    def detail(x,y):
        if not (torch.is_tensor(x) and torch.is_tensor(y)):return {'new':x,'historical':y}
        if x.shape!=y.shape or x.dtype!=y.dtype:return {'new_shape':list(x.shape),'historical_shape':list(y.shape),'new_dtype':str(x.dtype),'historical_dtype':str(y.dtype)}
        count=0;samples=[];maximum=0.
        for offset in range(0,x.numel(),1000000):
            u=x.reshape(-1)[offset:offset+1000000];v=y.reshape(-1)[offset:offset+1000000];idx=(u!=v).nonzero().reshape(-1)
            count+=idx.numel()
            if idx.numel():
                maximum=max(maximum,float((u.double()-v.double()).abs().max()))
                for j in idx[:max(0,8-len(samples))].tolist():samples.append({'flat_index':offset+j,'new':u[j].item(),'historical':v[j].item()})
        return {'different_elements':count,'maximum_absolute_difference':maximum,'first_differences':samples}
    state_hashes={k:{'new':fingerprint(a[k]),'historical_actual':fingerprint(b[k])} for k in COMPONENTS[:4]}
    model_keys=sorted(set(a['model'])|set(b['model']));model_diff=[]
    for k in model_keys:
        if k not in a['model'] or k not in b['model'] or fingerprint(a['model'][k])!=fingerprint(b['model'][k]):model_diff.append(k)
    optimizer_diff=[]
    for key in sorted(set(a['optimizer']['state'])|set(b['optimizer']['state'])):
        av=a['optimizer']['state'].get(key,{});bv=b['optimizer']['state'].get(key,{})
        for field in sorted(set(av)|set(bv)):
            if field not in av or field not in bv or fingerprint(av[field])!=fingerprint(bv[field]):optimizer_diff.append(f'state.{key}.{field}')
    groups_equal=fingerprint(a['optimizer']['param_groups'])==fingerprint(b['optimizer']['param_groups'])
    model_details={k:detail(a['model'].get(k),b['model'].get(k)) for k in model_diff}
    optimizer_details={}
    for path in optimizer_diff:
        _,index,field=path.split('.');key=int(index);optimizer_details[path]=detail(a['optimizer']['state'].get(key,{}).get(field),b['optimizer']['state'].get(key,{}).get(field))
    return {'component_signatures':state_hashes,'all_four_agree':all(v['new']==v['historical_actual'] for v in state_hashes.values()),
        'differing_model_keys':model_diff,'differing_optimizer_keys':optimizer_diff,'optimizer_parameter_groups_agree':groups_equal,
        'model_difference_details':model_details,'optimizer_difference_details':optimizer_details,
        'differing_scheduler_keys':[k for k in set(a['scheduler'])|set(b['scheduler']) if fingerprint(a['scheduler'].get(k))!=fingerprint(b['scheduler'].get(k))],
        'differing_rng_parts':[k for k in a['rng'] if fingerprint(a['rng'][k])!=fingerprint(b['rng'][k])],
        'progress_comparison':'Data order and scientific fields compared separately; timestamps/telemetry are not required equal.',
        'historical_embedded_expected_optimizer':b['state_digests']['optimizer'],'historical_cause':'unresolved'}

def main():
    import torch
    torch.set_num_threads(4);verify_plan();assert read_json(D/'REPLAY_BUDGET.json')['stages']['writer']['returncode']==0
    assert not (RUN/'RUNNING.lock').exists()
    verdicts=[]
    for p in sorted((RUN/'checkpoints').glob('generation_*.json')):
        if '.readback.' in p.name:continue
        proof,_=inspect(p);verdicts.append(proof)
    assert verdicts
    write_json(D/'audit/checkpoints_after_writer_exit.json',{'utc':utc(),'writer_exited':True,'checkpoints':verdicts,'all_verified':True})
    outcome=read_json(RUN/'attempts/1/outcome.json');complete=outcome['status']=='complete'
    preliminary={'utc':utc(),'internal_checkpoint_integrity':True,'writer_complete':complete,
        'selected_epoch':outcome['progress']['best_epoch'],'fresh_reload_eligible':complete and outcome['progress']['best_epoch']==5}
    if not preliminary['fresh_reload_eligible']:
        write_json(D/'audit/postwrite_result.json',preliminary);return
    new_meta=read_json(RUN/'checkpoints/epoch_05.json');new_path=RUN/'checkpoints'/new_meta['file']
    historical_path=PREVIOUS/'run/checkpoints/epoch_05.pt';historical_hash=file_hash(historical_path)
    assert historical_hash==read_json(PREVIOUS/'run/checkpoints/epoch_05.json')['sha256']
    a=torch.load(new_path,map_location='cpu',weights_only=False);b=torch.load(historical_path,map_location='cpu',weights_only=False)
    states=compare_states(a,b)
    assert file_hash(historical_path)==historical_hash
    trace=a['progress']['trace'];old_trace=b['progress']['trace']
    assert trace[:27000]==old_trace[:27000] and len(trace)==len(old_trace)==27935
    fields=['update','epoch','cursor_after','feature_ids_sha256','examples','valid_labels','learning_rate','next_learning_rate','loss','loss_sum','gradient_norm','padded_tokens']
    differences=[]
    for x,y in zip(trace[27000:],old_trace[27000:]):
        changed=[k for k in fields if x[k]!=y[k]]
        if changed:differences.append({'update':x['update'],'fields':changed,'new':{k:x[k] for k in changed},'historical':{k:y[k] for k in changed}})
    states['replay_trace_differences']=differences
    states['935_updates_and_order_verified']=len(trace[27000:])==935 and all(x['feature_ids_sha256']==y['feature_ids_sha256'] for x,y in zip(trace[27000:],old_trace[27000:]))
    write_json(D/'audit/historical_state_comparison.json',states)
    del a,b;gc.collect()
    new_rows=read_jsonl(RUN/'evaluations/epoch_05/predictions.jsonl');old_rows=read_jsonl(PREVIOUS/'run/evaluations/epoch_05/predictions.jsonl')
    predictions=compare_predictions(new_rows,old_rows)
    nm=read_json(RUN/'evaluations/epoch_05/metrics.json');om=read_json(PREVIOUS/'run/evaluations/epoch_05/metrics.json')
    predictions['scientific_metrics_match']=scientific_metrics(nm)==scientific_metrics(om)
    predictions['scientific_metric_differences']={k:{'new':scientific_metrics(nm).get(k),'historical':scientific_metrics(om).get(k)} for k in set(scientific_metrics(nm))|set(scientific_metrics(om)) if scientific_metrics(nm).get(k)!=scientific_metrics(om).get(k)}
    predictions['new_f1']=nm['f1'];predictions['historical_f1']=om['f1']
    write_json(D/'audit/historical_prediction_comparison.json',predictions)
    preliminary['historical_agreement']=states['all_four_agree'] and not differences and predictions['all_agree'] and predictions['scientific_metrics_match']
    preliminary['historical_cause']='unresolved';write_json(D/'audit/postwrite_result.json',preliminary)
    print(preliminary,flush=True)

if __name__=='__main__':main()
