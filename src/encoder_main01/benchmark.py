"""Fixed training-only throughput/memory comparison; no dev loader or score selection."""
import argparse
import os
import statistics
import time
from pilot.common import ROOT,read_json,write_json,write_jsonl,file_hash,digest
from .common import D,freeze,verify,identity
from .core import load_data,initialize,collator,order,update

SCHEDULE=[(2,1),(4,1),(8,1),(8,2),(4,2),(2,2)]

def prepare():
    data=load_data('train');indices=order(0,len(data))[:704]
    longest=sorted(range(len(data)),key=lambda i:(-len(data[i]['input_ids']),data[i]['id']))[:16]
    fixture={'ordered_feature_ids':[data[i]['id'] for i in indices],'ordered_indices':indices,
        'longest_distinct_indices':longest,'longest_feature_ids':[data[i]['id'] for i in longest],
        'longest_tokens':[len(data[i]['input_ids']) for i in longest],'repeated_longest_index':longest[0],
        'warmup_updates':4,'timed_updates':40,'memory_updates':2,'seed':13,'source_sha256':file_hash(ROOT/'stage_preflight/features/xlmr_ner_train.jsonl')}
    path=D/'benchmark_fixture.json'
    if path.exists():assert read_json(path)==fixture
    else:write_json(path,fixture)
    names=['encoder_main01/'+n for n in ['__init__.py','common.py','objective.py','core.py','checkpoints.py','benchmark.py','launch_benchmark.py',
        'PLAN.md','METHOD_DECISIONS.md','benchmark_fixture.json','audit/normalization_fp32.json']]
    names+=['stage_preflight/state.py','pilot/train_encoder.py','pilot/scoring.py','configs/models.lock.json','stage_preflight/views/manifest.json']
    print(freeze(names,'benchmark_plan.lock.json'))

def run(micro,repeat):
    verify('benchmark_plan.lock.json');assert (micro,repeat) in SCHEDULE
    import torch
    path=D/'benchmark'/f'micro_{micro}_repeat_{repeat}';assert not (path/'result.json').exists()
    fixture=read_json(D/'benchmark_fixture.json');data=load_data('train')
    assert file_hash(ROOT/'stage_preflight/features/xlmr_ner_train.jsonl')==fixture['source_sha256']
    identity(path,{'microbatch':micro,'accumulation':16//micro,'repeat':repeat,'task':'training_only_xlmr_ner',
        'objective':'sum_CE/complete_optimizer_batch_valid_first_subword_count','seed':13,'precision':'BF16 autocast / FP32 parameters',
        'gradient_checkpointing':True,'attention':'sdpa','optimizer':'AdamW','learning_rate':1e-5,'weight_decay':.01,
        'scheduler_horizon':27935,'warmup_steps':1676,'fixture_sha256':file_hash(D/'benchmark_fixture.json')},'benchmark_plan.lock.json')
    began=time.perf_counter();model,tok,optimizer,scheduler=initialize();torch.cuda.synchronize();load_seconds=time.perf_counter()-began
    c=collator(tok);steps=[];peak=reserved=0;min_free=torch.cuda.mem_get_info()[0];total=torch.cuda.get_device_properties(0).total_memory
    groups=[('warmup' if i<4 else 'timed',fixture['ordered_indices'][i*16:(i+1)*16]) for i in range(44)]
    groups += [('longest_distinct',fixture['longest_distinct_indices']),('longest_repeated',[fixture['repeated_longest_index']]*16)]
    try:
        for i,(phase,positions) in enumerate(groups):
            torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();started=time.perf_counter()
            result=update(model,optimizer,scheduler,[data[j] for j in positions],micro,c)
            torch.cuda.synchronize();seconds=time.perf_counter()-started
            row={'update':i+1,'phase':phase,'seconds':seconds,**result,'feature_ids':[data[j]['id'] for j in positions],
                'peak_gpu_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_gpu_reserved_bytes':torch.cuda.max_memory_reserved(),
                'free_gpu_bytes_after_update':torch.cuda.mem_get_info()[0]}
            steps.append(row);peak=max(peak,row['peak_gpu_allocated_bytes']);reserved=max(reserved,row['peak_gpu_reserved_bytes']);min_free=min(min_free,row['free_gpu_bytes_after_update'])
            write_jsonl(path/'steps.jsonl',steps)
        timed=[s['seconds'] for s in steps if s['phase']=='timed']
        result={'status':'complete','microbatch':micro,'repeat':repeat,'updates':len(steps),'timed_updates':len(timed),
            'timed_seconds':sum(timed),'features_per_second':640/sum(timed),'median_step_seconds':statistics.median(timed),
            'peak_gpu_allocated_bytes':peak,'peak_gpu_reserved_bytes':reserved,'minimum_free_gpu_bytes':min_free,'physical_gpu_bytes':total,
            'headroom_pass':reserved<=.8*total and min_free>=2*2**30,'model_load_seconds':load_seconds,
            'gpu':torch.cuda.get_device_name(),'long_shape_checks_passed':2}
    except BaseException as error:
        result={'status':'failed','microbatch':micro,'repeat':repeat,'completed_updates':len(steps),'error_type':type(error).__name__,'error':str(error)}
        write_json(path/'result.json',result);raise
    write_json(path/'result.json',result);print(result,flush=True)

def choose():
    verify('benchmark_plan.lock.json');records={m:[read_json(D/'benchmark'/f'micro_{m}_repeat_{r}/result.json') for r in [1,2]] for m in [2,4,8]}
    eligible={};summary={}
    for m,rs in records.items():
        good=all(r['status']=='complete' and r['headroom_pass'] for r in rs)
        rates=[r['features_per_second'] for r in rs] if good else []
        stable=good and max(rates)/min(rates)<=1.10
        summary[str(m)]={'results':rs,'eligible':stable,'median_features_per_second':statistics.median(rates) if rates else None}
        if stable:eligible[m]=statistics.median(rates)
    assert 2 in eligible,'Validated baseline failed the new objective/long-shape/stability gate; main blocked'
    base=eligible[2];base_fast=max(r['features_per_second'] for r in records[2])
    improvements=[m for m in [4,8] if m in eligible and eligible[m]>=1.10*base and min(r['features_per_second'] for r in records[m])>=1.05*base_fast]
    chosen=max(improvements,key=lambda m:eligible[m]) if improvements else 2
    decision={'microbatch':chosen,'gradient_accumulation':16//chosen,'effective_batch':16,'objective':'sum_CE/complete_optimizer_batch_valid_first_subword_count',
        'selection':'Training-only repeated throughput and memory gates; no development F1','baseline_retained':chosen==2,'summary':summary,
        'benchmark_plan_sha256':verify('benchmark_plan.lock.json')}
    path=D/'chosen_execution.json'
    if path.exists():assert read_json(path)==decision
    else:write_json(path,decision)
    print({'chosen':chosen,'rates':eligible,'baseline_retained':chosen==2})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');p.add_argument('--choose',action='store_true');p.add_argument('--microbatch',type=int,choices=[2,4,8]);p.add_argument('--repeat',type=int,choices=[1,2]);a=p.parse_args()
    if a.prepare:prepare()
    elif a.choose:choose()
    else:run(a.microbatch,a.repeat)
