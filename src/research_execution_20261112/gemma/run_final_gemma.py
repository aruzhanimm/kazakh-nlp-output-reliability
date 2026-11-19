"""Final Gemma ordinary4 on common panels, exclusively behind actual Freeze B.

Sequential independent greedy calls preserve the admitted train-pilot semantics.
Imports/metadata never read final data or load torch. No automatic retries.
"""
from pathlib import Path
import argparse,datetime,hashlib,importlib.metadata,json,os,sys,time,traceback

HERE=Path(__file__).resolve().parent
PROJECT=HERE.parents[1]
sys.path.insert(0,str(PROJECT/'ner_pipeline_20261111'))
from storage import canonical,digest,read_evidence,write_evidence,process_lock
sys.path.insert(0,str(PROJECT/'research_execution_20261112/final_evaluation'))
import output_adapter as adapter
from server_setup import REV,verified
from run_pilot8 import gpu_admission,EXPECTED_INPUT_SHA
from native_adapter import parse as parse_native

CONDITIONS={'ner':'ner_gemma4_12b_ordinary4','sa':'sa_gemma4_12b_ordinary4'}
POLICY={'model':'google/gemma-4-12B-it','revision':REV,'conditions':CONDITIONS,'shots':4,
        'dtype':'bfloat16','attention':'sdpa','do_sample':False,'enable_thinking':False,
        'seed':42,'max_new_tokens':{'ner':768,'sa':32},'input_ceiling':8192,
        'runtime_packages':{'transformers':'5.22.2','tokenizers':'0.23.1','safetensors':'0.8.0','huggingface-hub':'1.31.0','accelerate':'1.12.0','torchvision':'0.25.0+cu128','Pillow':'12.0.0'},
        'generation_workers':1,'batch_size':1,'request_order':'interleave_task_preserving_panel_and_window_order',
        'all_ner_windows_required':True,'panel_items_per_task':500,'automatic_retries':False,
        'sa_truncated_policy':'INVALID','ner_truncated_policy':'keep_valid_entities_flag_failure',
        'unexpected_nonempty_thinking':'save_evidence_stop_no_automatic_retry'}

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def sources():
    paths=[HERE/n for n in ['run_final_gemma.py','run_pilot8.py','server_setup.py','dependency_amendment.py','native_adapter.py','existing_alignment.py','existing_sa_scoring.py','HF_METADATA.json','INPUTS_8.json','RUNTIME_SOURCE_CHECK.json']]
    paths.append(PROJECT/'ner_pipeline_20261111/storage.py')
    return {**adapter.sources(),**{p.relative_to(PROJECT).as_posix():sha(p) for p in paths}}

def metadata():
    if sha(HERE/'INPUTS_8.json')!=EXPECTED_INPUT_SHA:raise ValueError('Fixed training messages changed')
    return {'status':'DRAFT_NOT_FINAL_AUTHORIZATION','gemma_policy':POLICY,'gemma_runtime_sources':sources(),
            'gemma_input_template_sha256':EXPECTED_INPUT_SHA,'no_final_data_opened':True,'no_model_or_GPU_loaded':True}

def freeze_admission(path,expected_sha,root):
    if sha(path)!=expected_sha:raise ValueError('Freeze B file SHA differs')
    f=read(path)
    if f.get('scope')!='final' or f.get('status')!='frozen_for_final_inference' or f.get('final_test_access_authorized')is not True or not f.get('decision_reference'):raise PermissionError('Actual Freeze B required before final panel access')
    if f.get('gemma_policy')!=POLICY or f.get('gemma_runtime_sources')!=sources() or f.get('gemma_input_template_sha256')!=EXPECTED_INPUT_SHA:raise ValueError('Frozen Gemma code/policy/templates differ')
    a=f.get('gemma_admission',{})
    if a.get('status')!='approved_final_gemma_inference' or not a.get('decision_reference') or a.get('pilot_technical_accepted')is not True:raise PermissionError('Actual technical pilot review required')
    if a.get('gpu_index')!=1 or not a.get('gpu_uuid') or Path(a.get('runtime_root','')).resolve()!=root:raise ValueError('Dedicated runtime/GPU admission differs')
    for name in ['RUNTIME_READY.json','RUNTIME_DEPENDENCY_AMENDMENT.json','CPU_INPUTS_READY.json','WEIGHTS_VERIFIED.json']:
        if sha(root/name)!=a['runtime_receipts'][name] or read(root/name)['status']!='PASS':raise ValueError('Runtime receipt differs: '+name)
    spec=a['pilot_complete'];pilot_path=Path(spec['path'])
    if not pilot_path.resolve().is_relative_to(root/'runs') or sha(pilot_path)!=spec['sha256']:raise ValueError('Actual pilot completion binding differs')
    pilot=read(pilot_path)
    if pilot.get('status')!='PASS_EXECUTION_NOT_QUALITY_ACCEPTANCE' or pilot.get('requests_completed')!=8 or pilot.get('no_dev_or_test_started')is not True:raise ValueError('Eight-call training pilot did not complete')
    protocol=read(pilot_path.parent/'PROTOCOL.json')
    if protocol['revision']!=REV or protocol['input_sha256']!=EXPECTED_INPUT_SHA or protocol['thinking']is not False or protocol['do_sample']is not False or protocol['dtype']!='bfloat16' or protocol['attention']!='sdpa' or protocol['max_new_tokens']!=POLICY['max_new_tokens']:raise ValueError('Final configuration differs from admitted pilot')
    if any(r.get('unexpected_nonempty_thinking') for r in pilot['outcomes']):raise ValueError('Unexpected thinking requires resolution before final')
    return f,a

def panel_read(path,spec,task,freeze_a_sha):
    if sha(path)!=spec['sha256']:raise ValueError('Server copy differs from common final panel')
    panel=read(path);ids=panel['expected_ids']
    if panel.get('format')!='encoder_common_view_panel_v1' or panel.get('split')!='test' or panel.get('task')!=task or panel.get('freeze_a_sha256')!=freeze_a_sha:raise ValueError('Common final panel provenance differs')
    if len(ids)!=500 or len(set(ids))!=500 or [x['id'] for x in panel['items']]!=ids or digest(ids)!=spec['ordered_ids_sha256'] or any(not x.startswith(task+':test:') for x in ids):raise ValueError('Exactly500 ordered unique final IDs required')
    for x in panel['items']:
        if any(k in x for k in ['label','labels','gold','gold_spans']):raise ValueError('Query gold forbidden in inference')
        if task=='ner':
            if any(w['tokens']!=x['tokens'][w['start']:w['end']] for w in x['windows']):raise ValueError('Shared NER window tokens differ')
            adapter.strict.merge_window_spans([(w['start'],w['end']) for w in x['windows']],[[] for _ in x['windows']],len(x['tokens']))
    return panel

def templates():
    if sha(HERE/'INPUTS_8.json')!=EXPECTED_INPUT_SHA:raise ValueError('Training input source changed')
    source=read(HERE/'INPUTS_8.json');prefixes={}
    for task in CONDITIONS:
        items=[x for x in source['items'] if x['task']==task and x['shots']==4]
        prefixes[task]=items[0]['messages'][:-1]
        if len(items)!=2 or any(x['messages'][:-1]!=prefixes[task] for x in items) or sum(m['role']=='assistant' for m in prefixes[task])!=4:raise ValueError('Identical fixed four-shot prefix required')
    return prefixes,source['allowed_ner_types']

def prepare_inputs(panels,processor,freeze_sha,freeze):
    prefixes,allowed=templates();by_task={}
    for task,panel in panels.items():
        jobs=[]
        for source in panel['items']:
            for wi in range(len(source['windows']) if task=='ner' else 1):
                text='Kazakh text:\n'+' '.join(source['windows'][wi]['tokens']) if task=='ner' else 'Review (JSON string):\n'+json.dumps(source['visible_text'],ensure_ascii=False)
                messages=[*prefixes[task],{'role':'user','content':text}]
                rendered=processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=False)
                inputs=processor.apply_chat_template(messages,tokenize=True,return_dict=True,return_tensors='pt',add_generation_prompt=True,enable_thinking=False)
                ids=inputs['input_ids'][0].tolist()
                if len(ids)>POLICY['input_ceiling']:raise ValueError('Final context exceeds frozen ceiling; no generation or silent truncation')
                identity={'freeze_b_sha256':freeze_sha,'panel_sha256':freeze['panels'][task]['sha256'],'id':source['id'],'task':task,'condition':CONDITIONS[task],'window_index':wi,'messages_sha256':digest(messages),'rendered_prompt_sha256':hashlib.sha256(rendered.encode()).hexdigest(),'input_token_ids_sha256':digest(ids)}
                jobs.append({**identity,'key':digest(identity),'messages':messages,'rendered_prompt':rendered,'input_ids':ids,**({'tokens':source['windows'][wi]['tokens']} if task=='ner' else {})})
        by_task[task]=jobs
    jobs=[]
    for i in range(max(map(len,by_task.values()))):
        for task in ['ner','sa']:
            if i<len(by_task[task]):jobs.append(by_task[task][i])
    return {'freeze_b_sha256':freeze_sha,'policy':POLICY,'jobs':jobs,'allowed_ner_types':allowed,'source_hashes':sources(),'request_count':len(jobs),'sample_count':1000}

def normalize(processor,ids,job,allowed,truncated):
    native=parse_native(processor,ids,job['input_ids'],job,allowed)
    if job['task']=='sa':
        parsed=adapter.parse_final_sa(native['task_text'],truncated=truncated,refusal=False,runtime_error=False)
    else:
        parsed={**native['task_parsed'],'failed':bool(native['task_parsed']['failed'] or truncated)}
    return {'native':native,'parsed':parsed,'truncated':truncated,'refusal':False,
            'refusal_policy':'No unvalidated natural-language refusal classifier; non-JSON refusal is a parse failure',
            'runtime_error':False,'terminal_failed':False,'unexpected_nonempty_thinking':native['unexpected_nonempty_thinking']}

def completed(folder,job,processor,allowed):
    if not (folder/'COMPLETE.json').exists():return None
    req=read_evidence(folder/'REQUEST.json');raw=read_evidence(folder/'RESPONSE_IDS.json');record=read_evidence(folder/'PARSED.json');done=read_evidence(folder/'COMPLETE.json')
    if digest(req)!=digest(job) or done['request_sha256']!=digest(job) or done['raw_ids_sha256']!=digest(raw):raise ValueError('Saved prompt/response identity differs')
    rebuilt=normalize(processor,raw['generated_token_ids'],job,allowed,raw['truncated'])
    if digest(record)!=digest(rebuilt) or done['parsed_sha256']!=digest(record):raise ValueError('Saved native/task parsing differs')
    return record

def export(prepared,panels,freeze,root):
    jobs=prepared['jobs'];records={j['key']:read_evidence(root/'requests'/j['key']/'PARSED.json') for j in jobs}
    if any(not (root/'requests'/j['key']/'COMPLETE.json').exists() for j in jobs):raise ValueError('Incomplete final run cannot be scored/exported')
    for task,panel in panels.items():
        rows=[]
        for source in panel['items']:
            members=[j for j in jobs if j['task']==task and j['id']==source['id']];windows=[]
            for j in members:
                r=records[j['key']];prefix='requests/'+j['key']+'/'
                windows.append({'window_index':j['window_index'],'raw':r['native']['task_text'],'status':'completed','truncated':r['truncated'],'refusal':r['refusal'],'runtime_error':False,'request_identity':j['key'],'raw_request_refs':[prefix+n for n in ['REQUEST.json','RESPONSE_IDS.json','NATIVE_RAW_TEXT.json','PARSED.json']]})
            if task=='ner':row=adapter.assemble_final_ner(source['id'],source['tokens'],source['windows'],windows,prepared['allowed_ner_types'])
            else:
                if len(members)!=1:raise ValueError('SA request coverage differs')
                row={'id':source['id'],**records[members[0]['key']]['parsed'],'raw_request_refs':windows[0]['raw_request_refs']}
            row.update(condition=CONDITIONS[task],panel_sha256=freeze['panels'][task]['sha256']);rows.append(row)
        if [r['id'] for r in rows]!=panel['expected_ids']:raise ValueError('Final output coverage differs')
        write_evidence(root/(task.upper()+'_PREDICTIONS.json'),{'task':task,'condition':CONDITIONS[task],'panel_sha256':freeze['panels'][task]['sha256'],'freeze_b_sha256':prepared['freeze_b_sha256'],'rows':rows})

def execute(args):
    root=Path(args.root).resolve();output=Path(args.output).resolve()
    if sys.platform!='linux' or root!=Path('/home/jovyan/work/kazakh_gemma4_20261112') or Path(sys.prefix).resolve()!=root/'venv':raise PermissionError('Dedicated Linux Gemma runtime required')
    if output!=root/'final':raise PermissionError('Only the dedicated final evidence directory is admitted')
    freeze,admission=freeze_admission(args.freeze,args.freeze_sha256,root)
    # All actual final-data access begins below the checked Freeze B boundary.
    panels={task:panel_read(getattr(args,task+'_panel'),freeze['panels'][task],task,freeze['freeze_a_sha256']) for task in ['ner','sa']}
    os.environ.update(CUDA_VISIBLE_DEVICES='1',HF_HOME=str(root/'hf_cache'),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_HUB_DISABLE_IMPLICIT_TOKEN='1',HF_HUB_DISABLE_TELEMETRY='1',PYTHONDONTWRITEBYTECODE='1',TOKENIZERS_PARALLELISM='false')
    output.mkdir(parents=True,exist_ok=True)
    with process_lock(output/'.run.lock'):
        if list((output/'requests').glob('*/ERROR.json')):raise RuntimeError('Prior error requires explicit reconciliation; no automatic retry')
        import torch,transformers
        from transformers import AutoProcessor,AutoModelForMultimodalLM
        if transformers.__version__!='5.22.2':raise ValueError('Pinned Transformers changed')
        for name,version in POLICY['runtime_packages'].items():
            if importlib.metadata.version(name)!=version:raise ValueError('Frozen package version differs: '+name)
        runtime=read(root/'RUNTIME_READY.json')
        if torch.__version__!=runtime['torch_reused']['torch_version'] or str(Path(torch.__file__).resolve())!=str(Path(runtime['torch_reused']['torch_file']).resolve()):raise ValueError('Verified parent torch changed')
        metadata=read(HERE/'HF_METADATA.json');files={x['rfilename']:x for x in metadata['siblings']};verification={}
        for name in ['config.json','generation_config.json','processor_config.json','chat_template.jinja','tokenizer.json','tokenizer_config.json']:
            verification[name]=verified(root/'model'/name,files[name])
        processor=AutoProcessor.from_pretrained(root/'model',local_files_only=True,trust_remote_code=False)
        prepared=prepare_inputs(panels,processor,args.freeze_sha256,freeze)
        write_evidence(output/'PREPARED.json',prepared);write_evidence(output/'FREEZE_BINDING.json',{'freeze_b_sha256':args.freeze_sha256,'admission':admission,'policy':POLICY})
        pending=[]
        for j in prepared['jobs']:
            folder=output/'requests'/j['key'];prior=completed(folder,j,processor,prepared['allowed_ner_types'])
            if prior is None:
                if folder.exists() and list(folder.iterdir()):raise RuntimeError('Unresolved prior request; no automatic repeat')
                pending.append(j)
        if not pending:
            export(prepared,panels,freeze,output);return {'status':'COMPLETE_REVERIFIED_NO_GENERATION','requests':len(prepared['jobs'])}
        gpu=gpu_admission(1)
        if gpu['uuid']!=admission['gpu_uuid']:raise ValueError('Physical GPU differs from frozen admission')
        verification['model.safetensors']=verified(root/'model/model.safetensors',files['model.safetensors'])
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():raise RuntimeError('BF16 CUDA required')
        torch.manual_seed(42);torch.cuda.manual_seed_all(42)
        start=time.perf_counter();model=AutoModelForMultimodalLM.from_pretrained(root/'model',dtype=torch.bfloat16,device_map={'':'cuda:0'},attn_implementation='sdpa',local_files_only=True,trust_remote_code=False);model.eval();torch.cuda.synchronize()
        eos=model.generation_config.eos_token_id;eos=set(eos if isinstance(eos,list) else [eos])
        execution_id=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
        write_evidence(output/'executions'/(execution_id+'.json'),{'utc':utc(),'gpu':gpu,'torch':torch.__version__,'transformers':transformers.__version__,'cuda_build':torch.version.cuda,'model_load_seconds':time.perf_counter()-start,'verified_model_files':verification,'pending_requests':len(pending)})
        for index,j in enumerate(pending):
            folder=output/'requests'/j['key'];write_evidence(folder/'REQUEST.json',j)
            try:
                inputs=processor.apply_chat_template(j['messages'],tokenize=True,return_dict=True,return_tensors='pt',add_generation_prompt=True,enable_thinking=False)
                if inputs['input_ids'][0].tolist()!=j['input_ids']:raise ValueError('Native input token drift')
                inputs=inputs.to('cuda:0');torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();start=time.perf_counter()
                with torch.inference_mode():tokens=model.generate(**inputs,max_new_tokens=POLICY['max_new_tokens'][j['task']],do_sample=False,temperature=None,top_p=None,top_k=None,use_cache=True)
                torch.cuda.synchronize();seconds=time.perf_counter()-start
                ids=tokens[0,len(j['input_ids']):].detach().cpu().tolist();cap=POLICY['max_new_tokens'][j['task']]
                truncated=len(ids)>=cap and (not ids or ids[-1] not in eos)
                raw={'generated_token_ids':ids,'input_tokens':len(j['input_ids']),'generated_tokens':len(ids),'truncated':truncated,'seconds':seconds,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved(),'execution_id':execution_id}
                write_evidence(folder/'RESPONSE_IDS.json',raw)  # Durably before decoding or parsing.
                text=processor.tokenizer.decode(ids,skip_special_tokens=False);write_evidence(folder/'NATIVE_RAW_TEXT.json',{'raw_native_text':text,'special_tokens_preserved':True})
                r=normalize(processor,ids,j,prepared['allowed_ner_types'],truncated);write_evidence(folder/'PARSED.json',r)
                if r['unexpected_nonempty_thinking']:raise RuntimeError('Unexpected thinking in disabled mode; preserve and investigate')
                write_evidence(folder/'COMPLETE.json',{'request_sha256':digest(j),'raw_ids_sha256':digest(raw),'parsed_sha256':digest(r),'model_revision':REV})
                if (index+1)%25==0:print(json.dumps({'completed_in_this_process':index+1,'remaining_in_this_process':len(pending)-index-1,'last_request_seconds':seconds}),flush=True)
            except Exception:
                write_evidence(folder/'ERROR.json',{'utc':utc(),'traceback':traceback.format_exc(),'terminal_failed':True,'automatic_retry':False});raise
        export(prepared,panels,freeze,output)
        result={'status':'COMPLETE_EXECUTION_NOT_QUALITY_JUDGEMENT','requests':len(prepared['jobs']),'sample_count':1000,'freeze_b_sha256':args.freeze_sha256,'source_hashes':sources()}
        write_evidence(output/'COMPLETE.json',result);return result

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['metadata','execute']);p.add_argument('--freeze');p.add_argument('--freeze-sha256');p.add_argument('--ner-panel');p.add_argument('--sa-panel');p.add_argument('--root');p.add_argument('--output');a=p.parse_args()
    if a.action=='metadata':r=metadata()
    else:
        if not all([a.freeze,a.freeze_sha256,a.ner_panel,a.sa_panel,a.root,a.output]):raise ValueError('Freeze B, panel-copy paths, dedicated root and output required')
        r=execute(a)
    print(json.dumps(r,ensure_ascii=True,indent=2))

if __name__=='__main__':main()
