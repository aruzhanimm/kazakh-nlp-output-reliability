"""Read-only final evidence reconstruction. Import performs no corpus I/O."""
from pathlib import Path
import hashlib,json,sys
import numpy as np
HERE=Path(__file__).resolve().parent;PROJECT=HERE.parents[1]
sys.path.insert(0,str(PROJECT/'ner_pipeline_20261111'))
from storage import canonical,digest
sys.path.insert(0,str(PROJECT/'research_execution_20261112/final_evaluation'))
import encoder_runner as encoder
import output_adapter as adapter
import project_freeze_b as projection
sys.path.insert(0,str(PROJECT/'research_execution_20261112/final_evaluation/llm'))
import final_local_llm as local
sys.path.insert(0,str(PROJECT/'research_execution_20261112/api/final_api'))
import final_api as api

class Reader:
    def __init__(self):self.files={}
    def read(self,path,expected=None):
        p=Path(path).resolve();raw=p.read_bytes();h=hashlib.sha256(raw).hexdigest()
        if expected is not None and h!=expected:raise ValueError('Bound evidence file hash differs: '+str(p))
        self.files[str(p)]=h;v=json.loads(raw)
        if isinstance(v,dict) and set(v)=={'payload','payload_sha256'}:
            if digest(v['payload'])!=v['payload_sha256']:raise ValueError('Evidence envelope corrupt')
            v=v['payload']
        return v
    def bound(self,spec):return self.read(spec['path'],spec['sha256'])
    def jsonl(self,path,expected):
        p=Path(path).resolve();raw=p.read_bytes();h=hashlib.sha256(raw).hexdigest()
        if h!=expected:raise ValueError('Raw generation trace hash differs')
        self.files[str(p)]=h
        return [json.loads(line) for line in raw.decode('utf-8').splitlines()]

def same(a,b,message):
    if canonical(a)!=canonical(b):raise ValueError(message)

def host_freeze(reader,spec,master,master_sha):
    f=reader.bound(spec)
    if spec['sha256']!=master_sha:projection.verify(master,master_sha,f)
    elif f!=master:raise ValueError('Scientific master mismatch')
    return f

def compare_prediction(saved,rebuilt):
    for name in ['id','spans','label','failed','truncated','refusal','terminal_failed']:
        if name in rebuilt:same(saved.get(name),rebuilt[name],'Saved prediction differs from raw reconstruction: '+name)

def predictions_index(pred,condition,panel,panel_sha,freeze_sha):
    if pred['condition']!=condition or pred['panel_sha256']!=panel_sha or pred.get('freeze_b_sha256',pred.get('freeze_sha256'))!=freeze_sha:raise ValueError('Prediction execution/panel/condition differs')
    if [r['id'] for r in pred['rows']]!=panel['expected_ids']:raise ValueError('Prediction ordered IDs differ')
    for r in pred['rows']:
        if r['condition']!=condition or r['panel_sha256']!=panel_sha:raise ValueError('Prediction row provenance differs')
    return {r['id']:r for r in pred['rows']}

def verify_encoder_logits(raw,feature,task,nlabels):
    z=np.asarray(raw['logits'],dtype=np.float64);p=np.asarray(raw['probabilities'],dtype=np.float64)
    shape=(len(feature['input_ids']),nlabels) if task=='ner' else (2,)
    if z.shape!=shape or p.shape!=shape or not np.all(np.isfinite(z)) or not np.all(np.isfinite(p)):raise ValueError('Encoder raw logits/probabilities invalid')
    q=np.exp(z-z.max(axis=-1,keepdims=True));q/=q.sum(axis=-1,keepdims=True)
    if not np.allclose(q,p,atol=2e-6,rtol=2e-5):raise ValueError('Encoder probabilities differ from raw logits')
    wanted=z.argmax(axis=-1).tolist()
    same(wanted,raw['token_pred_ids'] if task=='ner' else raw['label'],'Encoder argmax differs from raw logits')

def encoder_rows(reader,condition,spec,panel,freeze):
    root=Path(spec['root']);run=reader.read(root/'RUN.json');complete=reader.bound(spec['complete']);bindings=reader.bound(spec['runtime_bindings'])
    if spec['runtime_bindings']['sha256']!=freeze['runtime_bindings']['sha256'] or run['condition']!=condition or run['panel_sha256']!=spec['panel_sha256'] or run['freeze_sha256']!=spec['execution_freeze']['sha256']:raise ValueError('Encoder runtime/run/panel binding differs')
    if freeze['runtime_sources']!=encoder.sources() or run['runtime_sources']!=encoder.sources():raise ValueError('Encoder reconstruction source differs from executed source')
    if complete['status']!='complete_predictions_pending_independent_scoring' or complete['ordered_ids']!=panel['expected_ids'] or complete['n']!=500 or complete['optimizer_updates']!=0 or complete['model_signature_unchanged']is not True:raise ValueError('Encoder completion incomplete')
    if run['run_sha256']!=digest({k:v for k,v in run.items() if k!='run_sha256'}) or complete['run_sha256']!=run['run_sha256']:raise ValueError('Encoder run digest differs')
    binding=bindings['conditions'][condition]
    if binding['checkpoint_sha256']!=run['checkpoint_sha256']:raise ValueError('Encoder selected checkpoint binding differs')
    rows=[];hashes=[]
    for item in panel['items']:
        folder=root/'items'/digest(item['id']);start=reader.read(folder/'STARTED.json');r=reader.read(folder/'OUTPUT.json')
        if r['id']!=item['id'] or r['condition']!=condition or r['status']!='completed' or r['run_sha256']!=run['run_sha256'] or r['input_sha256']!=digest(item) or start['input_sha256']!=digest(item) or start['run_sha256']!=run['run_sha256']:raise ValueError('Encoder raw input identity differs')
        if r['record_sha256']!=digest({k:v for k,v in r.items() if k!='record_sha256'}):raise ValueError('Encoder record digest differs')
        expected_windows=len(item['windows']) if panel['task']=='ner' else 1
        if len(r['raw_windows'])!=expected_windows or r['windows']!=expected_windows:raise ValueError('Encoder raw window coverage differs')
        for raw,feature in zip(r['raw_windows'],item['features'][run['model']]):verify_encoder_logits(raw,feature,panel['task'],len(binding['labels']))
        if panel['task']=='ner':parsed=encoder.assemble_ner({**item,'_features':item['features'][run['model']]},r['raw_windows'],binding['labels'])
        else:
            raw=r['raw_windows'][0];parsed={'label':raw['label'],'failed':False}
            if type(parsed['label'])is not int or parsed['label']not in(0,1):raise ValueError('Invalid raw SA class')
        same(parsed,r['parsed'],'Encoder parsed output differs from raw windows')
        rows.append({'id':item['id'],**parsed,'truncated':False,'refusal':False,'terminal_failed':False,'seconds':r['seconds'],'peak_gpu_allocated_bytes':r['peak_gpu_allocated_bytes'],'peak_gpu_reserved_bytes':r['peak_gpu_reserved_bytes'],'input_token_count':r['input_token_count'],'cost_scope':r['cost_scope'],'raw_request_refs':[str(folder/'OUTPUT.json')]});hashes.append(r['record_sha256'])
    same(hashes,complete['record_sha256'],'Encoder COMPLETE records differ')
    return rows,{'level':'all_encoder_item_raw_windows_reconstructed','items':500,'weights_not_reloaded_by_analysis':True}

def local_window(reader,root,manifest,prompt,budget,config,freeze_sha):
    folder=root/'requests'/manifest['request_identity'];req=reader.read(folder/'REQUEST.json');r=reader.read(folder/'RECORD.json')
    same(req['manifest'],manifest,'Local request manifest mismatch');same(req['prepared_prompt'],prompt,'Local native prompt mismatch');same(req['context_budget'],budget,'Local context budget differs')
    same(req['backend_identity'],{'backend_config_sha256':digest(config),'freeze_b_sha256':freeze_sha},'Local actual backend binding differs')
    if req['backend_mode']!='local_hf':raise ValueError('Final local backend must be actual HF')
    local.execution._validate_prompt(prompt)
    ledger=[];last=None
    for index,attempt in enumerate(r['attempts'],1):
        if index!=1:raise ValueError('Final local generation has no automatic retries')
        stem=folder/'attempts'/f'{index:04d}';start=reader.read(stem.with_suffix('.started.json'));captured=reader.read(stem.with_suffix('.response.json'));outcome=reader.read(stem.with_suffix('.outcome.json'))
        identity=manifest['request_identity']+':'+str(index)
        if start['attempt_id']!=identity or captured['attempt_id']!=identity or start['request_identity']!=manifest['request_identity']:raise ValueError('Local attempt identity differs')
        last=captured['backend_response'];status=local.execution._classify(last)
        wanted={'attempt_id':identity,'status':status,'response_received':status!='transport_failure','started_utc':start['started_utc'],'captured_utc':captured['captured_utc'],'response_payload_sha256':digest(captured)}
        same(outcome,wanted,'Local outcome differs from raw response');same(attempt,outcome,'Local attempt ledger differs');ledger.append(outcome)
        m=last.get('metadata',{})
        if m.get('input_token_ids_sha256')!=manifest['input_token_ids_sha256'] or m.get('backend_config_sha256')!=req['backend_identity']['backend_config_sha256']:raise ValueError('Local actual backend input/config differs')
        obs=m['observation'];same(obs['generated_token_ids'],last['generated_token_ids'],'Local returned token IDs differ from observation');same(obs['raw'],last['raw'],'Local native raw text differs')
        steps=reader.jsonl(root/'traces'/(manifest['request_identity']+'.trace.jsonl'),m['trace_sha256'])
        same(steps,obs['steps'],'Local external trace differs from captured observation')
        same([x for s in steps for x in s['selected_token_ids']],last['generated_token_ids'],'Local trace output IDs differ')
        actual=dict(obs['trace_metadata']['actual_generation_config']);cap=budget['reserved_output_tokens'];length=actual.pop('max_length')
        if length!=len(prompt['input_token_ids'])+cap or actual!=config['generation_template']:raise ValueError('Local effective generation differs')
        if status=='completed':
            ids=last['generated_token_ids'];eos=config['generation_template']['eos_token_id']
            if not ids or len(ids)>cap or any(type(x)is not int or x<0 for x in ids) or (len(ids)<cap and ids[-1]not in eos):raise ValueError('Local generated output termination differs')
            if last['truncated']!=(len(ids)==cap and ids[-1]not in eos):raise ValueError('Local truncation flag differs from actual tokens')
    overflow=len(prompt['input_token_ids'])+budget['reserved_output_tokens']>budget['context_limit_tokens']
    if not ledger and not overflow:raise ValueError('Missing local actual response')
    same(r,local.execution._make_record(manifest,prompt,budget,ledger,last,overflow=overflow),'Local RECORD not reconstructible from raw response')
    measures=[]
    if last is not None:
        obs=last['metadata']['observation']
        measures=[{'request_identity':manifest['request_identity'],'input_tokens':len(prompt['input_token_ids']),'generated_tokens':len(last['generated_token_ids']),
                   **{k:obs[k] for k in ['seconds','peak_gpu_allocated_bytes','peak_gpu_reserved_bytes'] if k in obs},
                   'scope':'instrumented greedy generate with token trace, native decoding and CPU export; batch1; no clean speed ranking'}]
    return {**r,'runtime_error':bool(r.get('runtime_error') or r.get('execution_failure_reasons')),'raw_request_refs':[manifest['request_identity']],'measurements':measures}

def local_rows(reader,condition,spec,panel,freeze,allowed):
    root=Path(spec['root']);pack=reader.bound(spec['prepared_package']);pred=reader.bound(spec['predictions']);done=reader.bound(spec['complete']);binding=reader.read(root/'RUN_BINDING.json')
    if spec['prepared_package']['sha256']!=freeze['llm_prepared_packages'][condition]['sha256'] or pack['condition']!=condition or pack['panel_sha256']!=spec['panel_sha256'] or pred['panel_sha256']!=spec['panel_sha256'] or pred['freeze_sha256']!=spec['execution_freeze']['sha256']:raise ValueError('Local package/panel/freeze differs')
    if freeze['llm_runtime_sources']!=local.sources() or binding['runtime_sources']!=local.sources():raise ValueError('Local reconstruction source differs')
    if pack['runtime_sources']!=local.sources() or pack['binding_sha256']!=digest(pack['binding']) or pack['freeze_a_sha256']!=freeze['freeze_a_sha256']:raise ValueError('Local prepared provenance differs')
    if panel['task']=='ner':same(pack['allowed_types'],allowed,'Local ontology differs')
    if binding['freeze_sha256']!=spec['execution_freeze']['sha256'] or binding['package_sha256']!=spec['prepared_package']['sha256'] or binding['local_transport_retries']!=0:raise ValueError('Local run binding differs')
    if done['status']!='complete' or done['items']!=500 or done['requests']!=pack['request_count']:raise ValueError('Local completion incomplete')
    same(pack['expected_ids'],panel['expected_ids'],'Local source coverage differs')
    saved=predictions_index(pred,condition,panel,spec['panel_sha256'],spec['execution_freeze']['sha256']);rows=[];requests=0
    if len(pack['items'])!=500:raise ValueError('Local prepared items incomplete')
    config=pack['binding']['backend_config'];budget={'context_limit_tokens':config['context_limit'],'reserved_output_tokens':config['generation_template']['max_new_tokens']}
    for item,source in zip(pack['items'],panel['items']):
        if item['id']!=source['id']:raise ValueError('Local item order differs')
        wanted=local.request_manifest(item,pack['task'],pack['run_hash'],condition,item['prepared_prompts'],allowed)
        same(wanted,item['manifest'],'Local manifest changes')
        if len(item['manifest'])!=len(item['prepared_prompts']):raise ValueError('Local native/window coverage differs')
        records=[local_window(reader,root,m,p,budget,config,spec['execution_freeze']['sha256']) for m,p in zip(item['manifest'],item['prepared_prompts'])];requests+=len(records)
        if panel['task']=='ner':
            same(item['windows'],source['windows'],'Local windows differ from common panel');same(item['tokens'],source['tokens'],'Local original tokens differ')
            parsed=adapter.assemble_final_ner(item['id'],item['tokens'],item['windows'],records,allowed)
        else:
            if len(records)!=1 or item['visible_text']!=source['visible_text']:raise ValueError('Local SA input differs')
            parsed={'id':item['id'],**local.parse_sa_record(records[0])}
        compare_prediction(saved[item['id']],parsed)
        extra={'_fence_raw_windows':[{k:r[k] for k in ['window_index','raw','status','runtime_error','truncated','refusal','request_identity']} for r in records]} if panel['task']=='ner' and condition.endswith('_ordinary4') else {}
        rows.append({**saved[item['id']],**parsed,'measurements':[m for r in records for m in r['measurements']],**extra})
    if requests!=pack['request_count']:raise ValueError('Local total window coverage differs')
    return rows,{'level':'all_local_backend_attempts_trace_ID_config_invariants_and_task_parsing_reconstructed','items':500,'requests':requests,'native_token_generation_not_repeated':True,'full_logit_trace_replay':False}

def api_rows(reader,condition,spec,panel,freeze,allowed):
    root=Path(spec['root']);pack=reader.bound(spec['prepared_package']);pred=reader.bound(spec['predictions']);done=reader.bound(spec['complete']);run=reader.read(root/'BINDING.json');identity=reader.read(root/'MODEL_IDENTITY.json')
    if freeze['api_runtime_sources']!=api.sources() or pack['sources']!=api.sources():raise ValueError('API reconstruction source differs')
    same(pack['policy'],freeze['api_policy'],'API policy differs')
    same(pack['allowed_ner_types'],allowed,'API ontology differs')
    if pack['request_count']!=len(pack['requests']) or len({j['key'] for j in pack['requests']})!=len(pack['requests']):raise ValueError('API request accounting differs')
    if run['freeze_b_sha256']!=spec['execution_freeze']['sha256'] or pack['freeze_b_sha256']!=spec['execution_freeze']['sha256'] or run['package_sha256']!=pack['package_sha256'] or pack['package_sha256']!=digest({k:v for k,v in pack.items() if k!='package_sha256'}):raise ValueError('API run/package differs')
    if done['stopped']is not False or done['complete_requests']!=pack['request_count']:raise ValueError('API completion incomplete')
    if pred['panel_sha256']!=spec['panel_sha256'] or pack['panels'][panel['task']]['sha256']!=spec['panel_sha256']:raise ValueError('API common panel differs')
    saved=predictions_index(pred,condition,panel,spec['panel_sha256'],spec['execution_freeze']['sha256']);rows=[];_,templates=api.templates()
    for source in panel['items']:
        jobs=[x for x in pack['requests'] if x['task']==panel['task'] and x['id']==source['id']];records=[];measures=[]
        if len(jobs)!=(len(source['windows']) if panel['task']=='ner' else 1):raise ValueError('API missing required window')
        for j in jobs:
            same(j['request'],api.make_request(panel['task'],source,j['window_index'],templates[panel['task']]),'API prompt differs from frozen shared input/template')
            identity_fields={k:j[k] for k in ['freeze_b_sha256','panel_sha256','id','task','window_index','condition','request_sha256']}
            if j['request_sha256']!=digest(j['request']) or digest(identity_fields)!=j['key'] or j['condition']!=condition:raise ValueError('API request identity differs')
            if j['freeze_b_sha256']!=spec['execution_freeze']['sha256'] or j['panel_sha256']!=spec['panel_sha256']:raise ValueError('API window freeze/panel differs')
            folder=root/'items'/j['key'];start=reader.read(folder/'GENERATION_STARTED.json');raw=reader.read(folder/'HTTP_RESPONSE.json');record=reader.read(folder/'RECORD.json');complete=reader.read(folder/'COMPLETE.json');settle=reader.read(folder/'SETTLED.json');count=reader.read(folder/'COUNT_RESPONSE.json')
            same(start['request'],j['request'],'API request body differs');value=json.loads(raw['body_utf8'])
            if raw['status_code']!=200 or digest(raw)!=complete['http_sha256'] or digest(record)!=complete['record_sha256']:raise ValueError('API raw completion corrupt')
            same(identity,{'model':value.get('model'),'service_tier':value.get('service_tier'),'system_fingerprint':value.get('system_fingerprint')},'API identity drift')
            if value.get('model')!=api.MODEL or value.get('service_tier')!='default' or (value.get('reasoning')or{}).get('effort')!='low' or value.get('max_output_tokens')!=api.CAPS[panel['task']] or ((value.get('text')or{}).get('format')or{}).get('type')!='text':raise ValueError('Returned API configuration differs')
            same(api.usage(value,j,start['native_input_tokens']),settle,'API usage differs');same(record,api.normalize_response(value,j,allowed),'API parsed output differs')
            if json.loads(count['body_utf8'])['input_tokens']!=start['native_input_tokens']:raise ValueError('API native count differs')
            measures.append({'request_identity':j['key'],'input_tokens':settle['input_tokens'],'generated_tokens_including_reasoning':settle['output_tokens_including_reasoning'],
                'seconds':raw['seconds'],'count_endpoint_seconds':count['seconds'],'usage':settle['usage'],
                'conservative_charge_usd':settle['conservative_charge_usd'],'ordinary_uncached_rate_estimate_usd':settle['ordinary_uncached_rate_estimate_usd'],
                'scope':'HTTP request elapsed time including network; two concurrent workers; summed request times are not wall-clock makespan; costs are estimates, not invoice'})
            if panel['task']=='ner':same(j['tokens'],source['windows'][j['window_index']]['tokens'],'API window input differs')
            records.append({'window_index':j['window_index'],'raw':record['raw'],'status':'terminal_failure' if record['terminal_failure'] else 'completed','runtime_error':record['terminal_failure'],'truncated':record['truncated'],'incomplete':record['status']=='incomplete','refusal':record['refusal'],'request_identity':j['key']})
        parsed=adapter.assemble_final_ner(source['id'],source['tokens'],source['windows'],records,allowed) if panel['task']=='ner' else {'id':source['id'],**record['parsed']}
        compare_prediction(saved[source['id']],parsed);rows.append({**saved[source['id']],**parsed,'measurements':measures,**({'_fence_raw_windows':records} if panel['task']=='ner' else {})})
    return rows,{'level':'all_API_HTTP_usage_and_task_parsing_reconstructed','items':500,'provider_native_token_ids_available':False}

def gemma_rows(reader,condition,spec,panel,freeze,allowed):
    root=Path(spec['root']);pack=reader.bound(spec['prepared_package']);pred=reader.bound(spec['predictions']);done=reader.bound(spec['complete']);run=reader.read(root/'FREEZE_BINDING.json')
    if run['freeze_b_sha256']!=spec['execution_freeze']['sha256'] or pack['freeze_b_sha256']!=spec['execution_freeze']['sha256'] or pack['source_hashes']!=freeze['gemma_runtime_sources']:raise ValueError('Gemma run/source/freeze differs')
    same(pack['policy'],freeze['gemma_policy'],'Gemma policy differs')
    same(pack['allowed_ner_types'],allowed,'Gemma ontology differs')
    if pack['request_count']!=len(pack['jobs']) or len({j['key'] for j in pack['jobs']})!=len(pack['jobs']):raise ValueError('Gemma request accounting differs')
    if done['status']!='COMPLETE_EXECUTION_NOT_QUALITY_JUDGEMENT' or done['requests']!=pack['request_count'] or pred['panel_sha256']!=spec['panel_sha256']:raise ValueError('Gemma completion/panel differs')
    saved=predictions_index(pred,condition,panel,spec['panel_sha256'],spec['execution_freeze']['sha256']);rows=[]
    fixed=reader.read(PROJECT/'research_execution_20261112/gemma/INPUTS_8.json');prefixes={task:[x for x in fixed['items'] if x['task']==task and x['shots']==4][0]['messages'][:-1] for task in ['ner','sa']}
    for source in panel['items']:
        jobs=[j for j in pack['jobs'] if j['task']==panel['task'] and j['id']==source['id']];records=[];measures=[]
        if len(jobs)!=(len(source['windows']) if panel['task']=='ner' else 1):raise ValueError('Gemma missing required window')
        for j in jobs:
            text='Kazakh text:\n'+' '.join(source['windows'][j['window_index']]['tokens']) if panel['task']=='ner' else 'Review (JSON string):\n'+json.dumps(source['visible_text'],ensure_ascii=False)
            same(j['messages'],[*prefixes[panel['task']],{'role':'user','content':text}],'Gemma prompt differs from frozen shared input/template')
            identity_fields={k:j[k] for k in ['freeze_b_sha256','panel_sha256','id','task','condition','window_index','messages_sha256','rendered_prompt_sha256','input_token_ids_sha256']}
            if digest(identity_fields)!=j['key'] or digest(j['messages'])!=j['messages_sha256'] or digest(j['input_ids'])!=j['input_token_ids_sha256'] or hashlib.sha256(j['rendered_prompt'].encode()).hexdigest()!=j['rendered_prompt_sha256'] or j['condition']!=condition:raise ValueError('Gemma effective input identity differs')
            if j['freeze_b_sha256']!=spec['execution_freeze']['sha256'] or j['panel_sha256']!=spec['panel_sha256']:raise ValueError('Gemma window freeze/panel differs')
            if panel['task']=='ner':same(j['tokens'],source['windows'][j['window_index']]['tokens'],'Gemma query tokens differ')
            folder=root/'requests'/j['key'];request=reader.read(folder/'REQUEST.json');raw=reader.read(folder/'RESPONSE_IDS.json');native_text=reader.read(folder/'NATIVE_RAW_TEXT.json');r=reader.read(folder/'PARSED.json');complete=reader.read(folder/'COMPLETE.json')
            same(request,j,'Gemma request differs')
            if complete['request_sha256']!=digest(j) or complete['raw_ids_sha256']!=digest(raw) or complete['parsed_sha256']!=digest(r):raise ValueError('Gemma raw/prompt linkage corrupt')
            if complete['model_revision']!=freeze['gemma_policy']['revision']:raise ValueError('Gemma model revision differs')
            same(raw['generated_token_ids'],r['native']['raw_token_ids'],'Gemma native raw token IDs differ');same(native_text['raw_native_text'],r['native']['raw_native_text'],'Gemma raw native text differs')
            native=r['native']['native_parsed']
            if native.get('thinking') or native.get('tool_calls') or native.get('content')!=r['native']['task_text']:raise ValueError('Gemma nonthinking final channel differs')
            text=r['native']['task_text']
            parsed=adapter.parser.parse_mentions(text,j['tokens'],allowed) if panel['task']=='ner' else adapter.parse_final_sa(text,truncated=r['truncated'],refusal=False,runtime_error=False)
            if panel['task']=='ner':parsed={**parsed,'failed':bool(parsed['failed'] or r['truncated'])}
            same(parsed,r['parsed'],'Gemma task parse differs')
            if raw['truncated']!=r['truncated'] or raw['input_tokens']!=len(j['input_ids']) or raw['generated_tokens']!=len(raw['generated_token_ids']):raise ValueError('Gemma observed token counts/truncation differ')
            measures.append({'request_identity':j['key'],'input_tokens':raw['input_tokens'],'generated_tokens':raw['generated_tokens'],'seconds':raw['seconds'],
                'peak_gpu_allocated_bytes':raw['peak_allocated_bytes'],'peak_gpu_reserved_bytes':raw['peak_reserved_bytes'],
                'scope':'synchronized model.generate only; native decode and task parse excluded; batch1; no clean speed ranking'})
            records.append({'window_index':j['window_index'],'raw':text,'status':'completed','runtime_error':False,'truncated':r['truncated'],'refusal':False,'request_identity':j['key']})
        parsed=adapter.assemble_final_ner(source['id'],source['tokens'],source['windows'],records,allowed) if panel['task']=='ner' else {'id':source['id'],**r['parsed']}
        compare_prediction(saved[source['id']],parsed);rows.append({**saved[source['id']],**parsed,'measurements':measures,**({'_fence_raw_windows':records} if panel['task']=='ner' else {})})
    native_receipt=reader.bound(spec['native_cpu_reaudit']) if spec.get('native_cpu_reaudit') else None
    if native_receipt is not None and native_receipt.get('status')!='COMPLETE_REVERIFIED_NO_GENERATION':raise ValueError('Gemma native CPU replay did not pass')
    if native_receipt is not None and (native_receipt['freeze_b_sha256']!=spec['execution_freeze']['sha256'] or native_receipt['prepared_package_sha256']!=spec['prepared_package']['sha256'] or native_receipt['complete_sha256']!=spec['complete']['sha256'] or native_receipt['requests']!=pack['request_count']):raise ValueError('Gemma native replay refers to another run')
    return rows,{'level':'all_Gemma_raw_ID_native_text_linkage_and_task_parsing_reconstructed','items':500,'native_processor_replay_this_host':False,'server_native_cpu_reaudit_receipt':native_receipt,'limitation_if_absent':'Without a separate native CPU audit receipt, saved native-channel extraction is not independently re-executed here.'}
