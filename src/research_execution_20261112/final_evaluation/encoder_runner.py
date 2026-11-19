"""Selected-checkpoint inference only; all corpus/model access is behind an external gate.

No training, optimizer creation, automatic retries, CPU fallback or test selection.
Inputs are precomputed common views; tokenizer equality is checked, not redesigned.
"""
from pathlib import Path
import argparse
import datetime
import hashlib
import json
import os
import socket
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
sys.path.insert(0, str(PROJECT))
from pilot.inputs import first_word_predictions, owners, merge_window_tags
from pilot.scoring import bio_spans


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8')


def digest(value): return hashlib.sha256(canonical(value)).hexdigest()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b''): h.update(chunk)
    return h.hexdigest()


def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))


def write_new(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as f:
        f.write(canonical(value) + b'\n'); f.flush(); os.fsync(f.fileno())


def now(): return datetime.datetime.now(datetime.timezone.utc)


def aware(value):
    d = datetime.datetime.fromisoformat(value.replace('Z', '+00:00'))
    if d.tzinfo is None: raise ValueError('UTC-aware timestamp required')
    return d


def require(ok, message):
    if not ok: raise ValueError(message)


def sources():
    names = ['research_execution_20261112/final_evaluation/encoder_runner.py',
             'pilot/inputs.py', 'pilot/scoring.py', 'pilot/common.py', 'pilot/models.py',
             'stage_preflight/state.py', 'data/derived/ner_labels.json']
    return {name: sha(PROJECT/name) for name in names}


def check_gate(freeze, condition):
    """Check BEFORE opening a panel file; a draft never permits final-test access."""
    scope = freeze.get('scope')
    required = 'frozen_for_final_inference' if scope == 'final' else 'approved_dev_verification'
    require(scope in ('final', 'dev_verification') and freeze.get('status') == required,
            'External final freeze or explicit dev-verification gate required')
    require(bool(freeze.get('decision_reference')), 'Decision provenance missing')
    require(scope != 'final' or freeze.get('final_test_access_authorized') is True,
            'Final test is not authorized by this gate')
    require(condition in freeze.get('encoder_conditions', []), 'Condition absent from gate')
    require(freeze.get('runtime_sources') == sources(), 'Runtime source hashes changed')
    return scope


def bound_file(spec):
    p = Path(spec['path'])
    require(sha(p) == spec['sha256'], 'Bound file hash mismatch: ' + str(p))
    return p


def validate_registry(registry):
    rows = registry['rows']
    expected = {(t,m,s) for t in ('ner','sa') for m in ('xlmr','mbert') for s in (13,42,2026)}
    require(len(rows) == 12 and {(r['task'],r['model'],r['seed']) for r in rows} == expected,
            'All twelve accepted task/model/seed identities required')
    require(all(len(r['recorded_sha256']) == 64 for r in rows), 'Checkpoint hashes required')
    return {f"{r['task']}_{r['model']}_seed{r['seed']}": r for r in rows}


def validate_panel(panel, task, encoder, scope, labels):
    require(panel.get('format') == 'encoder_common_view_panel_v1' and panel['task'] == task,
            'Unknown common-view panel')
    split = 'test' if scope == 'final' else 'valid'
    ids = panel['expected_ids']; items = panel['items']
    require(panel['split'] == split and ids and len(set(ids)) == len(ids), 'Invalid split or duplicate IDs')
    require(scope != 'final' or len(ids) == 500, 'Deadline final scope requires exactly 500 items per task')
    require(all(x.startswith(task+':'+split+':') for x in ids), 'Out-of-scope input ID')
    require([x['id'] for x in items] == ids, 'Panel item order/coverage mismatch')
    if task == 'sa': require(labels == ['negative','positive'], 'SA label order changed')
    else:
        require(len(labels) == len(set(labels)) == 51 and 'O' in labels, 'Original BIO label list required')
        b = {x[2:] for x in labels if x.startswith('B-')}; i = {x[2:] for x in labels if x.startswith('I-')}
        require(len(b) == 25 and b == i, 'Original 25-type BIO ontology required')
    for item in items:
        require(not any(k in item for k in ('label','labels','gold','gold_spans')), 'Query gold must be separate')
        features = item['features'][encoder]
        if task == 'ner':
            tokens, windows = item['tokens'], item['windows']
            require(bool(tokens) and len(features) == len(windows), 'Missing NER windows')
            for wi, w in enumerate(windows):
                require(0 <= w['start'] < w['end'] <= len(tokens) and w['tokens'] == tokens[w['start']:w['end']],
                        'Window differs from original token coordinates')
                f = features[wi]
                require(f['id'] == item['id']+':w'+str(wi) and f['source_id'] == item['id'] and f['window'] == wi,
                        'Feature/window identity differs')
                require(set(x for x in f['word_ids'] if x is not None) == set(range(len(w['tokens']))),
                        'Tokenizer word coverage differs')
            require(item.get('owners', owners(windows,len(tokens))) == owners(windows,len(tokens)),
                    'Encoder word ownership changed')
        else:
            require(len(features) == 1 and features[0]['id'] == item['id'], 'SA feature identity differs')
            require(item['text'].startswith(item['visible_text']), 'SA input is not a literal prefix')
        for f in features:
            require(not any(k in f for k in ('label','labels','gold')), 'Labels cannot enter inference features')
            require(0 < len(f['input_ids']) <= 384 and len(f['attention_mask']) == len(f['input_ids']),
                    'Encoder context/attention mask invalid')
            require(all(type(x) is int and x >= 0 for x in f['input_ids']), 'Invalid input token ID')
            require(all(x in (0,1) for x in f['attention_mask']), 'Invalid attention mask')
            if 'token_type_ids' in f: require(len(f['token_type_ids']) == len(f['input_ids']), 'Token type length differs')
            if task == 'ner': require(len(f['word_ids']) == len(f['input_ids']), 'Subword/word map length differs')
    return len(ids)


def assemble_ner(item, raw_windows, labels):
    require(len(raw_windows) == len(item['windows']), 'Missing window prediction')
    local = []
    for window, feature, result in zip(item['windows'], item['_features'], raw_windows):
        word_ids = first_word_predictions(feature['word_ids'], result['token_pred_ids'], len(window['tokens']))
        local.append([labels[i] for i in word_ids])
    tags = merge_window_tags(item['windows'], local, len(item['tokens']))
    spans, repairs = bio_spans(tags, repair=True)
    return {'spans': spans, 'word_tags': tags, 'word_pred_ids': [labels.index(t) for t in tags],
            'repairs': repairs, 'failed': False, 'invalid_candidates': 0}


def load_bound_inputs(freeze_path, freeze_sha256, condition):
    require(sha(freeze_path) == freeze_sha256, 'External gate digest differs')
    freeze = read(freeze_path); scope = check_gate(freeze, condition)
    registry = validate_registry(read(bound_file(freeze['selected_checkpoints'])))
    require(condition in registry, 'Unknown accepted checkpoint identity')
    chosen = registry[condition]
    bindings = read(bound_file(freeze['runtime_bindings']))
    binding = bindings['conditions'][condition]
    require(binding['checkpoint_sha256'] == chosen['recorded_sha256'], 'Runtime checkpoint is not selected')
    require(binding['task'] == chosen['task'] and binding['encoder'] == chosen['model'] and binding['seed'] == chosen['seed'],
            'Runtime condition identity differs')
    require(binding['attention'] in ('sdpa','eager'), 'Explicit accepted attention implementation required')
    if chosen['task']=='ner':
        require(binding['labels']==read(PROJECT/'data/derived/ner_labels.json')['bio'], 'NER output label order differs from accepted training mapping')
    panel_path = bound_file(freeze['panels'][chosen['task']])
    panel = read(panel_path)
    validate_panel(panel,chosen['task'],chosen['model'],scope,binding['labels'])
    require(digest(panel['expected_ids']) == freeze['panels'][chosen['task']]['ordered_ids_sha256'],
            'Ordered panel IDs differ')
    return freeze, chosen, binding, panel, {'freeze_sha256':freeze_sha256,'condition':condition,
        'panel_sha256':sha(panel_path),'checkpoint_sha256':chosen['recorded_sha256'],
        'runtime_bindings_sha256':freeze['runtime_bindings']['sha256'], 'runtime_sources':sources()}


def check_admission(admission, run, out):
    require(admission.get('status') == 'approved_encoder_inference' and admission.get('freeze_sha256') == run['freeze_sha256']
            and admission.get('condition') == run['condition'], 'Matching resource admission required')
    require(admission.get('hostname') == socket.gethostname() and admission.get('device') == 'cuda:0', 'Host/device differs')
    require(aware(admission['issued_utc']) <= now() < aware(admission['expires_utc']), 'Resource admission expired')
    require(Path(out).resolve() == Path(admission['output_directory']).resolve(), 'Output directory differs')
    require(admission.get('exclusive_gpu_observed') is True and str(admission.get('gpu_uuid','')).startswith('GPU-'),
            'Exclusive GPU observation required')
    return True


def check_gpu_processes(active_pids, observed_names, admission):
    """Default strict idle; Windows permits only explicitly observed GUI PID/name pairs.

    Names are read without command lines. A new PID, reused PID or unreadable name
    blocks admission. This does not kill, suspend or modify any existing process.
    """
    allowed = {int(k): v for k,v in admission.get('gpu_gui_baseline', {}).items()}
    gui_names = {'ChatGPT','chrome','Code','codex-computer-use','codex-computer-use-swift',
                 'copilotapp','copilotapphost','CrossDeviceResume','dwm','explorer',
                 'msedge','msedgewebview2','PhoneExperienceHost','SearchHost',
                 'ShellExperienceHost','ShellHost','StartMenuExperienceHost'}
    require(all(type(k) is int and k > 0 and v in gui_names for k,v in allowed.items()), 'Invalid GUI-only baseline')
    require(not allowed or admission.get('gpu_process_policy') == 'windows_explicit_gui_baseline', 'Explicit GUI policy required')
    for pid in active_pids:
        require(pid in allowed and observed_names.get(pid)==allowed[pid],
                'GPU has unknown/changed/unreadable process; no process is touched: '+str(pid))
    return True


def windows_process_names(pids):
    if not pids:return {}
    require(os.name=='nt', 'GUI process baseline is Windows-only')
    # IDs originate from nvidia-smi and are converted to integers before interpolation.
    command = 'Get-Process -Id '+','.join(str(int(x)) for x in pids)+' -ErrorAction Stop | Select-Object Id,ProcessName | ConvertTo-Json -Compress'
    values=json.loads(subprocess.check_output(['powershell','-NoProfile','-Command',command],text=True))
    if isinstance(values,dict):values=[values]
    return {int(x['Id']):x['ProcessName'] for x in values}


def prepare_model(binding, chosen, admission):
    # Scope/admission were checked before importing torch or accessing checkpoint bytes.
    os.environ['CUDA_VISIBLE_DEVICES'] = admission['gpu_uuid']
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    os.environ['HF_HUB_OFFLINE'] = '1'; os.environ['TRANSFORMERS_OFFLINE'] = '1'
    gpu = subprocess.check_output(['nvidia-smi','--query-gpu=uuid,memory.free','--format=csv,noheader,nounits'],text=True)
    matching = [line.split(',') for line in gpu.splitlines() if line.split(',')[0].strip() == admission['gpu_uuid']]
    require(len(matching)==1 and int(matching[0][1].strip())*1024**2 >= admission['minimum_free_bytes'], 'Admitted GPU memory unavailable')
    active = subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True)
    pids=[int(line.split(',')[1].strip()) for line in active.splitlines() if line.split(',')[0].strip()==admission['gpu_uuid']]
    names=windows_process_names(pids) if admission.get('gpu_gui_baseline') else {}
    check_gpu_processes(pids,names,admission)
    import torch
    import transformers
    from transformers import AutoConfig, AutoModelForTokenClassification, AutoModelForSequenceClassification, AutoTokenizer
    from stage_preflight.state import fingerprint
    require(torch.cuda.is_available() and torch.cuda.device_count()==1 and torch.cuda.is_bf16_supported(), 'One BF16 GPU required; no CPU fallback')
    require(torch.__version__ == binding['environment']['torch'] and transformers.__version__ == binding['environment']['transformers'],
            'Runtime versions differ from frozen binding')
    torch.set_num_threads(binding.get('cpu_threads',4)); torch.use_deterministic_algorithms(True,warn_only=False)
    torch.backends.cudnn.benchmark=False
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    tokenizer_folder=Path(binding['tokenizer']['path'])
    for name,expected in binding['tokenizer']['files'].items(): require(sha(tokenizer_folder/name)==expected,'Tokenizer asset changed: '+name)
    tok=AutoTokenizer.from_pretrained(tokenizer_folder,local_files_only=True,use_fast=True,trust_remote_code=False)
    require(tok.is_fast, 'Fast tokenizer required for fixed word alignment')
    config_path=bound_file(binding['config'])
    cfg=AutoConfig.from_pretrained(config_path,local_files_only=True,trust_remote_code=False)
    labels=binding['labels'];cfg.num_labels=len(labels);cfg.id2label=dict(enumerate(labels));cfg.label2id={x:i for i,x in enumerate(labels)}
    ctor=AutoModelForTokenClassification if chosen['task']=='ner' else AutoModelForSequenceClassification
    model=ctor.from_config(cfg,attn_implementation=binding['attention'])
    checkpoint=Path(binding['checkpoint_path']);require(sha(checkpoint)==chosen['recorded_sha256'],'Selected checkpoint bytes differ')
    meta=read(bound_file(binding['checkpoint_metadata']))
    require(meta['sha256']==chosen['recorded_sha256'] and meta['epoch']==chosen['selected_epoch'], 'Checkpoint sidecar differs')
    require(meta['file']==checkpoint.name and meta['bytes']==checkpoint.stat().st_size, 'Checkpoint file/size differs')
    # Trusted own checkpoint, verified SHA before pickle deserialization. mmap avoids loading optimizer pages.
    state=torch.load(checkpoint,map_location='cpu',weights_only=False,mmap=True)
    require(state['run_hash']==meta['run_hash'] and state['state_digests']['model']==meta['state_digests']['model'], 'State identity differs')
    expected=meta['state_digests']['model'];require(fingerprint(state['model'])==expected,'Stored model signature differs')
    model.load_state_dict(state['model'],strict=True);del state
    require(fingerprint(model.state_dict())==expected,'Restored model differs')
    model.eval();model.requires_grad_(False);model.gradient_checkpointing_disable();model.to('cuda:0')
    return model,tok,expected,{'torch':torch.__version__,'transformers':transformers.__version__,
        'cuda':torch.version.cuda,'gpu_uuid':admission['gpu_uuid'],'gpu_name':torch.cuda.get_device_name(0),
        'precision':'FP32 stored parameters with BF16 autocast; FP32 exported logits',
        'attention':binding['attention'],'optimizer_created':False,'checkpoint_sha256':chosen['recorded_sha256']}


def verify_features(panel,encoder,tok):
    """Check prepared features against the pinned tokenizer; no changed input selection."""
    for item in panel['items']:
        for i,f in enumerate(item['features'][encoder]):
            enc=tok(item['windows'][i]['tokens'],is_split_into_words=True,truncation=False) if panel['task']=='ner' else tok(item['visible_text'],truncation=False)
            for key in ('input_ids','attention_mask','token_type_ids'):
                require(f.get(key)==dict(enc).get(key),'Prepared tokenizer feature differs: '+item['id']+'/'+key)
            if panel['task']=='ner': require(f['word_ids']==enc.word_ids(),'Prepared word IDs differ')


def forward(model,feature,task,labels):
    import torch
    inputs={k:torch.tensor([feature[k]],device='cuda:0') for k in ('input_ids','attention_mask','token_type_ids') if k in feature}
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16): z=model(**inputs).logits.float()[0]
    require(bool(torch.isfinite(z).all()),'Nonfinite logits')
    require(tuple(z.shape)==((len(feature['input_ids']),len(labels)) if task=='ner' else (2,)), 'Unexpected logits shape')
    p=z.softmax(-1)
    return {'logits':z.cpu().tolist(),'probabilities':p.cpu().tolist(),
            'token_pred_ids':z.argmax(-1).cpu().tolist() if task=='ner' else None,
            'label':int(z.argmax(-1).item()) if task=='sa' else None}


def execute(freeze,chosen,binding,panel,run,admission,out):
    check_admission(admission,run,out)
    require(len(panel['items'])<=admission['maximum_items'],'Admitted item bound exceeded')
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    lock=out/'PROCESS.lock'
    with lock.open('x',encoding='utf-8') as f:f.write(str(os.getpid()))
    # Lock remains after unexpected process death; reconcile rather than silently restart.
    try:
        identity={**run,'scope':freeze['scope'],'ordered_ids_sha256':digest(panel['expected_ids']),
                  'model':chosen['model'],'task':chosen['task'],'seed':chosen['seed']}
        identity['run_sha256']=digest(identity)
        if (out/'RUN.json').exists():require(read(out/'RUN.json')==identity,'Existing output belongs to another run')
        else:write_new(out/'RUN.json',identity)
        require(not (out/'COMPLETE.json').exists(),'Already complete; do not repeat inference')
        for item in panel['items']:
            folder=out/'items'/digest(item['id'])
            require(not ((folder/'STARTED.json').exists() and not (folder/'OUTPUT.json').exists()),'Unresolved prior item; no automatic retry')
            require(not (folder/'FAILURE.json').exists(),'Prior runtime failure requires explicit resolution')
        model,tok,model_signature,environment=prepare_model(binding,chosen,admission)
        verify_features(panel,chosen['model'],tok)
        import torch
        from stage_preflight.state import fingerprint
        session=out/'sessions'/now().strftime('%Y%m%dT%H%M%S%fZ')
        write_new(session/'ENVIRONMENT.json',environment)
        results=[]
        for item in panel['items']:
            check_admission(admission,run,out)
            require(now()<aware(admission['hard_deadline_utc']),'Hard deadline reached; run remains incomplete')
            folder=out/'items'/digest(item['id']); output=folder/'OUTPUT.json'
            if output.exists():
                record=read(output);expected=record['record_sha256']
                require(expected==digest({k:v for k,v in record.items() if k!='record_sha256'}) and
                        record['run_sha256']==identity['run_sha256'] and record['input_sha256']==digest(item), 'Saved item differs')
                require((folder/'STARTED.json').exists(),'Saved output lacks start provenance')
                results.append(record);continue
            write_new(folder/'STARTED.json',{'id':item['id'],'utc':now().isoformat(),'run_sha256':identity['run_sha256'],'input_sha256':digest(item)})
            try:
                torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
                raw=[forward(model,f,chosen['task'],binding['labels']) for f in item['features'][chosen['model']]]
                if chosen['task']=='ner': parsed=assemble_ner({**item,'_features':item['features'][chosen['model']]},raw,binding['labels'])
                else:parsed={'label':raw[0]['label'],'failed':False}
                torch.cuda.synchronize()
                record={'id':item['id'],'condition':run['condition'],'run_sha256':identity['run_sha256'],
                    'input_sha256':digest(item),'status':'completed','parsed':parsed,'raw_windows':raw,
                    'input_token_count':sum(len(f['input_ids']) for f in item['features'][chosen['model']]),
                    'windows':len(raw),'seconds':time.perf_counter()-start,
                    'peak_gpu_allocated_bytes':torch.cuda.max_memory_allocated(),
                    'peak_gpu_reserved_bytes':torch.cuda.max_memory_reserved(),
                    'cost_scope':'diagnostic batch1 pretokenized forward+CPU export+merge; not dedicated end-to-end benchmark'}
                record['record_sha256']=digest(record);write_new(output,record);results.append(record)
            except Exception as error:
                write_new(folder/'FAILURE.json',{'id':item['id'],'utc':now().isoformat(),'exception_type':type(error).__name__,
                    'message':str(error),'run_sha256':identity['run_sha256'],'status':'unresolved_runtime_failure_no_automatic_retry'})
                raise
        require([r['id'] for r in results]==panel['expected_ids'],'Incomplete or reordered final coverage')
        require(fingerprint(model.state_dict())==model_signature,'Model changed during inference')
        write_new(out/'COMPLETE.json',{'status':'complete_predictions_pending_independent_scoring',
            'run_sha256':identity['run_sha256'],'n':len(results),'ordered_ids':panel['expected_ids'],
            'record_sha256':[r['record_sha256'] for r in results],'model_signature_unchanged':True,
            'optimizer_updates':0,'training_started':False,'scientific_metrics_computed':False,
            'environment':environment,'utc':now().isoformat()})
        print(json.dumps({'status':'complete_predictions_pending_independent_scoring','items':len(results),'output':str(out)}))
    finally:
        lock.unlink()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['validate','execute'])
    p.add_argument('--freeze',required=True);p.add_argument('--freeze-sha256',required=True);p.add_argument('--condition',required=True)
    p.add_argument('--admission');p.add_argument('--output');a=p.parse_args()
    freeze,chosen,binding,panel,run=load_bound_inputs(a.freeze,a.freeze_sha256,a.condition)
    if a.mode=='validate':print(json.dumps({'status':'metadata_and_panel_validated_no_model_load','items':len(panel['items']),'condition':a.condition}));return
    require(a.admission and a.output,'Explicit admission and output required')
    execute(freeze,chosen,binding,panel,run,read(a.admission),a.output)


if __name__=='__main__':main()
