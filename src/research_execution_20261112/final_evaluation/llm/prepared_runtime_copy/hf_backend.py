"""Explicitly admitted local NER inference; import is stdlib-only and inert.

No loader is invoked by importing this module. Real execution requires a fresh,
bounded train/dev admission and an otherwise unoccupied explicitly selected GPU.
"""
from pathlib import Path
import copy
import datetime
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import re
import socket
import subprocess
import time

TRACE_SHA256 = '1f0d0e4a90795efd8823e01a0620e0af616ba647458704bbb72f4cf564f2bed2'
SCHEMA_SETTINGS = {'any_whitespace': True, 'strict_mode': True, 'any_order': False}
MODELS = {'Qwen/Qwen2.5-7B-Instruct': 'a09a35458c702b33eeacc393d103063234e8bc28',
          'inception42/Llama-3.1-Sherkala-8B-Chat': 'ab873dfe8b5c86598ed3ea9dcc548eb47a1e228f'}
GENERATION_SHA256 = {
    'Qwen/Qwen2.5-7B-Instruct':'7624d1f282d7e5dd6da649010d3d7477ec552044032a44025e192fd6940fac18',
    'inception42/Llama-3.1-Sherkala-8B-Chat':'f19b7865d4eec4aada542abd73c4d94977e8c574082181f3fbdc6077bc39bbf7'}
ARTIFACT_MANIFEST_SHA256 = {
    'Qwen/Qwen2.5-7B-Instruct':'aa51783be69a016512860b4ef01c20b2716f6db022929f3d049e5de513a255c4',
    'inception42/Llama-3.1-Sherkala-8B-Chat':'1ed9ca02ac657799b7625a6871f79536b34d9f81890b5e17dd84af5c15ce10b6'}
PROPERTY_ORDER = ['mention','type','occurrence']
PREPARED_TENSOR_KEYS = {'_bos_token_tensor','_eos_token_tensor','_pad_token_tensor','_decoder_start_token_tensor'}
ENTITY_TYPES = ['ADAGE','ART','CARDINAL','CONTACT','DATE','DISEASE','EVENT','FACILITY','GPE',
                'LANGUAGE','LAW','LOCATION','MISCELLANEOUS','MONEY','NON_HUMAN','NORP','ORDINAL',
                'ORGANISATION','PERCENTAGE','PERSON','POSITION','PRODUCT','PROJECT','QUANTITY','TIME']


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'),
                      allow_nan=False).encode('utf-8')


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def _sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''): h.update(block)
    return h.hexdigest()


def _utc(value):
    result = datetime.datetime.fromisoformat(value.replace('Z', '+00:00'))
    if result.tzinfo is None: raise ValueError('Admission timestamps require a timezone')
    return result


def validate_config(config):
    required = {'model_id', 'revision', 'model_path', 'tokenizer_path', 'model_files',
                'tokenizer_files', 'chat_template', 'chat_template_sha256',
                'generation_template', 'raw_decode_kwargs', 'gpu_uuid', 'trace_dir',
                'context_limit', 'seed', 'memory_fraction', 'schema', 'packages'}
    if set(config) != required: raise ValueError('Backend config keys differ from explicit contract')
    if MODELS.get(config['model_id']) != config['revision']: raise ValueError('Unpinned model')
    if config['context_limit'] != 4096 or config['seed'] != 13: raise ValueError('Context/seed differs from protocol')
    if not 0 < config['memory_fraction'] <= .6: raise ValueError('GPU fraction must be explicit and <=0.6')
    if not re.fullmatch(r'GPU-[0-9a-fA-F-]+', config['gpu_uuid']): raise ValueError('Explicit physical GPU UUID required')
    if not config['chat_template'] or hashlib.sha256(config['chat_template'].encode()).hexdigest() != config['chat_template_sha256']:
        raise ValueError('Template digest mismatch')
    artifact_identity={k:config[k] for k in ('model_files','tokenizer_files','chat_template_sha256')}
    if digest(artifact_identity) != ARTIFACT_MANIFEST_SHA256[config['model_id']]:
        raise ValueError('Weights/tokenizer/template manifest differs from audited artifacts')
    if config['raw_decode_kwargs'] != {'skip_special_tokens': True, 'clean_up_tokenization_spaces': False}:
        raise ValueError('Raw decoding differs from pinned protocol')
    g = config['generation_template']
    if digest(g) != GENERATION_SHA256[config['model_id']]:
        raise ValueError('Full generation template differs from historical effective configuration')
    fixed = {'max_new_tokens': 768, 'do_sample': False, 'num_beams': 1,
             'use_cache': True, 'repetition_penalty': 1.05, 'num_return_sequences': 1}
    if 'max_length' in g or any(g.get(k) != v for k,v in fixed.items()): raise ValueError('Generation policy mismatch')
    expected = [151645,151643] if config['model_id'].startswith('Qwen/') else [128001,128009]
    bos = 151643 if config['model_id'].startswith('Qwen/') else 128000
    if g.get('eos_token_id') != expected or g.get('pad_token_id') != expected[0] or g.get('bos_token_id') != bos:
        raise ValueError('Native BOS/EOS/PAD mismatch')
    if g.get('stop_strings') is not None or g.get('max_time') is not None or g.get('return_dict_in_generate') is not False:
        raise ValueError('Unexpected termination/output option')
    schema = config['schema']
    if set(schema) != {'enabled','definition','definition_sha256','compile_settings','property_order'} or type(schema['enabled']) is not bool:
        raise ValueError('Explicit schema condition required')
    if schema['property_order'] != PROPERTY_ORDER: raise ValueError('Pinned schema output property order required')
    if digest(schema['definition']) != schema['definition_sha256'] or schema['compile_settings'] != SCHEMA_SETTINGS:
        raise ValueError('Schema digest/settings mismatch')
    d = schema['definition']; item = d.get('items', {}); props = item.get('properties', {})
    if d.get('type') != 'array' or item.get('type') != 'object' or item.get('additionalProperties') is not False or item.get('required') != ['mention','type','occurrence']:
        raise ValueError('NER schema structure changed')
    if set(d) != {'type','items'} or set(item) != {'type','properties','required','additionalProperties'}:
        raise ValueError('Unexpected schema controls')
    if set(props) != set(PROPERTY_ORDER) or props['mention'] != {'type':'string','minLength':1} or props['occurrence'] != {'type':'integer','minimum':1}:
        raise ValueError('NER schema fields changed')
    types = props['type'].get('enum', [])
    if set(props['type']) != {'type','enum'} or props['type'].get('type') != 'string' or types != ENTITY_TYPES:
        raise ValueError('Exact pinned 25-type ontology and order required')
    wanted = {'torch':'2.10.0+cu128', 'transformers':'4.57.6', 'tokenizers':'0.22.2'}
    if schema['enabled']: wanted['xgrammar'] = '0.2.9'
    if config['packages'] != wanted: raise ValueError('Runtime package pin differs')
    for group in ('model_files','tokenizer_files'):
        if not isinstance(config[group], dict) or not config[group]: raise ValueError('File manifest is empty')
        for name, v in config[group].items():
            if Path(name).name != name or '/' in name or '\\' in name or ':' in name:
                raise ValueError('Manifest must use plain filenames')
            if set(v) != {'sha256','bytes'} or type(v['bytes']) is not int or v['bytes'] < 1 or not re.fullmatch('[0-9a-f]{64}', v['sha256']):
                raise ValueError('Invalid file hash/size')
    if not {'config.json','generation_config.json','model.safetensors.index.json'} <= set(config['model_files']):
        raise ValueError('Incomplete model metadata manifest')
    if not any(n.endswith('.safetensors') for n in config['model_files']): raise ValueError('No pinned safetensors weights')
    if not {'tokenizer.json','tokenizer_config.json','special_tokens_map.json'} <= set(config['tokenizer_files']):
        raise ValueError('Incomplete tokenizer manifest')
    canonical(config)
    return True


def ordered_schema(config):
    """Canonical evidence storage sorts dict keys; the explicit list binds grammar order."""
    schema=copy.deepcopy(config['schema']['definition'])
    props=schema['items']['properties']
    schema['items']['properties']={name:props[name] for name in config['schema']['property_order']}
    return schema


def compile_ner_grammar(config, tokenizer_info):
    import xgrammar as xgr
    helper = Path(__file__).with_name('escape_safe_ner_grammar.py')
    if _sha(helper) != '0ccb5e46a05dee7146e0de875da758532e74e1b5036a9dc292692fb9ce33c33c':
        raise ValueError('Corrected grammar helper source changed')
    from escape_safe_ner_grammar import compile_escape_safe_ner
    compiled, receipt = compile_escape_safe_ner(xgr.GrammarCompiler(tokenizer_info, max_threads=4),
        ordered_schema(config), config['schema']['compile_settings'])
    receipt['helper_source_sha256'] = _sha(helper)
    return compiled, receipt


def build_generation_config(config, generation_class):
    """Check a pure GenerationConfig roundtrip before any model allocation or call."""
    template=config['generation_template']
    values={k:v for k,v in template.items() if k not in PREPARED_TENSOR_KEYS}
    generation=generation_class.from_dict(copy.deepcopy(values))
    actual=generation.to_dict()
    if actual.pop('max_length',None) != 20 or actual != values:
        raise ValueError('Installed GenerationConfig changes frozen generation values')
    return generation


def verify_admission(admission, config, item_id=None, *, now=None, hostname=None, environ=None):
    now = now or datetime.datetime.now(datetime.timezone.utc)
    hostname = hostname or socket.gethostname(); environ = os.environ if environ is None else environ
    if admission.get('status') != 'approved_train_dev_execution' or admission.get('task') != 'ner' or admission.get('split') not in ('train','valid'):
        raise PermissionError('Only explicitly admitted NER train/dev execution is allowed')
    if not (_utc(admission['issued_utc']) <= now <= _utc(admission['expires_utc'])): raise PermissionError('Admission is not current')
    if admission.get('backend_config_sha256') != digest(config): raise PermissionError('Admission belongs to another backend config')
    if not re.fullmatch('[0-9a-f]{64}', admission.get('panel_sha256','')) or not admission.get('condition'):
        raise PermissionError('Bounded condition and panel identity required')
    ids = admission.get('allowed_item_ids')
    if not isinstance(ids,list) or not ids or len(ids)!=len(set(ids)) or any(not re.fullmatch('ner:'+admission['split']+r':\d+', x) for x in ids):
        raise PermissionError('Admission requires an exact unique train/dev item list')
    if item_id is not None and item_id not in ids: raise PermissionError('Request is not admitted')
    if type(admission.get('maximum_model_calls')) is not int or admission['maximum_model_calls'] < 1:
        raise PermissionError('Explicit call limit required')
    resource = admission.get('resource', {})
    if resource.get('hostname') != hostname or resource.get('device') != 'cuda:0' or resource.get('gpu_uuid') != config['gpu_uuid'] or resource.get('no_sa_process_on_device') is not True:
        raise PermissionError('Resource isolation decision does not match this process')
    if environ.get('CUDA_VISIBLE_DEVICES') != config['gpu_uuid'] or environ.get('NER_ISOLATED_PROCESS') != '1':
        raise PermissionError('A dedicated explicitly isolated NER process is required')
    if environ.get('HF_HUB_OFFLINE') != '1' or environ.get('TRANSFORMERS_OFFLINE') != '1':
        raise PermissionError('Offline model loading is required')
    return True


def verify_files(folder, manifest, *, model=False):
    folder = Path(folder).resolve(); verified = {}
    for name, expected in manifest.items():
        path = folder/name
        if path.stat().st_size != expected['bytes'] or _sha(path) != expected['sha256']:
            raise ValueError('File verification failed: '+name)
        verified[name] = dict(expected)
    if model:
        index = json.loads((folder/'model.safetensors.index.json').read_text())
        referenced = set(index['weight_map'].values())
        if not referenced or not referenced <= set(manifest) or any(not x.endswith('.safetensors') for x in referenced):
            raise ValueError('Model index references unverified weight shards')
        actual = {x.name for x in folder.glob('*.safetensors')}
        if actual != referenced: raise ValueError('Unpinned or missing safetensors shard')
    else:
        assets = {'tokenizer.json','tokenizer_config.json','special_tokens_map.json','added_tokens.json',
                  'chat_template.jinja','vocab.json','vocab.txt','merges.txt','config.json',
                  'sentencepiece.bpe.model','spiece.model'}
        if not {n for n in assets if (folder/n).is_file()} <= set(manifest):
            raise ValueError('Unpinned loader-relevant tokenizer asset')
    return verified


def verify_gpu_exclusive(gpu_uuid, *, own_pid=None, run=subprocess.run):
    """Read-only check; no processes are killed or paused. Namespace mismatch fails closed."""
    own_pid = os.getpid() if own_pid is None else own_pid
    inventory = run(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader,nounits'],
                    capture_output=True,text=True,check=True,timeout=10)
    physical = [x.strip() for x in inventory.stdout.splitlines() if x.strip()]
    if gpu_uuid not in physical: raise PermissionError('Admitted physical GPU is absent from current inventory')
    result = run(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],
                 capture_output=True,text=True,check=True,timeout=10)
    seen = []
    for line in result.stdout.splitlines():
        if not line.strip(): continue
        fields = [x.strip() for x in line.split(',')]
        if len(fields) != 2: raise PermissionError('Cannot establish GPU process isolation')
        if fields[0] == gpu_uuid:
            pid = int(fields[1]); seen.append(pid)
            if pid != own_pid: raise PermissionError('Another compute process occupies the admitted GPU')
    return {'gpu_uuid':gpu_uuid,'own_pid':own_pid,'visible_compute_pids':seen,'physical_gpu_verified':True}


def execution_failures(steps, constrained):
    reasons = []
    for step in steps:
        raw, processed = step['raw'], step['processed']
        if any(raw[x] for x in ('nan','positive_inf','negative_inf')) or not raw['finite']:
            reasons.append('unexpected_raw_nonfinite')
        if raw['finite'] and raw['finite_min']==raw['finite_max']==0: reasons.append('all_zero_raw_logits')
        if processed['nan'] or processed['positive_inf'] or not processed['finite'] or (processed['negative_inf'] and not constrained):
            reasons.append('invalid_processed_logits')
    return list(dict.fromkeys(reasons))


def validate_observation(observation, prompt, config):
    ids = observation['generated_token_ids']; steps = observation['steps']; meta = observation['trace_metadata']
    if any(type(x) is not int or x < 0 for x in ids): raise ValueError('Invalid generated IDs')
    if [x for s in steps for x in s['selected_token_ids']] != ids: raise ValueError('Trace tokens differ from returned output')
    if not ids: raise ValueError('Generation returned no tokens')
    if len(ids)>768: raise ValueError('Generation exceeded fixed output cap')
    actual = dict(meta['actual_generation_config'])
    if actual.pop('max_length') != len(prompt['input_token_ids'])+768 or actual != config['generation_template']:
        raise ValueError('Effective generation config differs from frozen template')
    expected = ['RepetitionPenaltyLogitsProcessor'] + (['CheckedMask'] if config['schema']['enabled'] else [])
    if meta['native_logits_processors'] != expected: raise ValueError('Unexpected native logits processors')
    eos = config['generation_template']['eos_token_id']
    if len(ids)<768 and ids[-1] not in eos: raise ValueError('Unexpected early termination without native EOS')
    reasons = execution_failures(steps,config['schema']['enabled'])
    truncated = len(ids)==768 and ids[-1] not in eos
    if config['schema']['enabled'] and not truncated and observation.get('grammar_complete') is not True:
        reasons.append('schema_did_not_complete')
    return truncated, reasons


class HFBackend:
    mode = 'local_hf'

    @classmethod
    def load_verified(cls, config, admission):
        validate_config(config); verify_admission(admission,config)
        isolation = verify_gpu_exclusive(config['gpu_uuid'])
        files = {'model':verify_files(config['model_path'],config['model_files'],model=True),
                 'tokenizer':verify_files(config['tokenizer_path'],config['tokenizer_files'])}
        for package, version in config['packages'].items():
            if importlib.metadata.version(package) != version: raise ValueError('Package version differs: '+package)
        # Full weight hashing may take time; admission and resource state must
        # still hold immediately before heavy imports and model allocation.
        verify_admission(admission,config)
        final_isolation = verify_gpu_exclusive(config['gpu_uuid'])
        # All heavy imports and device activity are behind preceding admission checks.
        runtime = _HFRuntime(config)
        instance = cls(config,admission,runtime)
        instance.load_receipt = {'model_files_freshly_hashed':files,'resource_check_before_hashing':isolation,
                                 'resource_check_before_model_load':final_isolation,
                                 'runtime':runtime.load_receipt}
        return instance

    def __init__(self, config, admission, runtime):
        validate_config(config)
        self.config=copy.deepcopy(config); self.admission=copy.deepcopy(admission)
        self.runtime=runtime; self.calls=0
        self.identity={'kind':'local_hf','backend_config_sha256':digest(self.config),
                       'model_id':config['model_id'],'revision':config['revision'],
                       'gpu_uuid':config['gpu_uuid'],
                       'precision':'bfloat16','attention':'eager','schema_constrained':config['schema']['enabled'],
                       'source_sha256':_sha(Path(__file__)), 'trace_source_sha256':TRACE_SHA256,
                       'grammar_implementation':runtime.grammar_receipt}

    def generate(self, request):
        c=self.config; manifest=request['manifest']; prompt=request['prepared_prompt']
        verify_admission(self.admission,c,manifest['id'])
        if manifest['condition'] != self.admission['condition']: raise PermissionError('Condition mismatch')
        if self.calls>=self.admission['maximum_model_calls']: raise PermissionError('Model call budget exhausted')
        if len(prompt['input_token_ids'])+768>4096: raise ValueError('Overflow must be handled before backend invocation')
        shots=sum(m['role']=='assistant' for m in prompt['messages'])
        if shots not in (0,4) or (c['schema']['enabled'] and shots!=4): raise ValueError('Unsupported shot/schema condition')
        self.runtime.verify_input(prompt,c)
        self.runtime.check_resource(c['gpu_uuid'])
        identity=manifest['request_identity']
        if not re.fullmatch('[0-9a-f]{64}',identity): raise ValueError('Invalid request identity')
        trace=Path(c['trace_dir'])/(identity+'.trace.jsonl')
        if trace.exists(): raise FileExistsError('Refusing duplicate execution trace')
        self.calls+=1
        observation=self.runtime.generate(prompt,c,trace)
        ids=observation.get('generated_token_ids',[]); raw=observation.get('raw','')
        reasons=list(observation.get('runtime_reasons',[])); truncated=False
        try:
            if not reasons: truncated, reasons=validate_observation(observation,prompt,c)
        except Exception as exc:
            reasons.append('post_generation_contract:'+type(exc).__name__+': '+str(exc))
        return {'kind':'runtime_failure' if reasons else 'completed','raw':raw,
                'generated_token_ids':ids,'truncated':truncated,'refusal':False,
                'runtime_error':'; '.join(reasons) if reasons else None,
                'execution_failure_reasons':reasons,
                'metadata':{'backend_identity':self.identity,'input_token_ids_sha256':digest(prompt['input_token_ids']),
                            'backend_config_sha256':self.identity['backend_config_sha256'],
                            'refusal_detection':'not inferred automatically; non-schema refusals are parser diagnostics',
                            'trace_path':str(trace),'trace_sha256':_sha(trace) if trace.exists() else None,
                            'actual_generation':observation.get('trace_metadata',{}),
                            'grammar_complete':observation.get('grammar_complete'),
                            'schema_instance_fresh':observation.get('schema_instance_fresh',False),
                            'seconds':observation.get('seconds'),'peak_gpu_allocated_bytes':observation.get('peak_gpu_allocated_bytes'),
                            'peak_gpu_reserved_bytes':observation.get('peak_gpu_reserved_bytes')}}


class _HFRuntime:
    def __init__(self, config):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, set_seed
        self.torch=torch; self.GenerationConfig=GenerationConfig
        build_generation_config(config,GenerationConfig)
        if torch.cuda.is_initialized(): raise RuntimeError('Refusing to reuse an already active CUDA process')
        torch.set_num_threads(4); torch.set_num_interop_threads(1)
        set_seed(config['seed']); torch.use_deterministic_algorithms(True,warn_only=True)
        if torch.cuda.device_count()!=1 or not torch.cuda.is_bf16_supported(): raise RuntimeError('Expected one BF16-capable admitted GPU')
        torch.cuda.set_per_process_memory_fraction(config['memory_fraction'],0)
        self.tok=AutoTokenizer.from_pretrained(config['tokenizer_path'],local_files_only=True,trust_remote_code=False,use_fast=True)
        start=time.perf_counter()
        self.model=AutoModelForCausalLM.from_pretrained(config['model_path'],torch_dtype=torch.bfloat16,
            device_map={'':0},attn_implementation='eager',local_files_only=True,use_safetensors=True,trust_remote_code=False).eval()
        torch.cuda.synchronize()
        if {str(p.dtype) for p in self.model.parameters()}!={'torch.bfloat16'} or {str(p.device) for p in self.model.parameters()}!={'cuda:0'} or getattr(self.model,'is_quantized',False) or self.model.config._attn_implementation!='eager':
            raise RuntimeError('Loaded model precision/device/attention differs')
        trace_path=Path(__file__).with_name('vendor')/'trace_schema.py'
        if _sha(trace_path)!=TRACE_SHA256: raise ValueError('Audited trace implementation changed')
        spec=importlib.util.spec_from_file_location('ner_audited_trace',trace_path)
        self.trace=importlib.util.module_from_spec(spec); spec.loader.exec_module(self.trace)
        self.compiled=None; self.xgr=None; self.grammar_receipt=None
        if config['schema']['enabled']:
            import xgrammar as xgr
            self.xgr=xgr
            info=xgr.TokenizerInfo.from_huggingface(self.tok,vocab_size=self.model.config.vocab_size,
                                                 stop_token_ids=config['generation_template']['eos_token_id'])
            self.compiled,self.grammar_receipt=compile_ner_grammar(config,info)
        self.load_receipt={'load_seconds':time.perf_counter()-start,'model_config':self.model.config.to_dict(),
            'original_generation_config':self.model.generation_config.to_dict(),'attention':self.trace.attention_info(self.model),
            'packages':config['packages'],'schema_compiled':self.compiled is not None,
              'grammar_implementation':self.grammar_receipt}

    def check_resource(self,gpu_uuid): return verify_gpu_exclusive(gpu_uuid)

    def verify_input(self,prompt,config):
        text=self.tok.apply_chat_template(prompt['messages'],chat_template=config['chat_template'],tokenize=False,add_generation_prompt=True)
        ids=self.tok(text,add_special_tokens=False,truncation=False)['input_ids']
        if text!=prompt['rendered_prompt'] or ids!=prompt['input_token_ids']: raise ValueError('Actual model tokenizer input differs from frozen request')

    def generate(self,prompt,config,trace_path):
        torch=self.torch; processors=[]
        if self.compiled is not None:
            from transformers import LogitsProcessor
            from xgrammar.contrib.hf import LogitsProcessor as NativeMask
            compiled=self.compiled
            class CheckedMask(LogitsProcessor):
                def __init__(self): self.native=NativeMask(compiled)
                def __call__(self,input_ids,scores):
                    if not torch.isfinite(scores).all(): raise RuntimeError('Nonfinite logits before grammar constraint')
                    result=self.native(input_ids,scores)
                    if not torch.isfinite(result).any() or torch.isnan(result).any() or torch.isposinf(result).any():
                        raise RuntimeError('Invalid constrained logits')
                    return result
            processors=[CheckedMask()]
        # Internal prepared tensors are regenerated by HF; full effective config is checked afterward.
        generation=build_generation_config(config,self.GenerationConfig)
        inputs=torch.tensor([prompt['input_token_ids']],dtype=torch.long,device='cuda:0')
        attention=torch.ones_like(inputs)
        result={'generated_token_ids':[],'raw':'','steps':[],'trace_metadata':{},'runtime_reasons':[],
                'schema_instance_fresh':len(processors)==1}
        trace=None; torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); start=time.perf_counter()
        try:
            with torch.inference_mode(),self.trace.Trace(self.model,trace_path) as trace:
                output=self.model.generate(input_ids=inputs,attention_mask=attention,generation_config=generation,
                    logits_processor=processors,streamer=trace)
            ids=output[0,len(prompt['input_token_ids']):].detach().cpu().tolist()
            result['generated_token_ids']=ids; result['raw']=self.tok.decode(ids,**config['raw_decode_kwargs'])
        except Exception as exc:
            result['runtime_reasons']=['local_generation:'+type(exc).__name__+': '+str(exc)]
        finally:
            if trace is not None: result['trace_metadata']=trace.metadata
            if trace_path.exists():
                try:
                    for line in trace_path.read_text(encoding='utf-8').splitlines():
                        result['steps'].append(json.loads(line))
                except Exception as exc:
                    result['runtime_reasons'].append('trace_recovery:'+type(exc).__name__+': '+str(exc))
            if result['runtime_reasons']:
                if not result['generated_token_ids']:
                    result['generated_token_ids']=[x for s in result['steps'] for x in s['selected_token_ids']]
                try: result['raw']=self.tok.decode(result['generated_token_ids'],**config['raw_decode_kwargs'])
                except Exception as exc: result['runtime_reasons'].append('partial_decode:'+type(exc).__name__+': '+str(exc))
            try:
                torch.cuda.synchronize()
                result.update(peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),
                              peak_gpu_reserved_bytes=torch.cuda.max_memory_reserved())
            except Exception as exc: result['runtime_reasons'].append('cuda_finalize:'+type(exc).__name__)
            result['seconds']=time.perf_counter()-start
        if self.compiled is not None and not result['runtime_reasons']:
            try:
                matcher=self.xgr.GrammarMatcher(self.compiled)
                result['grammar_complete']=bool(matcher.accept_string(result['raw']) and matcher.is_completed())
            except Exception as exc:
                result['runtime_reasons'].append('grammar_verification:'+type(exc).__name__+': '+str(exc))
        return result
