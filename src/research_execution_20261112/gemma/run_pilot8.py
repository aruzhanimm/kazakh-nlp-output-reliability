"""Exactly eight fixed TRAIN Gemma replies; no automatic dev/test continuation."""
from pathlib import Path
import argparse,hashlib,json,os,subprocess,sys,time,traceback
from datetime import datetime,timezone

HERE=Path(__file__).resolve().parent
EXPECTED_INPUT_SHA='f1ec4b9059fbdb5f555a463dd7c90cdcc836731527f6ebb377849edc1e7b5723'
CAPS={'ner':768,'sa':32}

def emit(path,value):
    data=json.dumps(value,ensure_ascii=False,sort_keys=True,indent=2).encode()
    with path.open('xb') as f:f.write(data);f.flush();os.fsync(f.fileno())

def gpu_admission(index):
    rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.total,memory.free','--format=csv,noheader,nounits'],text=True)
    chosen=[x.strip().split(', ') for x in rows.splitlines() if x.strip().split(',')[0]==str(index)]
    if len(chosen)!=1:raise RuntimeError('Explicit GPU index not found')
    idx,uuid,total,free=chosen[0]
    processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True)
    if any(x.strip().split(',')[0]==uuid for x in processes.splitlines()):raise RuntimeError('Chosen GPU has an existing compute process; nothing interrupted')
    if int(float(free))<30*1024:raise RuntimeError('Chosen GPU has under 30 GiB free; BF16 admission stopped')
    return {'physical_index':int(idx),'uuid':uuid,'total_MiB':int(float(total)),'free_MiB':int(float(free)),'existing_compute_processes':0}

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',required=True);p.add_argument('--gpu-index',type=int,required=True,choices=[1]);a=p.parse_args()
    root=Path(a.root).resolve()
    if sys.platform!='linux' or root!=Path('/home/jovyan/work/kazakh_gemma4_20261112'):raise RuntimeError('Dedicated server root required')
    if Path(sys.prefix).resolve()!=(root/'venv').resolve():raise RuntimeError('Use the isolated Gemma venv')
    for name in ['RUNTIME_READY.json','ASSETS_VERIFIED.json','CPU_INPUTS_READY.json','WEIGHTS_VERIFIED.json']:
        if json.loads((root/name).read_text())['status']!='PASS':raise RuntimeError('Missing completed admission: '+name)
    raw_source=(HERE/'INPUTS_8.json').read_bytes()
    if hashlib.sha256(raw_source).hexdigest()!=EXPECTED_INPUT_SHA:raise RuntimeError('Fixed train source binding changed')
    source=json.loads(raw_source);items=source['items'];prepared=json.loads((root/'PREPARED_8.json').read_text())
    if len(items)!=8 or len(prepared['items'])!=8:raise RuntimeError('Exactly eight requests required')
    keys=[(x['id'],x['shots']) for x in items]
    if len(set(keys))!=8 or any(':train:' not in x['id'] for x in items):raise RuntimeError('Train-only fixed request identity failed')
    if any(x['messages']!=y['messages'] or x['id']!=y['id'] or x['shots']!=y['shots'] for x,y in zip(items,prepared['items'])):raise RuntimeError('CPU-prepared messages differ')
    os.environ.update(CUDA_VISIBLE_DEVICES=str(a.gpu_index),HF_HOME=str(root/'hf_cache'),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_HUB_DISABLE_IMPLICIT_TOKEN='1',HF_HUB_DISABLE_TELEMETRY='1',PYTHONDONTWRITEBYTECODE='1',TOKENIZERS_PARALLELISM='false')
    admission=gpu_admission(a.gpu_index)
    from server_setup import verified,REV
    metadata=json.loads((HERE/'HF_METADATA.json').read_text(encoding='utf-8'));infos={x['rfilename']:x for x in metadata['siblings']}
    weight_receipt=verified(root/'model/model.safetensors',infos['model.safetensors'])
    import torch,transformers
    from transformers import AutoProcessor,AutoModelForMultimodalLM
    from native_adapter import parse
    if transformers.__version__!='5.22.2' or not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():raise RuntimeError('Pinned runtime / BF16 CUDA unavailable')
    torch.manual_seed(42);torch.cuda.manual_seed_all(42)
    processor=AutoProcessor.from_pretrained(root/'model',local_files_only=True,trust_remote_code=False)
    run_dir=root/'runs'/('train8_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
    run_dir.mkdir(parents=True,exist_ok=False)
    protocol={'purpose':'technical_train_pilot_not_model_selection','requests':8,'model':'google/gemma-4-12B-it','revision':REV,'input_sha256':EXPECTED_INPUT_SHA,'thinking':False,'do_sample':False,'dtype':'bfloat16','attention':'sdpa','max_new_tokens':CAPS,'input_ceiling':8192,'seed':42,'auto_dev':False,'test_access':False,'gpu':admission,'weights_verified':weight_receipt,'torch':torch.__version__,'transformers':transformers.__version__,'runtime_cuda':torch.version.cuda,'source_hashes':{f:hashlib.sha256((HERE/f).read_bytes()).hexdigest() for f in ['run_pilot8.py','native_adapter.py','existing_alignment.py','existing_sa_scoring.py']},'failure_policy':'SA runtime/truncation -> INVALID; NER retains parseable spans with failure flags; no JSON repair; native-channel exception stops pilot'}
    emit(run_dir/'PROTOCOL.json',protocol)
    try:
        start=time.perf_counter()
        model=AutoModelForMultimodalLM.from_pretrained(root/'model',dtype=torch.bfloat16,device_map={'':'cuda:0'},attn_implementation='sdpa',local_files_only=True,trust_remote_code=False)
        model.eval();torch.cuda.synchronize();load_seconds=time.perf_counter()-start
        eos=model.generation_config.eos_token_id;eos=set(eos if isinstance(eos,list) else [eos])
        outcomes=[]
        for index,(item,ready) in enumerate(zip(items,prepared['items'])):
            request_dir=run_dir/f'{index:02d}';request_dir.mkdir()
            inputs=processor.apply_chat_template(item['messages'],tokenize=True,return_dict=True,return_tensors='pt',add_generation_prompt=True,enable_thinking=False)
            prefix=inputs['input_ids'][0].tolist()
            if prefix!=ready['input_ids'] or len(prefix)>8192:raise RuntimeError('Native tokens differ from CPU admission')
            cap=CAPS[item['task']]
            emit(request_dir/'REQUEST.json',{'id':item['id'],'task':item['task'],'shots':item['shots'],'input_ids':prefix,'messages':item['messages'],'max_new_tokens':cap,'protocol_ref':'../PROTOCOL.json'})
            inputs=inputs.to('cuda:0');torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();start=time.perf_counter()
            try:
                with torch.inference_mode():
                    output=model.generate(**inputs,max_new_tokens=cap,do_sample=False,temperature=None,top_p=None,top_k=None,use_cache=True)
                torch.cuda.synchronize();seconds=time.perf_counter()-start
                token_ids=output[0,len(prefix):].detach().cpu().tolist()
                truncated=len(token_ids)>=cap and (not token_ids or token_ids[-1] not in eos)
                raw={'id':item['id'],'task':item['task'],'shots':item['shots'],'generated_token_ids':token_ids,'raw_native_text':processor.tokenizer.decode(token_ids,skip_special_tokens=False),'input_tokens':len(prefix),'generated_tokens':len(token_ids),'truncated':truncated,'seconds':seconds,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved(),'request_ref':'REQUEST.json'}
                emit(request_dir/'RAW.json',raw)  # Evidence is durable BEFORE interpretation.
                result=parse(processor,token_ids,prefix,item,source['allowed_ner_types'])
                task=result['task_parsed'];failed=bool(task['failed'] or truncated)
                if item['task']=='sa' and truncated:task={**task,'label':None,'failed':True,'errors':[*task['errors'],'generation_truncated']}
                elif truncated:task={**task,'failed':True,'errors':[*task['errors'],'generation_truncated']}
                result.update(task_parsed=task,truncated=truncated,failed=failed,runtime_error=False,refusal=None,refusal_assessment='not_automatically_inferred',terminal_failed=False,raw_request_refs=['REQUEST.json','RAW.json'])
                emit(request_dir/'PARSED.json',result)
                outcomes.append({'id':item['id'],'task':item['task'],'shots':item['shots'],'failed':failed,'truncated':truncated,'unexpected_nonempty_thinking':result['unexpected_nonempty_thinking'],'generated_tokens':len(token_ids),'seconds':seconds,'path':request_dir.name})
                print(json.dumps(outcomes[-1]),flush=True)
                if result['unexpected_nonempty_thinking']:raise RuntimeError('Unexpected nonempty thinking in disabled mode; remaining calls stopped')
            except Exception:
                emit(request_dir/'ERROR.json',{'traceback':traceback.format_exc(),'terminal_failed':True,'no_automatic_retry':True})
                raise
        emit(run_dir/'COMPLETE.json',{'status':'PASS_EXECUTION_NOT_QUALITY_ACCEPTANCE','requests_completed':len(outcomes),'outcomes':outcomes,'model_load_seconds':load_seconds,'no_dev_or_test_started':True})
        print('Eight train calls complete; STOP. Evidence: '+str(run_dir),flush=True)
    except Exception:
        emit(run_dir/'STOPPED.json',{'status':'STOPPED','traceback':traceback.format_exc(),'preserved_evidence':True,'no_automatic_retry':True})
        raise

if __name__=='__main__':main()
