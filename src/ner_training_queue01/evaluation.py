import time
from collections import Counter
from datetime import datetime,timezone
from pilot.common import ROOT,read_json,read_jsonl,write_json,write_jsonl,file_hash,digest
from pilot.evaluate import summarize
from pilot.inputs import first_word_predictions
from pilot.scoring import bio_spans
from encoder_main01.control import status
from .core import labels
from .common import RUN

def dev_rows():
    p=ROOT/'stage_preflight/views/ner_valid.jsonl';m=read_json(ROOT/'stage_preflight/views/manifest.json')
    assert file_hash(p)==m['files']['views/ner_valid.jsonl']
    rows=read_jsonl(p);assert len(rows)==11100 and all(r['id'].startswith('ner:valid:') for r in rows)
    return rows

def predict(feature,row,model,labs,device='cuda'):
    import torch
    # Use frozen shared-text features. Never feed labels or use labels for alignment.
    b={k:torch.tensor([feature[k]],device=device) for k in ['input_ids','attention_mask','token_type_ids'] if k in feature}
    with torch.inference_mode(),torch.autocast(device_type=device,dtype=torch.bfloat16,enabled=device=='cuda'):
        logits=model(**b).logits.float()[0]
    if not bool(torch.isfinite(logits).all()):raise FloatingPointError('Nonfinite development logits')
    ids=logits.argmax(-1).cpu().tolist()
    words=first_word_predictions(feature['word_ids'],ids,len(row['tokens']));tags=[labs[i] for i in words]
    spans,repairs=bio_spans(tags,repair=True)
    first=[];seen=set()
    for i,w in enumerate(feature['word_ids']):
        if w is not None and w not in seen:first.append(i);seen.add(w)
    conf=logits[first].softmax(-1).amax(-1).cpu().tolist()
    return {'spans':spans,'invalid_candidates':0,'failed':False,'repairs':repairs},{'word_tags':tags,'word_confidence':conf}

def evaluate(model,features,rows,progress,run_hash,deadline):
    import torch
    epoch=progress['epoch']+1;path=RUN/'evaluations'/f'epoch_{epoch:02d}';items=path/'items';items.mkdir(parents=True,exist_ok=True)
    model.eval();model.gradient_checkpointing_disable();labs=labels();predictions=[]
    # Identical discarded warmup in every evaluation process, excluded from prediction timing.
    predict(features[0],rows[0],model,labs)
    for i,(f,row) in enumerate(zip(features,rows)):
        if time.time()>=deadline-60:return None
        target=items/(digest(row['id'])+'.json')
        if target.exists():
            item=read_json(target)
            assert item['run_hash']==run_hash and item['input_sha256']==digest(row) and item['model_sha256']==progress['evaluation_model_sha256']
        else:
            torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();started=time.perf_counter()
            try:parsed,raw=predict(f,row,model,labs);error=None
            except Exception as exc:
                parsed={'spans':[],'invalid_candidates':0,'failed':True,'repairs':0};raw={};error={'type':type(exc).__name__,'message':str(exc)}
            torch.cuda.synchronize()
            item={'id':row['id'],'input_sha256':digest(row),'run_hash':run_hash,'model_sha256':progress['evaluation_model_sha256'],
                'epoch':epoch,'parsed':parsed,'raw':raw,'seconds':time.perf_counter()-started,
                'peak_gpu_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_gpu_reserved_bytes':torch.cuda.max_memory_reserved()}
            if error:item['runtime_error']=error
            write_json(target,item)
            if error:raise RuntimeError('Development execution failed; failed output preserved: '+str(error))
        predictions.append(item)
        if (i+1)%500==0:
            write_json(RUN/'status.json',{**status(progress),'dev_predictions':i+1,'utc':datetime.now(timezone.utc).isoformat()})
            print({'epoch':epoch,'dev_predictions':i+1,'total':len(rows)},flush=True)
    metrics=summarize('ner',rows,predictions)
    metrics['stage']='full_data_encoder_exploratory_development';metrics['timing']['scope']='batch 1, warmed, synchronized; finite-logit checks and confidence collection included; model loading and item writes excluded'
    tag_counts=Counter(t for p in predictions for t in p['raw'].get('word_tags',[]))
    entities=Counter(s[2] for p in predictions for s in p['parsed']['spans'])
    confidence=[c for p in predictions for c in p['raw'].get('word_confidence',[])]
    hist=[0]*10
    for c in confidence:hist[min(9,int(c*10))]+=1
    distributions={'word_tags':{l:tag_counts[l] for l in labs},'entity_types':{l:entities[l] for l in read_json(ROOT/'data/derived/ner_labels.json')['types']},
        'confidence_bins_0_to_1_step_point1':hist,'mean_max_probability':sum(confidence)/len(confidence) if confidence else None,
        'words':sum(tag_counts.values()),'entities':sum(entities.values())}
    write_jsonl(path/'predictions.jsonl',predictions);write_json(path/'metrics.json',metrics);write_json(path/'distributions.json',distributions)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False});model.train()
    return metrics
