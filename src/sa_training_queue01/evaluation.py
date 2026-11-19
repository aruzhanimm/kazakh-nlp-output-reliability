import time,math
from collections import Counter
from pilot.common import ROOT,read_json,read_jsonl,file_hash,digest,write_json,write_jsonl
from pilot.evaluate import summarize
from pilot.scoring import score_sa
from .common import RUN,utc,config
from .control import status

def dev_rows():
    p=ROOT/'stage_preflight/views/sa_valid.jsonl';m=read_json(ROOT/'stage_preflight/views/manifest.json')
    assert file_hash(p)==m['files']['views/sa_valid.jsonl'];rows=read_jsonl(p)
    assert len(rows)==len({r['id'] for r in rows})==16766
    return rows

def predict(feature,row,model):
    import torch
    assert feature['id']==row['id'] and feature['labels']==int(row['label'])
    inputs={k:torch.tensor([feature[k]],device='cuda') for k in ['input_ids','attention_mask','token_type_ids'] if k in feature}
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):z=model(**inputs).logits.float()
    assert z.shape==(1,2) and bool(torch.isfinite(z).all()),'Nonfinite or invalid SA logits'
    probs=z.softmax(-1);pred=int(z.argmax(-1).item());loss=float(-z.log_softmax(-1)[0,int(row['label'])])
    return {'label':pred,'failed':False},{'logits':z[0].cpu().tolist(),'probabilities':probs[0].cpu().tolist(),'cross_entropy':loss}

def evaluate(model,features,rows,progress,run_hash,deadline):
    import torch
    assert len(features)==len(rows)==16766 and [f['id'] for f in features]==[r['id'] for r in rows]
    epoch=progress['epoch']+1;path=RUN/'evaluations'/f'epoch_{epoch:02d}';items=path/'items';items.mkdir(parents=True,exist_ok=True)
    model.eval();model.gradient_checkpointing_disable();predictions=[]
    predict(features[0],rows[0],model)
    for i,(f,row) in enumerate(zip(features,rows)):
        if time.time()>=deadline-60:return None
        target=items/(digest(row['id'])+'.json')
        if target.exists():
            item=read_json(target)
            assert item['run_hash']==run_hash and item['input_sha256']==digest(row) and item['model_sha256']==progress['evaluation_model_sha256']
            assert not item.get('runtime_error'),'Previously failed output cannot be accepted by retry'
        else:
            torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();started=time.perf_counter();error=None
            try:parsed,raw=predict(f,row,model)
            except Exception as exc:parsed={'label':None,'failed':True};raw={};error={'type':type(exc).__name__,'message':str(exc)}
            torch.cuda.synchronize()
            item={'id':row['id'],'input_sha256':digest(row),'visible_text_sha256':digest(row['visible_text']),'run_hash':run_hash,'model_sha256':progress['evaluation_model_sha256'],
                'epoch':epoch,'parsed':parsed,'raw':raw,'seconds':time.perf_counter()-started,'peak_gpu_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_gpu_reserved_bytes':torch.cuda.max_memory_reserved()}
            if error:item['runtime_error']=error
            write_json(target,item)
            if error:raise RuntimeError('Development failure preserved: '+str(error))
        predictions.append(item)
        if (i+1)%500==0:write_json(RUN/'status.json',{**status(progress),'dev_predictions':i+1,'utc':utc()});print({'epoch':epoch,'dev_predictions':i+1},flush=True)
    metrics=summarize('sa',rows,predictions);metrics['majority_baseline']=score_sa([int(r['label']) for r in rows],[1]*len(rows))
    metrics['dev_cross_entropy']=sum(p['raw']['cross_entropy'] for p in predictions)/len(rows)
    metrics['stage']='full_data_SA_exploratory_development';metrics['timing']['scope']='batch1 warmed synchronized, finite checks/logits/probabilities collection included; loading/item writes excluded'
    counts=Counter(p['parsed']['label'] for p in predictions);hist=[0]*10;per={str(i):[0]*10 for i in [0,1]};confidence=[]
    for p in predictions:
        c=max(p['raw']['probabilities']);confidence.append(c);j=min(9,int(c*10));hist[j]+=1;per[str(p['parsed']['label'])][j]+=1
    distributions={'predicted_labels':dict(counts),'gold_labels':dict(Counter(int(r['label']) for r in rows)),'confidence_bins_step_0_1':hist,'confidence_by_predicted_class':per,'mean_max_probability':sum(confidence)/len(confidence)}
    write_jsonl(path/'predictions.jsonl',predictions);write_json(path/'metrics.json',metrics);write_json(path/'distributions.json',distributions)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False});model.train();return metrics

def scientific_metrics(metrics):
    return {k:v for k,v in metrics.items() if k not in ['timing','peak_gpu_allocated_bytes','peak_gpu_reserved_bytes']}

def compare_predictions(new,old):
    assert len(new)==len(old)
    counts={k:0 for k in ['ids','input_hashes','visible_text_hashes','labels_and_failures','logits','probabilities','cross_entropy']};ids={k:[] for k in counts}
    for a,b in zip(new,old):
        checks={'ids':a['id']==b['id'],'input_hashes':a['input_sha256']==b['input_sha256'],'visible_text_hashes':a['visible_text_sha256']==b['visible_text_sha256'],
            'labels_and_failures':a['parsed']==b['parsed'],'logits':a['raw'].get('logits')==b['raw'].get('logits'),
            'probabilities':a['raw'].get('probabilities')==b['raw'].get('probabilities'),'cross_entropy':a['raw'].get('cross_entropy')==b['raw'].get('cross_entropy')}
        for k,ok in checks.items():
            if not ok:counts[k]+=1;ids[k].append(a['id'])
    return {'n':len(old),'all_agree':not any(counts.values()),'mismatch_counts':counts,'mismatch_ids':ids,'comparison':'exact values; only timing/telemetry/identities excluded'}
