"""Exactly one full-data condition. No test loader and no search/grid entry point."""
import argparse
from collections import Counter
import json
import os
import shutil
import time
from datetime import datetime,timezone
from pilot.common import ROOT,read_json,read_jsonl,write_json,write_jsonl,file_hash,digest,canonical,RunLock
from pilot.evaluate import summarize
from pilot.inputs import first_word_predictions
from pilot.scoring import bio_spans
from stage_preflight.state import fingerprint
from .common import D,freeze,verify,identity
from .core import load_data,initialize,collator,order,update,labels
from .checkpoints import save,restore
from .control import new_progress,status,select

RUN=D/'run_xlmr_ner_seed13_lr1e-5'

def dev_rows():
    p=ROOT/'stage_preflight/views/ner_valid.jsonl';m=read_json(ROOT/'stage_preflight/views/manifest.json')
    assert file_hash(p)==m['files']['views/ner_valid.jsonl']
    rows=read_jsonl(p);assert len(rows)==11100 and all(r['id'].startswith('ner:valid:') for r in rows)
    return rows

def prepare():
    chosen=read_json(D/'chosen_execution.json');assert chosen['effective_batch']==16
    train=load_data('train');valid=load_data('valid');rows=dev_rows();labs=labels()
    assert len(labs)==51 and len(read_json(ROOT/'data/derived/ner_labels.json')['types'])==25
    assert len({f['id'] for f in train})==89386
    assert all(f['source_id']==r['id'] and f['word_ids'].count(None)>=2 and len(r['windows'])==1 for f,r in zip(valid,rows))
    n=sum(sum(x!=-100 for x in f['labels']) for f in train);assert n==1034279
    spec={'task':'ner','model':'xlmr_base','seed':13,'learning_rate':1e-5,'training_sentences':89386,'training_features':89386,
        'training_first_subword_labels':n,'development_sentences':11100,'types':25,'labels':labs,
        'microbatch':chosen['microbatch'],'gradient_accumulation':chosen['gradient_accumulation'],'effective_batch':16,
        'epochs_max':5,'updates_per_epoch':5587,'scheduler_horizon_updates':27935,'warmup_updates':1676,
        'scheduler':'linear_warmup_then_linear_decay','optimizer':'AdamW','weight_decay':.01,'gradient_clip':1.,
        'precision':'BF16 autocast; FP32 parameters, gradients and optimizer states; FP32 cross-entropy',
        'gradient_checkpointing':{'enabled':True,'use_reentrant':False},'attention':'sdpa','strict_determinism':True,
        'objective':'summed_CE / all_nonignored_first_subword_labels_in_complete_optimizer_batch',
        'evaluation':'all_retained_dev_after_each_epoch','selection':'strictly highest exact-span entity micro-F1; earliest ties',
        'early_stop':{'patience':2,'minimum_epochs':3,'min_delta':.0001},'operational_ceiling_seconds':28800,
        'safe_stop_reserve_seconds':60,'rolling_slots':2,'rolling_every_updates':1000,'epoch_checkpoints':'retain all',
        'planned_fresh_process_restart_update':1000,'initialization':read_json(ROOT/'configs/models.lock.json')['encoder'],
        'final_test_locked':True,'development_status':'exploratory_repeatedly_used_development_data'}
    p=D/'main_config.json'
    if p.exists():assert read_json(p)==spec
    else:write_json(p,spec)
    names=['encoder_main01/'+n for n in ['__init__.py','common.py','objective.py','core.py','checkpoints.py','control.py','main.py','launch_main.py',
        'PLAN.md','METHOD_DECISIONS.md','main_config.json','chosen_execution.json','benchmark_plan.lock.json','audit/normalization_fp32.json',
        'tests/test_objective.py','tests/test_checkpoints.py','tests/test_control.py','tests/test_evaluation.py']]
    names+=['pilot/common.py','pilot/models.py','pilot/inputs.py','pilot/train_encoder.py','pilot/evaluate.py','pilot/scoring.py',
        'stage_preflight/state.py','stage_preflight/views/manifest.json','configs/models.lock.json','data/derived/ner_labels.json']
    print(freeze(names,'main_plan.lock.json'))
    usage=shutil.disk_usage(D);reserve=40*2**30
    write_json(D/'audit/disk_before_main.json',{'utc':datetime.now(timezone.utc).isoformat(),'free_bytes':usage.free,'required_reserve_bytes':reserve,'passed':usage.free>=reserve})
    assert usage.free>=reserve,'Insufficient capacity for retained epoch and two rolling checkpoints'

def predict(feature,row,model,labs,device='cuda'):
    import torch
    # Use frozen shared-text features. Never feed labels or use labels for alignment.
    b={k:torch.tensor([feature[k]],device=device) for k in ['input_ids','attention_mask']}
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

def run(deadline,attempt):
    import torch
    verify('main_plan.lock.json');spec=read_json(D/'main_config.json');RUN.mkdir(exist_ok=True)
    run_hash=identity(RUN,spec,'main_plan.lock.json');attempt_dir=RUN/'attempts'/str(attempt);attempt_dir.mkdir(parents=True,exist_ok=False)
    checkpoint_dir=RUN/'checkpoints';checkpoint_dir.mkdir(exist_ok=True)
    with RunLock(RUN):
        if (RUN/'completed.json').exists():raise RuntimeError('Completed immutable condition; no rerun')
        if time.time()>=deadline-60:raise RuntimeError('Operational budget exhausted before loading')
        began=time.perf_counter();train=load_data('train');valid=load_data('valid');rows=dev_rows()
        model,tok,opt,sched=initialize();torch.cuda.synchronize()
        write_json(attempt_dir/'load.json',{'seconds':time.perf_counter()-began,'gpu':torch.cuda.get_device_name(),'vram_bytes':torch.cuda.get_device_properties(0).total_memory,'initialization':'pinned pretrained, then compatible recovery only if present'})
        progress,recovery=restore(checkpoint_dir,model,opt,sched,run_hash)
        write_json(attempt_dir/'recovery.json',recovery)
        if progress is None:
            if list(checkpoint_dir.glob('*.pt')) or recovery['rejected']:raise RuntimeError('Recovery artifacts exist but none validated; refusing fresh start')
            progress=new_progress()
        c=collator(tok);torch.cuda.reset_peak_memory_stats();safe=True
        try:
            with (attempt_dir/'training.jsonl').open('x',encoding='utf-8',newline='\n',buffering=1) as journal:
                while progress['phase']!='complete':
                    epoch=progress['epoch'];indices=order(epoch,len(train))
                    order_path=RUN/'orders'/f'epoch_{epoch+1:02d}.json'
                    ordering={'epoch':epoch+1,'seed':13+epoch,'feature_ids':[train[i]['id'] for i in indices]}
                    if order_path.exists():assert read_json(order_path)==ordering
                    else:write_json(order_path,ordering)
                    if progress['phase']=='train':
                        for cursor in range(progress['cursor'],len(train),16):
                            if time.time()>=deadline-60:
                                save(checkpoint_dir,model,opt,sched,progress,run_hash)
                                write_json(attempt_dir/'outcome.json',{'status':'incomplete_budget','progress':status(progress)});return
                            positions=indices[cursor:cursor+16];torch.cuda.synchronize();started=time.perf_counter();safe=False
                            values=update(model,opt,sched,[train[i] for i in positions],spec['microbatch'],c)
                            torch.cuda.synchronize();seconds=time.perf_counter()-started
                            progress['updates']+=1;progress['cursor']=cursor+len(positions);progress['epoch_loss_sum']+=values['loss_sum'];progress['epoch_valid_labels']+=values['valid_labels'];progress['active_training_seconds']+=seconds
                            progress['peak_gpu_allocated_bytes']=max(progress['peak_gpu_allocated_bytes'],torch.cuda.max_memory_allocated());progress['peak_gpu_reserved_bytes']=max(progress['peak_gpu_reserved_bytes'],torch.cuda.max_memory_reserved())
                            record={'update':progress['updates'],'epoch':epoch+1,'cursor_after':progress['cursor'],'seconds':seconds,
                                'feature_ids_sha256':digest([train[i]['id'] for i in positions]),**values}
                            progress['trace'].append(record);journal.write(canonical(record)+'\n');safe=True
                            if progress['updates']%100==0:
                                write_json(RUN/'status.json',{**status(progress),'utc':datetime.now(timezone.utc).isoformat()})
                                print({'epoch':epoch+1,'updates':progress['updates'],'epoch_examples':progress['cursor'],'loss':values['loss'],'training_seconds':progress['active_training_seconds']},flush=True)
                            if progress['updates']%1000==0:
                                restart=progress['updates']==1000 and not progress['planned_restart_done']
                                if restart:progress['planned_restart_done']=True
                                meta=save(checkpoint_dir,model,opt,sched,progress,run_hash)
                                write_json(attempt_dir/f'checkpoint_{progress["updates"]}.json',meta)
                                if restart:
                                    write_json(attempt_dir/'outcome.json',{'status':'planned_restart_ready','progress':status(progress)});return
                        assert progress['cursor']==89386 and progress['epoch_valid_labels']==1034279
                        progress['phase']='evaluate';progress['evaluation_model_sha256']=fingerprint(model.state_dict())
                        save(checkpoint_dir,model,opt,sched,progress,run_hash)
                    evaluation_started=time.perf_counter();metrics=evaluate(model,valid,rows,progress,run_hash,deadline)
                    if metrics is None:
                        save(checkpoint_dir,model,opt,sched,progress,run_hash)
                        write_json(attempt_dir/'outcome.json',{'status':'incomplete_budget','progress':status(progress)});return
                    score=metrics['f1'];select(progress,score,epoch+1)
                    progress['history'].append({'epoch':epoch+1,'updates':progress['updates'],'training_loss':progress['epoch_loss_sum']/progress['epoch_valid_labels'],
                        'training_valid_labels':progress['epoch_valid_labels'],'dev_micro_f1':score,'dev_precision':metrics['precision'],'dev_recall':metrics['recall'],
                        'bio_repairs':metrics['bio_repairs'],'evaluation_wall_seconds_this_attempt':time.perf_counter()-evaluation_started,
                        'active_training_seconds_cumulative':progress['active_training_seconds'],'best_epoch':progress['best_epoch'],'bad_epochs':progress['bad_epochs']})
                    progress.update(epoch=epoch+1,cursor=0,epoch_loss_sum=0.,epoch_valid_labels=0,phase='complete' if epoch+1>=5 or progress['stop'] else 'train')
                    meta=save(checkpoint_dir,model,opt,sched,progress,run_hash,epoch=epoch+1)
                    write_json(RUN/'learning_curve.json',progress['history']);write_json(RUN/'status.json',status(progress));print(progress['history'][-1],flush=True)
                write_jsonl(RUN/'training_committed.jsonl',progress['trace'])
                selected=RUN/'evaluations'/f'epoch_{progress["best_epoch"]:02d}'
                shutil.copy2(selected/'predictions.jsonl',RUN/'predictions.jsonl');shutil.copy2(selected/'metrics.json',RUN/'metrics.json');shutil.copy2(selected/'distributions.json',RUN/'distributions.json')
                write_json(RUN/'completed.json',{'run_hash':run_hash,**status(progress),'selected_checkpoint':f'epoch_{progress["best_epoch"]:02d}.pt',
                    'final_test_locked':True,'utc':datetime.now(timezone.utc).isoformat()})
                write_json(attempt_dir/'outcome.json',{'status':'complete','progress':status(progress)})
        except BaseException as exc:
            # Never commit a possibly half-applied optimizer update. Last committed recovery remains intact.
            write_json(attempt_dir/'failure.json',{'error_type':type(exc).__name__,'error':str(exc),'at_safe_boundary':safe,'progress':status(progress)})
            write_json(attempt_dir/'outcome.json',{'status':'blocked','error':str(exc),'progress':status(progress)})
            raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');p.add_argument('--deadline',type=float);p.add_argument('--attempt',type=int);a=p.parse_args()
    if a.prepare:prepare()
    else:
        if a.deadline is None or a.attempt is None:p.error('Use the bounded launcher')
        ledger=read_json(D/'main_budget.json');assert a.deadline==ledger['deadline_unix'] and time.time()<=a.deadline
        run(a.deadline,a.attempt)
