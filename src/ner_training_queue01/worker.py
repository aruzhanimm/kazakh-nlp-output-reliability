"""One initialized process per stage. Explicit fresh start or checked exact resume."""
import argparse,time,math,traceback,shutil,gc
from pathlib import Path
from unittest.mock import patch
from pilot.common import ROOT,read_json,read_jsonl,write_json,write_jsonl,digest,file_hash
from stage_preflight.state import fingerprint,digests
from encoder_main01.control import new_progress,select,status
from .common import D,RUN,utc,durable,append,config,verify_plan
from .core import initialize,load_data,order,collator,update
from .checkpoints import save,committed_meta
from .restore import restore

def forbidden(*a,**kw):raise AssertionError('Update/forward prohibited in zero-update validation')

def zero(mode):
    import torch
    verify_plan();c=config();model,tok,opt,sched=initialize();torch.cuda.synchronize()
    if mode=='capture':
        p=new_progress();before=digests(model,opt,sched,p)
        head={n:v for n,v in model.state_dict().items() if n.startswith('classifier.')}
        base={n:v for n,v in model.state_dict().items() if not n.startswith('classifier.')}
        durable(D/'validation/INITIAL.json',{'utc':utc(),'all_components':before,'seed':c['seed'],'head_signature':fingerprint(head),'pretrained_encoder_signature':fingerprint(base),
            'thread_counts':[torch.get_num_threads(),torch.get_num_interop_threads()],'attention':model.config._attn_implementation,'gpu':torch.cuda.get_device_name(),'vram_bytes':torch.cuda.get_device_properties(0).total_memory})
        with patch.object(model,'forward',forbidden),patch.object(opt,'step',forbidden),patch.object(type(sched),'step',forbidden):
            meta=save(D/'validation/checkpoints',model,opt,sched,p,c['run_hash'],lineage={'fresh_pretrained':True,'optimizer_updates':0})
        after=digests(model,opt,sched,p);same={k:before[k]==after[k] for k in ['model','optimizer','scheduler','rng']}
        durable(D/'validation/CAPTURE.json',{'passed':all(same.values()),'actual_before':before,'actual_after':after,'unchanged':same,
            'bookkeeping_only_progress_changes':['checkpoint_sequence:0->1','rolling_sequence:0->1'],'metadata':meta})
        assert all(same.values())
    else:
        initial=read_json(D/'validation/INITIAL.json');before=digests(model,opt,sched,new_progress())
        durable(D/'validation/FRESH_INITIALIZATION.json',{'actual':before,'expected':initial['all_components'],'passed':before==initial['all_components']})
        assert before==initial['all_components'],'Fresh initialization not reproducible'
        metadata,_=committed_meta(D/'validation/checkpoints')
        with patch.object(model,'forward',forbidden),patch.object(opt,'step',forbidden),patch.object(type(sched),'step',forbidden):
            p,source,meta=restore(metadata,model,opt,sched,D/'validation/gpu_restore')
        assert p['updates']==0 and not opt.state
        durable(D/'validation/ROUNDTRIP.json',{'utc':utc(),'passed':True,'optimizer_updates':0,'fresh_process':True,'all_five_components':True,'source_unchanged':True,'metadata':meta})
        del source

def train(stage,deadline,resume=None):
    import torch
    from . import evaluation
    verify_plan();assert read_json(D/'validation/ROUNDTRIP.json')['passed'];c=config();folder=RUN/'attempts'/stage
    folder.mkdir(parents=True,exist_ok=False);tick=time.perf_counter();p=None;safe=False
    try:
        features=load_data('train');valid=load_data('valid');rows=evaluation.dev_rows();model,tok,opt,sched=initialize();torch.cuda.synchronize()
        if resume:
            p,source,meta=restore(resume,model,opt,sched,folder/'restore');del source
        else:
            assert not (RUN/'checkpoints/latest.json').exists(),'Fresh start cannot replace an existing run'
            p=new_progress();actual=digests(model,opt,sched,p);expected=read_json(D/'validation/INITIAL.json')['all_components']
            durable(folder/'fresh_initialization.json',{'actual':actual,'expected':expected,'passed':actual==expected});assert actual==expected
        durable(folder/'load.json',{'utc':utc(),'seconds':time.perf_counter()-tick,'resume':str(resume) if resume else None,
            'threads':[torch.get_num_threads(),torch.get_num_interop_threads()],'attention':model.config._attn_implementation,'gpu':torch.cuda.get_device_name()})
        collate=collator(tok);safe=True;torch.cuda.reset_peak_memory_stats()
        while p['phase']!='complete':
            epoch=p['epoch'];indices=order(epoch,len(features));ordering={'epoch':epoch+1,'seed':c['seed']+epoch,'feature_ids':[features[i]['id'] for i in indices]}
            op=RUN/'orders'/f'epoch_{epoch+1:02d}.json'
            if op.exists():assert read_json(op)==ordering
            else:durable(op,ordering)
            if p['phase']=='train':
                for cursor in range(p['cursor'],len(features),16):
                    if time.time()>=deadline-150:
                        meta=save(RUN/'checkpoints',model,opt,sched,p,c['run_hash'],lineage={'reason':'time_ceiling'})
                        durable(folder/'RESULT.json',{'utc':utc(),'status':'incomplete_budget','progress':status(p),'latest':meta});return
                    positions=indices[cursor:cursor+16];ids=[features[i]['id'] for i in positions]
                    torch.cuda.synchronize();started=utc();began=time.perf_counter();safe=False
                    values=update(model,opt,sched,[features[i] for i in positions],c['microbatch'],collate)
                    torch.cuda.synchronize();seconds=time.perf_counter()-began
                    assert all(math.isfinite(v) for v in values.values() if isinstance(v,(int,float))),'Nonfinite update'
                    p['updates']+=1;p['cursor']=cursor+len(positions);p['epoch_loss_sum']+=values['loss_sum'];p['epoch_valid_labels']+=values['valid_labels']
                    record={'update':p['updates'],'epoch':epoch+1,'cursor_after':p['cursor'],'feature_ids_sha256':digest(ids),**values}
                    p['trace'].append(record);append(folder/'scientific_trace.jsonl',record);append(folder/'batch_order.jsonl',{'update':p['updates'],'feature_ids':ids})
                    append(folder/'telemetry.jsonl',{'update':p['updates'],'seconds':seconds,'started_utc':started,'finished_utc':utc(),
                        'peak_gpu_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_gpu_reserved_bytes':torch.cuda.max_memory_reserved()});safe=True
                    if p['updates']%100==0:
                        write_json(RUN/'status.json',{'utc':utc(),**status(p)});print({'condition':D.name,'update':p['updates'],'epoch':epoch+1,'loss':values['loss']},flush=True)
                    if p['updates']%1000==0:
                        restart=p['updates']==1000 and not p['planned_restart_done']
                        if restart:p['planned_restart_done']=True
                        meta=save(RUN/'checkpoints',model,opt,sched,p,c['run_hash'])
                        if restart:
                            durable(folder/'RESULT.json',{'utc':utc(),'status':'planned_restart_ready','progress':status(p),'latest':meta});return
                assert p['cursor']==len(features) and p['epoch_valid_labels']==c['training_first_subword_labels']
                p['phase']='evaluate';p['evaluation_model_sha256']=fingerprint(model.state_dict());save(RUN/'checkpoints',model,opt,sched,p,c['run_hash'],lineage={'reason':'pre_evaluation'})
            began=time.perf_counter();metrics=evaluation.evaluate(model,valid,rows,p,c['run_hash'],deadline)
            append(folder/'evaluation_telemetry.jsonl',{'epoch':epoch+1,'seconds':time.perf_counter()-began,'utc':utc(),'complete':metrics is not None})
            if metrics is None:
                meta=save(RUN/'checkpoints',model,opt,sched,p,c['run_hash'],lineage={'reason':'evaluation_ceiling'})
                durable(folder/'RESULT.json',{'utc':utc(),'status':'incomplete_budget','progress':status(p),'latest':meta});return
            assert fingerprint(model.state_dict())==p['evaluation_model_sha256'],'Evaluation changed model'
            select(p,metrics['f1'],epoch+1)
            p['history'].append({'epoch':epoch+1,'updates':p['updates'],'training_loss':p['epoch_loss_sum']/p['epoch_valid_labels'],
                'training_valid_labels':p['epoch_valid_labels'],'dev_micro_f1':metrics['f1'],'dev_precision':metrics['precision'],'dev_recall':metrics['recall'],
                'bio_repairs':metrics['bio_repairs'],'best_epoch':p['best_epoch'],'bad_epochs':p['bad_epochs']})
            p.update(epoch=epoch+1,cursor=0,epoch_loss_sum=0.,epoch_valid_labels=0,phase='complete' if epoch+1>=5 or p['stop'] else 'train')
            meta=save(RUN/'checkpoints',model,opt,sched,p,c['run_hash'],epoch=epoch+1,lineage={'reason':'post_evaluation'})
            write_json(RUN/'learning_curve.json',p['history']);write_json(RUN/'status.json',{'utc':utc(),**status(p)});print(p['history'][-1],flush=True)
        write_jsonl(RUN/'training_committed.jsonl',p['trace']);selected=RUN/'evaluations'/f'epoch_{p["best_epoch"]:02d}'
        for name in ['predictions.jsonl','metrics.json','distributions.json']:shutil.copy2(selected/name,RUN/name)
        durable(folder/'RESULT.json',{'utc':utc(),'status':'complete','progress':status(p),'latest':meta,'seconds':time.perf_counter()-tick})
    except BaseException as e:
        durable(folder/'FAILURE.json',{'utc':utc(),'type':type(e).__name__,'error':str(e),'traceback':traceback.format_exc(),'safe_boundary':safe,'progress':status(p) if p else None});raise

def postwrite():
    from .reader import inspect
    verify_plan();result=read_json(D/'WRITER_RESULT.json');records=[]
    for receipt in sorted((RUN/'checkpoints').glob('generation_*.committed.json')):
        r=read_json(receipt);metadata=receipt.parent/r['metadata'];m=read_json(metadata);payload=metadata.parent/m['file']
        if not payload.exists():
            retired=read_json(metadata.parent/f'generation_{m["sequence"]:06d}.retired.json')
            assert retired['sha256']==m['sha256'] and m['checkpoint_identity']['logical_name'].startswith('rolling_')
            records.append({'metadata':str(metadata),'retired':True,'sha256':m['sha256']});continue
        proof=inspect(metadata,D/'audit/postwrite'/f'generation_{m["sequence"]:06d}.json');assert proof['accepted']
        records.append({'metadata':str(metadata),'retired':False,'sha256':m['sha256'],'verified':True})
    assert records
    durable(D/'audit/POSTWRITE.json',{'utc':utc(),'all_verified':True,'writer_complete':result['status']=='complete',
        'selected_epoch':result['progress']['best_epoch'],'retained_and_retired':records})

def benchmark(microbatch):
    import torch,statistics
    verify_plan();c=config();data=load_data('train');model,tok,opt,sched=initialize();collate=collator(tok);indices=order(0,len(data));times=[]
    result={'microbatch':microbatch,'accumulation':16//microbatch,'updates':0,'status':'running','score_free':True}
    try:
        torch.cuda.reset_peak_memory_stats()
        for u in range(25):
            torch.cuda.synchronize();t=time.perf_counter();values=update(model,opt,sched,[data[i] for i in indices[u*16:(u+1)*16]],microbatch,collate);torch.cuda.synchronize()
            assert all(math.isfinite(v) for v in values.values() if isinstance(v,(int,float)))
            if u>=5:times.append(time.perf_counter()-t)
            result['updates']+=1
        longest=sorted(range(len(data)),key=lambda i:(-len(data[i]['input_ids']),i))[:16]
        values=update(model,opt,sched,[data[i] for i in longest],microbatch,collate);torch.cuda.synchronize()
        assert all(math.isfinite(v) for v in values.values() if isinstance(v,(int,float)))
        peak=torch.cuda.max_memory_reserved();total=torch.cuda.get_device_properties(0).total_memory
        result.update(status='passed',timed_seconds=times,mean_seconds=statistics.mean(times),cv=statistics.stdev(times)/statistics.mean(times),
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=peak,total_vram_bytes=total,headroom_fraction=1-peak/total,
            longest_batch_ids=[data[i]['id'] for i in longest],longest_batch_lengths=[len(data[i]['input_ids']) for i in longest])
    except torch.cuda.OutOfMemoryError as e:result.update(status='memory_rejected',error=str(e))
    except BaseException as e:result.update(status='failed',error=str(e));raise
    finally:durable(D/f'BENCHMARK_{microbatch}.json',result)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['capture','restore','train','postwrite','benchmark']);p.add_argument('--stage');p.add_argument('--deadline',type=float);p.add_argument('--resume');p.add_argument('--microbatch',type=int);a=p.parse_args()
    if a.mode in ['capture','restore']:zero(a.mode)
    elif a.mode=='train':train(a.stage,a.deadline,a.resume)
    elif a.mode=='postwrite':postwrite()
    else:benchmark(a.microbatch)
