"""Frozen scientific update/evaluation functions, external telemetry, bounded recovery."""
import argparse,math,time,traceback,shutil
from pathlib import Path
from pilot.common import ROOT,read_json,read_jsonl,write_json,write_jsonl,digest,file_hash
from stage_preflight.state import fingerprint
from encoder_main01.core import load_data,order,collator,update
from encoder_main01.control import select,status
from .common import D,RUN,META,utc,durable,append,initialize,verify_plan,config
from .restore import restore
from .checkpoints import save,committed_meta

def run(mode,deadline):
    import torch
    import encoder_main01.main as evaluation
    verify_plan();c=config();spec=read_json(ROOT/'encoder_lr_screen02/main_config.json')
    folder=D/'trajectories'/mode if mode!='continue' else RUN/'execution'
    folder.mkdir(parents=True,exist_ok=False);start=time.perf_counter();progress=None;safe=False
    output=D/'trajectories/C/checkpoints' if mode=='C' else D/'trajectories/R/checkpoints' if mode.startswith('R') else RUN/'checkpoints'
    output.mkdir(parents=True,exist_ok=True)
    try:
        train=load_data('train');model,tok,opt,sched=initialize();torch.cuda.synchronize()
        if mode in ['C','R1']:metadata=META
        elif mode=='R2':metadata=committed_meta(D/'trajectories/R/checkpoints')[0]
        else:
            assert read_json(D/'validation/EQUIVALENCE.json')['passed']
            metadata=committed_meta(D/'trajectories/C/checkpoints')[0]
        restore_start=time.perf_counter();progress,source,input_meta=restore(metadata,model,opt,sched,folder/'restore')
        required={'C':1000,'R1':1000,'R2':1050,'continue':1100}[mode]
        assert progress['updates']==required and progress['epoch']==0 and progress['cursor']==required*16
        inherited={k:progress[k] for k in ['active_training_seconds','peak_gpu_allocated_bytes','peak_gpu_reserved_bytes']}
        del source  # Source retained until all synchronized restoration verification finished.
        durable(folder/'load.json',{'utc':utc(),'data_initialize_restore_seconds':time.perf_counter()-start,'restore_seconds':time.perf_counter()-restore_start,
            'input_metadata':str(metadata),'input_sha256':input_meta['sha256'],'frozen_legacy_telemetry_fields':inherited,'threads':[torch.get_num_threads(),torch.get_num_interop_threads()],
            'gpu':torch.cuda.get_device_name(0),'vram_bytes':torch.cuda.get_device_properties(0).total_memory,'attention':model.config._attn_implementation})
        run_hash=input_meta['run_hash'];collate=collator(tok);safe=True;target=1100 if mode in ['C','R2'] else 1050 if mode=='R1' else 27935
        if mode=='continue':
            valid=load_data('valid');rows=evaluation.dev_rows();evaluation.RUN=RUN
            write_json(RUN/'identity.json',{'scientific_original_identity':read_json(ROOT/'encoder_lr_screen02/identity.json'),
                'scientific_run_hash':run_hash,'recovery_configuration_sha256':file_hash(D/'CONFIG.lock.json'),'predefined_branch':'C','parent':input_meta})
        torch.cuda.reset_peak_memory_stats();latest_meta=None
        while progress['phase']!='complete':
            epoch=progress['epoch'];indices=order(epoch,len(train));ordering={'epoch':epoch+1,'seed':13+epoch,'feature_ids':[train[i]['id'] for i in indices]}
            original=ROOT/'encoder_lr_screen02/run/orders'/f'epoch_{epoch+1:02d}.json'
            if original.exists():assert read_json(original)==ordering,'Frozen data order mismatch'
            order_path=folder/'orders'/f'epoch_{epoch+1:02d}.json'
            if not order_path.exists():durable(order_path,ordering)
            if progress['phase']=='train':
                for cursor in range(progress['cursor'],len(train),16):
                    if time.time()>=deadline-150:
                        latest_meta=save(output,model,opt,sched,progress,run_hash,lineage={'parent':input_meta,'reason':'time_ceiling'})
                        durable(folder/'RESULT.json',{'utc':utc(),'status':'incomplete_budget','progress':status(progress),'latest':latest_meta,'seconds':time.perf_counter()-start});return
                    positions=indices[cursor:cursor+16];ids=[train[i]['id'] for i in positions]
                    torch.cuda.synchronize();began=utc();tick=time.perf_counter();safe=False
                    values=update(model,opt,sched,[train[i] for i in positions],spec['microbatch'],collate)
                    torch.cuda.synchronize();seconds=time.perf_counter()-tick
                    assert all(math.isfinite(v) for v in values.values() if isinstance(v,(float,int))),'Nonfinite update trace'
                    progress['updates']+=1;progress['cursor']=cursor+len(positions)
                    progress['epoch_loss_sum']+=values['loss_sum'];progress['epoch_valid_labels']+=values['valid_labels']
                    record={'update':progress['updates'],'epoch':epoch+1,'cursor_after':progress['cursor'],'feature_ids_sha256':digest(ids),**values}
                    # All inherited fields/prefix remain intact. New telemetry has no canonical state entry.
                    progress['trace'].append(record);append(folder/'scientific_trace.jsonl',record)
                    append(folder/'batch_order.jsonl',{'update':progress['updates'],'feature_ids':ids})
                    append(folder/'telemetry.jsonl',{'update':progress['updates'],'seconds':seconds,'started_utc':began,'finished_utc':utc(),
                        'peak_gpu_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_gpu_reserved_bytes':torch.cuda.max_memory_reserved()})
                    assert all(progress[k]==v for k,v in inherited.items());safe=True
                    if progress['updates']%100==0:
                        write_json(folder/'status.json',{'utc':utc(),**status(progress)});print({'mode':mode,'updates':progress['updates'],'epoch':epoch+1,'cursor':progress['cursor'],'loss':values['loss']},flush=True)
                    checkpoint=(progress['updates'] in [1050,1100]) if mode!='continue' else progress['updates']%1000==0
                    if checkpoint:
                        latest_meta=save(output,model,opt,sched,progress,run_hash,lineage={'parent_metadata':str(metadata),'parent_sha256':input_meta['sha256']})
                        durable(folder/f'checkpoint_{progress["updates"]}.json',latest_meta)
                    if mode!='continue' and progress['updates']==target:
                        assert latest_meta and latest_meta['updates']==target
                        durable(folder/'RESULT.json',{'utc':utc(),'status':'trajectory_complete','progress':status(progress),'latest':latest_meta,'seconds':time.perf_counter()-start});return
                assert progress['cursor']==89386 and progress['epoch_valid_labels']==1034279
                progress['phase']='evaluate';progress['evaluation_model_sha256']=fingerprint(model.state_dict())
                latest_meta=save(output,model,opt,sched,progress,run_hash,lineage={'parent':input_meta,'reason':'pre_evaluation'})
            assert mode=='continue'
            tick=time.perf_counter();metrics=evaluation.evaluate(model,valid,rows,progress,run_hash,deadline)
            append(folder/'evaluation_telemetry.jsonl',{'epoch':epoch+1,'seconds':time.perf_counter()-tick,'utc':utc(),'complete':metrics is not None})
            if metrics is None:
                latest_meta=save(output,model,opt,sched,progress,run_hash,lineage={'reason':'evaluation_ceiling'})
                durable(folder/'RESULT.json',{'utc':utc(),'status':'incomplete_budget','progress':status(progress),'latest':latest_meta,'seconds':time.perf_counter()-start});return
            assert fingerprint(model.state_dict())==progress['evaluation_model_sha256'],'Model changed during evaluation'
            score=metrics['f1'];select(progress,score,epoch+1)
            progress['history'].append({'epoch':epoch+1,'updates':progress['updates'],'training_loss':progress['epoch_loss_sum']/progress['epoch_valid_labels'],
                'training_valid_labels':progress['epoch_valid_labels'],'dev_micro_f1':score,'dev_precision':metrics['precision'],'dev_recall':metrics['recall'],
                'bio_repairs':metrics['bio_repairs'],'best_epoch':progress['best_epoch'],'bad_epochs':progress['bad_epochs']})
            progress.update(epoch=epoch+1,cursor=0,epoch_loss_sum=0.,epoch_valid_labels=0,phase='complete' if epoch+1>=5 or progress['stop'] else 'train')
            latest_meta=save(output,model,opt,sched,progress,run_hash,epoch=epoch+1,lineage={'reason':'post_evaluation'})
            write_json(RUN/'learning_curve.json',progress['history']);write_json(folder/'status.json',{'utc':utc(),**status(progress)})
            print(progress['history'][-1],flush=True)
        write_jsonl(RUN/'training_committed.jsonl',progress['trace'])
        selected=RUN/'evaluations'/f'epoch_{progress["best_epoch"]:02d}'
        for name in ['predictions.jsonl','metrics.json','distributions.json']:shutil.copy2(selected/name,RUN/name)
        durable(folder/'RESULT.json',{'utc':utc(),'status':'complete','progress':status(progress),'latest':latest_meta,'seconds':time.perf_counter()-start})
    except BaseException as e:
        durable(folder/'FAILURE.json',{'utc':utc(),'type':type(e).__name__,'error':str(e),'traceback':traceback.format_exc(),'safe_boundary':safe,'progress':status(progress) if progress else None})
        raise  # No failure-triggered save or retry of a possibly half-applied update.

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['C','R1','R2','continue'],required=True);p.add_argument('--deadline',type=float,required=True);a=p.parse_args();run(a.mode,a.deadline)
