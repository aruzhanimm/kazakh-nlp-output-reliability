"""Score-independent execution controls and the previously proposed stopping rule."""
def select(progress,score,epoch):
    previous=progress.get('best_f1',-1.)
    progress['bad_epochs']=0 if score>previous+.0001 else progress.get('bad_epochs',0)+1
    if score>previous:
        progress.update(best_f1=score,best_epoch=epoch)
    progress['stop']=epoch>=3 and progress['bad_epochs']>=2
    return progress

def new_progress():
    return {'epoch':0,'cursor':0,'updates':0,'phase':'train','checkpoint_sequence':0,'rolling_sequence':0,
        'trace':[],'history':[],'epoch_loss_sum':0.,'epoch_examples':0,'active_training_seconds':0.,
        'peak_gpu_allocated_bytes':0,'peak_gpu_reserved_bytes':0,'planned_restart_done':False,
        'bad_epochs':0,'best_f1':-1.,'best_epoch':None,'stop':False}

def status(progress):
    return {k:v for k,v in progress.items() if k not in ['trace']}
