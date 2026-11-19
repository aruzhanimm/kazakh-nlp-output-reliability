"""Read-only compact progress and conservative remaining-to-five-epochs estimate."""
import time
from pilot.common import read_json
from .common import D

def main():
    budget=read_json(D/'main_budget.json');path=D/'run_xlmr_ner_seed13_lr1e-5';state_path=path/'status.json'
    state=read_json(state_path) if state_path.exists() else {}
    updates=state.get('updates',0);seconds=state.get('active_training_seconds',0)
    rate=updates/seconds if seconds else None
    remaining=(27935-updates)/rate if rate else None
    print({'status':budget['status'],'phase':state.get('phase'),'epoch_index':state.get('epoch'),'updates':updates,
        'epoch_examples':state.get('cursor'),'dev_predictions':state.get('dev_predictions'),'wall_minutes':round((time.time()-budget['start_unix'])/60,1),
        'active_features_per_second':round(16*rate,2) if rate else None,
        'remaining_training_minutes_to_max5':round(remaining/60,1) if remaining else None,
        'epoch_f1_percent':[(h['epoch'],round(100*h['dev_micro_f1'],3)) for h in state.get('history',[])],
        'deadline_remaining_minutes':round((budget['deadline_unix']-time.time())/60,1)})

if __name__=='__main__':main()
