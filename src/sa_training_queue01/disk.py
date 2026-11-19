import math,shutil
from .common import Q,ROOT,read_json,utc,durable

def estimate():
    sizes={enc:(ROOT/f'stage_preflight/calibrations/{enc}_sa_s13/checkpoint_100.pt').stat().st_size for enc in ['xlmr','mbert']}
    # Five epochs + two rolling full states + initial validation (~one third state).
    # Progress/trace growth covered by 5%; 20% aggregate margin and 8 GiB predictions/logs.
    per={k:math.ceil(v*1.05*(7+1/3)) for k,v in sizes.items()}
    transient=math.ceil(max(sizes.values())*1.05*2)
    return {'measured_calibration_full_state_bytes':sizes,'per_condition_persistent_projection':per,'two_inflight_payload_allowance':transient,
        'fixed_prediction_log_allowance':8*2**30,'aggregate_margin':1.2,'source':'file sizes only; calibration weights never loaded'}

def required_remaining(q):
    e=estimate();remaining=sum(e['per_condition_persistent_projection'][x['encoder']] for x in q['conditions'] if x['status']!='verified_complete')
    return math.ceil(1.2*(remaining+e['two_inflight_payload_allowance']+e['fixed_prediction_log_allowance']))

def audit():
    e=estimate();required=math.ceil(1.2*(4*sum(e['per_condition_persistent_projection'].values())+e['two_inflight_payload_allowance']+e['fixed_prediction_log_allowance']))
    free=shutil.disk_usage(Q).free;e.update(utc=utc(),free_bytes=free,required_bytes=required,passed=free>=required)
    durable(Q/'DISK_PLAN.json',e);assert e['passed'],'Full authorized SA queue does not fit without deleting protected artifacts';return e
