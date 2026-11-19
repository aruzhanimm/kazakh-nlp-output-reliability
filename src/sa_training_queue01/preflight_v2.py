"""Preparation-only adapter: append a new disk audit instead of replacing evidence."""
import math,shutil
from pilot.common import file_hash,read_json
from .common import Q,utc,durable
from .disk import estimate
from . import preflight

def disk_recheck():
    e=estimate();required=math.ceil(1.2*(4*sum(e['per_condition_persistent_projection'].values())+e['two_inflight_payload_allowance']+e['fixed_prediction_log_allowance']))
    free=shutil.disk_usage(Q).free;e.update(utc=utc(),free_bytes=free,required_bytes=required,passed=free>=required)
    durable(Q/'DISK_BEFORE_PREFLIGHT.json',e)
    assert e['passed'],'Full authorized SA queue does not fit without deleting protected artifacts'
    return e

if __name__=='__main__':
    amendment=read_json(Q/'PREPARATION_AMENDMENT01.json')
    assert file_hash(Q/'preflight_v2.py')==amendment['adapter_sha256']
    assert file_hash(Q/'SOURCE.lock.json')==amendment['original_source_lock_sha256']
    preflight.audit=disk_recheck
    preflight.main()
