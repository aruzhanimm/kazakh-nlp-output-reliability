import time
from datetime import datetime,timezone
from pilot.common import ROOT,read_json,write_json,file_hash
D=ROOT/'checkpoint_integrity01'
OLD=ROOT/'encoder_cont01'
CK=OLD/'run/checkpoints'
COMPONENTS=['model','optimizer','scheduler','rng','progress']

def utc():return datetime.now(timezone.utc).isoformat()
def guard():
    budget=read_json(D/'phase_a_budget.json')
    assert time.time()<budget['deadline_unix'],'Phase A 60-minute ceiling reached'
    assert budget['phase_b_authorized_to_start'] is False

def start():
    assert not (D/'phase_a_budget.json').exists()
    start=datetime(2026,11,4,11,50,34,tzinfo=timezone.utc).timestamp()
    write_json(D/'phase_a_budget.json',{'started_utc':'2026-11-04T11:50:34+00:00','start_unix':start,
        'deadline_unix':start+3600,'limit_seconds':3600,'status':'diagnosing','phase_b_authorized_to_start':False,
        'phase_b_condition':'Established cause and validated fix confined to saving/verification; one replay only',
        'shutdown_enabled':False,'final_test_locked':True})

if __name__=='__main__':start()
