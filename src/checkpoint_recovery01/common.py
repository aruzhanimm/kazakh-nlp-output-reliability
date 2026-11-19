import time
from datetime import datetime,timezone
from pilot.common import ROOT,read_json,write_json,file_hash,digest
D=ROOT/'checkpoint_recovery01'
RUN=D/'run'
PREVIOUS=ROOT/'encoder_cont01'
INPUT=PREVIOUS/'run/checkpoints/rolling_1.pt'
INPUT_SHA='bead52a8c91bfcd481a4dd062594b2ff62cd29b9e4e31269de8285d64233b899'
RUN_HASH='e44d8c77d5944029b0fa239e62cea963dd96b1a598d7cef5e535285b7032701c'
COMPONENTS=['model','optimizer','scheduler','rng','progress']

def utc():return datetime.now(timezone.utc).isoformat()
def prepare_guard():assert time.time()<read_json(D/'PREPARATION_BUDGET.json')['deadline_unix'],'Preparation ceiling reached'
def verify_plan():
    from encoder_cont01.common import verify_plan as previous_plan
    previous_plan()
    p=read_json(D/'plan.lock.json')
    for name,h in p['files'].items():assert file_hash(ROOT/name)==h,name
    return digest(p)

def start():
    assert not (D/'PREPARATION_BUDGET.json').exists()
    start=datetime(2026,11,4,12,35,14,tzinfo=timezone.utc).timestamp()
    write_json(D/'PREPARATION_BUDGET.json',{'started_utc':'2026-11-04T12:35:14+00:00','start_unix':start,
        'deadline_unix':start+3600,'limit_seconds':3600,'status':'preparing','historical_cause':'unresolved',
        'replay_attempts':0,'shutdown_enabled':False,'final_test_locked':True})

if __name__=='__main__':start()
