from datetime import datetime,timezone
from pilot.common import ROOT,read_json,write_json,file_hash,digest
D=ROOT/'encoder_lr_screen02'
RUN=D/'run'
COMPONENTS=('model','optimizer','scheduler','rng','progress')
def utc():return datetime.now(timezone.utc).isoformat()
def verify_plan(unused=None):
    from encoder_main01.common import verify
    verify('main_plan.lock.json')
    binding=read_json(D/'plan.lock.json')
    for name,h in binding['files'].items():assert file_hash(ROOT/name)==h,name
    return digest(binding)
def identity(path,spec,unused=None):
    assert path==RUN and spec==read_json(D/'main_config.json')
    value=read_json(D/'identity.json');assert value['plan_sha256']==verify_plan()
    target=path/'identity.json'
    if target.exists():assert read_json(target)==value
    else:write_json(target,value)
    return digest(value)
