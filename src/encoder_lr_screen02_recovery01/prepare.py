import ast,difflib,shutil,subprocess,sys,time
from datetime import datetime
from pilot.common import ROOT,read_json,file_hash,environment,digest
from encoder_lr_screen02.common import verify_plan as original_verify
from .common import D,META,INPUT,PIN,utc,durable,env

def main():
    assert not (D/'CONFIG.lock.json').exists();start=datetime.fromisoformat('2026-11-09T08:26:58+00:00').timestamp()
    assert time.time()<start+2700;original_verify()
    for p in D.glob('*.py'):ast.parse(p.read_text(encoding='utf-8'))
    with (D/'tests.log').open('x',encoding='utf-8') as f:
        code=subprocess.run([sys.executable,'-B','-X','utf8','-m','unittest','encoder_lr_screen02_recovery01.test_streams','encoder_lr_screen02_recovery01.test_checkpoint','-v'],cwd=ROOT,env=env(),stdout=f,stderr=subprocess.STDOUT).returncode
    assert code==0,'Unexpected preparation test failure; stop'
    before={}
    for name in ['encoder_lr_screen02','encoder_lr_screen02_diag01','encoder_lr_screen02_diag02','encoder_lr_screen02_diag03','encoder_lr_screen02_diag04']:
        for p in (ROOT/name).rglob('*'):
            if p.is_file():before[p.relative_to(ROOT).as_posix()]={'sha256':file_hash(p),'bytes':p.stat().st_size}
    durable(D/'audit/preservation_before.json',before)
    scientific=read_json(ROOT/'encoder_lr_screen02/main_config.json');identity=read_json(ROOT/'encoder_lr_screen02/identity.json');meta=read_json(META)
    assert meta['sha256']==PIN==file_hash(INPUT) and digest(identity)==meta['run_hash']
    current=environment();assert current['packages']==identity['environment']['packages']
    c={'start_unix':start,'total_deadline_unix':start+14400,'preparation_deadline_unix':start+2700,'preparation_limit_seconds':2700,
        'continuation_limit_seconds':10800,'report_reserve_seconds':900,'postwriter_reserve_seconds':600,'frozen_utc':utc(),
        'input':INPUT.relative_to(ROOT).as_posix(),'input_sha256':PIN,'sidecar_sha256':file_hash(META),'scientific':scientific,'scientific_run_hash':meta['run_hash'],
        'packages':current['packages'],'threads':[16,16],'policy':'initialize before CPU checkpoint load; retain source through synchronized all-five restore verification',
        'trajectories':{'C':{'start':1001,'end':1100,'saves':[1050,1100]},'R1':{'start':1001,'end':1050,'saves':[1050]},'R2':{'start':1051,'end':1100,'saves':[1100]}},
        'stage_order':['roundtrip_capture','roundtrip_restore','roundtrip_gate','C','R1','compare1050','R2','compare1100','continue_C','postwrite','fresh_reload'],
        'continuation_branch':'C','equality':'all five components and entire progress; exact traces and batch IDs; no excluded progress fields',
        'telemetry_policy':'Inherited fields and trace prefix retained; new telemetry external only','new_checkpoint_retention':'all committed generations; no historical or new payload deletion',
        'lr_selection_rule':'best full-dev exact-span/type micro-F1; exact tie retains 1e-5','historical_cause':'unresolved; not an acceptance gate',
        'shutdown_enabled':False,'final_test_locked':True,'other_conditions_authorized':False}
    durable(D/'CONFIG.lock.json',c)
    protected=[]
    for folder in ['pilot','stage_preflight','encoder_main01','encoder_cont01','checkpoint_integrity01','checkpoint_recovery01','encoder_lr_screen02']:
        protected.extend((ROOT/folder).glob('*.py'))
    protected += [ROOT/n for n in ['encoder_lr_screen02/main_config.json','encoder_lr_screen02/identity.json','encoder_lr_screen02/COMPARATOR.json',
        'stage_preflight/views/manifest.json','configs/models.lock.json','data/derived/ner_labels.json','artifacts/encoder_snapshot_hashes.json',
        'stage_preflight/features/xlmr_ner_train.jsonl','stage_preflight/features/xlmr_ner_valid.jsonl','stage_preflight/views/ner_valid.jsonl',
        'encoder_lr_screen02_diag01/evidence.py','encoder_lr_screen02_diag01/common.py','encoder_lr_screen02_diag04/streams.py']]
    snapshot=ROOT/'.cache/huggingface/models--FacebookAI--xlm-roberta-base/snapshots'/scientific['initialization']['revision']
    pins=read_json(ROOT/'artifacts/encoder_snapshot_hashes.json')
    for n,h in pins['files'].items():assert file_hash(snapshot/n)==h
    protected+=list(snapshot.iterdir())+list((ROOT/'artifacts/tokenizers/encoder').iterdir())+[INPUT,META]
    files={p.relative_to(ROOT).as_posix():file_hash(p) for p in protected if p.is_file()}
    for p in list(D.glob('*.py'))+[D/'PLAN.md',D/'CONFIG.lock.json']:files[p.relative_to(ROOT).as_posix()]=file_hash(p)
    durable(D/'SOURCE.lock.json',{'utc':utc(),'files':files})
    for p in protected+list(D.glob('*.py'))+[D/'PLAN.md',D/'CONFIG.lock.json']:
        if p.suffix not in ['.py','.md','.json'] or p.stat().st_size>10*1024**2:continue
        target=D/'source'/p.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target)
    diffs=[]
    for old,new in [('encoder_main01/main.py','train.py'),('encoder_lr_screen02/checkpoints.py','checkpoints.py'),('encoder_lr_screen02/restore.py','restore.py')]:
        diffs.extend(difflib.unified_diff((ROOT/old).read_text(encoding='utf-8').splitlines(True),(D/new).read_text(encoding='utf-8').splitlines(True),fromfile=old,tofile=D.name+'/'+new))
    (D/'SOURCE_DIFF.patch').write_text(''.join(diffs),encoding='utf-8')
    free=shutil.disk_usage(D).free;assert free>=160*2**30
    durable(D/'PREPARATION_GATE.json',{'utc':utc(),'passed':True,'source_tests_passed':6,'preserved_files':len(before),'disk_free_bytes':free,'disk_reserve_bytes':160*2**30,
        'packages_equal_original':True,'scientific_configuration_unchanged':True,'historical_cause_unresolved':True,'elapsed_seconds':time.time()-start})
    print({'prepared':True,'elapsed_seconds':time.time()-start},flush=True)

if __name__=='__main__':main()
