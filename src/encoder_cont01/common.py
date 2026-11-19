import os
from pathlib import Path
from datetime import datetime,timezone
from pilot.common import ROOT,read_json,write_json,file_hash,digest
D=ROOT/'encoder_cont01'
OLD=ROOT/'encoder_main01'
OLD_RUN=OLD/'run_xlmr_ner_seed13_lr1e-5'
RUN=D/'run'
PAYLOAD=OLD_RUN/'checkpoints/rolling_1.pt'
EXPECTED_SHA='fffcaf4caa0eff676da382b506d349dad32d5d92120387a9b54a995f9f499e95'
RUN_HASH='e44d8c77d5944029b0fa239e62cea963dd96b1a598d7cef5e535285b7032701c'

def utc():return datetime.now(timezone.utc).isoformat()

def old_files(check=False):
    target=D/'audit/previous_files.json'
    if check:
        original=read_json(target)['files'];bad=[p for p,h in original.items() if not (ROOT/p).exists() or file_hash(ROOT/p)!=h]
        result={'utc':utc(),'files':len(original),'changed_or_missing':bad};write_json(D/'audit/previous_files_verified.json',result);assert not bad,bad;print(result);return
    assert not target.exists()
    out={}
    for folder,dirs,names in os.walk(ROOT,followlinks=False):
        dirs[:]=sorted(n for n in dirs if n not in {'encoder_cont01','.venv','.cache','.git','__pycache__','.pytest_cache','tmp'} and not (Path(folder)/n).is_symlink() and not (Path(folder)/n).is_junction())
        for n in sorted(names):
            p=Path(folder)/n
            if not n.endswith(('.pyc','.tmp')):out[p.relative_to(ROOT).as_posix()]=file_hash(p)
    write_json(target,{'utc':utc(),'files':out});print({'preserved_files':len(out)})

def verify_plan():
    from encoder_main01.common import verify
    verify('main_plan.lock.json');verify('benchmark_plan.lock.json')
    binding=read_json(D/'plan.lock.json')
    for p,h in binding['files'].items():assert file_hash(ROOT/p)==h,p
    return digest(binding)

def no_old_watcher():
    import psutil
    assert (OLD/'shutdown_watch/CANCEL').exists(),'Old watcher cancellation guard is missing'
    found=[]
    for p in psutil.process_iter(['pid','name','cmdline']):
        args=p.info['cmdline'] or []
        if any(args[i]=='-m' and args[i+1]=='encoder_main01.completion_watcher' for i in range(len(args)-1)):
            found.append({'pid':p.pid,'cmdline':args})
    assert not found,('Old watcher still active',found)
    return {'utc':utc(),'old_watchers':found,'cancel_guard':True,'shutdown_enabled':False}

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('action',choices=['preserve','verify']);a=p.parse_args();old_files(a.action=='verify')
