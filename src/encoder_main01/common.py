import os
import shutil
from pathlib import Path
from datetime import datetime,timezone
from pilot.common import ROOT,read_json,write_json,file_hash,digest,environment
D=ROOT/'encoder_main01'

def preserve(check=False):
    target=D/'preservation.json'
    if check:
        old=read_json(target);bad=[n for n,h in old['files'].items() if not (ROOT/n).exists() or file_hash(ROOT/n)!=h]
        out={'utc':datetime.now(timezone.utc).isoformat(),'files':len(old['files']),'changed_or_missing':bad};write_json(D/'audit/preservation_verified.json',out);assert not bad,bad;print(out);return
    if target.exists():return
    files=[]
    for folder,dirs,names in os.walk(ROOT,followlinks=False):
        dirs[:]=sorted(n for n in dirs if n not in {'encoder_main01','.venv','.cache','.git','__pycache__','.pytest_cache','tmp'}
            and not (Path(folder)/n).is_symlink() and not (Path(folder)/n).is_junction())
        files.extend(Path(folder)/n for n in sorted(names) if not n.endswith(('.pyc','.tmp')))
    write_json(target,{'utc':datetime.now(timezone.utc).isoformat(),'files':{p.relative_to(ROOT).as_posix():file_hash(p) for p in files}})
    print({'preserved_files':len(files)})

def freeze(names,filename):
    binding={'files':{n:file_hash(ROOT/n) for n in names}}
    target=D/filename
    if target.exists():assert read_json(target)==binding,'Frozen plan/source changed'
    else:write_json(target,binding)
    return digest(binding)

def verify(filename):
    binding=read_json(D/filename)
    for n,h in binding['files'].items():assert file_hash(ROOT/n)==h,n
    return digest(binding)

def identity(path,spec,lock):
    assert Path(path).resolve().is_relative_to(D.resolve())
    value={**spec,'plan_sha256':verify(lock),'environment':environment()};target=path/'identity.json'
    if target.exists():assert read_json(target)==value,'Incompatible run identity'
    else:
        write_json(target,value)
        for n in read_json(D/lock)['files']:
            if n.endswith(('.py','.md','.txt','.json')) and not n.endswith('preservation.json'):
                dest=path/'source'/n;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/n,dest)
    return digest(value)

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('action',choices=['preserve','verify-preservation']);a=p.parse_args();preserve(a.action=='verify-preservation')
