import os
import shutil
from pathlib import Path
from datetime import datetime, timezone
from pilot.common import ROOT, digest, file_hash, read_json, write_json, environment

D = ROOT/'stage_preflight'

def preserve(check=False):
    path=D/'preservation.json'
    if check:
        original=read_json(path); bad=[n for n,h in original['files'].items() if not (ROOT/n).exists() or file_hash(ROOT/n)!=h]
        note={'utc':datetime.now(timezone.utc).isoformat(),'files':len(original['files']),'changed_or_missing':bad}
        write_json(D/'audit/preservation_verified.json',note); assert not bad,bad; print(note); return
    if path.exists(): return
    paths=[]
    for folder,dirs,names in os.walk(ROOT,followlinks=False):
        dirs[:]=sorted(n for n in dirs if n not in {'stage_preflight','.venv','.cache','.git','__pycache__','.pytest_cache','tmp'}
            and not (Path(folder)/n).is_symlink() and not (Path(folder)/n).is_junction())
        paths += [Path(folder)/n for n in sorted(names) if not n.endswith(('.pyc','.tmp'))]
    write_json(path,{'utc':datetime.now(timezone.utc).isoformat(),'files':{p.relative_to(ROOT).as_posix():file_hash(p) for p in paths}})
    print({'prior_files':len(paths)})

def freeze(names, filename):
    binding={'files':{n:file_hash(ROOT/n) for n in names}}
    path=D/filename
    if path.exists():
        if read_json(path)!=binding: raise ValueError('Frozen source/input changed: '+filename)
    else: write_json(path,binding)
    return digest(binding)

def verify(filename):
    binding=read_json(D/filename)
    for n,h in binding['files'].items():
        if file_hash(ROOT/n)!=h: raise ValueError('Frozen source/input changed: '+n)
    return digest(binding)

def identity(path, spec, lock):
    path=Path(path); assert path.resolve().is_relative_to(D.resolve())
    spec={**spec,'plan_sha256':verify(lock),'environment':environment()}
    target=path/'identity.json'
    if target.exists():
        if read_json(target)!=spec: raise ValueError('Incompatible run identity')
    else:
        write_json(target,spec)
        for name in read_json(D/lock)['files']:
            if name.endswith(('.py','.md','.txt','.json')) and not name.endswith('preservation.json'):
                dest=path/'source'/name; dest.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(ROOT/name,dest)
    return digest(spec)

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(); p.add_argument('action',choices=['preserve','verify-preservation']); a=p.parse_args(); preserve(a.action=='verify-preservation')
