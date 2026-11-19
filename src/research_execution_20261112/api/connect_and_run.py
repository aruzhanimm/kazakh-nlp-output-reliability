"""Interactive local-only key entry; no key file or chat message is required."""
from pathlib import Path
import argparse
import getpass
import json
import os
import subprocess
import sys
from api_common import ROOT, read_evidence, validate_package
from run_api import authorization

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for flag in ('package','store','approval'):p.add_argument('--'+flag,required=True)
    p.add_argument('--stage',choices=['smoke','dev'],required=True)
    args=p.parse_args();package=read_evidence(args.package)
    validate_package(package)
    approved=json.loads(Path(args.approval).read_text(encoding='utf-8'))
    authorization(approved,package,args.store,args.stage)
    if approved.get('execution_mode')!='live_openai':raise PermissionError('A live approval is required')
    if not sys.stdin.isatty():raise RuntimeError('Open an interactive local terminal for hidden key entry')
    secret=getpass.getpass('OpenAI project API key (hidden; never paste into chat): ')
    if not secret.strip():raise RuntimeError('No key entered')
    env=dict(os.environ);env['OPENAI_API_KEY']=secret
    try:
        code=subprocess.call([sys.executable,'-B',str(ROOT/'run_api.py'),'execute',
            '--package',args.package,'--store',args.store,'--approval',args.approval,
            '--stage',args.stage,'--execute-paid-api'],env=env)
    finally:
        env.pop('OPENAI_API_KEY',None);secret=None
    raise SystemExit(code)
