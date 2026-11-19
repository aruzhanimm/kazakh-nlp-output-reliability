"""Server-only isolated Gemma runtime and pinned downloads; never starts inference."""
from pathlib import Path
import argparse,hashlib,importlib.metadata,json,os,shutil,subprocess,sys,urllib.request

HERE=Path(__file__).resolve().parent
REPO='google/gemma-4-12B-it'
REV='43c9d014b231474fa7212dbb48a762566c2b255d'
TF_URL='https://files.pythonhosted.org/packages/ff/bb/cc0b27a3000ddcf4903d3a26983c8b4c9a36050bbaf05675f353e8bcb08c/transformers-5.22.2-py3-none-any.whl#sha256=829c2820aff91730b8ec45a4b8ff4be07f586a140b68a9073998b36fa35297dc'
REQUIREMENTS=[TF_URL,'huggingface-hub==1.31.0','tokenizers==0.23.1','safetensors==0.8.0','accelerate==1.12.0','jinja2==3.1.6','regex==2025.11.3']

def save(path,value):
    raw=json.dumps(value,ensure_ascii=False,sort_keys=True,indent=2).encode()
    if path.exists():
        if path.read_bytes()!=raw:raise RuntimeError('Existing evidence differs: '+str(path))
    else:
        with path.open('xb') as f:f.write(raw)

def run(args,**kwargs):return subprocess.run(args,check=True,**kwargs)

def environment(root):
    env=dict(os.environ);env.update(HF_HOME=str(root/'hf_cache'),HF_HUB_DISABLE_IMPLICIT_TOKEN='1',HF_HUB_DISABLE_TELEMETRY='1',PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES='')
    return env

BASELINE="""import torch,importlib.metadata as m,json
p={}
for n in ['torch','torchvision','torchaudio','numpy']:
 try:p[n]=m.version(n)
 except m.PackageNotFoundError:pass
print(json.dumps({'torch_version':torch.__version__,'torch_file':torch.__file__,'cuda_build':torch.version.cuda,'protected':p}))
"""

def setup(root):
    if sys.version_info<(3,11):raise RuntimeError('Pinned Transformers requires Python >=3.11; choose a matching existing torch interpreter')
    env=environment(root);baseline=json.loads(subprocess.check_output([sys.executable,'-c',BASELINE],env=env,text=True))
    save(root/'BASE_ENVIRONMENT.json',baseline)
    constraint=root/'protect_existing_runtime.txt'
    expected=''.join(k+'=='+v+'\n' for k,v in baseline['protected'].items())
    if constraint.exists() and constraint.read_text()!=expected:raise RuntimeError('Protected runtime changed')
    if not constraint.exists():constraint.write_text(expected)
    venv=root/'venv';python=venv/'bin/python'
    if not venv.exists():run([sys.executable,'-m','venv','--system-site-packages',str(venv)],env=env)
    if not python.is_file():raise RuntimeError('Expected isolated Linux venv')
    # A nested venv does not inherit its parent venv's site-packages. Add only
    # the verified source interpreter's package roots to this NEW venv; its
    # own pinned packages remain first. Never write into the source runtime.
    roots=json.loads(subprocess.check_output([sys.executable,'-c',"import json,site; print(json.dumps(site.getsitepackages()))"],env=env,text=True))
    torch_path=Path(baseline['torch_file']).resolve()
    torch_site=next((str(x) for x in torch_path.parents if x.name in ('site-packages','dist-packages')),None)
    if torch_site and torch_site not in roots:roots.append(torch_site)
    local_site=Path(subprocess.check_output([str(python),'-c',"import sysconfig; print(sysconfig.get_paths()['purelib'])"],env=env,text=True).strip())
    if not local_site.is_relative_to(venv):raise RuntimeError('Unexpected isolated package directory')
    roots=[str(Path(x).resolve()) for x in roots if Path(x).is_dir() and Path(x).resolve()!=local_site.resolve()]
    pth=local_site/'zz_verified_research_parent_packages.pth';content=''.join(x+'\n' for x in dict.fromkeys(roots))
    if pth.exists() and pth.read_text()!=content:raise RuntimeError('Existing parent package bridge differs')
    if not pth.exists():pth.write_text(content)
    save(root/'PARENT_PACKAGE_BRIDGE.json',{'new_venv_only_pth':str(pth),'read_only_source_paths':roots,'source_interpreter':sys.executable})
    try:before=json.loads(subprocess.check_output([str(python),'-c',BASELINE],env=env,text=True))
    except subprocess.CalledProcessError as error:raise RuntimeError('Cannot reuse the verified source torch from isolated venv; no installation attempted') from error
    if before!=baseline:raise RuntimeError('Isolated venv does not resolve the exact verified source torch; no installation attempted')
    report=root/'PIP_DRY_RUN.json'
    run([str(python),'-m','pip','install','--dry-run','--report',str(report),'-c',str(constraint),*REQUIREMENTS],env=env)
    planned=json.loads(report.read_text())
    for item in planned.get('install',[]):
        name=item['metadata']['name'].lower().replace('_','-')
        if name in ('torch','torchvision','torchaudio','triton') or name.startswith(('nvidia-','cuda-')):
            raise RuntimeError('Refusing torch/CUDA changes, even inside the new venv: '+name)
    run([str(python),'-m','pip','install','-c',str(constraint),*REQUIREMENTS],env=env)
    after=json.loads(subprocess.check_output([str(python),'-c',BASELINE],env=env,text=True))
    if after!=baseline:raise RuntimeError('Shared torch/runtime provenance changed')
    run([str(python),'-c',"import transformers; from transformers import AutoProcessor, AutoModelForMultimodalLM, Gemma4UnifiedForConditionalGeneration; assert transformers.__version__=='5.22.2'; print('CPU native import PASS; no model loaded')"],env=env)
    save(root/'RUNTIME_READY.json',{'status':'PASS','transformers':'5.22.2','torch_reused':baseline,'GPU_used':False,'weights_loaded':False})
    freeze=subprocess.check_output([str(python),'-m','pip','freeze'],env=env,text=True)
    (root/'INSTALLED_FREEZE.txt').write_text(freeze)

def verified(path,info):
    if path.stat().st_size!=info['size']:raise RuntimeError('Size mismatch: '+path.name)
    h=hashlib.sha256() if 'lfs' in info else hashlib.sha1(b'blob '+str(info['size']).encode()+b'\0')
    with path.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    expected=info['lfs']['sha256'] if 'lfs' in info else info['blobId']
    if h.hexdigest()!=expected:raise RuntimeError('Official content digest mismatch: '+path.name)
    return {'size':info['size'],'digest':h.hexdigest(),'algorithm':'sha256' if 'lfs' in info else 'git_blob_sha1'}

def download(root,weights):
    if Path(sys.prefix).resolve()!=(root/'venv').resolve():raise RuntimeError('Use the new isolated venv for downloads')
    if not (root/'RUNTIME_READY.json').is_file():raise RuntimeError('Native CPU import must pass first')
    if weights and not (root/'CPU_INPUTS_READY.json').is_file():raise RuntimeError('CPU tokenizer/template preparation must pass before weight download')
    os.environ.update(environment(root))
    metadata=json.load(urllib.request.urlopen('https://huggingface.co/api/models/'+REPO+'/revision/'+REV+'?blobs=true',timeout=60))
    pinned_raw=(HERE/'HF_METADATA.json').read_bytes()
    source=json.loads((HERE/'RUNTIME_SOURCE_CHECK.json').read_text(encoding='utf-8'))
    if hashlib.sha256(pinned_raw).hexdigest()!=source['metadata_sha256']:raise RuntimeError('Bundled official metadata checksum differs')
    pinned=json.loads(pinned_raw)
    if metadata['sha']!=REV or metadata.get('gated') is not False or metadata.get('private') is not False:raise RuntimeError('Unexpected repository access/revision')
    known={x['rfilename']:x for x in pinned['siblings']};current={x['rfilename']:x for x in metadata['siblings']}
    names=['model.safetensors'] if weights else ['README.md','config.json','generation_config.json','processor_config.json','chat_template.jinja','tokenizer.json','tokenizer_config.json']
    if any(current[n]!=known[n] for n in names):raise RuntimeError('Pinned file metadata differs')
    save(root/'HF_REVISION_VERIFIED.json',{'repo':REPO,'sha':REV,'files':known,'gated':False})
    model=root/'model';model.mkdir(exist_ok=True)
    needed=sum(known[n]['size'] for n in names if not (model/n).exists())
    if shutil.disk_usage(root).free<needed+2*1024**3:raise RuntimeError('Insufficient free disk for selected files plus 2 GiB margin')
    from huggingface_hub import hf_hub_download
    result={}
    for name in names:
        path=model/name
        if not path.exists():path=Path(hf_hub_download(REPO,name,revision=REV,local_dir=model,token=False))
        result[name]=verified(path,known[name])
    save(root/('WEIGHTS_VERIFIED.json' if weights else 'ASSETS_VERIFIED.json'),{'status':'PASS','repo':REPO,'revision':REV,'files':result})

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['setup','assets','weights']);p.add_argument('--root',required=True);a=p.parse_args()
    root=Path(a.root).resolve()
    if sys.platform!='linux' or root.parent!=Path('/home/jovyan/work') or root.name!='kazakh_gemma4_20261112':raise RuntimeError('Only the dedicated research server directory is admitted')
    root.mkdir(exist_ok=True)
    save(root/'OWNERSHIP.json',{'project':'Kazakh NLP Gemma bounded research','root':str(root),'does_not_modify_base_environment':True})
    if a.action=='setup':setup(root)
    else:download(root,a.action=='weights')

if __name__=='__main__':main()
