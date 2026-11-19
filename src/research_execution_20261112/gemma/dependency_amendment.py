"""Only NEW Gemma venv: add compatible vision support, CPU admit, fetch weights."""
from pathlib import Path
import argparse,hashlib,importlib.metadata,json,os,platform,subprocess,sys

HERE=Path(__file__).resolve().parent
WHEELS={
 '311':('cp311-cp311','ebf2b495c76097796b9a2eac9290efbcae96e0fd9e5ae52c40eff188610bb440'),
 '312':('cp312-cp312','1255a0ca2bf987acf9f103b96c5c4cfe3415fc4a1eef17fa08af527a04a4f573'),
 '313':('cp313-cp313','a9c0de893dce9c2913c9c7ae88a916910f92d02b99da149678806d18e8079f29'),
 '314':('cp314-cp314','5b7ad3fb6cf03ef2a2fd617cb4b4e41efa9bb0143c67f506c2a3e6765c7b12ad')}
def save(path,value):
    raw=json.dumps(value,ensure_ascii=False,sort_keys=True,indent=2).encode()
    if path.exists():
        if path.read_bytes()!=raw:raise RuntimeError('Existing amendment evidence differs: '+str(path))
    else:
        with path.open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
def state():
    import torch,numpy
    packages={}
    for n in ['torch','numpy','torchvision','Pillow','transformers','torchaudio']:
        try:packages[n]=importlib.metadata.version(n)
        except importlib.metadata.PackageNotFoundError:packages[n]=None
    return {'torch_version':torch.__version__,'torch_file':str(Path(torch.__file__).resolve()),'cuda_build':torch.version.cuda,'numpy_version':numpy.__version__,'numpy_file':str(Path(numpy.__file__).resolve()),'packages':packages}
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',required=True);a=p.parse_args();root=Path(a.root).resolve()
    if sys.platform!='linux' or platform.machine()!='x86_64' or root!=Path('/home/jovyan/work/kazakh_gemma4_20261112') or Path(sys.prefix).resolve()!=root/'venv':raise RuntimeError('Dedicated new Linux x86_64 Gemma venv required')
    cfg=(root/'venv/pyvenv.cfg').read_text().lower().replace(' ','')
    if 'include-system-site-packages=false' not in cfg:raise RuntimeError('First isolate NEW venv from unrelated global torchvision/torchaudio')
    os.environ.update(CUDA_VISIBLE_DEVICES='',HF_HOME=str(root/'hf_cache'),HF_HUB_DISABLE_IMPLICIT_TOKEN='1',HF_HUB_DISABLE_TELEMETRY='1',PYTHONDONTWRITEBYTECODE='1')
    if not (root/'RUNTIME_READY.json').is_file() or not (root/'ASSETS_VERIFIED.json').is_file():raise RuntimeError('Existing setup and small assets receipts required; do not forge them')
    before=state();baseline=json.loads((root/'BASE_ENVIRONMENT.json').read_text())
    if before['torch_version']!='2.10.0+cu128' or before['torch_file']!=str(Path(baseline['torch_file']).resolve()) or before['packages']['numpy']!=baseline['protected']['numpy']:raise RuntimeError('Verified parent torch/numpy changed')
    before_path=root/'RUNTIME_DEPENDENCY_BEFORE.json'
    if not before_path.exists():save(before_path,before)
    else:before=json.loads(before_path.read_text())
    key=str(sys.version_info.major)+str(sys.version_info.minor)
    if key not in WHEELS:raise RuntimeError('No inspected official wheel for this Python ABI')
    tag,hash_=WHEELS[key]
    url='https://download-r2.pytorch.org/whl/cu128/torchvision-0.25.0%2Bcu128-'+tag+'-manylinux_2_28_x86_64.whl#sha256='+hash_
    current=state()
    if current['packages']['torchvision'] not in (None,'0.25.0+cu128'):raise RuntimeError('Unexpected torchvision remains visible; do not overwrite parent/global packages')
    if current['packages']['torchvision'] is None:
        subprocess.run([sys.executable,'-m','pip','install','--no-deps','--report',str(root/'TORCHVISION_INSTALL_REPORT.json'),url],check=True)
    try:import PIL
    except ImportError:
        subprocess.run([sys.executable,'-m','pip','install','--no-deps','--only-binary=:all:','--index-url','https://pypi.org/simple','--report',str(root/'PILLOW_INSTALL_REPORT.json'),'Pillow==12.0.0'],check=True)
        import PIL
    import torch,torchvision,transformers
    from transformers import AutoProcessor,Gemma4UnifiedForConditionalGeneration
    if torchvision.__version__!='0.25.0+cu128' or not Path(torchvision.__file__).resolve().is_relative_to(root/'venv'):raise RuntimeError('Vision support must be exact compatible version inside only NEW venv')
    processor=AutoProcessor.from_pretrained(root/'model',local_files_only=True,trust_remote_code=False)
    after=state()
    for k in ['torch_version','torch_file','cuda_build','numpy_version','numpy_file']:
        if before[k]!=after[k]:raise RuntimeError('Protected inherited runtime changed')
    save(root/'RUNTIME_DEPENDENCY_AMENDMENT.json',{'status':'PASS','reason':'Gemma4 Unified AutoProcessor requires vision import even for text-only tasks','official_compatibility_source':'https://pytorch.org/get-started/previous-versions/','official_wheel_url':url,'optional_Pillow_pin':'12.0.0 only if previously missing','before':before,'after':after,'actual_processor_class':type(processor).__name__,'torchvision_file':torchvision.__file__,'Pillow_file':PIL.__file__,'base_environment_unchanged':True,'new_venv_only':True,'GPU_used':False,'model_weights_loaded':False,'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    (root/'DEPENDENCY_AMENDED_FREEZE.txt').write_text(subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True))
    # Existing failed stage logs/receipts remain untouched. Each child uses its
    # original readiness and provenance checks; no synthetic PASS marker is made.
    subprocess.run([sys.executable,'-B',str(HERE/'cpu_prepare.py'),'--root',str(root)],check=True)
    subprocess.run([sys.executable,'-B',str(HERE/'server_setup.py'),'weights','--root',str(root)],check=True)
    print('Compatible isolated vision dependency, actual CPU8 admission, pinned weight download complete',flush=True)

if __name__=='__main__':main()
