import os,time,json,subprocess,sys
from datetime import datetime,timezone
from pathlib import Path
from pilot.common import ROOT,read_json,write_json,file_hash,digest
from encoder_lr_screen02_diag01.evidence import durable
D=ROOT/'encoder_lr_screen02_recovery01';RUN=D/'run'
INPUT=ROOT/'encoder_lr_screen02/run/checkpoints/rolling_0_g000001.pt'
META=INPUT.parent/'generation_000001.json'
PIN='bd481f3fd47088b236aa2705aab5f4870afce7c1663c40b1de7c2854c9c7bf01'
COMPONENTS=('model','optimizer','scheduler','rng','progress')
def utc():return datetime.now(timezone.utc).isoformat()
def config():return read_json(D/'CONFIG.lock.json')
def verify_plan():
    lock=read_json(D/'SOURCE.lock.json')
    for n,h in lock['files'].items():assert file_hash(ROOT/n)==h,'Protected file changed: '+n
    return digest(lock)
def env():
    temp=D/'runtime/tmp';temp.mkdir(parents=True,exist_ok=True)
    return dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONUTF8='1',PYTHONUNBUFFERED='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
        HF_HUB_DISABLE_TELEMETRY='1',TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='16',MKL_NUM_THREADS='16',OPENBLAS_NUM_THREADS='16',
        CUBLAS_WORKSPACE_CONFIG=':4096:8',TEMP=str(temp),TMP=str(temp),CUDA_CACHE_PATH=str(D/'runtime/cuda'),TORCHINDUCTOR_CACHE_DIR=str(D/'runtime/inductor'))
def append(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a',encoding='utf-8',newline='\n') as f:f.write(json.dumps(value,ensure_ascii=False,sort_keys=True)+'\n');f.flush()
def initialize():
    import torch
    from encoder_lr_screen02.core import initialize as original
    torch.set_num_threads(16);torch.set_num_interop_threads(16)
    return original()
def reader_process(metadata,output):
    started=time.perf_counter();log=Path(output).with_suffix('.log');log.parent.mkdir(parents=True,exist_ok=True)
    with log.open('x',encoding='utf-8') as f:
        p=subprocess.run([sys.executable,'-B','-X','utf8','-m','encoder_lr_screen02_recovery01.reader','--metadata',str(metadata),'--output',str(output)],
            cwd=ROOT,env=env(),stdout=f,stderr=subprocess.STDOUT,timeout=180)
    assert p.returncode==0,'Independent checkpoint reader failed: '+str(output)
    proof=read_json(output);assert proof['accepted'];return time.perf_counter()-started
