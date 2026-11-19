import os,sys,time,json,subprocess
from pathlib import Path
from datetime import datetime,timezone
from pilot.common import ROOT,read_json,write_json,file_hash,digest
from encoder_lr_screen02_diag01.evidence import durable
Q=ROOT/'ner_training_queue01'
D=Path(os.environ.get('NER_CONDITION_DIR',str(Q/'preparation'))).resolve()
assert D.is_relative_to(Q.resolve()) and D!=Q.resolve()
RUN=D/'run'
COMPONENTS=('model','optimizer','scheduler','rng','progress')
def utc():return datetime.now(timezone.utc).isoformat()
def config():return read_json(D/'CONFIG.lock.json')
def append(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a',encoding='utf-8',newline='\n') as f:f.write(json.dumps(value,ensure_ascii=False,sort_keys=True)+'\n');f.flush()
def env(directory=None):
    directory=Path(directory or D);tmp=directory/'runtime/tmp';tmp.mkdir(parents=True,exist_ok=True)
    return dict(os.environ,NER_CONDITION_DIR=str(directory),PYTHONDONTWRITEBYTECODE='1',PYTHONUTF8='1',PYTHONUNBUFFERED='1',
        HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_HUB_DISABLE_TELEMETRY='1',TOKENIZERS_PARALLELISM='false',
        OMP_NUM_THREADS='16',MKL_NUM_THREADS='16',OPENBLAS_NUM_THREADS='16',CUBLAS_WORKSPACE_CONFIG=':4096:8',
        TEMP=str(tmp),TMP=str(tmp),CUDA_CACHE_PATH=str(directory/'runtime/cuda'),TORCHINDUCTOR_CACHE_DIR=str(directory/'runtime/inductor'))
def verify_plan():
    lock=read_json(Q/'SOURCE.lock.json')
    for name,h in lock['files'].items():assert file_hash(ROOT/name)==h,'Protected source changed: '+name
    master=lock['context_master_plan'];assert file_hash(Path(master['path']))==master['sha256'],'Context master plan changed'
    if (D/'CONFIG.sha256').exists():assert file_hash(D/'CONFIG.lock.json')==(D/'CONFIG.sha256').read_text().strip()
    return digest(lock)
def reader_process(metadata,output):
    tick=time.perf_counter();log=Path(output).with_suffix('.log');log.parent.mkdir(parents=True,exist_ok=True)
    with log.open('x',encoding='utf-8') as f:
        p=subprocess.run([sys.executable,'-B','-X','utf8','-m','ner_training_queue01.reader','--metadata',str(metadata),'--output',str(output)],
            cwd=ROOT,env=env(),stdout=f,stderr=subprocess.STDOUT,timeout=180)
    assert p.returncode==0 and read_json(output)['accepted'],'Independent checkpoint reader failed: '+str(output)
    return time.perf_counter()-tick
