import os,random,math
from pathlib import Path
from pilot.common import ROOT,read_json,read_jsonl,file_hash
from encoder_main01.core import update,collator,labels
from .common import config

def load_data(split):
    assert split in ['train','valid'],'Final test data locked'
    c=config();p=ROOT/'stage_preflight/features'/f'{c["encoder"]}_ner_{split}.jsonl'
    m=read_json(ROOT/'stage_preflight/views/manifest.json')
    assert file_hash(p)==m['files'][f'features/{p.name}']
    rows=read_jsonl(p);assert len(rows)==c['feature_counts'][split]
    assert all(r['source_id'].startswith(f'ner:{split}:') for r in rows)
    return rows

def order(epoch,n,seed=None):
    if seed is None:seed=config()['seed']
    result=list(range(n));random.Random(seed+epoch).shuffle(result);return result

def initialize():
    import torch
    from transformers import AutoModelForTokenClassification,AutoTokenizer,set_seed,get_linear_schedule_with_warmup
    c=config();torch.set_num_threads(16);torch.set_num_interop_threads(16)
    assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    set_seed(c['seed']);torch.use_deterministic_algorithms(True,warn_only=False);torch.backends.cudnn.benchmark=False
    labs=labels();spec=c['initialization'];path=Path(spec['snapshot'])
    assert file_hash(path/'model.safetensors')==spec['weights_sha256']
    if c['encoder']=='mbert':
        for name,h in read_json(ROOT/'stage_preflight/mbert.lock.json')['files'].items():assert file_hash(path/name)==h
    tok=AutoTokenizer.from_pretrained(c['tokenizer_path'],local_files_only=True,use_fast=True,trust_remote_code=False)
    model=AutoModelForTokenClassification.from_pretrained(path,local_files_only=True,use_safetensors=True,trust_remote_code=False,
        num_labels=len(labs),id2label=dict(enumerate(labs)),label2id={x:i for i,x in enumerate(labs)},attn_implementation=c['attention']).to('cuda')
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False});model.train()
    opt=torch.optim.AdamW(model.parameters(),lr=c['learning_rate'],weight_decay=c['weight_decay'])
    sched=get_linear_schedule_with_warmup(opt,c['warmup_updates'],c['scheduler_horizon_updates'])
    return model,tok,opt,sched
