import os
import random
from pilot.common import ROOT,read_json,read_jsonl,file_hash
from pilot.models import cache_dir,load_tokenizer
from pilot.train_encoder import collate as old_collate
from .common import D

def load_data(split):
    if split not in ['train','valid']:raise ValueError('Final test inference/data loading locked')
    p=ROOT/'stage_preflight/features'/f'xlmr_ner_{split}.jsonl'
    manifest=read_json(ROOT/'stage_preflight/views/manifest.json')
    assert file_hash(p)==manifest['files'][f'features/xlmr_ner_{split}.jsonl']
    rows=read_jsonl(p);assert len(rows)==(89386 if split=='train' else 11100)
    assert all(r['source_id'].startswith(f'ner:{split}:') for r in rows)
    return rows

def labels():return read_json(ROOT/'data/derived/ner_labels.json')['bio']

def initialize():
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    import torch
    from transformers import AutoModelForTokenClassification,set_seed
    assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    set_seed(13);torch.use_deterministic_algorithms(True,warn_only=False);torch.backends.cudnn.benchmark=False
    tok=load_tokenizer('encoder');spec=read_json(ROOT/'configs/models.lock.json')['encoder'];labs=labels()
    model=AutoModelForTokenClassification.from_pretrained(spec['model_id'],revision=spec['revision'],cache_dir=cache_dir(),local_files_only=True,
        use_safetensors=True,trust_remote_code=False,num_labels=len(labs),id2label=dict(enumerate(labs)),label2id={x:i for i,x in enumerate(labs)},attn_implementation='sdpa').to('cuda')
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False});model.train()
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-5,weight_decay=.01)
    from transformers import get_linear_schedule_with_warmup
    scheduler=get_linear_schedule_with_warmup(optimizer,int(27935*.06),27935)
    return model,tok,optimizer,scheduler

def collator(tokenizer):
    def apply(features,device):
        return old_collate('ner',[{k:v for k,v in f.items() if k in ['input_ids','attention_mask','token_type_ids','labels']} for f in features],tokenizer,device)
    return apply

def order(epoch,n):
    result=list(range(n));random.Random(13+epoch).shuffle(result);return result

def update(model,optimizer,scheduler,features,microbatch,collate):
    import torch
    from .objective import accumulate
    optimizer.zero_grad(set_to_none=True)
    result=accumulate(model,features,microbatch,collate)
    missing=[n for n,p in model.named_parameters() if p.requires_grad and p.grad is None]
    if missing:raise RuntimeError('Missing gradients: '+str(missing))
    norm=float(torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True));lr=optimizer.param_groups[0]['lr']
    optimizer.step();scheduler.step();optimizer.zero_grad(set_to_none=True)
    return {**result,'gradient_norm':norm,'learning_rate':lr,'next_learning_rate':scheduler.get_last_lr()[0]}
