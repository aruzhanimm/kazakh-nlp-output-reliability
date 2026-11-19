"""CPU-only view checks and immutable source/configuration preparation."""
import math,os,shutil,json,difflib,importlib.metadata
from pathlib import Path
from pilot.common import ROOT,read_json,read_jsonl,file_hash,digest,write_json
from .common import Q,utc,durable
MASTER=Path('C:/Users/Aru/Downloads/Kazakh_NLP_Master_Plan (1).md')

def audit_views():
    from stage_preflight.views import tokenizers
    toks=tokenizers();m=read_json(ROOT/'stage_preflight/views/manifest.json');labs=read_json(ROOT/'data/derived/ner_labels.json')['bio'];result={}
    assert len(labs)==51
    for split in ['train','valid']:
        rp=ROOT/'stage_preflight/views'/f'ner_{split}.jsonl';assert file_hash(rp)==m['files'][f'views/{rp.name}']
        rows=read_jsonl(rp);assert len(rows)==(89386 if split=='train' else 11100)
        for enc,tok in toks.items():
            fp=ROOT/'stage_preflight/features'/f'{enc}_ner_{split}.jsonl';assert file_hash(fp)==m['files'][f'features/{fp.name}']
            feats=read_jsonl(fp);assert len(feats)==len(rows) and len({f['id'] for f in feats})==len(feats)
            valid=0;maxlen=0;subwords=0;unknowns=0;types=set()
            for f,r in zip(feats,rows):
                assert f['source_id']==r['id'] and len(r['windows'])==1
                w=r['windows'][0];assert w['start']==0 and w['end']==len(r['tokens']) and w['tokens']==r['tokens']
                assert w['text']==r['text']==' '.join(r['tokens']) and r['owners']==[0]*len(r['tokens'])
                e=tok(r['tokens'],is_split_into_words=True,truncation=False)
                assert all(e[k]==f[k] for k in ['input_ids','attention_mask']) and e.word_ids()==f['word_ids']
                if 'token_type_ids' in f:assert e['token_type_ids']==f['token_type_ids'] and not any(f['token_type_ids'])
                seen=set()
                for word,label in zip(f['word_ids'],f['labels']):
                    if word is None or word in seen:assert label==-100
                    else:assert labs[label]==r['labels'][word];seen.add(word);valid+=1;types.add(r['labels'][word][2:])
                assert seen==set(range(len(r['tokens']))) and len(f['input_ids'])<=384<=tok.model_max_length
                maxlen=max(maxlen,len(f['input_ids']));subwords+=len(f['input_ids']);unknowns+=f['input_ids'].count(tok.unk_token_id)
            result.setdefault(enc,{})[split]={'features':len(feats),'sentences':len(rows),'supervised_words':valid,'max_tokens':maxlen,
                'subword_tokens':subwords,'unknown_tokens':unknowns,'all_features_reencoded':True,'one_window_exact_shared_text':True,'sha256':file_hash(fp)}
    durable(Q/'VIEW_AUDIT.json',result);return result

def make_config(directory,enc,seed,lr,microbatch):
    directory=Path(directory);assert not (directory/'CONFIG.lock.json').exists();directory.mkdir(parents=True,exist_ok=True)
    a=read_json(Q/'VIEW_AUDIT.json')[enc];base=read_json(ROOT/'encoder_lr_screen02/main_config.json')
    n=a['train']['features'];u=math.ceil(n/16);h=5*u
    if enc=='xlmr':
        pin=read_json(ROOT/'configs/models.lock.json')['encoder'];snapshot=ROOT/'.cache/huggingface'/('models--'+pin['model_id'].replace('/','--'))/'snapshots'/pin['revision']
        weights=read_json(ROOT/'artifacts/encoder_snapshot_hashes.json')
        # The pinned weight hash is also checked by the frozen source lock.
        assert file_hash(snapshot/'model.safetensors')==weights['files']['model.safetensors']
        init={**pin,'snapshot':str(snapshot),'weights_sha256':weights['files']['model.safetensors']};tok=ROOT/'artifacts/tokenizers/encoder'
    else:
        pin=read_json(ROOT/'stage_preflight/mbert.lock.json');init={'model_id':pin['model_id'],'revision':pin['revision'],'snapshot':pin['snapshot'],'weights_sha256':pin['files']['model.safetensors']};tok=Path(pin['snapshot'])
    c={**base,'encoder':enc,'model':enc+'_base','seed':seed,'learning_rate':lr,'initialization':init,'tokenizer_path':str(tok),
        'training_features':n,'training_sentences':a['train']['sentences'],'development_sentences':a['valid']['sentences'],
        'feature_counts':{s:a[s]['features'] for s in a},'training_first_subword_labels':a['train']['supervised_words'],
        'updates_per_epoch':u,'scheduler_horizon_updates':h,'warmup_updates':int(h*.06),'microbatch':microbatch,'gradient_accumulation':16//microbatch,
        'operational_ceiling_seconds':14400,'safe_stop_reserve_seconds':150,'initialization_policy':'fresh pinned pretrained with seed-specific head; never a fine-tuned parent',
        'condition_id':directory.name,'queue':'ner_training_queue01','telemetry_policy':'outside canonical state; all canonical components compared without exclusions',
        'checkpoint_policy':'owned snapshot, independent precommit reader, all epochs plus newest two verified rolling payloads',
        'planned_fresh_process_restart_update':1000,'final_test_locked':True,'shutdown_enabled':False}
    c['run_hash']=digest(c);durable(directory/'CONFIG.lock.json',c);(directory/'CONFIG.sha256').write_text(file_hash(directory/'CONFIG.lock.json')+'\n')
    return c

def freeze():
    assert read_json(Q/'PREPARATION_TESTS.json')['passed']
    names=[p.relative_to(ROOT).as_posix() for p in Q.glob('*.py')]+['ner_training_queue01/PLAN.md','ner_training_queue01/VIEW_AUDIT.json',
        'stage_preflight/MAIN_ENCODER_SCHEDULE.md','stage_preflight/views/manifest.json','stage_preflight/mbert.lock.json',
        'configs/models.lock.json','data/derived/ner_labels.json','data/pilot/ner_valid.jsonl']
    for folder in ['pilot','encoder_main01','encoder_lr_screen02','encoder_lr_screen02_recovery01','checkpoint_recovery01','checkpoint_integrity01','stage_preflight','encoder_lr_screen02_diag01','encoder_cont01']:
        names += [p.relative_to(ROOT).as_posix() for p in (ROOT/folder).glob('*.py')]
    for enc in ['xlmr','mbert']:
        names += [f'stage_preflight/features/{enc}_ner_{s}.jsonl' for s in ['train','valid']]
    names += [f'stage_preflight/views/ner_{s}.jsonl' for s in ['train','valid']]
    names += ['encoder_lr_screen02/main_config.json','encoder_lr_screen02/COMPARATOR.json','checkpoint_recovery01/FINAL_STATUS.json','artifacts/encoder_snapshot_hashes.json']
    names += [p.relative_to(ROOT).as_posix() for p in (ROOT/'artifacts/tokenizers/encoder').glob('*') if p.is_file()]
    files={n:file_hash(ROOT/n) for n in sorted(set(names))}
    for n in files:
        p=ROOT/n
        if p.suffix=='.py':dest=Q/'source'/n;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
    durable(Q/'SOURCE.lock.json',{'utc':utc(),'files':files,'context_master_plan':{'path':str(MASTER),'sha256':file_hash(MASTER),'authority':'context only; latest user request governs; never modified'}})
    packages={x:importlib.metadata.version(x) for x in ['torch','transformers','numpy','safetensors','tokenizers','scikit-learn','seqeval','psutil']}
    durable(Q/'PACKAGES.json',packages)
    patch=(Q/'ADAPTER_DIFF.patch').read_text(encoding='utf-8')
    for p in Q.glob('*.py'):
        if p.name not in ['checkpoints.py','reader.py','restore.py','fresh_reload.py','evaluation.py']:
            patch+=''.join(difflib.unified_diff([],p.read_text(encoding='utf-8').splitlines(True),fromfile='/dev/null',tofile=p.relative_to(ROOT).as_posix()))
    (Q/'SOURCE_DIFF.patch').write_text(patch,encoding='utf-8')
    durable(Q/'PREPARATION.json',{'utc':utc(),'passed':True,'views':read_json(Q/'VIEW_AUDIT.json'),'source_lock_sha256':file_hash(Q/'SOURCE.lock.json'),
        'existing_recovery_unchanged':True,'master_plan_edited':False,'final_test_loaded':False,'gpu_preflight_during_active_recovery':False})

if __name__=='__main__':
    import sys
    if sys.argv[1]=='views':audit_views()
    elif sys.argv[1]=='freeze':freeze()
    else:raise ValueError('Unknown preparation operation')
