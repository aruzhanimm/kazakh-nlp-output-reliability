import math,os,shutil,json,difflib,importlib.metadata,time,random
from pathlib import Path
from collections import Counter
from pilot.common import ROOT,read_json,read_jsonl,file_hash,digest
from .common import Q,utc,durable
MASTER=Path('C:/Users/Aru/Downloads/Kazakh_NLP_Master_Plan (1).md')

def audit_views():
    from stage_preflight.views import tokenizers,common_prefix
    toks=tokenizers();m=read_json(ROOT/'stage_preflight/views/manifest.json');inherited=read_json(Q/'INHERITED_INPUT_AUDIT.json');result={}
    for name,h in inherited['hashes'].items():assert file_hash(Path(name))==h
    for split in ['train','valid']:
        rp=ROOT/'stage_preflight/views'/f'sa_{split}.jsonl';assert file_hash(rp)==m['files'][f'views/{rp.name}']
        rows=read_jsonl(rp);expected=133894 if split=='train' else 16766
        assert len(rows)==len({r['id'] for r in rows})==expected
        counts=Counter(int(r['label']) for r in rows);assert counts==({0:23764,1:110130} if split=='train' else {0:2981,1:13785})
        for r in rows:
            visible,fallback=common_prefix(r['text'],toks)
            assert visible.encode('utf-8')==r['visible_text'].encode('utf-8') and fallback==r['character_fallback']
            assert r['text'].startswith(r['visible_text']) and r['capped']==(r['text']!=r['visible_text'])
        for enc,tok in toks.items():
            fp=ROOT/'stage_preflight/features'/f'{enc}_sa_{split}.jsonl';assert file_hash(fp)==m['files'][f'features/{fp.name}']
            feats=read_jsonl(fp);assert len(feats)==len(rows) and len({f['id'] for f in feats})==expected
            lengths=[];unk=0
            for f,r in zip(feats,rows):
                assert f['id']==f['source_id']==r['id'] and f['labels']==int(r['label']) and type(f['labels']) is int
                e=tok(r['visible_text'],truncation=False)
                assert all(e[k]==f[k] for k in e) and set(f)==set(e)|{'id','source_id','labels'}
                assert len(f['input_ids'])<=384<=tok.model_max_length and all(x==1 for x in f['attention_mask'])
                if 'token_type_ids' in e:assert not any(e['token_type_ids'])
                lengths.append(len(f['input_ids']));unk+=f['input_ids'].count(tok.unk_token_id)
            result.setdefault(enc,{})[split]={'features':expected,'sentences':expected,'labels':dict(counts),'max_tokens':max(lengths),'mean_tokens':sum(lengths)/expected,
                'unknown_tokens':unk,'all_features_reencoded':True,'all_prefixes_recomputed':True,'literal_visible_text_preserved':True,'sha256':file_hash(fp)}
            if split=='train':
                order=list(range(expected));random.Random(13).shuffle(order);longest=sorted(range(expected),key=lambda i:(-lengths[i],i))[:16]
                durable(Q/'preflight'/enc/'FIXTURE.json',{'seed':13,'ordered_indices':order[:400],'ordered_ids':[rows[i]['id'] for i in order[:400]],
                    'longest_indices':longest,'longest_ids':[rows[i]['id'] for i in longest],'longest_lengths':[lengths[i] for i in longest],
                    'planned_microbatches':[2,4,8],'warmup_updates':5,'measured_updates':20,'separate_longest_updates':1})
        print({'split':split,'rows':expected,'reencoding':'passed'},flush=True)
    assert read_json(Q/'preflight/xlmr/FIXTURE.json')['ordered_ids']==read_json(Q/'preflight/mbert/FIXTURE.json')['ordered_ids']
    full={r['id']:r for r in rows};panel=read_jsonl(ROOT/'data/pilot/sa_valid.jsonl')
    assert len(panel)==200 and all(full[r['id']]['visible_text']==r['visible_text'] and int(full[r['id']]['label'])==int(r['label']) for r in panel)
    durable(Q/'VIEW_AUDIT.json',result)

def make_config(directory,enc,seed,lr,microbatch):
    directory=Path(directory);assert not (directory/'CONFIG.lock.json').exists();directory.mkdir(parents=True,exist_ok=True)
    a=read_json(Q/'VIEW_AUDIT.json')[enc];n=a['train']['features'];u=math.ceil(n/16);h=5*u
    assert (n,u,h,int(h*.06),n%16)==(133894,8369,41845,2510,6)
    if enc=='xlmr':
        pin=read_json(ROOT/'configs/models.lock.json')['encoder'];snapshot=ROOT/'.cache/huggingface'/('models--'+pin['model_id'].replace('/','--'))/'snapshots'/pin['revision'];weights=read_json(ROOT/'artifacts/encoder_snapshot_hashes.json')
        assert pin['revision']=='e73636d4f797dec63c3081bb6ed5c7b0bb3f2089' and file_hash(snapshot/'model.safetensors')==weights['files']['model.safetensors']
        init={**pin,'snapshot':str(snapshot),'weights_sha256':weights['files']['model.safetensors']};tok=ROOT/'artifacts/tokenizers/encoder'
    else:
        pin=read_json(ROOT/'stage_preflight/mbert.lock.json');assert pin['revision']=='3f076fdb1ab68d5b2880cb87a0886f315b8146f8'
        init={'model_id':pin['model_id'],'revision':pin['revision'],'snapshot':pin['snapshot'],'weights_sha256':pin['files']['model.safetensors']};tok=Path(pin['snapshot'])
    c={'task':'sa','encoder':enc,'model':enc+'_base','seed':seed,'learning_rate':lr,'initialization':init,'tokenizer_path':str(tok),
       'training_features':n,'training_examples':n,'development_sentences':16766,'feature_counts':{s:a[s]['features'] for s in a},
       'labels':['negative','positive'],'id2label':{'0':'negative','1':'positive'},'updates_per_epoch':u,'epochs_max':5,'scheduler_horizon_updates':h,'warmup_updates':int(h*.06),
       'microbatch':microbatch,'gradient_accumulation':16//microbatch,'effective_batch':16,'weight_decay':.01,'gradient_clip':1.,'optimizer':'AdamW',
       'objective':'sum review CE / actual reviews in complete optimizer batch','decision':'argmax of two logits','class_weights':None,'sampling':'train-only shuffle(seed+epoch)',
       'attention':'sdpa','strict_determinism':True,'precision':'BF16 autocast; FP32 parameters/gradients/optimizer/CE','gradient_checkpointing':{'enabled':True,'use_reentrant':False},
       'selection':'exact highest full-dev macro-F1; earliest exact tie','early_stop':{'minimum_epochs':3,'patience':2,'min_delta':.0001,'threshold_affects':'patience only'},
       'operational_ceiling_seconds':28800,'safe_stop_reserve_seconds':150,'condition_id':directory.name,'queue':'sa_training_queue01',
       'initialization_policy':'fresh pinned pretrained and seed-specific sequence-classification head; full encoder trainable',
       'checkpoint_policy':'owned snapshot, independent precommit reader, all epochs plus two newest committed rolling full states',
       'planned_fresh_process_restart_update':1000,'telemetry_policy':'outside canonical state; all five state components checked without exclusions',
       'final_test_locked':True,'shutdown_enabled':False}
    c['run_hash']=digest(c);durable(directory/'CONFIG.lock.json',c);(directory/'CONFIG.sha256').write_text(file_hash(directory/'CONFIG.lock.json')+'\n');return c

def freeze():
    assert read_json(Q/'CPU_TESTS.json')['passed']
    names=[p.relative_to(ROOT).as_posix() for p in Q.glob('*.py')]+['sa_training_queue01/PLAN.md','sa_training_queue01/VIEW_AUDIT.json','sa_training_queue01/BUDGET.json',
        'stage_preflight/MAIN_ENCODER_SCHEDULE.md','encoder_main01/METHOD_DECISIONS.md','stage_preflight/views/manifest.json','stage_preflight/mbert.lock.json','configs/models.lock.json','data/pilot/sa_valid.jsonl','artifacts/encoder_snapshot_hashes.json']
    for folder in ['pilot','encoder_main01','encoder_lr_screen02','encoder_lr_screen02_recovery01','checkpoint_recovery01','checkpoint_integrity01','stage_preflight','encoder_lr_screen02_diag01','encoder_cont01']:
        names += [p.relative_to(ROOT).as_posix() for p in (ROOT/folder).glob('*.py')]
    for enc in ['xlmr','mbert']:names += [f'stage_preflight/features/{enc}_sa_{s}.jsonl' for s in ['train','valid']]+[f'sa_training_queue01/preflight/{enc}/FIXTURE.json']
    names += [f'stage_preflight/views/sa_{s}.jsonl' for s in ['train','valid']]+[p.relative_to(ROOT).as_posix() for p in (ROOT/'artifacts/tokenizers/encoder').glob('*') if p.is_file()]
    names += ['sa_training_queue01/LLM_REFERENCE.json']+[p.relative_to(ROOT).as_posix() for p in (Q/'llm_reference').rglob('*') if p.is_file()]
    files={n:file_hash(ROOT/n) for n in sorted(set(names))}
    for n in files:
        if Path(n).suffix=='.py':dest=Q/'source'/n;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/n,dest)
    durable(Q/'SOURCE.lock.json',{'utc':utc(),'files':files,'context_master_plan':{'path':str(MASTER),'sha256':file_hash(MASTER),'authority':'context only; unchanged'}})
    prior=['ner_training_queue01/FINAL_STATUS.json','ner_training_queue01/DELIVERY_CHECK.json','ner_training_queue01/AUTHORITATIVE_ARTIFACT_MANIFEST.json','ner_training_queue01/COMPLETE_BUNDLE_VERIFICATION.json',
        'encoder_lr_screen02/FINAL_AUDIT.json','checkpoint_recovery01/FINAL_STATUS.json','encoder_lr_screen02_recovery01/FINAL_STATUS.json']
    durable(Q/'PRIOR_EVIDENCE.json',{'files':{n:file_hash(ROOT/n) for n in prior},'scope':'Protected prior reports/manifests plus all source dependencies; prior weights are never opened for adaptation or written.'})
    durable(Q/'PACKAGES.json',{x:importlib.metadata.version(x) for x in ['torch','transformers','numpy','safetensors','tokenizers','scikit-learn','psutil']})
    patch=''
    for p in Q.glob('*.py'):
        old=ROOT/'ner_training_queue01'/p.name
        patch+=''.join(difflib.unified_diff(old.read_text(encoding='utf-8').splitlines(True) if old.exists() else [],p.read_text(encoding='utf-8').splitlines(True),fromfile=str(old.relative_to(ROOT)) if old.exists() else '/dev/null',tofile=str(p.relative_to(ROOT))))
    (Q/'SOURCE_DIFF.patch').write_text(patch,encoding='utf-8')
    durable(Q/'CPU_PREPARATION.json',{'utc':utc(),'passed':True,'source_lock_sha256':file_hash(Q/'SOURCE.lock.json'),'views_reencoded':True,'final_test_read':False,'master_plan_edited':False})

if __name__=='__main__':
    import sys
    {'views':audit_views,'freeze':freeze}[sys.argv[1]]()
