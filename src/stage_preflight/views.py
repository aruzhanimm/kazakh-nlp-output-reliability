"""Common gold-independent text selection and fully validated retained feature views."""
import argparse
from collections import Counter
import json
import random
import re
import time
from pilot.common import ROOT, file_hash, read_json, read_jsonl, write_json, write_jsonl, digest
from pilot.inputs import owners, align_first_subwords, load_panel
from pilot.models import load_tokenizer
from pilot.scoring import bio_spans
from .common import D, freeze, verify

def tokenizers():
    from transformers import AutoTokenizer
    lock=read_json(D/'mbert.lock.json')
    folder=__import__('pathlib').Path(lock['snapshot'])
    for n,h in lock['files'].items():
        if n in ['tokenizer.json','tokenizer_config.json','vocab.txt','special_tokens_map.json','config.json']:
            assert file_hash(folder/n)==h
    result={'xlmr':load_tokenizer('encoder'),'mbert':AutoTokenizer.from_pretrained(folder,local_files_only=True,use_fast=True,trust_remote_code=False)}
    assert all(t.is_fast and t.model_max_length>=384 for t in result.values())
    assert result['mbert'].do_lower_case is False
    return result

def common_windows(tokens, toks, max_tokens=384, max_words=96, overlap=24):
    result=[]; start=0
    while start<len(tokens):
        end=min(len(tokens),start+max_words)
        while end>start:
            encoded={name:tok(tokens[start:end],is_split_into_words=True,truncation=False) for name,tok in toks.items()}
            if all(len(e['input_ids'])<=max_tokens for e in encoded.values()): break
            end-=1
        if end==start: raise ValueError('A single original word exceeds the common encoder context')
        for name,e in encoded.items():
            if set(i for i in e.word_ids() if i is not None)!=set(range(end-start)):
                raise ValueError('Tokenizer lost an original word: '+name)
        result.append({'start':start,'end':end,'tokens':tokens[start:end],'text':' '.join(tokens[start:end]),
                       'token_lengths':{n:len(e['input_ids']) for n,e in encoded.items()}})
        if end==len(tokens): break
        start=max(start+1,end-overlap)
    if not result: raise ValueError('Empty NER sentence cannot be silently omitted')
    return result

def common_prefix(text,toks,max_tokens=384,max_words=128):
    matches=list(re.finditer(r'\S+',text)); count=min(len(matches),max_words)
    while count:
        visible=text if count==len(matches) else text[:matches[count-1].end()]
        if all(len(tok(visible,truncation=False)['input_ids'])<=max_tokens for tok in toks.values()): return visible,False
        count-=1
    if matches:
        for end in range(matches[0].end(),0,-1):
            visible=text[:end]
            if all(len(tok(visible,truncation=False)['input_ids'])<=max_tokens for tok in toks.values()): return visible,True
        raise ValueError('No nonempty common literal prefix fits')
    return text,False

def source_lock():
    names=['data/derived/manifest.json','data/derived/ner_labels.json','data/derived/ner_exclusions.jsonl','data/derived/sa_exclusions.jsonl',
           'configs/models.lock.json','interface_pilot/plan.lock.json','interface_pilot/ner_mentions.txt','interface_pilot/alignment.py',
           'interface_pilot/prompts.py','prompts/sa_v1.txt','pilot/scoring.py','pilot/inputs.py']
    names += [f'data/derived/{t}_{s}.jsonl' for t in ['ner','sa'] for s in ['train','valid']]
    names += ['stage_preflight/PLAN.md','stage_preflight/views.py','stage_preflight/common.py','stage_preflight/mbert.lock.json']
    return freeze(names,'views_plan.lock.json')

def prepare():
    if (D/'views/manifest.json').exists():
        validate_files(); print('Existing full views verified'); return
    source_lock(); toks=tokenizers(); label_list=read_json(ROOT/'data/derived/ner_labels.json')['bio']; label2id={v:i for i,v in enumerate(label_list)}
    retained=read_json(ROOT/'data/derived/manifest.json')['tasks']; all_stats={}; files={}; calibration={}; continuity={}
    for task in ['ner','sa']:
        old_by_id={r['id']:r for r in load_panel(task,'valid')}
        for split in ['train','valid']:
            started=time.perf_counter(); source=ROOT/'data/derived'/f'{task}_{split}.jsonl'; rows=read_jsonl(source)
            assert len(rows)==retained[task]['retained'][split]['rows'] and len({r['id'] for r in rows})==len(rows)
            stats={'rows':len(rows),'windows':0,'training_features':0,'zero_owned_windows':0,'capped_rows':0,'character_fallbacks':0,
                'cross_window_entities':0,'entities_without_one_full_window':0,'entities_crossing_owners':0,
                'words':0,'labels':Counter(),'models':{n:{'features':0,'tokens':0,'max_tokens':0,'unknown_tokens':0,'length_histogram':Counter()} for n in toks}}
            out_path=D/'views'/f'{task}_{split}.jsonl'; out_path.parent.mkdir(exist_ok=True)
            features_paths={n:D/'features'/f'{n}_{task}_{split}.jsonl' for n in toks}
            for p in features_paths.values(): p.parent.mkdir(exist_ok=True)
            feature_ids=[]; changes=[]; panel=[]
            with out_path.open('x',encoding='utf-8') as stream:
                streams={n:p.open('x',encoding='utf-8') for n,p in features_paths.items()}
                try:
                    for ri,row in enumerate(rows):
                        assert row['id'].startswith(f'{task}:{split}:')
                        row={**row,'retained_source_sha256':digest(row)}
                        if task=='ner':
                            w=common_windows(row['tokens'],toks); own=owners(w,len(row['tokens'])); row.update(windows=w,owners=own)
                            stats['windows']+=len(w); stats['words']+=len(row['tokens']); stats['labels'].update(row['labels'])
                            gold,repairs=bio_spans(row['labels']); assert repairs==0
                            for a,b,k in gold:
                                stats['cross_window_entities']+=any(a<x<b or a<y<b for x,y in [(win['start'],win['end']) for win in w])
                                stats['entities_without_one_full_window']+=not any(win['start']<=a and b<=win['end'] for win in w)
                                stats['entities_crossing_owners']+=len(set(own[a:b]))>1
                            for wi,win in enumerate(w):
                                count=sum(o==wi for o in own); stats['zero_owned_windows']+=count==0
                                # Zero-owned windows are retained for complete inference but not training supervision.
                                keep=split=='valid' or count>0
                                for name,tok in toks.items():
                                    enc=tok(win['tokens'],is_split_into_words=True,truncation=False); word_ids=enc.word_ids()
                                    labels=align_first_subwords(word_ids,row['labels'][win['start']:win['end']],label2id)
                                    for i,word in enumerate(word_ids):
                                        if word is not None and own[win['start']+word]!=wi: labels[i]=-100
                                    assert sum(x!=-100 for x in labels)==count
                                    if keep: emit_feature(streams[name],stats['models'][name],tok,{'id':f'{row["id"]}:w{wi}','source_id':row['id'],'window':wi,
                                        **dict(enc),'word_ids':word_ids,'labels':labels})
                                if keep: feature_ids.append(f'{row["id"]}:w{wi}')
                            assert sum(sum(o==wi for o in own) for wi in range(len(w)))==len(row['tokens'])
                        else:
                            visible,fallback=common_prefix(row['text'],toks); row.update(visible_text=visible,capped=visible!=row['text'],character_fallback=fallback)
                            stats['capped_rows']+=row['capped']; stats['character_fallbacks']+=fallback; stats['labels'].update([str(row['label'])])
                            if row['capped']: changes.append({'id':row['id'],'original_chars':len(row['text']),'visible_chars':len(visible),'character_fallback':fallback})
                            for name,tok in toks.items(): emit_feature(streams[name],stats['models'][name],tok,{'id':row['id'],'source_id':row['id'],**dict(tok(visible,truncation=False)),'labels':int(row['label'])})
                            feature_ids.append(row['id'])
                        stream.write(json.dumps(row,ensure_ascii=False)+'\n')
                        if split=='valid' and row['id'] in old_by_id:
                            old=old_by_id[row['id']]
                            if task=='ner':
                                assert [(x['start'],x['end'],x['tokens']) for x in row['windows']]==[(x['start'],x['end'],x['tokens']) for x in old['windows']]
                                assert row['owners']==old['owners'] and row['tokens']==old['tokens']
                            else: assert row['visible_text'].encode('utf-8')==old['visible_text'].encode('utf-8')
                            panel.append(row)
                        if (ri+1)%10000==0: print({'task':task,'split':split,'rows':ri+1},flush=True)
                finally:
                    for f in streams.values(): f.close()
            stats['features']=len(feature_ids); stats['preparation_seconds']=time.perf_counter()-started
            stats['training_features']=len(feature_ids) if split=='train' else None
            all_stats[f'{task}_{split}']=stats
            for p in [out_path,*features_paths.values()]: files[p.relative_to(D).as_posix()]=file_hash(p)
            if task=='sa': write_jsonl(D/'views'/f'sa_{split}_caps.jsonl',changes)
            if split=='train':
                selected=list(feature_ids); random.Random(13).shuffle(selected); selected=selected[:1600]; assert len(selected)==1600
                calibration[task]={'feature_ids':selected,'sha256':digest(selected),'source':'retained_training_only','seed':13}
                for name in toks:
                    selected_set=set(selected); subset={r['id']:r for r in read_jsonl(features_paths[name]) if r['id'] in selected_set}
                    assert len(subset)==1600
                    p=D/'calibration_data'/f'{name}_{task}.jsonl'; write_jsonl(p,[subset[i] for i in selected]); files[p.relative_to(D).as_posix()]=file_hash(p)
            else:
                index={r['id']:r for r in panel}; panel=[index[i] for i in old_by_id]; assert len(panel)==200
                p=D/'views'/f'{task}_panel.jsonl'; write_jsonl(p,panel); files[p.relative_to(D).as_posix()]=file_hash(p)
                continuity[task]={'n':200,'same_visible_text':True,'ids':list(old_by_id)}
    write_json(D/'calibration_data/selection.json',calibration)
    files['calibration_data/selection.json']=file_hash(D/'calibration_data/selection.json')
    write_json(D/'views/manifest.json',{'plan_sha256':verify('views_plan.lock.json'),'stats':all_stats,'files':files,'continuity':continuity,
        'tokenizers':{n:{'class':type(t).__name__,'is_fast':t.is_fast,'model_max_length':t.model_max_length,'vocab_size':t.vocab_size,'do_lower_case':getattr(t,'do_lower_case',None)} for n,t in toks.items()},
        'policy':'common 384 subwords / NER 96 words overlap 24 / SA 128 words literal prefix; both tokenizers; no gold-assisted text selection'})
    print({n:{k:v for k,v in s.items() if k not in ['models','labels']} for n,s in all_stats.items()},flush=True)

def emit_feature(stream,stats,tok,feature):
    n=len(feature['input_ids']); assert 0<n<=384
    stats['features']+=1; stats['tokens']+=n; stats['max_tokens']=max(stats['max_tokens'],n)
    stats['unknown_tokens']+=feature['input_ids'].count(tok.unk_token_id); stats['length_histogram'][n]+=1
    stream.write(json.dumps(feature,ensure_ascii=False)+'\n')

def validate_files():
    verify('views_plan.lock.json'); manifest=read_json(D/'views/manifest.json')
    for name,h in manifest['files'].items(): assert file_hash(D/name)==h,name
    return manifest

if __name__=='__main__': prepare()
