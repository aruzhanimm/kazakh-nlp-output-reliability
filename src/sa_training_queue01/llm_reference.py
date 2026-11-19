"""Read ONLY completed local exports; never connect to or operate a server."""
import hashlib,json,zipfile
from pilot.common import ROOT,read_json,read_jsonl,file_hash,digest
from pilot.evaluate import summarize
from .common import Q,utc,durable

def prepare():
    panel=read_jsonl(ROOT/'data/pilot/sa_valid.jsonl');records=[];sources={};members={}
    specs=[('qwen_bf16','server_qwen_bf16_20261109T114050Z_d5cac6d0','outputs/bf16'),
           ('sherkala_bf16','server_sherkala_resume_20261109T150428Z_21b68e41','outputs/main')]
    for model,folder,prefix in specs:
        local=ROOT.parent/folder;archive=local/'server_export/review.zip';audit=local/'LOCAL_VERIFICATION.json'
        expected=read_json(audit)['archive_sha256'];assert file_hash(archive)==expected
        sources[str(archive)]=expected;sources[str(audit)]=file_hash(audit)
        with zipfile.ZipFile(archive) as z:
            saved=[json.loads(x) for x in z.read('payload/data/pilot/sa_valid.jsonl').decode('utf-8').splitlines()]
            assert saved==panel,'LLM evaluation text/labels/IDs differ'
            for shots in [0,4]:
                base=f'{prefix}/sa_{shots}shot';ib=z.read(base+'/identity.json');identity=json.loads(ib)
                assert identity['ids']==[r['id'] for r in panel]
                if model=='qwen_bf16':
                    actual=json.loads(z.read(f'outputs/actual_model_sa_{shots}.json'))
                    assert set(actual['parameter_dtypes'])=={'torch.bfloat16'} and not actual['is_quantized']
                    assert identity['config']['model_dtype']=='bfloat16' and identity['config']['quantization'] is None
                    precision={'actual_parameter_dtypes':actual['parameter_dtypes'],'quantized':False}
                else:
                    candidates=[n for n in z.namelist() if n.startswith('NATIVE_FREEZE') and n.endswith('.json')]
                    # Original runner records digest(parsed JSON), not the file-byte hash.
                    match=[n for n in candidates if digest(json.loads(z.read(n)))==identity['native_freeze_sha256']]
                    assert len(match)==1
                    native=json.loads(z.read(match[0]));assert native['model_dtype']=='bfloat16' and native['quantization'] is None
                    precision={'native_freeze':match[0],'dtype':native['model_dtype'],'quantization':None}
                pb=z.read(base+'/predictions.jsonl');pred=[json.loads(x) for x in pb.decode('utf8').splitlines()]
                assert [p['id'] for p in pred]==[r['id'] for r in panel]
                assert all(p['input_sha256']==digest(r) for p,r in zip(pred,panel))
                metrics=summarize('sa',panel,pred)
                target=Q/'llm_reference'/f'{model}_{shots}shot';target.mkdir(parents=True,exist_ok=True)
                for name,b in [('predictions.jsonl',pb),('identity.json',ib)]:
                    dest=target/name
                    if dest.exists():assert dest.read_bytes()==b
                    else:dest.write_bytes(b)
                if (target/'metrics.json').exists():assert read_json(target/'metrics.json')==metrics
                else:durable(target/'metrics.json',metrics)
                for p in target.iterdir():members[p.relative_to(Q).as_posix()]=file_hash(p)
                records.append({'condition':f'{model}_{shots}shot','model':model,'shots':shots,'n':200,
                    'macro_f1':metrics['macro_f1'],'accuracy':metrics['accuracy'],'invalid':metrics['invalid'],
                    'format_failures':metrics['format_failures'],'runtime_failures':metrics['runtime_failures'],
                    'precision_provenance':precision,'archive':str(archive),'member':base+'/predictions.jsonl',
                    'predictions_sha256':hashlib.sha256(pb).hexdigest(),'same_ids_text_labels':True})
    durable(Q/'LLM_REFERENCE.json',{'utc':utc(),'records':records,'source_hashes':sources,'copied_files':members,
        'new_model_calls':0,'server_accessed':False,'scope':'Completed local export bytes only; original failures retained; descriptive comparison, unequal adaptation budgets.'})

def comparison(q):
    audit=read_json(Q/'LLM_REFERENCE.json')
    for n,h in audit['copied_files'].items():assert file_hash(Q/n)==h
    rows=[{'system':r['condition'],'model':r['model'],'lr':None,'seed':None,'shots':r['shots'],'selected_epoch':None,
        'n':r['n'],'macro_f1':r['macro_f1'],'accuracy':r['accuracy'],'invalid':r['invalid'],
        'format_failures':r['format_failures'],'runtime_failures':r['runtime_failures'],'source_sha256':r['predictions_sha256']} for r in audit['records']]
    for e in q['conditions']:
        if e['status']!='verified_complete':continue
        from pathlib import Path
        d=Path(e['directory']);final=read_json(d/'FINAL_STATUS.json');epoch=final['selected_epoch'];panel=read_json(d/'panel'/f'epoch_{epoch:02d}.json');m=panel['metrics']
        rows.append({'system':e['id'],'model':e['encoder'],'lr':e['learning_rate'],'seed':e['seed'],'shots':None,'selected_epoch':epoch,
            'n':m['n'],'macro_f1':m['macro_f1'],'accuracy':m['accuracy'],'invalid':m['invalid'],'format_failures':m['format_failures'],
            'runtime_failures':m['runtime_failures'],'source_sha256':panel['source_sha256']})
    return rows

if __name__=='__main__':prepare()
