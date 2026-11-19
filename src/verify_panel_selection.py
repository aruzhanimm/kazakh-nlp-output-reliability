"""Verify archived primary panel IDs from pinned official source files; no model calls.

Run from the released repository with --ner IOB2_test.txt --sa 05_pc_test.zip.
The sources are read only. No tokenizer, checkpoint or paid service is needed.
"""
from pathlib import Path
import argparse,csv,hashlib,io,json,zipfile

def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def population(path,task):
    if task=='ner':
        ids=[];occupied=False
        for line in path.read_text(encoding='utf-8-sig').splitlines()+['']:
            if line.strip():occupied=True
            elif occupied:
                ids.append(f'ner:test:{len(ids)}');occupied=False
        return ids
    with zipfile.ZipFile(path) as z:
        names=[n for n in z.namelist() if n.endswith('.csv')]
        if len(names)!=1:raise ValueError('Exactly one official CSV required')
        with z.open(names[0]) as f:
            rows=csv.DictReader(io.TextIOWrapper(f,encoding='utf-8-sig'))
            return ['sa:test:'+r['custom_id'] for r in rows]

def verify(root,ner,sa):
    cfg=json.loads((root/'configs/PRIMARY_PROTOCOL.json').read_text(encoding='utf-8'))
    s=cfg['selection'];assert s['items_per_task']==500 and s['seed']==20261110 and s['namespace']=='kazakh-final-v2'
    report={'status':'PASS_PANEL_IDS_RECONSTRUCTED_FROM_PINNED_SOURCES','selection':s,'tasks':{},'no_inference_or_scoring':True}
    for task,path in [('ner',ner),('sa',sa)]:
        actual_sha=digest(path)
        if actual_sha!=cfg['datasets'][task]['sha256']:raise ValueError('Pinned source SHA differs: '+task)
        ids=population(path,task)
        if len(ids)!=len(set(ids)):raise ValueError('Duplicate official IDs')
        def key(rid):
            payload=json.dumps([s['namespace'],s['seed'],task,rid],ensure_ascii=False,separators=(',',':')).encode('utf-8')
            return hashlib.sha256(payload).hexdigest(),rid
        selected=sorted(ids,key=key)[:s['items_per_task']]
        saved=json.loads((root/f'configs/{task}_primary500_ids.json').read_text(encoding='utf-8'))
        if selected!=saved['ids']:raise ValueError('Panel IDs/order differ: '+task)
        evidence=[json.loads(x)['id'] for x in (root/f'evidence/{task}_items.jsonl').read_text(encoding='utf-8').splitlines() if x.strip()]
        if set(selected)!=set(evidence):raise ValueError('Evidence panel membership differs: '+task)
        report['tasks'][task]={'source_sha256':actual_sha,'official_items':len(ids),'selected_items':len(selected),'ids_and_order_match':True,'released_evidence_membership_matches':True,'first_rank_serialized':json.dumps([s['namespace'],s['seed'],task,selected[0]],ensure_ascii=False,separators=(',',':')),'first_rank_sha256':key(selected[0])[0]}
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--repository',type=Path,default=Path(__file__).resolve().parents[1]);p.add_argument('--ner',required=True,type=Path);p.add_argument('--sa',required=True,type=Path);p.add_argument('--output',type=Path)
    a=p.parse_args();result=verify(a.repository,a.ner,a.sa);text=json.dumps(result,ensure_ascii=False,indent=2)+'\n'
    if a.output:
        with a.output.open('x',encoding='utf-8') as f:f.write(text)
    print(text)
