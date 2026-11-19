"""Build fixed API train/dev inputs offline; never reads test or imports a model."""
from pathlib import Path
from collections import Counter
import argparse
import datetime
import json
from api_common import *


def rows(path):
    if Path(path).name not in ('sa_train.jsonl','sa_valid.jsonl','sa_demos.jsonl'):
        raise PermissionError('Only explicit SA train/dev sources are read here')
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]


def ranked(items, namespace):
    return sorted(items,key=lambda x:(digest([namespace,20261112,x['id']]),x['id']))


def sa_messages(text, demos, definition):
    messages=[{'role':'system','content':definition}]
    for r in demos:
        if not r['id'].startswith('sa:train:'):raise ValueError('Only training demonstrations')
        messages += [{'role':'user','content':'Review (JSON string):\n'+json.dumps(r['visible_text'],ensure_ascii=False)},
                     {'role':'assistant','content':json.dumps({'label':'positive' if r['label']=='1' else 'negative'},ensure_ascii=False,separators=(',',':'))}]
    messages.append({'role':'user','content':'Review (JSON string):\n'+json.dumps(text,ensure_ascii=False)})
    return messages


def prepare(output):
    native=load_module('api_existing_ner_messages',PROJECT/'ner_pipeline_20261111/native_inputs.py')
    protocol_path=PROJECT/'ner_runtime_20261111/NER_DEV_PROTOCOL.json'
    protocol=json.loads(protocol_path.read_text(encoding='utf-8'))
    if sha(protocol_path)!='a15fa0a62eed383faf5b91511d379afe499cb9b94b358244dc7cdb207fd6c13e':raise ValueError('Bounded NER protocol changed')
    prep_root=PROJECT/'ner_runtime_20261111/prepared/20261111T145039Z'
    receipt_path=prep_root/'PREPARATION_RECEIPT.json'
    if sha(receipt_path)!='bdb0e15312c6e75d5ad3bdf4896e7e1d0ce6e15abdef6ab656bf44dc06166a42':
        raise ValueError('Previously verified NER preparation receipt changed')
    receipt=json.loads(receipt_path.read_text(encoding='utf-8'))
    source_paths=[protocol_path,PROJECT/protocol['definitions']['path'],PROJECT/'prompts/sa_v1.txt',
                  PROJECT/'data/pilot/sa_train.jsonl',PROJECT/'data/pilot/sa_valid.jsonl',PROJECT/'data/pilot/sa_demos.jsonl']
    definitions=(PROJECT/protocol['definitions']['path']).read_text(encoding='utf-8')
    if sha(PROJECT/protocol['definitions']['path'])!=protocol['definitions']['sha256']:raise ValueError('DATE definition changed')
    ner={};gold={};ner_native4={}
    for stage in ('smoke','dev'):
        pp=prep_root/stage/'qwen_ordinary4/PANEL.json';gp=pp.with_name('GOLD.json')
        original=next(x for x in receipt['panels'] if x['stage']==stage and x['condition']=='qwen_ordinary4')
        if sha(pp)!=original['panel_file_sha256'] or sha(gp)!=original['gold_file_sha256']:
            raise ValueError('NER native input or gold file changed')
        source_paths += [pp,gp]
        panel=read_evidence(pp);g=read_evidence(gp)
        if g['panel_sha256']!=panel['panel_sha256']:raise ValueError('NER panel/gold mismatch')
        chosen=panel['items'] if stage=='smoke' else ranked(panel['items'],'kazakh-api-dev-v1')[:100]
        for item in chosen:
            if len(item['windows'])!=1 or item['windows'][0]['start']!=0 or item['windows'][0]['end']!=len(item['tokens']):
                raise ValueError('Current API preparation supports only verified complete single windows')
            ner_native4[item['id']]=item['prepared_prompts'][0]['messages']
        ner[stage]=chosen;gold.update({r['id']:r for r in g['rows'] if r['id'] in {i['id'] for i in chosen}})
    sa_demos=rows(PROJECT/'data/pilot/sa_demos.jsonl')
    if len(sa_demos)!=4 or Counter(r['label'] for r in sa_demos)!={'0':2,'1':2}:raise ValueError('SA primary pack changed')
    sa_definition=(PROJECT/'prompts/sa_v1.txt').read_text(encoding='utf-8')
    demo_ids={r['id'] for r in sa_demos};demo_keys={r['text_key'] for r in sa_demos}
    train=[r for r in rows(PROJECT/'data/pilot/sa_train.jsonl') if r['id'] not in demo_ids and r['text_key'] not in demo_keys]
    chosen=[];seen=set()
    for label in ('0','1'):
        for r in ranked([r for r in train if r['label']==label],'kazakh-api-sa-smoke-v1'):
            if r['text_key'] in seen:continue
            chosen.append(r);seen.add(r['text_key'])
            if sum(x['label']==label for x in chosen)==5:break
    sa={'smoke':chosen,'dev':ranked(rows(PROJECT/'data/pilot/sa_valid.jsonl'),'kazakh-api-dev-v1')[:100]}
    requests=[]
    for task,panels in [('ner',ner),('sa',sa)]:
        for stage,items in panels.items():
            split='train' if stage=='smoke' else 'valid'
            for shots in (0,4):
                for r in items:
                    if not r['id'].startswith(task+':'+split+':'):raise ValueError('Split leakage')
                    demos=protocol['demonstrations']['primary']['items'] if task=='ner' else sa_demos
                    messages=native.build_messages(r['tokens'],definitions,demos if shots else [],protocol['allowed_types']) if task=='ner' else sa_messages(r['visible_text'],demos if shots else [],sa_definition)
                    if task=='ner' and shots and messages!=ner_native4[r['id']]:raise ValueError('API NER content differs from local primary condition')
                    request=request_body(task,messages)
                    item={'id':r['id'],'task':task,'split':split,'stage':stage,'shots':shots,
                          'request':request,'request_sha256':digest(request),
                          'input_text_utf8_bytes':sum(len(m['content'].encode('utf-8')) for m in messages),
                          'input_tokens':None,'native_token_count_status':'requires_authorized_API_count_not_HF_estimate'}
                    item.update({'tokens':r['tokens']} if task=='ner' else {'visible_text':r['visible_text']})
                    item['key']=digest({'id':r['id'],'task':task,'shots':shots,'request':request})
                    requests.append(item)
                    if task=='sa':gold[r['id']]={'id':r['id'],'visible_text':r['visible_text'],'label':r['label']}
    gold_rows=[gold[k] for k in sorted(gold)]
    package={'format':'kazakh_api_train_dev_v1','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
             'status':'offline_prepared_not_authorized','generation_policy':POLICY,'output_caps':CAPS,
             'input_token_ceiling':INPUT_CEILING,'allowed_ner_types':protocol['allowed_types'],
             'sources':sources(),'input_sources':{str(p.relative_to(PROJECT)):sha(p) for p in source_paths},
             'demonstration_ids':{'ner':[r['id'] for r in protocol['demonstrations']['primary']['items']],
                                  'sa':[r['id'] for r in sa_demos]},
             'selection':{'dev':'first100 by SHA256 canonical [kazakh-api-dev-v1,20261112,id] from existing dev200, no labels/scores',
                          'ner_smoke':'reuse fixed10 independent train queries from NER bounded protocol',
                          'sa_smoke':'five per binary label, hash order, exclude demo IDs/text_key and duplicate text_key'},
             'gold_rows_sha256':digest(gold_rows),'requests':requests,'final_test_access':False,'paid_calls':0}
    package['package_sha256']=digest(package);validate_package(package)
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    write_evidence(output/'REQUESTS.json',package)
    write_evidence(output/'GOLD.json',{'package_sha256':package['package_sha256'],'rows':gold_rows})
    summary={'status':'OFFLINE_PREPARATION_PASS','package_sha256':package['package_sha256'],
             'request_count':len(requests),'unique_items':len(gold_rows),'train_requests':40,'dev_requests':400,
             'api_native_input_tokens_verified':False,'max_input_text_utf8_bytes':max(r['input_text_utf8_bytes'] for r in requests),
             'source_parity':'NER four-shot equals prepared local primary messages; SA builder matches existing pilot format',
             'model_calls':0,'network_calls':0,'GPU_calls':0,'test_data_access':False,'torch_imported':'torch' in sys.modules}
    write_evidence(output/'PREPARATION_RECEIPT.json',summary)
    return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',required=True)
    print(json.dumps(prepare(parser.parse_args().output),ensure_ascii=False,indent=2))
