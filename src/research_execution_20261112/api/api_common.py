"""Dependency-light API study helpers; imports never make a network request."""
from pathlib import Path
from decimal import Decimal, ROUND_CEILING
import hashlib
import importlib.util
import json
import sys

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[1]
sys.path.insert(0, str(PROJECT / 'ner_pipeline_20261111'))
from storage import canonical, digest, read_evidence, write_evidence, process_lock

MODEL = 'gpt-6.1-sol'
INPUT_CEILING = 8192
CAPS = {'ner': 8192, 'sa': 4096}
POLICY = {'model': MODEL, 'reasoning': {'effort': 'low'}, 'tools': [],
          'tool_choice': 'none', 'text': {'format': {'type': 'text'}},
          'store': False, 'stream': False, 'service_tier': 'default',
          'truncation': 'disabled'}
# Conservative generation reservation: cache-write input rate plus 10% regional
# surcharge. This is a client accounting assumption, not an invoice guarantee.
RESERVE_INPUT = Decimal('2.75')
RESERVE_OUTPUT = Decimal('11')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


alignment = load_module('api_existing_alignment', PROJECT / 'interface_pilot/alignment.py')
scoring = load_module('api_existing_sa_scoring', PROJECT / 'pilot/scoring.py')
strict = load_module('api_existing_strict_ner', PROJECT / 'ner_evidence_20261111/analysis/strict_ner.py')


def sources():
    paths = [ROOT / 'api_common.py', ROOT / 'prepare_api.py', ROOT / 'run_api.py', ROOT / 'connect_and_run.py',
             PROJECT / 'interface_pilot/alignment.py', PROJECT / 'pilot/scoring.py',
             PROJECT / 'ner_evidence_20261111/analysis/strict_ner.py',
             PROJECT / 'ner_pipeline_20261111/storage.py',
             PROJECT / 'ner_pipeline_20261111/native_inputs.py']
    return {str(p.relative_to(PROJECT)): sha(p) for p in paths}


def request_body(task, messages):
    if task not in CAPS:
        raise ValueError('Only NER and SA are supported')
    if not messages or any(set(m) != {'role', 'content'} or
                           m['role'] not in ('system', 'user', 'assistant') or
                           not isinstance(m['content'], str) for m in messages):
        raise ValueError('Only explicit text messages are supported')
    return {**POLICY, 'input': messages, 'max_output_tokens': CAPS[task]}


def reserve_usd(task, input_tokens=INPUT_CEILING):
    return ((Decimal(input_tokens) * RESERVE_INPUT + Decimal(CAPS[task]) * RESERVE_OUTPUT)
            / Decimal(1000000)).quantize(Decimal('0.00000001'), rounding=ROUND_CEILING)


def validate_package(package):
    if package.get('format') != 'kazakh_api_train_dev_v1' or package.get('sources') != sources():
        raise ValueError('Package/code provenance mismatch')
    if digest({k: v for k, v in package.items() if k != 'package_sha256'}) != package['package_sha256']:
        raise ValueError('Package digest mismatch')
    if package['generation_policy'] != POLICY or package['output_caps'] != CAPS:
        raise ValueError('Generation policy drift')
    seen = set()
    for item in package['requests']:
        if item['split'] not in ('train', 'valid') or not item['id'].startswith(item['task']+':'+item['split']+':'):
            raise PermissionError('Final test is not admitted by this package')
        if item['shots'] not in (0, 4) or item['stage'] != ('smoke' if item['split'] == 'train' else 'dev'):
            raise ValueError('Unexpected experimental condition')
        if item['request'] != request_body(item['task'], item['request']['input']):
            raise ValueError('Unplanned API parameter')
        if item['request_sha256'] != digest(item['request']):
            raise ValueError('Request digest mismatch')
        identity = (item['task'], item['id'], item['shots'])
        if identity in seen: raise ValueError('Duplicate planned request')
        seen.add(identity)
        if item['task'] == 'ner' and item['request']['input'][-1]['content'] != 'Kazakh text:\n'+' '.join(item['tokens']):
            raise ValueError('NER query/input mismatch')
        if item['task'] == 'sa' and item['request']['input'][-1]['content'] != 'Review (JSON string):\n'+json.dumps(item['visible_text'],ensure_ascii=False):
            raise ValueError('SA query/input mismatch')
    counts = {(t,s,n):sum(x['task']==t and x['stage']==s and x['shots']==n for x in package['requests'])
              for t in CAPS for s in ('smoke','dev') for n in (0,4)}
    if any(v != (10 if k[1]=='smoke' else 100) for k,v in counts.items()):
        raise ValueError('Incomplete fixed 440-request panel')
    return {'requests':len(seen),'smoke':40,'dev':400}


def parse_response(response, item, allowed_types):
    """Preserve raw output and classify; never recover JSON by editing text."""
    if not isinstance(response,dict): raise ValueError('Response must be an object')
    status=response.get('status');texts=[];refusals=[];unexpected=[]
    for output in response.get('output',[]):
        if output.get('type')=='reasoning': continue
        if output.get('type')!='message' or output.get('role')!='assistant':
            unexpected.append(output.get('type'));continue
        for part in output.get('content',[]):
            if part.get('type')=='output_text': texts.append(part.get('text',''))
            elif part.get('type')=='refusal': refusals.append(part.get('refusal',''))
            else: unexpected.append(part.get('type'))
    raw=''.join(texts)
    terminal = status in ('failed','cancelled')
    if status not in ('completed','incomplete','failed','cancelled') or unexpected:
        raise ValueError('Unexpected nonterminal response or tool/content output')
    effective_raw='' if terminal else raw
    parsed = alignment.parse_mentions(effective_raw,item['tokens'],allowed_types) if item['task']=='ner' else scoring.parse_sa(effective_raw)
    return {'raw':raw,'parsed':parsed,'status':status,'terminal_failure':terminal,
            'truncated':status=='incomplete' and (response.get('incomplete_details') or {}).get('reason')=='max_output_tokens',
            'refusal':bool(refusals),'refusals':refusals,'incomplete_details':response.get('incomplete_details'),
            'usage':response.get('usage'),'returned_model':response.get('model'),
            'response_id':response.get('id'),'system_fingerprint':response.get('system_fingerprint')}


def score_records(package, records, gold):
    """Score only complete fixed conditions; altered/missing gold is rejected."""
    if gold['package_sha256'] != package['package_sha256'] or digest(gold['rows']) != package['gold_rows_sha256']:
        raise ValueError('Gold differs from frozen package')
    gold_by_id={r['id']:r for r in gold['rows']}
    results={}
    for task in CAPS:
        for stage in ('smoke','dev'):
            for shots in (0,4):
                items=[i for i in package['requests'] if (i['task'],i['stage'],i['shots'])==(task,stage,shots)]
                keys=[i['key'] for i in items]
                if not any(k in records for k in keys):continue
                if not all(k in records for k in keys):raise ValueError('Incomplete condition cannot be scored')
                if task=='sa':
                    m=scoring.score_sa([int(gold_by_id[i['id']]['label']) for i in items],
                                       [records[k]['parsed']['label'] for k in keys])
                else:
                    counts=[0,0,0];types={t:[0,0,0] for t in package['allowed_ner_types']}
                    for item,key in zip(items,keys):
                        g=strict.bio_to_spans(gold_by_id[item['id']]['labels'])[0];p=records[key]['parsed']['spans']
                        counts=[a+b for a,b in zip(counts,strict.exact_counts(g,p))]
                        for t in types:
                            types[t]=[a+b for a,b in zip(types[t],strict.exact_counts([s for s in g if s[2]==t],[s for s in p if s[2]==t]))]
                    m={'n':len(items),**strict.prf(*counts),'per_type':{t:strict.prf(*v) for t,v in types.items()},
                       'invalid_candidate_penalty_in_primary':False}
                m.update(failed_items=sum(records[k]['parsed']['failed'] for k in keys),
                         truncated_items=sum(records[k]['truncated'] for k in keys),
                         refusal_items=sum(records[k]['refusal'] for k in keys))
                results[f'{task}_{stage}_{shots}shot']=m
    return results
