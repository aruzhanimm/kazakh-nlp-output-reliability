"""Frozen final GPT reference; two workers, shared bounded ledger, no retries.

Metadata and imports never read final panels. prepare/execute require actual
Freeze B. Only explicit execute constructs the authenticated live transport.
"""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from decimal import Decimal
from pathlib import Path
import argparse,datetime,getpass,json,os,sys,threading

HERE=Path(__file__).resolve().parent
API=HERE.parent
PROJECT=API.parents[1]
sys.path.insert(0,str(API))
from api_common import canonical,digest,sha,read_evidence,write_evidence,process_lock,parse_response,MODEL
from run_api import OpenAIHTTP
sys.path.insert(0,str(PROJECT/'research_execution_20261112/final_evaluation'))
import output_adapter as adapter

CAPS={'ner':4096,'sa':1024}
INPUT_CEILING=8192
INPUT_RATE=Decimal('5.5')
OUTPUT_RATE=Decimal('11')
CONDITIONS={'ner':'ner_gpt61sol_ordinary4','sa':'sa_gpt61sol_ordinary4'}
POLICY={'model':MODEL,'reasoning_effort':'low','shots':4,'ordinary_text_only':True,
        'conditions':CONDITIONS,'caps':CAPS,'input_ceiling':INPUT_CEILING,'workers':2,
        'max_total_generation_ledger_usd':'20','reservation_input_per_million':'5.5',
        'reservation_output_per_million':'11','automatic_retries':False,
        'sa_incomplete_policy':'INVALID','ner_incomplete_policy':'retain_parseable_entities_flag_failure',
        'panel_items_per_task':500,'all_ner_windows_required':True}
PILOT_SHA='ed9daeb3343dc48e3480d741f4e02a2d94ee53c7f58904d0154212a5fcca27ea'

def utc():return datetime.datetime.now(datetime.timezone.utc)
def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def fee(n,o):return (Decimal(n)*INPUT_RATE+Decimal(o)*OUTPUT_RATE)/Decimal(1000000)
def sources():
    paths=[Path(__file__),API/'api_common.py',API/'run_api.py',PROJECT/'ner_pipeline_20261111/storage.py']
    return {**adapter.sources(),**{p.relative_to(PROJECT).as_posix():sha(p) for p in paths}}

def templates():
    p=read_evidence(API/'pilot8/REQUESTS.json')
    if p['package_sha256']!=PILOT_SHA or p['package_sha256']!=digest({k:v for k,v in p.items() if k!='package_sha256'}):raise ValueError('Pilot template binding differs')
    result={}
    for task in CAPS:
        candidates=[r for r in p['requests'] if r['task']==task and r['shots']==4]
        prefix=candidates[0]['request']['input'][:-1]
        if any(r['request']['input'][:-1]!=prefix for r in candidates) or sum(m['role']=='assistant' for m in prefix)!=4:raise ValueError('Frozen four-shot prefix differs')
        result[task]=deepcopy(candidates[0]['request'])
    return p,result

def metadata():
    p,_=templates();review=read_evidence(HERE/'PILOT8_INDEPENDENT_REVIEW.json')
    return {'status':'DRAFT_NOT_FINAL_AUTHORIZATION','api_policy':POLICY,'api_runtime_sources':sources(),
            'api_prompt_templates':{'path':str(API/'pilot8/REQUESTS.json'),'sha256':sha(API/'pilot8/REQUESTS.json'),'package_sha256':p['package_sha256']},
            'api_prior_usage_review':{'path':str(HERE/'PILOT8_INDEPENDENT_REVIEW.json'),'sha256':sha(HERE/'PILOT8_INDEPENDENT_REVIEW.json')},
            'prior_conservative_usage_usd':review['total_conservative_usage_estimate_usd'],
            'api_ledger_root':str(HERE/'live'),'api_budget_required':{'status':'user_approved','approval_reference':'ACTUAL_USER_USD20_APPROVAL_REFERENCE','limit_total_usd':'20','expires_utc':'ACTUAL_EXPIRY','execution_mode':'live_openai'},
            'no_final_panel_read':True,'no_API_calls':True}

def bound(spec):
    if sha(spec['path'])!=spec['sha256']:raise ValueError('Bound file SHA mismatch')
    return read(spec['path'])

def check_freeze(path,expected_sha):
    if sha(path)!=expected_sha:raise ValueError('Freeze B SHA differs')
    f=read(path)
    if f.get('scope')!='final' or f.get('status')!='frozen_for_final_inference' or f.get('final_test_access_authorized')is not True or not f.get('decision_reference'):raise PermissionError('Actual Freeze B required before any final panel access')
    if f.get('api_policy')!=POLICY or f.get('api_runtime_sources')!=sources():raise ValueError('Final API code/policy differs from Freeze B')
    m=metadata()
    for k in ['api_prompt_templates','api_prior_usage_review']:
        if f.get(k)!=m[k]:raise ValueError('Pilot/template/prior usage binding differs')
    if Path(f.get('api_ledger_root','')).resolve()!=(HERE/'live').resolve():raise ValueError('Only this dedicated final API ledger is admitted')
    budget=f.get('api_budget',{})
    if budget.get('status')!='user_approved' or not budget.get('approval_reference') or budget.get('execution_mode')!='live_openai':raise PermissionError('Actual USD20 user approval required')
    limit=Decimal(str(budget.get('limit_total_usd','0')))
    if not limit.is_finite() or not 0<limit<=20:raise PermissionError('Explicit positive limit cannot exceed USD20')
    expiry=datetime.datetime.fromisoformat(budget['expires_utc'].replace('Z','+00:00'))
    if expiry.tzinfo is None or utc()>=expiry:raise PermissionError('Final API authorization expired')
    prior=read_evidence(HERE/'PILOT8_INDEPENDENT_REVIEW.json')
    if prior['status']!='INDEPENDENT_RAW_REPARSE_PASS' or prior['pilot_package_sha256']!=PILOT_SHA:raise ValueError('Completed pilot review required')
    return f,limit,Decimal(prior['total_conservative_usage_estimate_usd'])

def make_request(task,source,window_index,template):
    body=deepcopy(template)
    text='Kazakh text:\n'+' '.join(source['windows'][window_index]['tokens']) if task=='ner' else 'Review (JSON string):\n'+json.dumps(source['visible_text'],ensure_ascii=False)
    body['input'][-1]={'role':'user','content':text}
    if body['max_output_tokens']!=CAPS[task] or body['model']!=MODEL or body['reasoning']!={'effort':'low'} or body['tools']!=[]:raise ValueError('Frozen API template settings differ')
    return body

def prepare(freeze_path,freeze_sha):
    f,limit,prior=check_freeze(freeze_path,freeze_sha)
    pilot,t=templates();panels={};rows=[]
    for task in CAPS:
        panel=bound(f['panels'][task]);ids=panel['expected_ids']
        if panel['format']!='encoder_common_view_panel_v1' or panel['task']!=task or panel['split']!='test' or panel['freeze_a_sha256']!=f['freeze_a_sha256']:raise ValueError('Wrong final panel binding')
        if len(ids)!=500 or len(set(ids))!=500 or [r['id'] for r in panel['items']]!=ids or any(not i.startswith(task+':test:') for i in ids):raise ValueError('Final panel must contain exactly500 unique ordered test IDs')
        if digest(ids)!=f['panels'][task]['ordered_ids_sha256']:raise ValueError('Ordered panel IDs differ from Freeze B')
        panels[task]=panel
        for source in panel['items']:
            if any(k in source for k in ('label','labels','gold','gold_spans')):raise ValueError('Query gold must not enter inference')
            if task=='ner':
                bounds=[(w['start'],w['end']) for w in source['windows']]
                adapter.strict.merge_window_spans(bounds,[[] for _ in bounds],len(source['tokens']))
                if any(w['tokens']!=source['tokens'][w['start']:w['end']] for w in source['windows']):raise ValueError('Common window token mismatch')
            for wi in range(len(source['windows']) if task=='ner' else 1):
                body=make_request(task,source,wi,t[task]);identity={'freeze_b_sha256':freeze_sha,'panel_sha256':f['panels'][task]['sha256'],'id':source['id'],'task':task,'window_index':wi,'condition':CONDITIONS[task],'request_sha256':digest(body)}
                rows.append({**identity,'key':digest(identity),'request':body,**({'tokens':source['windows'][wi]['tokens']} if task=='ner' else {})})
    # Interleave tasks by the already-fixed within-task order. No output-based ordering.
    ner=[r for r in rows if r['task']=='ner'];sa=[r for r in rows if r['task']=='sa'];ordered=[]
    for i in range(max(len(ner),len(sa))):
        if i<len(ner):ordered.append(ner[i])
        if i<len(sa):ordered.append(sa[i])
    payload={'format':'kazakh_final_api_v1','freeze_b_sha256':freeze_sha,'policy':POLICY,'sources':sources(),'panels':f['panels'],'allowed_ner_types':pilot['allowed_ner_types'],'requests':ordered,'request_count':len(ordered),'sample_count':1000,'prior_charge_reserve_usd':str(prior),'limit_total_usd':str(limit),'gold_present':False}
    payload['package_sha256']=digest(payload);write_evidence(HERE/'PREPARED_REQUESTS.json',payload)
    return {'status':'PREPARED_NO_API_CALLS','package_sha256':payload['package_sha256'],'requests':len(ordered),'NER_windows':len(ner),'SA_calls':len(sa),'examples':1000}

def normalize_response(value,item,allowed):
    r=parse_response(value,item,allowed)
    incomplete=r['status']=='incomplete'
    if item['task']=='sa':
        r['parsed']=adapter.parse_final_sa(r['raw'],truncated=r['truncated'],refusal=r['refusal'],runtime_error=r['terminal_failure'],terminal_failed=r['terminal_failure'] or incomplete)
    elif r['terminal_failure'] or r['refusal']:
        r['parsed']={**r['parsed'],'spans':[],'failed':True}
    if incomplete:r['parsed']={**r['parsed'],'failed':True}
    return r

def usage(value,item,counted):
    u=value.get('usage') or {};n=u.get('input_tokens');o=u.get('output_tokens')
    if type(n)is not int or type(o)is not int or n!=counted or not 0<n<=INPUT_CEILING or not 0<=o<=CAPS[item['task']]:raise ValueError('Unverified usage/caps; full reservation retained')
    return {'input_tokens':n,'output_tokens_including_reasoning':o,'usage':u,'conservative_charge_usd':str(fee(n,o)),
            'ordinary_uncached_rate_estimate_usd':str((Decimal(n)*2+Decimal(o)*10)/1000000),
            'pricing_note':'Conservative ledger prices every input at12h cache-write+10pct and every output including reasoning atstandard+10pct; not an invoice assertion'}

class Ledger:
    """One process OS-lock outside, one thread mutex inside; immutable events."""
    def __init__(self,root,prior,limit):
        self.root=Path(root);self.prior=prior;self.limit=limit;self.lock=threading.RLock();self.charges={};self.stop=threading.Event()
        for d in (self.root/'items').glob('*'):
            if (d/'SETTLED.json').exists():self.charges[d.name]=Decimal(read_evidence(d/'SETTLED.json')['conservative_charge_usd'])
            elif (d/'GENERATION_STARTED.json').exists():self.charges[d.name]=Decimal(read_evidence(d/'GENERATION_STARTED.json')['reserved_usd'])
            elif (d/'COUNT_STARTED.json').exists():self.charges[d.name]=Decimal(read_evidence(d/'COUNT_STARTED.json')['reserved_usd'])
        if self.total()>self.limit:raise PermissionError('Existing ledger already exceeds configured limit')
    def total(self):
        with self.lock:return self.prior+sum(self.charges.values(),Decimal(0))
    def reserve_count(self,item,package_sha):
        with self.lock:
            if self.stop.is_set():return False
            if item['key'] in self.charges:raise RuntimeError('Prior unresolved attempt; do not retry')
            reserve=fee(INPUT_CEILING,CAPS[item['task']])
            if self.total()+reserve>self.limit:self.stop.set();raise PermissionError('Budget stop BEFORE count/generation')
            write_evidence(self.root/'items'/item['key']/'COUNT_STARTED.json',{'utc':utc().isoformat(),'request_sha256':item['request_sha256'],'package_sha256':package_sha,'reserved_usd':str(reserve)})
            self.charges[item['key']]=reserve;return True
    def start_generation(self,item,n):
        with self.lock:
            if self.stop.is_set():return False
            reserve=fee(n,CAPS[item['task']])
            if reserve>self.charges[item['key']]:raise ValueError('Native count increased reserved maximum')
            write_evidence(self.root/'items'/item['key']/'GENERATION_STARTED.json',{'utc':utc().isoformat(),'request':item['request'],'request_sha256':item['request_sha256'],'native_input_tokens':n,'reserved_usd':str(reserve),'mode':'live_openai'})
            self.charges[item['key']]=reserve;return True
    def settle(self,item,settlement):
        with self.lock:
            charge=Decimal(settlement['conservative_charge_usd'])
            if charge>self.charges[item['key']]:raise ValueError('Usage exceeds full-cap reservation')
            write_evidence(self.root/'items'/item['key']/'SETTLED.json',settlement);self.charges[item['key']]=charge
    def observe_identity(self,value):
        with self.lock:
            identity={'model':value.get('model'),'service_tier':value.get('service_tier'),'system_fingerprint':value.get('system_fingerprint')}
            path=self.root/'MODEL_IDENTITY.json'
            if path.exists() and read_evidence(path)!=identity:raise ValueError('Observed model/tier/fingerprint changed; preserve response and halt before pooling')
            write_evidence(path,identity)

def run_one(item,package,ledger,expiry,transport_factory=OpenAIHTTP):
    folder=ledger.root/'items'/item['key']
    if ledger.stop.is_set():return None
    try:
        if utc()>=expiry:raise PermissionError('Budget approval expired')
        if not ledger.reserve_count(item,package['package_sha256']):return None
        transport=transport_factory()
        count_body={k:item['request'][k] for k in ('model','input','reasoning','text','tools','tool_choice','truncation')}
        counted=transport.post('responses/input_tokens',count_body);write_evidence(folder/'COUNT_RESPONSE.json',counted)
        if counted['status_code']!=200:raise RuntimeError('Count HTTP failure retained; no generation')
        n=json.loads(counted['body_utf8']).get('input_tokens')
        if type(n)is not int or not 0<n<=INPUT_CEILING:raise ValueError('Input context ceiling exceeded; no generation')
        if not ledger.start_generation(item,n):return None
        raw=transport.post('responses',item['request']);write_evidence(folder/'HTTP_RESPONSE.json',raw)
        if raw['status_code']!=200:raise RuntimeError('Response HTTP failure retained; unknown cost stays reserved; no retry')
        value=json.loads(raw['body_utf8']);settlement=usage(value,item,n)
        # Confirmed usage releases unused reservation even if interpretation later fails.
        ledger.settle(item,settlement)
        r=normalize_response(value,item,package['allowed_ner_types']);write_evidence(folder/'RECORD.json',r)
        if value.get('model')!=MODEL or value.get('service_tier')!='default' or (value.get('reasoning') or {}).get('effort')!='low' or value.get('max_output_tokens')!=CAPS[item['task']] or ((value.get('text') or {}).get('format') or {}).get('type')!='text':raise ValueError('Returned model/tier/configuration differs; preserve and stop')
        ledger.observe_identity(value)
        # A malformed/truncated/refused task answer is a scientific outcome, not a retry trigger.
        write_evidence(folder/'COMPLETE.json',{'key':item['key'],'record_sha256':digest(r),'http_sha256':digest(raw),'returned_model':value['model'],'system_fingerprint':value.get('system_fingerprint'),'response_id':value.get('id'),'status':value.get('status')})
        return r
    except Exception as error:
        ledger.stop.set()
        write_evidence(folder/'STOPPED.json',{'utc':utc().isoformat(),'exception_type':type(error).__name__,'reason':str(error),'automatic_retry':False,'unknown_cost_keeps_reservation':True})
        return None

def revalidate_completed(item,package,root):
    folder=root/'items'/item['key']
    if not (folder/'COMPLETE.json').exists():return None
    start=read_evidence(folder/'GENERATION_STARTED.json');raw=read_evidence(folder/'HTTP_RESPONSE.json');saved=read_evidence(folder/'RECORD.json');done=read_evidence(folder/'COMPLETE.json')
    if start['request']!=item['request'] or start['request_sha256']!=item['request_sha256']:raise ValueError('Saved request differs')
    value=json.loads(raw['body_utf8']);rebuilt=normalize_response(value,item,package['allowed_ner_types'])
    if digest(saved)!=digest(rebuilt) or digest(saved)!=done['record_sha256'] or digest(raw)!=done['http_sha256']:raise ValueError('Raw/parsed/receipt integrity differs')
    if read_evidence(folder/'SETTLED.json')!=usage(value,item,start['native_input_tokens']):raise ValueError('Saved usage settlement differs')
    identity={'model':value.get('model'),'service_tier':value.get('service_tier'),'system_fingerprint':value.get('system_fingerprint')}
    if read_evidence(root/'MODEL_IDENTITY.json')!=identity:raise ValueError('Completed response identity differs from run binding')
    return saved

def export_predictions(package,root):
    records={r['key']:revalidate_completed(r,package,root) for r in package['requests']}
    if any(r is None for r in records.values()):raise ValueError('Incomplete run cannot be exported or scored')
    for task in CAPS:
        panel=bound(package['panels'][task]);predictions=[]
        for source in panel['items']:
            items=[x for x in package['requests'] if x['task']==task and x['id']==source['id']];window_records=[]
            for item in items:
                r=records[item['key']];ref=str(Path('items')/item['key'])
                window_records.append({'window_index':item['window_index'],'raw':r['raw'],'status':'terminal_failure' if r['terminal_failure'] else 'completed','truncated':r['truncated'],'incomplete':r['status']=='incomplete','refusal':r['refusal'],'runtime_error':r['terminal_failure'],'request_identity':item['key'],'raw_request_refs':[ref+'/GENERATION_STARTED.json',ref+'/HTTP_RESPONSE.json',ref+'/RECORD.json']})
            if task=='ner':row=adapter.assemble_final_ner(source['id'],source['tokens'],source['windows'],window_records,package['allowed_ner_types'])
            else:
                if len(items)!=1:raise ValueError('SA request coverage differs')
                row={'id':source['id'],**records[items[0]['key']]['parsed'],'raw_request_refs':window_records[0]['raw_request_refs']}
            row.update(condition=CONDITIONS[task],panel_sha256=package['panels'][task]['sha256']);predictions.append(row)
        if [r['id'] for r in predictions]!=panel['expected_ids']:raise ValueError('Prediction coverage differs')
        write_evidence(root/(task.upper()+'_PREDICTIONS.json'),{'task':task,'condition':CONDITIONS[task],'panel_sha256':package['panels'][task]['sha256'],'freeze_b_sha256':package['freeze_b_sha256'],'rows':predictions})

def execute(freeze_path,freeze_sha):
    f,limit,prior=check_freeze(freeze_path,freeze_sha);package=read_evidence(HERE/'PREPARED_REQUESTS.json')
    if package['freeze_b_sha256']!=freeze_sha or package['policy']!=POLICY or package['sources']!=sources() or package['package_sha256']!=digest({k:v for k,v in package.items() if k!='package_sha256'}):raise ValueError('Prepared API package differs')
    # Recompile deterministically from the same frozen panels, without changing files.
    prepare(freeze_path,freeze_sha)
    root=HERE/'live';root.mkdir(parents=True,exist_ok=True)
    with process_lock(root/'.final_api.lock'):
        write_evidence(root/'BINDING.json',{'freeze_b_sha256':freeze_sha,'package_sha256':package['package_sha256'],'budget':f['api_budget'],'prior_charge_reserve_usd':str(prior),'policy':POLICY})
        if list((root/'items').glob('*/STOPPED.json')):raise RuntimeError('Prior stop requires explicit reconciliation; no automatic retry')
        ledger=Ledger(root,prior,limit);pending=[]
        for item in package['requests']:
            complete=revalidate_completed(item,package,root)
            if complete is None:
                if item['key'] in ledger.charges:raise RuntimeError('Unresolved counted/started attempt; preserve without retry')
                pending.append(item)
        expiry=datetime.datetime.fromisoformat(f['api_budget']['expires_utc'].replace('Z','+00:00'))
        with ThreadPoolExecutor(max_workers=2) as pool:
            for index,r in enumerate(pool.map(lambda x:run_one(x,package,ledger,expiry),pending)):
                if r is not None and (index+1)%25==0:print(json.dumps({'processed_in_this_resume':index+1,'conservative_total_usd':str(ledger.total()),'limit_usd':str(limit)}),flush=True)
        summary={'package_sha256':package['package_sha256'],'total_requests':package['request_count'],'complete_requests':sum((root/'items'/x['key']/'COMPLETE.json').exists() for x in package['requests']),'prior_charge_reserve_usd':str(prior),'conservative_total_committed_usd':str(ledger.total()),'limit_usd':str(limit),'stopped':ledger.stop.is_set(),'no_auto_retry':True}
        write_evidence(root/('STOP_SUMMARY_'+utc().strftime('%Y%m%dT%H%M%SZ')+'.json' if ledger.stop.is_set() else 'COMPLETE.json'),summary)
        if not ledger.stop.is_set():export_predictions(package,root)
        return summary

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['metadata','prepare','execute']);p.add_argument('--freeze');p.add_argument('--freeze-sha256');p.add_argument('--interactive-key',action='store_true');a=p.parse_args()
    if a.action=='metadata':result=metadata()
    elif not a.freeze or not a.freeze_sha256:raise ValueError('Explicit Freeze B path and SHA required')
    elif a.action=='prepare':result=prepare(a.freeze,a.freeze_sha256)
    else:
        check_freeze(a.freeze,a.freeze_sha256)
        if a.interactive_key:
            if not sys.stdin.isatty():raise RuntimeError('Hidden key entry requires your local interactive terminal')
            secret=getpass.getpass('OpenAI API key (hidden; never paste into chat): ')
            if not secret.strip():raise RuntimeError('No key provided')
            previous=os.environ.get('OPENAI_API_KEY');os.environ['OPENAI_API_KEY']=secret
            try:result=execute(a.freeze,a.freeze_sha256)
            finally:
                if previous is None:os.environ.pop('OPENAI_API_KEY',None)
                else:os.environ['OPENAI_API_KEY']=previous
                secret=None
        else:result=execute(a.freeze,a.freeze_sha256)
    print(json.dumps(result,ensure_ascii=True,indent=2))

if __name__=='__main__':main()
