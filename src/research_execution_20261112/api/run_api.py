"""Explicit paid runner, disabled without a user-approved, package-bound budget.

No automatic retries. Ambiguous attempts stop the run; raw HTTP bodies are saved
before parsing. Imports, validate and report modes are offline.
"""
from pathlib import Path
from decimal import Decimal
import argparse
import datetime
import json
import os
import time
import urllib.request
import urllib.error
from api_common import *


def now(): return datetime.datetime.now(datetime.timezone.utc)


def authorization(auth, package, root, stage):
    if auth.get('status')!='user_approved' or not auth.get('approval_reference'):
        raise PermissionError('A real user budget decision is required; templates do not authorize calls')
    if auth.get('package_sha256')!=package['package_sha256'] or Path(auth.get('ledger_root','')).resolve()!=Path(root).resolve():
        raise PermissionError('Approval belongs to another package or ledger')
    if stage not in auth.get('allowed_stages',[]) or stage not in ('smoke','dev'):
        raise PermissionError('Stage not authorized')
    if auth.get('generation_reservation_rates')!={'input_per_million':'2.75','output_per_million':'11'}:
        raise PermissionError('Budget pricing assumptions differ')
    if auth.get('model')!=MODEL or auth.get('allow_automatic_retries') is not False:
        raise PermissionError('Model or retry policy differs')
    expires=datetime.datetime.fromisoformat(auth['expires_utc'].replace('Z','+00:00'))
    if expires.tzinfo is None or now()>=expires:raise PermissionError('Budget approval expired')
    limit=Decimal(str(auth['generation_limit_usd']))
    if not limit.is_finite() or limit<=0:raise PermissionError('Positive finite explicit limit required')
    return limit


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self,*args,**kwargs): raise RuntimeError('API redirect refused')


class OpenAIHTTP:
    def __init__(self):
        # Only this explicitly constructed live transport reads a key. No key is
        # written to files, printed, included in errors, or sent to other hosts.
        self.key=os.environ.get('OPENAI_API_KEY')
        if not self.key:raise RuntimeError('Connect OPENAI_API_KEY locally; never send it in chat')
        self.opener=urllib.request.build_opener(NoRedirect())

    def post(self,path,body):
        if path not in ('responses','responses/input_tokens'):raise ValueError('Unexpected API endpoint')
        request=urllib.request.Request('https://api.openai.com/v1/'+path,data=canonical(body),method='POST',
            headers={'Authorization':'Bearer '+self.key,'Content-Type':'application/json'})
        start=time.monotonic()
        try:response=self.opener.open(request,timeout=900)
        except urllib.error.HTTPError as error:response=error
        with response:
            payload=response.read()
            return {'status_code':response.status,'body_utf8':payload.decode('utf-8'),
                    'headers':{k:v for k,v in response.headers.items() if k.lower() in ('x-request-id','date','openai-processing-ms')},
                    'seconds':time.monotonic()-start}


def records_from_store(package, root):
    root=Path(root);out={}
    for item in package['requests']:
        folder=root/'items'/item['key'];started=folder/'STARTED.json';response=folder/'HTTP_RESPONSE.json';record=folder/'RECORD.json'
        if record.exists():
            if not started.exists() or not response.exists():raise ValueError('Record lost raw provenance')
            s=read_evidence(started);raw=read_evidence(response)
            if s['request_sha256']!=item['request_sha256'] or s['package_sha256']!=package['package_sha256']:
                raise ValueError('Request binding differs')
            rebuilt=parse_response(json.loads(raw['body_utf8']),item,package['allowed_ner_types'])
            if read_evidence(record)!=rebuilt:raise ValueError('Parsed record differs from raw response')
            out[item['key']]=rebuilt
        elif started.exists():
            raise RuntimeError('Unresolved started attempt; preserve it and reconcile, never regenerate automatically')
    return out


def require_smoke(package,root,auth):
    gate=auth.get('dev_gate',{})
    if gate.get('status')!='approved_after_complete_smoke' or not gate.get('decision'):
        raise PermissionError('Dev needs a documented real smoke review, not a score threshold')
    receipt=read_evidence(Path(root)/'SMOKE_COMPLETE.json')
    if gate.get('smoke_receipt_sha256')!=digest(receipt) or receipt.get('package_sha256')!=package['package_sha256']:
        raise PermissionError('Smoke review is not bound to this package')
    rs=records_from_store(package,root)
    smoke=[i for i in package['requests'] if i['stage']=='smoke']
    if len(smoke)!=40 or any(i['key'] not in rs for i in smoke):raise PermissionError('Smoke evidence incomplete')
    if receipt['mode']!='live_openai':raise PermissionError('Synthetic responses cannot pass the live gate')


def execute(package, root, auth, stage, transport_factory=OpenAIHTTP):
    validate_package(package);root=Path(root)
    intended_mode='live_openai' if transport_factory is OpenAIHTTP else 'synthetic_test_transport'
    if auth.get('execution_mode')!=intended_mode:
        raise PermissionError('Synthetic budget fixtures cannot authorize a live transport')
    limit=authorization(auth,package,root,stage)
    root.mkdir(parents=True,exist_ok=True)
    with process_lock(root/'.api.lock'):
        write_evidence(root/'PACKAGE.json',package)
        if (root/'STOPPED.json').exists():
            read_evidence(root/'STOPPED.json');raise RuntimeError('Prior stop requires explicit investigation')
        records=records_from_store(package,root)
        if stage=='dev':require_smoke(package,root,auth)
        items=[i for i in package['requests'] if i['stage']==stage]
        transport=None
        for item in items:
            if item['key'] in records:continue
            authorization(auth,package,root,stage)
            spent=sum((Decimal(read_evidence(p)['reserved_usd']) for p in (root/'items').glob('*/STARTED.json')),Decimal(0)) if (root/'items').exists() else Decimal(0)
            reserve=reserve_usd(item['task'])
            if spent+reserve>limit:raise PermissionError('Client reservation budget would be exceeded; no request sent')
            folder=root/'items'/item['key']
            # Count requests require access/approval too. Their server billing was
            # not established from the public docs; see explicit proposal caveat.
            if (folder/'COUNT_STARTED.json').exists():raise RuntimeError('Ambiguous count preflight; investigate before continuing')
            if transport is None:transport=transport_factory()
            write_evidence(folder/'COUNT_STARTED.json',{'utc':now().isoformat(),'request_sha256':item['request_sha256']})
            count_body={k:item['request'][k] for k in ('model','input','reasoning','text','tools','tool_choice','truncation')}
            count=transport.post('responses/input_tokens',count_body)
            write_evidence(folder/'COUNT_RESPONSE.json',count)
            if count['status_code']!=200:raise RuntimeError('Token-count preflight failed; no generation sent')
            counted=json.loads(count['body_utf8']).get('input_tokens')
            if type(counted)is not int or not 0<counted<=INPUT_CEILING:raise RuntimeError('Native input count exceeds pre-agreed ceiling or is missing')
            write_evidence(root/'approvals'/(digest(auth)+'.json'),auth)
            write_evidence(folder/'STARTED.json',{'utc':now().isoformat(),'request_sha256':item['request_sha256'],
                'package_sha256':package['package_sha256'],'reserved_usd':str(reserve),'native_input_tokens':counted,
                'request':item['request'],'mode':'live_openai' if isinstance(transport,OpenAIHTTP) else 'synthetic_test_transport'})
            try:
                raw=transport.post('responses',item['request'])
                write_evidence(folder/'HTTP_RESPONSE.json',raw)
                if raw['status_code']!=200:raise RuntimeError('HTTP failure preserved; no automatic retry')
                value=json.loads(raw['body_utf8']);record=parse_response(value,item,package['allowed_ner_types'])
                write_evidence(folder/'RECORD.json',record)
                if value.get('model')!=MODEL or value.get('service_tier')!='default':
                    raise RuntimeError('Returned model/tier differs; do not pool this condition')
                if (value.get('reasoning') or {}).get('effort')!='low' or value.get('max_output_tokens')!=CAPS[item['task']] or \
                        ((value.get('text') or {}).get('format') or {}).get('type')!='text':
                    raise RuntimeError('Returned reasoning/output configuration differs or is unverified')
                usage=value.get('usage') or {}
                if type(usage.get('input_tokens')) is not int or type(usage.get('output_tokens')) is not int or \
                        not 0<=usage['input_tokens']<=INPUT_CEILING or not 0<=usage['output_tokens']<=CAPS[item['task']]:
                    raise RuntimeError('Usage missing or outside declared ceiling')
                identity={'model':value['model'],'system_fingerprint':value.get('system_fingerprint')}
                if (root/'RETURNED_IDENTITY.json').exists():
                    if read_evidence(root/'RETURNED_IDENTITY.json')!=identity:raise RuntimeError('Observable API version drift')
                else:write_evidence(root/'RETURNED_IDENTITY.json',identity)
                if value['status'] in ('failed','cancelled'):raise RuntimeError('Terminal runtime outcome preserved; stop before next request')
                records[item['key']]=record
            except Exception as error:
                write_evidence(root/'STOPPED.json',{'utc':now().isoformat(),'key':item['key'],
                    'exception_type':type(error).__name__,'automatic_retry':False,'reservation_retained':str(reserve)})
                raise
        mode='live_openai' if all(read_evidence(root/'items'/i['key']/'STARTED.json')['mode']=='live_openai' for i in items) else 'synthetic_test_transport'
        complete={'package_sha256':package['package_sha256'],'stage':stage,'items':len(items),
                  'record_hashes':{i['key']:digest(records[i['key']]) for i in items},'mode':mode}
        write_evidence(root/(stage.upper()+'_COMPLETE.json'),complete)
        return complete


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='cmd',required=True)
    for name in ('validate','execute','report'):
        p=sub.add_parser(name);p.add_argument('--package',required=True)
        if name!='validate':p.add_argument('--store',required=True)
        if name=='execute':
            p.add_argument('--approval',required=True);p.add_argument('--stage',choices=['smoke','dev'],required=True)
            p.add_argument('--execute-paid-api',action='store_true',required=True)
        if name=='report':p.add_argument('--gold',required=True);p.add_argument('--output',required=True)
    args=parser.parse_args();package=read_evidence(args.package)
    if args.cmd=='validate':result=validate_package(package)
    elif args.cmd=='execute':result=execute(package,args.store,json.loads(Path(args.approval).read_text(encoding='utf-8')),args.stage)
    else:
        validate_package(package)
        if (Path(args.store)/'STOPPED.json').exists():raise RuntimeError('Resolve execution/provenance failure before reporting')
        result=score_records(package,records_from_store(package,args.store),read_evidence(args.gold))
        write_evidence(args.output,result)
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
