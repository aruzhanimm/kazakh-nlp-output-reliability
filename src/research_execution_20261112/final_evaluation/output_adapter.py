"""Gold-free final output normalization; callers validate raw request provenance.

API providers do not expose native input token IDs. This adapter deliberately
does not invent them. It reuses the same literal parser and whole-span merger
as local inference, while each caller retains its own source-bound ledger.
"""
from pathlib import Path
import hashlib,importlib.util,json

PROJECT=Path(__file__).resolve().parents[2]
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
parser=load('final_literal_ner',PROJECT/'interface_pilot/alignment.py')
strict=load('final_strict_ner',PROJECT/'ner_evidence_20261111/analysis/strict_ner.py')
sa=load('final_sa_parser',PROJECT/'pilot/scoring.py')

def sources():
    paths=[Path(__file__).resolve(),PROJECT/'interface_pilot/alignment.py',PROJECT/'ner_evidence_20261111/analysis/strict_ner.py',PROJECT/'pilot/scoring.py']
    return {p.relative_to(PROJECT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}

def assemble_final_ner(item_id,tokens,windows,records,allowed_types):
    """Records: window_index, raw, status completed/terminal_failure, booleans
    truncated/refusal/runtime_error. Exactly one terminal record per window.
    A completed truncated response keeps valid entities; runtime failures do not.
    """
    if not allowed_types or len(set(allowed_types))!=len(allowed_types):raise ValueError('Invalid ontology')
    parser.text_offsets(tokens)
    if len(records)!=len(windows) or not records:raise ValueError('Missing window records')
    by={r['window_index']:r for r in records}
    if len(by)!=len(records) or set(by)!=set(range(len(windows))):raise ValueError('Duplicate/missing window identity')
    local=[];details=[]
    for wi,w in enumerate(windows):
        if w['tokens']!=tokens[w['start']:w['end']]:raise ValueError('Window differs from original tokens')
        r=by[wi]
        if r['status'] not in ('completed','terminal_failure'):raise ValueError('Unfinished window cannot be scored')
        if any(type(r[k]) is not bool for k in ['truncated','refusal','runtime_error']):raise ValueError('Explicit failure booleans required')
        terminal=r['status']=='terminal_failure'
        if r['runtime_error'] and not terminal:raise ValueError('Runtime failure must be terminal')
        if not isinstance(r['raw'],str):raise ValueError('Raw text required even on failure')
        parsed=None if terminal else parser.parse_mentions(r['raw'],w['tokens'],allowed_types)
        local.append(None if terminal else ([] if r['refusal'] else parsed['spans']))
        if type(r.get('incomplete',False)) is not bool:raise ValueError('Explicit optional incomplete flag required')
        details.append({'window_index':wi,'status':r['status'],'truncated':r['truncated'],'incomplete':r.get('incomplete',False),'refusal':r['refusal'],
                        'raw_sha256':hashlib.sha256(r['raw'].encode()).hexdigest(),'parsed':parsed,
                        'request_identity':r.get('request_identity'),'raw_request_refs':r.get('raw_request_refs',[])})
    merged=strict.merge_window_spans([(w['start'],w['end']) for w in windows],local,len(tokens))
    conflicts=[r for r in merged['rejected'] if r['reason']=='overlap_first_valid_wins']
    parser_failed=any(d['parsed'] is not None and d['parsed']['failed'] for d in details)
    truncated=any(d['truncated'] for d in details);refusal=any(d['refusal'] for d in details);incomplete=any(d['incomplete'] for d in details)
    provenance=[]
    for span in merged['spans']:
        found=[]
        owner=max((min(span[0]-w['start'],w['end']-span[1]),-wi,wi) for wi,w in enumerate(windows) if w['start']<=span[0] and span[1]<=w['end'])[2]
        for ci,candidate in enumerate(local[owner] or []):
            if (candidate[0]+windows[owner]['start'],candidate[1]+windows[owner]['start'],candidate[2])==span:
                aligned=[x for x in details[owner]['parsed']['diagnostics'] if x['category']=='aligned']
                found.append({'span':span,'window_index':owner,'candidate_index':ci,'raw_record_index':aligned[ci]['record_index'],'request_identity':details[owner]['request_identity']})
        if len(found)!=1:raise ValueError('Accepted span must trace to exactly one owning candidate')
        provenance.extend(found)
    return {'id':item_id,'spans':merged['spans'],'failed':bool(parser_failed or merged['failed_windows'] or truncated or incomplete or refusal or conflicts),
            'parser_failed':parser_failed,'terminal_failed':bool(merged['failed_windows']),'terminal_failed_windows':merged['failed_windows'],
            'partial_runtime_failure':bool(merged['failed_windows']) and len(merged['failed_windows'])<len(windows),
            'truncated':truncated,'incomplete':incomplete,'refusal':refusal,'merge_alignment_failure':bool(conflicts),'merge_rejections':merged['rejected'],
            'invalid_candidates':sum(d['parsed']['invalid_candidates'] for d in details if d['parsed'] is not None),
            'invalid_candidate_penalty_in_primary':False,'window_results':details,'span_provenance':provenance,
            'raw_request_refs':[ref for d in details for ref in d['raw_request_refs']]}

def parse_final_sa(raw,*,truncated,refusal,runtime_error,terminal_failed=False):
    if any(type(x)is not bool for x in [truncated,refusal,runtime_error,terminal_failed]):raise ValueError('Explicit failure booleans required')
    parsed=sa.parse_sa(raw)
    if truncated or refusal or runtime_error or terminal_failed:
        parsed={**parsed,'label':None,'failed':True,'errors':[*parsed['errors'],'runtime_truncation_refusal_or_terminal_invalid']}
    return {**parsed,'truncated':truncated,'refusal':refusal,'terminal_failed':terminal_failed or runtime_error}
