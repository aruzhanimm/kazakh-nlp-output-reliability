"""One descriptive serialization sensitivity; primary raw outputs stay intact.

Only a sole whole-string triple-backtick fence can be unwrapped. Its content
must already be a valid JSON array under the SAME strict JSON hooks. No prose
extraction, bracket fix, record/schema change, or text/mention normalization.
"""
import json,re
from collections import Counter

CONDITIONS=('ner_qwen_ordinary4','ner_sherkala_ordinary4','ner_gemma4_12b_ordinary4','ner_gpt61sol_ordinary4')
POLICY={'scope':'descriptive sensitivity only, all four ordinary prompted NER systems',
    'conditions':list(CONDITIONS),
    'eligibility':'completed non-refusal non-runtime-failed response; after outer Unicode str.strip, sole triple-backtick fence; opening/closing delimiter on separate lines',
    'language_tag':'absent or ASCII json, case-insensitive; only spaces/tabs after opening tag',
    'body':'must be valid JSON array under unchanged duplicate-key/nonfinite-constant rejecting parser; any interior triple-backtick sequence rejects unwrapping',
    'changes':'remove only surrounding fence and outer response whitespace; preserve body bytes and every other generation flag',
    'alignment_and_merge':'unchanged literal whole-token alignment, ontology and all-window ownership merge',
    'no_repair':['no prose extraction','no bracket repair','no type/mention edits','no schema repair','no fuzzy alignment'],
    'primary_predictions_metrics_and_six_comparisons':'unchanged',
    'inference':'descriptive exact typed P/R/F1 only; no new model calls, CI or confirmatory hypothesis',
    'motivation':'Pretest train-only observation: root reported 3 of 4 Gemma NER technical-pilot responses across zero/four-shot conditions with sole JSON fences. Not a prevalence estimate or final-performance observation.',
    'interpretation':'Sensitivity to this fixed output wrapper under the same parser; not latent language comprehension'}
FENCE=re.compile(r'```(?:json)?[ \t]*\r?\n(?P<body>[\s\S]*?)\r?\n```',re.IGNORECASE|re.ASCII)

def unwrap(raw,parser):
    if not isinstance(raw,str):raise ValueError('Original raw response string required')
    match=FENCE.fullmatch(raw.strip())
    if match is None:return raw,{'sole_fence':False,'eligible_valid_list':False,'unwrapped':False,'reason':'not_sole_supported_fence'}
    body=match.group('body')
    if '```'in body:return raw,{'sole_fence':False,'eligible_valid_list':False,'unwrapped':False,'reason':'interior_or_multiple_fence'}
    try:value=json.loads(body,object_pairs_hook=parser.strict_object,parse_constant=parser.reject_constant)
    except (ValueError,TypeError):return raw,{'sole_fence':True,'eligible_valid_list':False,'unwrapped':False,'reason':'body_not_strict_JSON'}
    if not isinstance(value,list):return raw,{'sole_fence':True,'eligible_valid_list':False,'unwrapped':False,'reason':'body_not_array'}
    return body,{'sole_fence':True,'eligible_valid_list':True,'unwrapped':True,'reason':'sole_valid_array_fence_removed'}

def item(item_id,tokens,windows,records,allowed,strict_prediction,adapter):
    revised=[];notes=[]
    for record in records:
        copy=dict(record)
        if record['status']!='completed' or record['runtime_error'] or record['refusal']:
            note={'sole_fence':False,'eligible_valid_list':False,'unwrapped':False,'reason':'terminal_runtime_or_refusal_preserved'}
        else:copy['raw'],note=unwrap(record['raw'],adapter.parser)
        notes.append({'window_index':record['window_index'],**note});revised.append(copy)
    parsed=adapter.assemble_final_ner(item_id,tokens,windows,revised,allowed)
    strict_spans=set(map(tuple,strict_prediction['spans']));secondary_spans=set(map(tuple,parsed['spans']))
    return {'id':item_id,'spans':parsed['spans'],'failed':parsed['failed'],
        'changed_prediction':strict_spans!=secondary_spans,
        'changed_failure':strict_prediction['failed']!=parsed['failed'],
        'windows':notes,'original_responses_modified':False}

def summarize(rows,gold,strict_metrics,adapter):
    counts=Counter();reasons=Counter();tp=fp=fn=0
    for row in rows:
        g=set(map(tuple,gold[row['id']]));p=set(map(tuple,row['spans']))
        tp+=len(g&p);fp+=len(p-g);fn+=len(g-p)
        counts.update(changed_prediction_items=int(row['changed_prediction']),changed_failure_items=int(row['changed_failure']),
                      any_change_items=int(row['changed_prediction'] or row['changed_failure']),secondary_failed_items=int(row['failed']),
                      items_with_sole_fence=int(any(w['sole_fence'] for w in row['windows'])),
                      items_with_eligible_fence=int(any(w['eligible_valid_list'] for w in row['windows'])),
                      items_unwrapped=int(any(w['unwrapped'] for w in row['windows'])))
        for w in row['windows']:
            counts.update(window_responses=1,sole_fence_windows=int(w['sole_fence']),eligible_valid_list_windows=int(w['eligible_valid_list']),unwrapped_windows=int(w['unwrapped']))
            reasons[w['reason']]+=1
    return {'policy':POLICY,'items':len(rows),'counts':dict(counts),'eligibility_reason_counts':dict(reasons),
        'strict_primary':{k:strict_metrics[k] for k in ['tp','fp','fn','support','precision','recall','f1']},
        'fence_only_secondary':adapter.strict.prf(tp,fp,fn),'secondary_items':rows,
        'original_responses_modified':False,'primary_scores_changed':False}
