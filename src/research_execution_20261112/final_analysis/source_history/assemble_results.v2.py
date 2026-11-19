"""Fixed final analysis, from verified raw evidence; no I/O at import.

Metadata mode reads source/train contracts only. Run mode requires a real,
externally approved Freeze B before opening any final panel or gold file.
"""
from pathlib import Path
from collections import Counter
from decimal import Decimal
import argparse,csv,datetime,hashlib,io,json,sys,time
import numpy as np
import evidence_inputs as evidence
import offline_kernel as kernel
import ner_boundary_diagnostics as boundary_diagnostics

HERE=Path(__file__).resolve().parent;PROJECT=HERE.parents[1]
SEEDS=(13,42,2026)
ENCODERS={f'{task}_{model}_seed{seed}':{'task':task,'kind':'encoder','model':model,'seed':seed}
          for task in ('ner','sa') for model in ('xlmr','mbert') for seed in SEEDS}
LOCAL={c:{'task':c.split('_')[0],'kind':'local_llm'} for c in (
    'ner_qwen_ordinary4','ner_qwen_schema4','ner_sherkala_ordinary4','ner_sherkala_schema4',
    'sa_qwen_ordinary4','sa_sherkala_ordinary4')}
MODERN={f'{task}_{model}_ordinary4':{'task':task,'kind':kind} for task in ('ner','sa')
        for model,kind in (('gemma4_12b','gemma'),('gpt61sol','api'))}
CONDITIONS={**ENCODERS,**LOCAL,**MODERN}
PRIMARY=[
 ('Qwen_NER_schema4-minus-XLMR_seedmean','ner','ner_qwen_schema4','ner_xlmr_seedmean'),
 ('Sherkala_NER_schema4-minus-XLMR_seedmean','ner','ner_sherkala_schema4','ner_xlmr_seedmean'),
 ('Qwen_SA_ordinary4-minus-XLMR_seedmean','sa','sa_qwen_ordinary4','sa_xlmr_seedmean'),
 ('Sherkala_SA_ordinary4-minus-XLMR_seedmean','sa','sa_sherkala_ordinary4','sa_xlmr_seedmean'),
 ('Qwen_NER_schema4-minus-ordinary4','ner','ner_qwen_schema4','ner_qwen_ordinary4'),
 ('Sherkala_NER_schema4-minus-ordinary4','ner','ner_sherkala_schema4','ner_sherkala_ordinary4')]
DESCRIPTIVE=[(f'{model}_{task.upper()}_ordinary4-minus-XLMR_seedmean',task,
              f'{task}_{model}_ordinary4',f'{task}_xlmr_seedmean')
             for task in ('ner','sa') for model in ('gemma4_12b','gpt61sol')]
POLICY={'format':'kazakh_final_analysis_v1','items_per_task':500,'conditions':CONDITIONS,
        'primary_comparisons':[list(x) for x in PRIMARY],
        'descriptive_comparisons':[list(x) for x in DESCRIPTIVE],
        'replicates':50000,'random_seed':20261111,'batch_size':500,'primary_family_K':6,
        'primary_quantiles':[.05/12,1-.05/12],
        'pointwise_quantiles':[.025,.975],
        'cluster':'Unicode NFC + collapsed whitespace of COMPLETE original text',
        'seed_summary':'mean of three separate metrics; sample SD; no pooled counts or ensemble',
        'NER_metric':'exact typed-span micro-F1; rejected candidates add no extra primary FP',
        'SA_metric':'fixed negative/positive macro-F1, 2x3 confusion with INVALID column',
        'raw_reconstruction_required':True,'missing_conditions_or_records':'stop; never silently omit',
        'cost_comparison':'descriptive measured scopes only; no clean cross-backend ranking',
        'human_validation':'separate actual judgments; never inferred from automatic analysis',
        'ner_boundary_type_descriptive_diagnostics':boundary_diagnostics.POLICY}

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def sources():
    paths=[HERE/'assemble_results.py',HERE/'evidence_inputs.py',HERE/'offline_kernel.py',HERE/'ner_boundary_diagnostics.py',
           PROJECT/'research_execution_20261112/final_evaluation/project_freeze_b.py',
           PROJECT/'research_execution_20261112/gemma/INPUTS_8.json',
           PROJECT/'research_execution_20261112/api/pilot8/REQUESTS.json']
    return {**evidence.encoder.sources(),**evidence.local.sources(),**evidence.api.sources(),
            **{p.relative_to(PROJECT).as_posix():sha(p) for p in paths}}

def metadata():
    return {'status':'DRAFT_METADATA_NO_TEST_ACCESS','analysis_policy':POLICY,
            'analysis_sources':sources(),'bind_in_freeze_A_and_B':True,
            'test_inputs_read':False,'model_or_API_calls':0}

def validate_gate(reader,binding):
    if binding.get('format')!='kazakh_final_analysis_locator_v1' or binding.get('status')!='ready_for_fixed_analysis':raise PermissionError('Concrete analysis evidence locator required')
    master=reader.bound(binding['master_freeze'])
    if master.get('scope')!='final' or master.get('status')!='frozen_for_final_inference' or master.get('final_test_access_authorized')is not True or not master.get('decision_reference') or 'host_projection' in master:raise PermissionError('Actual scientific master Freeze B required before final data access')
    evidence.same(master.get('analysis_policy'),POLICY,'Frozen analysis settings differ')
    evidence.same(master.get('analysis_sources'),sources(),'Frozen analysis sources differ')
    if set(binding['conditions'])!=set(CONDITIONS):raise ValueError('Exactly the22 prespecified conditions required')
    protocol=reader.bound(binding['scientific_protocol'])
    if binding['scientific_protocol']['sha256']!=master['scientific_protocol']['sha256']:raise ValueError('Scientific protocol binding differs')
    stats=protocol['statistics']
    if stats['bootstrap_replicates']!=50000 or stats['seed']!=20261111 or stats['primary_family_K']!=6 or stats['primary_comparisons']!=[x[0] for x in PRIMARY] or stats['modern_reference_in_primary_family']is not False:raise ValueError('Scientific statistics contract differs')
    evidence.same(stats['bonferroni_quantiles'],POLICY['primary_quantiles'],'Primary quantiles differ')
    if protocol['selection']['items_per_task']!=500 or protocol['models']['encoder_seeds']!=list(SEEDS):raise ValueError('Scientific sample/seed scope differs')
    return master,protocol

def panels_and_gold(reader,binding,master,allowed):
    panels={};gold={}
    for task in ('ner','sa'):
        ps=binding['panels'][task];gs=binding['gold'][task]
        if ps['sha256']!=master['panels'][task]['sha256'] or gs['sha256']!=master['gold'][task]['sha256']:raise ValueError('Final panel/gold differs from Freeze B')
        p=reader.bound(ps);g=reader.bound(gs);ids=p['expected_ids']
        if p['format']!='encoder_common_view_panel_v1' or p['split']!='test' or p['task']!=task or p['freeze_a_sha256']!=master['freeze_a_sha256']:raise ValueError('Final panel provenance differs')
        if len(ids)!=500 or len(set(ids))!=500 or [x['id'] for x in p['items']]!=ids or any(not x.startswith(task+':test:') for x in ids) or evidence.digest(ids)!=master['panels'][task]['ordered_ids_sha256']:raise ValueError('Exactly500 ordered common test IDs required')
        if g['task']!=task or g['split']!='test' or g['panel_sha256']!=ps['sha256'] or [x['id'] for x in g['rows']]!=ids:raise ValueError('Original gold coverage/provenance differs')
        gold[task]={}
        for item,row in zip(p['items'],g['rows']):
            if hashlib.sha256(kernel.normalized_text(item['original_text']).encode()).hexdigest()!=item['cluster_key']:raise ValueError('Complete original-text cluster key differs')
            if task=='ner':
                evidence.same(row['tokens'],item['tokens'],'Original NER gold tokens differ')
                if item['original_text']!=' '.join(item['tokens']) or len(row['labels'])!=len(item['tokens']):raise ValueError('NER original text/gold length differs')
                spans,repairs=evidence.adapter.strict.bio_to_spans(row['labels'],repair_prediction=False)
                if repairs or any(t not in allowed for a,b,t in spans):raise ValueError('Strict original25 gold required')
                gold[task][item['id']]=spans
            else:
                if type(row['label'])is not str or row['label']not in ('0','1'):raise ValueError('Original binary SA gold required')
                gold[task][item['id']]=int(row['label'])
        panels[task]=p
    return panels,gold

def prf(tp,fp,fn):return evidence.adapter.strict.prf(int(tp),int(fp),int(fn))

def score_rows(task,rows,gold,allowed):
    """Pure descriptive scores; rejects unknown labels/spans before counting."""
    counts=Counter();pertype=[];error_reasons=Counter();failures=Counter()
    for row in rows:
        for flag in ('failed','truncated','refusal','terminal_failed','parser_failed','incomplete','partial_runtime_failure','merge_alignment_failure'):
            failures[flag]+=int(bool(row.get(flag,False)))
        failures['invalid_candidates']+=row.get('invalid_candidates',0)
    if task=='ner':
        bytype={t:Counter() for t in allowed}
        for r in rows:
            g=set(map(tuple,gold[r['id']]));p=set(map(tuple,r['spans']))
            if len(p)!=len(r['spans']) or any(t not in allowed for _,_,t in p):raise ValueError('Duplicate/unknown-type predictions')
            counts.update(tp=len(g&p),fp=len(p-g),fn=len(g-p),exact=int(g==p),failure_aware_exact=int(g==p and not r['failed']))
            if not g:counts.update(gold_empty_items=1,gold_empty_items_with_fp=int(bool(p)),gold_empty_fp=len(p))
            for t in allowed:
                gt={x for x in g if x[2]==t};pt={x for x in p if x[2]==t}
                bytype[t].update(tp=len(gt&pt),fp=len(pt-gt),fn=len(gt-pt))
            all_diagnostics=[d for w in r.get('window_results',[]) if w.get('parsed') is not None for d in w['parsed']['diagnostics']]
            syntax=any(d.get('reason')=='json_syntax_or_root' for d in all_diagnostics)
            contract=any(d.get('category')=='format' and d.get('reason')!='json_syntax_or_root' for d in all_diagnostics)
            alignment=any(d.get('category')=='alignment' for d in all_diagnostics) or bool(r.get('merge_alignment_failure'))
            failures.update(syntax_or_root_items=int(syntax),contract_items=int(contract),alignment_items=int(alignment),partial_valid_items=int(r['failed'] and bool(p)))
            error_reasons.update(d.get('reason','unknown') for d in all_diagnostics if d.get('category')not in ('aligned','duplicate'))
        result={**prf(counts['tp'],counts['fp'],counts['fn']),'items':len(rows),'exact_set_accuracy':counts['exact']/len(rows),
                'failure_aware_exact_set_accuracy':counts['failure_aware_exact']/len(rows),
                **{k:counts[k] for k in ['gold_empty_items','gold_empty_items_with_fp','gold_empty_fp']}}
        pertype=[{'type':t,**prf(c['tp'],c['fp'],c['fn'])} for t,c in bytype.items()]
    else:
        matrix=np.zeros((2,3),dtype=np.int64)
        for r in rows:
            label=r['label']
            if not (label is None or type(label)is int and label in (0,1)):raise ValueError('Invalid SA prediction domain')
            matrix[gold[r['id']],2 if label is None else label]+=1
            error_reasons.update(map(str,r.get('errors',[])))
        pertype=[]
        for c,name in enumerate(('negative','positive')):
            tp=matrix[c,c];fp=matrix[1-c,c];fn=matrix[c].sum()-tp
            pertype.append({'type':name,**prf(tp,fp,fn)})
        result={'items':len(rows),'f1':float(kernel.f1_from_counts('sa',matrix)),
                'macro_precision':float(np.mean([x['precision'] for x in pertype])),
                'macro_recall':float(np.mean([x['recall'] for x in pertype])),
                'accuracy':float(np.trace(matrix[:,:2])/len(rows)),
                'invalid':int(matrix[:,2].sum()),'confusion':matrix.tolist(),
                'confusion_columns':['negative','positive','INVALID'],'confusion_rows':['negative','positive']}
    output={'metrics':result,'per_type_or_class':pertype,'failures':dict(failures),'parser_reason_counts':dict(error_reasons)}
    if task=='ner':output['ner_boundary_type_diagnostics']=boundary_diagnostics.analyze(rows,gold,allowed)
    return output

def statistical_tables(task,ids,clusters,aligned,*,replicates=50000):
    counts=kernel.align_prediction_rows(task,ids,aligned)
    seed_groups={f'{task}_{m}_seedmean':{s:f'{task}_{m}_seed{s}' for s in SEEDS} for m in ('xlmr','mbert')}
    planned=[x for x in PRIMARY+DESCRIPTIVE if x[1]==task]
    result=kernel.paired_analysis(task,ids,clusters,counts,seed_groups=seed_groups,
        contrasts=[(x[2],x[3]) for x in planned],replicates=replicates,random_seed=20261111,batch_size=500)
    for c,definition in zip(result['paired_differences'],planned):
        c['comparison_id']=definition[0];c['family']='primary_K6' if definition in PRIMARY else 'descriptive_only'
        if c['family']=='descriptive_only':c.pop('ci_bonferroni_K6')
    return result

def cost_summary(condition,rows):
    definition=CONDITIONS[condition];measurements=[]
    for r in rows:
        if definition['kind']=='encoder':
            measurements.append({'input_tokens':r['input_token_count'],'seconds':r['seconds'],
                'peak_gpu_allocated_bytes':r['peak_gpu_allocated_bytes'],'peak_gpu_reserved_bytes':r['peak_gpu_reserved_bytes'],
                'scope':r['cost_scope']})
        else:measurements.extend(r.get('measurements',[]))
    result={'condition':condition,'task':definition['task'],'kind':definition['kind'],
            'items':len(rows),'measurement_records':len(measurements),
            'scope':' | '.join(dict.fromkeys(m['scope'] for m in measurements)),
            'comparison_limit':'descriptive instrumentation; different hosts/backends/scopes; no clean speed ranking'}
    for k in ('input_tokens','generated_tokens','generated_tokens_including_reasoning'):
        if measurements and all(k in x for x in measurements):result[k+'_total']=sum(x[k] for x in measurements)
    if measurements and all('seconds'in x for x in measurements):
        seconds=[x['seconds'] for x in measurements]
        if any(not np.isfinite(x) or x<0 for x in seconds):raise ValueError('Invalid measured duration')
        result.update(sum_request_or_item_seconds=float(sum(seconds)),median_request_or_item_seconds=float(np.median(seconds)),p95_request_or_item_seconds=float(np.quantile(seconds,.95)))
    for k in ('peak_gpu_allocated_bytes','peak_gpu_reserved_bytes'):
        if measurements and all(k in x for x in measurements):result[k+'_maximum']=max(x[k] for x in measurements)
    for k in ('conservative_charge_usd','ordinary_uncached_rate_estimate_usd'):
        if measurements and all(k in x for x in measurements):result[k+'_total']=str(sum((Decimal(x[k]) for x in measurements),Decimal(0)))
    return result

def write_new(path,data):
    path=Path(path);raw=data if isinstance(data,bytes) else evidence.canonical(data)+b'\n'
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('xb') as stream:stream.write(raw)

def csv_bytes(rows):
    buf=io.StringIO(newline='');fields=list(dict.fromkeys(k for row in rows for k in row))
    writer=csv.DictWriter(buf,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    return buf.getvalue().encode('utf-8')

def execute(locator_path,locator_sha,output):
    reader=evidence.Reader();binding=reader.read(locator_path,locator_sha);master,protocol=validate_gate(reader,binding)
    allowed=reader.read(PROJECT/'data/derived/ner_labels.json')['types'];panels,gold=panels_and_gold(reader,binding,master,allowed)
    rows_by={};reports={};reconstruction={};start=time.perf_counter()
    funcs={'encoder':evidence.encoder_rows,'local_llm':evidence.local_rows,'api':evidence.api_rows,'gemma':evidence.gemma_rows}
    for condition,definition in CONDITIONS.items():
        spec=binding['conditions'][condition];task=definition['task']
        if spec['kind']!=definition['kind'] or spec['task']!=task or spec['panel_sha256']!=binding['panels'][task]['sha256']:raise ValueError('Condition locator differs from frozen scope')
        f=evidence.host_freeze(reader,spec['execution_freeze'],master,binding['master_freeze']['sha256'])
        args=[reader,condition,spec,panels[task],f]
        if definition['kind']!='encoder':args.append(allowed)
        rows,receipt=funcs[definition['kind']](*args)
        if [r['id'] for r in rows]!=panels[task]['expected_ids']:raise ValueError('Reconstructed coverage differs')
        if task=='ner':
            for r,item in zip(rows,panels[task]['items']):
                if any(not 0<=a<b<=len(item['tokens']) for a,b,t in r['spans']):raise ValueError('Reconstructed spans outside original text')
        rows_by[condition]=rows;reports[condition]=score_rows(task,rows,gold[task],allowed);reconstruction[condition]=receipt
    stats={}
    for task in ('ner','sa'):
        aligned={}
        for condition,rows in rows_by.items():
            if CONDITIONS[condition]['task']!=task:continue
            aligned[condition]=[{'id':r['id'],**({'gold_spans':gold[task][r['id']],'pred_spans':r['spans']} if task=='ner' else {'gold_label':gold[task][r['id']],'pred_label':r['label']})} for r in rows]
        stats[task]=statistical_tables(task,panels[task]['expected_ids'],[x['cluster_key'] for x in panels[task]['items']],aligned)
        for c in aligned:
            if not np.isclose(stats[task]['conditions'][c]['f1'],reports[c]['metrics']['f1'],atol=1e-14):raise ValueError('Independent descriptive/statistical score mismatch')
    primary=[c for s in stats.values() for c in s['paired_differences'] if c['family']=='primary_K6']
    if {x['comparison_id'] for x in primary}!={x[0] for x in PRIMARY}:raise ValueError('Primary family incomplete')
    table=[];pertype=[];failures=[];seedrows=[];costs=[];untyped=[];type_confusion=[]
    for c,report in reports.items():
        task=CONDITIONS[c]['task'];interval=stats[task]['conditions'][c]['ci95_pointwise']
        table.append({'task':task,'condition':c,'summary':'individual_fixed_system',**{k:v for k,v in report['metrics'].items() if not isinstance(v,list)},'ci95_low':interval[0],'ci95_high':interval[1]})
        pertype.extend({'task':task,'condition':c,**r} for r in report['per_type_or_class'])
        failures.append({'task':task,'condition':c,**report['failures']})
        costs.append(cost_summary(c,rows_by[c]))
        if task=='ner':
            d=report['ner_boundary_type_diagnostics']
            untyped.append({'condition':c,**d['untyped_exact_boundary'],**d['counts'],
                **{'conditional_type_'+k:v for k,v in d['conditional_type_assignment'].items()}})
            type_confusion.extend({'condition':c,**cell} for cell in d['type_confusion']['cells'])
    for task in ('ner','sa'):
        for model in ('xlmr','mbert'):
            name=f'{task}_{model}_seedmean';members=[f'{task}_{model}_seed{s}' for s in SEEDS]
            values=[reports[c]['metrics'] for c in members];summary=stats[task]['conditions'][name];interval=summary['ci95_pointwise']
            mean={k:float(np.mean([v[k] for v in values])) for k in (['precision','recall','f1'] if task=='ner' else ['macro_precision','macro_recall','f1','accuracy'])}
            sd={k+'_seed_sample_sd':float(np.std([v[k] for v in values],ddof=1)) for k in mean}
            item={'task':task,'condition':name,'summary':'mean_of_three_individual_seed_metrics','items':500,**mean,**sd,'ci95_low':interval[0],'ci95_high':interval[1]}
            table.append(item);seedrows.append(item)
    result={'status':'COMPLETE_RAW_RECONSTRUCTED_FIXED_ANALYSIS','publication_ready':False,
            'publication_ready_reason':'Human review, manuscript claim audit and final presentation remain separate requirements.',
            'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'locator_sha256':locator_sha,
            'master_freeze_sha256':binding['master_freeze']['sha256'],'scientific_protocol_sha256':binding['scientific_protocol']['sha256'],
            'policy':POLICY,'source_hashes':sources(),'conditions':reports,'statistics':stats,
            'reconstruction':reconstruction,'measured_cost_summaries':costs,'analysis_seconds':time.perf_counter()-start,
            'provenance_binding_verified':True,'protocol_freeze_verified':True,
            'human_review_performed_by_this_code':False,'original_gold_modified':False,
            'limitations':['Bootstrap uncertainty conditions on fixed trained systems/prompts; seed SD is separate.',
                          'API internal token IDs unavailable; exact request and raw HTTP are verified.',
                          'Saved Gemma native token/text mapping requires its separately captured CPU replay receipt for independent native-parser verification.',
                          'Automatic parser categories are observed symptoms, not human-validated causal explanations.']}
    out=Path(output)
    if out.exists():raise FileExistsError('Choose a fresh analysis output directory; do not replace prior results')
    write_new(out/'RESULTS.json',result)
    write_new(out/'TABLE_MAIN.csv',csv_bytes(table));write_new(out/'TABLE_PER_TYPE_CLASS.csv',csv_bytes(pertype))
    write_new(out/'TABLE_FAILURES.csv',csv_bytes(failures));write_new(out/'TABLE_ENCODER_SEEDS.csv',csv_bytes(seedrows))
    write_new(out/'TABLE_COSTS.csv',csv_bytes(costs))
    write_new(out/'TABLE_NER_UNTYPED.csv',csv_bytes(untyped));write_new(out/'TABLE_NER_TYPE_CONFUSION.csv',csv_bytes(type_confusion))
    paired=[]
    for task,s in stats.items():
        for c in s['paired_differences']:
            row={'task':task,**{k:v for k,v in c.items() if not isinstance(v,list)},'ci95_low':c['ci95_pointwise'][0],'ci95_high':c['ci95_pointwise'][1]}
            if 'ci_bonferroni_K6'in c:row.update(ci_K6_low=c['ci_bonferroni_K6'][0],ci_K6_high=c['ci_bonferroni_K6'][1])
            paired.append(row)
    write_new(out/'TABLE_PAIRED.csv',csv_bytes(paired))
    for task in ('ner','sa'):
        data={'task':task,'panel_sha256':binding['panels'][task]['sha256'],'expected_ids':panels[task]['expected_ids'],
              'conditions':{c:r for c,r in rows_by.items() if CONDITIONS[c]['task']==task}}
        write_new(out/(task.upper()+'_VERIFIED_PREDICTIONS.json'),data)
    write_new(out/'INPUT_FILE_CLOSURE.json',{'files':reader.files,'note':'Exact bytes accessed for evidence validation, plus separate source hash closure.'})
    outputs={p.name:sha(p) for p in out.iterdir() if p.is_file()}
    write_new(out/'COMPLETE.json',{'status':result['status'],'conditions':22,'items_per_task':500,'output_sha256':outputs,'analysis_seconds':result['analysis_seconds']})
    return {'status':result['status'],'output':str(out),'conditions':22,'analysis_seconds':result['analysis_seconds']}

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);s=p.add_subparsers(dest='command',required=True)
    m=s.add_parser('metadata');m.add_argument('--output',type=Path,required=True)
    r=s.add_parser('run');r.add_argument('--locator',type=Path,required=True);r.add_argument('--locator-sha256',required=True);r.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.command=='metadata':write_new(a.output,metadata());print(json.dumps({'status':'METADATA_ONLY_NO_TEST_ACCESS','path':str(a.output),'sha256':sha(a.output)}))
    else:print(json.dumps(execute(a.locator,a.locator_sha256,a.output)))
