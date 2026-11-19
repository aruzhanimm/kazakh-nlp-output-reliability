"""Deterministic presentation of completed saved statistics; no model or new test."""
from pathlib import Path
import argparse, hashlib, json, statistics

SEEDS=(13,42,2026)
LABELS={'qwen':'Qwen','sherkala':'Sherkala','gemma4_12b':'Gemma 4 12B','gpt61sol':'GPT-6.1-Sol'}
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()


def table(caption,label,cols,header,rows,wide=False):
    env='table*' if wide else 'table'
    return '\n'.join([r'\begin{'+env+'}[t]',r'\centering\footnotesize',r'\caption{'+caption+'}',
        r'\label{'+label+'}',r'\begin{tabular}{'+cols+'}',r'\toprule',header+r'\\',r'\midrule',
        *[x+r'\\' for x in rows],r'\bottomrule',r'\end{tabular}',r'\end{'+env+'}'])+'\n'


def build(result,out):
    if result['status']!='COMPLETE_RAW_RECONSTRUCTED_FIXED_ANALYSIS' or len(result['conditions'])!=22 or not result['provenance_binding_verified']:
        raise ValueError('Actual complete fixed analysis required')
    c=result['conditions'];out=Path(out)
    if out.exists():raise FileExistsError('Fresh table output directory required')
    out.mkdir(parents=True)
    rows=[]
    for model in LABELS:
        d=c[f'ner_{model}_ordinary4']['ner_fence_only_sensitivity'];s=d['fence_only_secondary']
        rows.append(f'{LABELS[model]} & {100*d["strict_primary"]["f1"]:.2f} & {100*s["precision"]:.2f} & {100*s["recall"]:.2f} & {100*s["f1"]:.2f} & {d["counts"]["items_unwrapped"]}')
    (out/'fence_results.tex').write_text(table(
        'Prespecified NER wrapper sensitivity on the same 500 saved ordinary responses per model. Strict F1 is the primary result; the secondary branch removes only one whole-answer Markdown JSON fence. No tokens, entities, type labels or alignment rules are otherwise changed. All scores are percentages.',
        'tab:fence','lrrrrr',r'System & Strict F1 & \multicolumn{3}{c}{Fence-only secondary} & Unwrapped\\\cmidrule(lr){3-5} & & P & R & F1 & /500',rows,True),encoding='utf-8')
    rows=[]
    for model,mode in [('qwen','ordinary4'),('qwen','schema4'),('sherkala','ordinary4'),('sherkala','schema4'),('gemma4_12b','ordinary4'),('gpt61sol','ordinary4')]:
        v=c[f'ner_{model}_{mode}'];d=v['ner_boundary_type_diagnostics'];t=d['conditional_type_assignment']
        rows.append(f'{LABELS[model]} {"schema" if mode=="schema4" else "ordinary"} & {100*v["metrics"]["f1"]:.2f} & {100*d["untyped_exact_boundary"]["f1"]:.2f} & {t["correct_type"]}/{t["eligible_boundaries"]}')
    (out/'boundary_results.tex').write_text(table(
        'Descriptive diagnostics of primary accepted NER spans. Boundary F1 ignores type but still requires exact spans. The last column counts correct types only among exact boundary matches with a unique predicted type; it is not overall tagging accuracy.',
        'tab:boundary','lrrr',r'System & Typed F1 & Boundary F1 & Type match',rows),encoding='utf-8')
    rows=[]
    for task in ('ner','sa'):
        for model,name in [('xlmr','XLM-R'),('mbert','mBERT')]:
            fs=[c[f'{task}_{model}_seed{s}']['metrics']['f1']*100 for s in SEEDS]
            rows.append(f'{task.upper()} & {name} & '+ ' & '.join(f'{x:.2f}' for x in fs)+f' & {statistics.mean(fs):.2f} & {statistics.stdev(fs):.2f}')
    (out/'final_seed_results.tex').write_text(table(
        'Final encoder F1 by seed (NER micro; SA fixed-binary macro). Mean and sample SD summarize the three separately trained systems; predictions are not ensembled.',
        'tab:seeds','llrrrrr','Task & Model & 13 & 42 & 2026 & Mean & SD',rows,True),encoding='utf-8')
    rows=[]
    for model,name in [('xlmr','XLM-R'),('mbert','mBERT'),*LABELS.items()]:
        ids=[f'sa_{model}_seed{s}' for s in SEEDS] if model in ('xlmr','mbert') else [f'sa_{model}_ordinary4']
        f=[statistics.mean(c[x]['per_type_or_class'][j]['f1'] for x in ids)*100 for j in (0,1)]
        inv={c[x]['metrics']['invalid'] for x in ids}
        if len(inv)!=1:raise ValueError('Do not present averaged invalid counts as counts')
        rows.append(f'{name} & {f[0]:.2f} & {f[1]:.2f} & {next(iter(inv))}')
    (out/'sa_class_results.tex').write_text(table(
        'SA class F1 (percent) and invalid predictions on 500 reviews, with 94 negative and 406 positive gold labels. Encoder class F1 values average three seeds; every encoder seed has zero invalid outputs.',
        'tab:sa-classes','lrrr',r'System & Negative F1 & Positive F1 & INVALID',rows),encoding='utf-8')
    rows=[]
    for model,name in [('xlmr','XLM-R'),('mbert','mBERT'),*LABELS.items()]:
        ids=[(f'sa_{model}_seed{s}',f'{name} seed {s}') for s in SEEDS] if model in ('xlmr','mbert') else [(f'sa_{model}_ordinary4',name)]
        for ident,label in ids:
            cf=c[ident]['metrics']['confusion'];rows.append(label+' & '+' & '.join(str(x) for row in cf for x in row))
    (out/'sa_confusions.tex').write_text(table(
        'SA confusion counts. For each gold class, predicted columns are negative (N), positive (P), and invalid (I). Rows retain all 500 examples and all three encoder seeds.',
        'tab:sa-confusions','lrrrrrr',r'System & \multicolumn{3}{c}{Gold negative (94)} & \multicolumn{3}{c}{Gold positive (406)}\\\cmidrule(lr){2-4}\cmidrule(lr){5-7} & N & P & I & N & P & I',rows,True),encoding='utf-8')
    rows=[]
    for v in result['measured_cost_summaries']:
        if v['kind']=='encoder':continue
        ident=v['condition'];name=next(LABELS[x] for x in LABELS if f'_{x}_' in ident)
        mode='schema' if 'schema' in ident else 'ordinary'
        mem=v.get('peak_gpu_allocated_bytes_maximum');memtext='--' if mem is None else f'{mem/2**30:.2f}'
        toks=v.get('generated_tokens_total',v.get('generated_tokens_including_reasoning_total'))
        rows.append(f'{v["task"].upper()} & {name} {mode} & {v["median_request_or_item_seconds"]:.2f} & {v["p95_request_or_item_seconds"]:.2f} & {memtext} & {toks:,}')
    (out/'operational_results.tex').write_text(table(
        'Observed generation timings and allocated memory. Local Qwen/Sherkala: instrumented greedy generation including token traces, native decoding and CPU export; Gemma: synchronized model.generate only; API: HTTP elapsed time including network, two concurrent workers. Thus columns describe runs, not a controlled cross-system speed ranking. API output tokens include reasoning; GPU memory is unavailable for API.',
        'tab:operations','llrrrr','Task & System & Median s & P95 s & Peak GiB & Output tokens',rows,True),encoding='utf-8')
    files={p.name:sha(p) for p in out.iterdir()}
    (out/'TABLE_MANIFEST.json').write_text(json.dumps({'status':'TABLES_FROM_VERIFIED_SAVED_STATISTICS','new_inference_or_bootstrap':False,'source_results_sha256':RESULTS_SHA,'builder_sha256':sha(__file__),'output_sha256':files},indent=2),encoding='utf-8')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--results',required=True);p.add_argument('--results-sha256',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();RESULTS_SHA=a.results_sha256
    if sha(a.results)!=RESULTS_SHA:raise ValueError('RESULTS hash differs')
    build(json.loads(Path(a.results).read_text(encoding='utf-8')),a.output)
