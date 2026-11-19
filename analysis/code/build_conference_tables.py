"""Conference presentation tables from unchanged saved statistics; no new scoring."""
from pathlib import Path
import argparse,hashlib,json,statistics
SEEDS=(13,42,2026)
ORDER=['xlmr_seedmean','mbert_seedmean','qwen_ordinary4','qwen_schema4','sherkala_ordinary4','sherkala_schema4','gemma4_12b_ordinary4','gpt61sol_ordinary4']
LABELS=['XLM-R (3-seed mean)','mBERT (3-seed mean)','Qwen ordinary','Qwen schema','Sherkala ordinary','Sherkala schema','Gemma strict','GPT-6.1-Sol']
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def build(r,out):
    assert r['status']=='COMPLETE_RAW_RECONSTRUCTED_FIXED_ANALYSIS' and len(r['conditions'])==22
    out=Path(out);out.mkdir(parents=True,exist_ok=True);c=r['conditions']
    lines=[r'\begin{table*}[t]',r'\centering\footnotesize',r'\caption{All final conditions on 500 shared items per task. NER uses exact typed-span micro metrics; SA uses fixed-binary macro metrics. Encoder values average three separate seeds (not ensembles); SD is seed variation. Intervals are pointwise paired 95\% intervals conditional on fitted systems. Gemma secondary removes only an eligible sole fence from unchanged answers; it has no added interval. Scores are percentages.}',r'\label{tab:conference-main}',r'\begin{tabular}{llrrrrr}',r'\toprule',r'Task & System & Precision & Recall & F1 & 95\% interval & Seed SD\\',r'\midrule']
    for task in ['ner','sa']:
        for model,label in zip(ORDER,LABELS):
            if task=='sa' and 'schema' in model:continue
            if task=='sa' and model=='gemma4_12b_ordinary4':label='Gemma 4 12B'
            ids=[f'{task}_{model.split("_")[0]}_seed{s}' for s in SEEDS] if model.endswith('seedmean') else [f'{task}_{model}']
            pk,rk=('precision','recall') if task=='ner' else ('macro_precision','macro_recall')
            p,rec=[statistics.mean(c[i]['metrics'][key] for i in ids)*100 for key in [pk,rk]]
            stat=r['statistics'][task]['conditions'][f'{task}_{model}'];lo,hi=[100*x for x in stat['ci95_pointwise']]
            sd=f'{statistics.stdev(c[i]["metrics"]["f1"] for i in ids)*100:.2f}' if len(ids)>1 else '--'
            lines.append(f'{task.upper()} & {label} & {p:.2f} & {rec:.2f} & {stat["f1"]*100:.2f} & [{lo:.2f}, {hi:.2f}] & {sd}'+r'\\')
            if task=='ner' and model=='gemma4_12b_ordinary4':
                s=c[f'{task}_{model}']['ner_fence_only_sensitivity']['fence_only_secondary']
                lines.append(f'NER & Gemma fence-only (secondary) & {s["precision"]*100:.2f} & {s["recall"]*100:.2f} & {s["f1"]*100:.2f} & -- & --'+r'\\')
        if task=='ner':lines.append(r'\midrule')
    lines += [r'\bottomrule',r'\end{tabular}',r'\end{table*}']
    (out/'conference_main_table.tex').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    lines=[r'\begin{table*}[t]',r'\centering\footnotesize',r'\caption{Matched NER decoding intervention. Syntax/root, contract and alignment columns count affected items out of 500 and can overlap. TP and FP count accepted typed entity spans against the unchanged benchmark; symptom counts are not additional false-positive penalties.}',r'\label{tab:schema-diagnostics}',r'\begin{tabular}{lrrrrr}',r'\toprule',r'Condition & Syntax/root & Contract & Alignment & TP & FP\\',r'\midrule']
    for model,label in zip(ORDER[2:6],LABELS[2:6]):
        v=c['ner_'+model];f=v['failures'];m=v['metrics']
        lines.append(label+' & '+' & '.join(str(x) for x in [f['syntax_or_root_items'],f['contract_items'],f['alignment_items'],m['tp'],m['fp']])+r'\\')
    lines += [r'\bottomrule',r'\end{tabular}',r'\end{table*}']
    (out/'schema_diagnostics_table.tex').write_text('\n'.join(lines)+'\n',encoding='utf-8')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--results',required=True);p.add_argument('--results-sha256',required=True);p.add_argument('--output',required=True);a=p.parse_args();assert sha(a.results)==a.results_sha256
    build(json.loads(Path(a.results).read_text(encoding='utf-8')),a.output)
    names=['conference_main_table.tex','schema_diagnostics_table.tex']
    m={'status':'CONFERENCE_TABLES_FROM_UNCHANGED_RESULTS','results_sha256':a.results_sha256,'builder_sha256':sha(__file__),'output_sha256':{n:sha(Path(a.output)/n) for n in names},'new_scoring_or_inference':False}
    (Path(a.output).parent/'CONFERENCE_TABLE_MANIFEST.json').write_text(json.dumps(m,indent=2),encoding='utf-8')
