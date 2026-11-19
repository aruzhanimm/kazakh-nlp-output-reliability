"""Recreate illustrative manuscript table from a verified non-private excerpt."""
from pathlib import Path
import argparse,hashlib,json
def escape(s):
    for a,b in [('\\',r'\textbackslash{}'),('&',r'\&'),('%',r'\%'),('_',r'\_'),('#',r'\#')]:s=s.replace(a,b)
    return s
def kz(s):return r'\kz{'+escape(s)+'}'
def build(d,out):
    assert d['status']=='ACTUAL_AUTOMATED_ILLUSTRATIVE_EXCERPT' and d['not_human_annotations']
    cs={c['category']:c for c in d['cases']}
    for c in cs.values():
        assert ' '.join(c['tokens'])==c['text_space_joined_original_tokens']
        for o in c['outputs']:
            assert hashlib.sha256(o['raw_task_answer'].encode()).hexdigest()==o['raw_task_answer_sha256']
    g,q,s,t=[cs[x] for x in ('gemma','qwen','sherkala','gpt')]
    assert g['outputs'][0]['primary_spans']==[] and g['outputs'][0]['prespecified_fence_only_secondary_spans']==g['benchmark_spans']
    assert 'phalt' in q['outputs'][0]['raw_task_answer'] and len(q['outputs'])==2
    assert [x['type'] for x in s['outputs'][0]['primary_spans']]==['MONEY','MONEY']
    descriptions={
      'gemma':'The raw answer has one whole-answer JSON fence around the correct entity record. Strict parsing accepts no entities; the prespecified fence-only branch exactly matches the benchmark.',
      'qwen':'Ordinary output uses the wrong key ``phalt\'\' in two records; only the extra DISEASE '+kz(q['outputs'][0]['primary_spans'][0]['mention'])+' is accepted. Schema decoding recovers NORP but also predicts extra DISEASE and PERSON spans and misses CARDINAL '+kz(q['benchmark_spans'][0]['mention'])+'. This is a generation intervention, not passive repair.',
      'sherkala':'The ordinary output splits the expression into two MONEY spans. The JSON answer is accepted but disagrees with the benchmark span.',
      'gpt':'The predicted CARDINAL span is '+kz(t['outputs'][0]['primary_spans'][0]['mention'])+', a boundary disagreement with the corpus. No claim is made here that this resolves the linguistic ambiguity.'}
    lines=[r'% Actual archived cases, automatically selected; no human judgments or translations.',r'\begin{table*}[t]',r'\centering\footnotesize',
      r'\caption{Post-hoc illustrations from actual archived NER answers. Source IDs are official test-row identities. Quotes retain source tokens and spacing. The benchmark spans are comparison targets, not new human adjudications.}',r'\label{tab:illustrations}',
      r'\begin{tabular}{@{}p{23mm}p{66mm}p{74mm}@{}}',r'\toprule',r'System / row & Source and benchmark & Observed output behavior\\',r'\midrule']
    for i,(key,label) in enumerate([('gemma','Gemma'),('qwen','Qwen'),('sherkala','Sherkala'),('gpt','GPT')]):
        c=cs[key]
        if i:lines.append(r'\addlinespace')
        benchmark='; '.join(escape(x['type'])+' '+kz(x['mention']) for x in c['benchmark_spans'])
        lines.append(label+' / '+c['source_id'].split(':')[-1]+r' & {\raggedright '+kz(c['text_space_joined_original_tokens'])+r'\newline Benchmark: '+benchmark+r'.\par} & {\raggedright '+descriptions[key]+r'\par}\\')
    lines += [r'\bottomrule',r'\end{tabular}',r'\end{table*}']
    Path(out).write_text('\n'.join(lines)+'\n',encoding='utf-8')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--cases',required=True);p.add_argument('--cases-sha256',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    raw=Path(a.cases).read_bytes();assert hashlib.sha256(raw).hexdigest()==a.cases_sha256
    build(json.loads(raw),a.output)
