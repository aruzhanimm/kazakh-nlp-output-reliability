"""Presentation-only figures from fixed-analysis RESULTS.json; no data/model access.

All intervals are consumed as saved. No bootstrap, model selection, significance
test, human-label inference, or cross-device runtime ranking is performed here.
"""
from pathlib import Path
import argparse, datetime, hashlib, importlib.metadata, json, math, os, statistics, sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / 'vendor_packages'))
os.environ.setdefault('MPLCONFIGDIR', str(HERE / 'matplotlib_cache'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

SEEDS = (13, 42, 2026)
PRIMARY = [
    ('Qwen_NER_schema4-minus-XLMR_seedmean', 'NER: Qwen schema - XLM-R'),
    ('Sherkala_NER_schema4-minus-XLMR_seedmean', 'NER: Sherkala schema - XLM-R'),
    ('Qwen_SA_ordinary4-minus-XLMR_seedmean', 'SA: Qwen - XLM-R'),
    ('Sherkala_SA_ordinary4-minus-XLMR_seedmean', 'SA: Sherkala - XLM-R'),
    ('Qwen_NER_schema4-minus-ordinary4', 'NER: Qwen schema - ordinary'),
    ('Sherkala_NER_schema4-minus-ordinary4', 'NER: Sherkala schema - ordinary'),
]
MODEL_ORDER = ['xlmr_seedmean', 'mbert_seedmean', 'qwen_ordinary4', 'qwen_schema4',
               'sherkala_ordinary4', 'sherkala_schema4', 'gemma4_12b_ordinary4', 'gpt61sol_ordinary4']
LABELS = {'xlmr_seedmean':'XLM-R (3-seed mean)', 'mbert_seedmean':'mBERT (3-seed mean)',
          'qwen_ordinary4':'Qwen ordinary', 'qwen_schema4':'Qwen schema',
          'sherkala_ordinary4':'Sherkala ordinary', 'sherkala_schema4':'Sherkala schema',
          'gemma4_12b_ordinary4':'Gemma 4 12B', 'gpt61sol_ordinary4':'GPT-6.1-Sol'}
FAILURES = [('syntax_or_root_items','Syntax / root'), ('contract_items','Contract'),
            ('alignment_items','Alignment'), ('partial_valid_items','Partial valid'),
            ('truncated','Truncated')]
EXPECTED = {f'{t}_{m}_seed{s}' for t in ['ner','sa'] for m in ['xlmr','mbert'] for s in SEEDS}
EXPECTED |= {f'{t}_{m}' for t in ['ner','sa'] for m in MODEL_ORDER[2:] if t == 'ner' or 'schema' not in m}

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def finite(value, low, high):
    if isinstance(value, bool) or not isinstance(value, (int,float)) or not math.isfinite(value) or not low <= value <= high:
        raise ValueError(f'Invalid numeric value {value!r}')

def interval(values, low=-1, high=1):
    if not isinstance(values, list) or len(values) != 2:
        raise ValueError('Two-element interval required')
    for x in values: finite(x, low, high)
    if values[0] > values[1]: raise ValueError('Reversed interval')

def validate(result, synthetic=False):
    expected_status = 'SYNTHETIC_ONLY_NO_RESEARCH_RESULTS' if synthetic else 'COMPLETE_RAW_RECONSTRUCTED_FIXED_ANALYSIS'
    if result.get('status') != expected_status: raise ValueError('Wrong evidence status')
    if bool(result.get('synthetic_fixture', False)) != synthetic: raise ValueError('Synthetic flag mismatch')
    if not synthetic and not (result.get('provenance_binding_verified') is True and result.get('protocol_freeze_verified') is True):
        raise ValueError('Fixed verified analysis required')
    if set(result['conditions']) != EXPECTED: raise ValueError('Exact 22 condition identities required')
    all_primary = []
    for task in ['ner','sa']:
        stats = result['statistics'][task]
        for row in stats['conditions'].values():
            finite(row['f1'], 0, 1); interval(row['ci95_pointwise'],0,1)
        for row in stats['paired_differences']:
            finite(row['delta_f1'],-1,1);interval(row['ci95_pointwise'])
            if row['family'] == 'primary_K6':
                all_primary.append(row['comparison_id']);interval(row['ci_bonferroni_K6'])
            elif row['family'] != 'descriptive_only' or 'ci_bonferroni_K6' in row:
                raise ValueError('Unknown family or descriptive comparison given primary correction')
        for c, row in result['conditions'].items():
            if not c.startswith(task+'_'): continue
            n = row['metrics']['items']
            if not synthetic and n != 500: raise ValueError('Unaccounted coverage')
            if not isinstance(n,int) or n < 1: raise ValueError('Positive item denominator required')
            finite(row['metrics']['f1'],0,1)
            for p in row['per_type_or_class']:
                finite(p['f1'],0,1)
                for key in ['support','tp','fp','fn']:
                    if not isinstance(p[key],int) or p[key] < 0: raise ValueError('Invalid per-type count')
            for key,_ in FAILURES:
                count = row['failures'].get(key,0)
                if not isinstance(count,int) or not 0 <= count <= n: raise ValueError('Invalid item-level failure count')
    if sorted(all_primary) != sorted(x[0] for x in PRIMARY): raise ValueError('Six exact primary contrasts required')

def saved_f1(result,task,model):
    return result['statistics'][task]['conditions'][f'{task}_{model}']

def entity_matrix(result):
    base = result['conditions']['ner_xlmr_seed13']['per_type_or_class']
    types = [x['type'] for x in base]
    if len(types) != len(set(types)): raise ValueError('Duplicate entity type')
    support = [x['support'] for x in base]
    values=[]
    for model in MODEL_ORDER:
        ids = [f'ner_{model.split("_")[0]}_seed{s}' for s in SEEDS] if model.endswith('seedmean') else [f'ner_{model}']
        per=[]
        for c in ids:
            rows={p['type']:p for p in result['conditions'][c]['per_type_or_class']}
            if set(rows)!=set(types) or [rows[t]['support'] for t in types]!=support:
                raise ValueError('Per-type supports differ across systems')
            per.append([rows[t]['f1'] for t in types])
        # Explicitly average per-seed per-type F1; do not pool predictions/counts.
        values.append([statistics.mean(x) for x in zip(*per)])
    return types,support,np.asarray(values).T

def prepare_style():
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.titlesize':9,
                        'axes.labelsize':8,'xtick.labelsize':7,'ytick.labelsize':8,
                        'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none',
                        'axes.spines.top':False,'axes.spines.right':False,
                        'savefig.dpi':300,'figure.facecolor':'white'})

def save(fig,out,name,synthetic):
    if synthetic:
        fig.text(.5,1.08,'SYNTHETIC FIXTURE - NOT RESEARCH RESULTS',ha='center',va='top',
                 color='#a01818',fontsize=10,weight='bold',bbox={'facecolor':'white','edgecolor':'#a01818','pad':3})
    prefix='SYNTHETIC_' if synthetic else ''
    for ext in ['pdf','svg','png']:
        fig.savefig(out/f'{prefix}{name}.{ext}',bbox_inches='tight',metadata={'Creator':'Kazakh NLP fixed-result presentation builder'} if ext=='pdf' else None)
    plt.close(fig)

def primary_forest(result,out,synthetic):
    rows={r['comparison_id']:r for t in ['ner','sa'] for r in result['statistics'][t]['paired_differences']}
    fig,ax=plt.subplots(figsize=(7.0,3.0));fig.subplots_adjust(left=.38,right=.97,bottom=.18,top=.84)
    for i,(key,label) in enumerate(PRIMARY):
        row=rows[key];y=5-i;lo,hi=np.asarray(row['ci95_pointwise'])*100
        alo,ahi=np.asarray(row['ci_bonferroni_K6'])*100
        color='#0072B2' if i<4 else '#B45F06'
        ax.plot([alo,ahi],[y,y],color=color,lw=2.8,alpha=.5)
        ax.plot([lo,hi],[y,y],color=color,lw=1.0)
        ax.plot(row['delta_f1']*100,y,'o',color=color,ms=4)
    ax.axvline(0,color='.35',lw=.8,ls='--');ax.set_yticks(range(5,-1,-1),[x[1] for x in PRIMARY])
    ax.set_xlabel('F1 difference (percentage points; left system minus right system)')
    ax.set_title('Six prespecified contrasts: broad line = family-adjusted interval',pad=15)
    ax.grid(axis='x',alpha=.15);ax.set_ylim(-.5,5.5)
    save(fig,out,'primary_paired_intervals',synthetic)

def modern_forest(result,out,synthetic):
    rows=[(task,r) for task in ['ner','sa'] for r in result['statistics'][task]['paired_differences'] if r['family']=='descriptive_only']
    if len(rows)!=4:raise ValueError('Four descriptive comparisons required')
    fig,ax=plt.subplots(figsize=(7.0,2.5));fig.subplots_adjust(left=.37,right=.97,bottom=.22,top=.80)
    labels=[]
    for i,(task,row) in enumerate(rows):
        label=LABELS[row['left'][len(task)+1:]];labels.append(f'{task.upper()}: {label} - XLM-R')
        lo,hi=np.asarray(row['ci95_pointwise'])*100;y=3-i
        ax.plot([lo,hi],[y,y],color='#5D5D5D',lw=1.5);ax.plot(row['delta_f1']*100,y,'o',color='#5D5D5D',ms=4)
    ax.axvline(0,color='.35',lw=.8,ls='--');ax.set_yticks(range(3,-1,-1),labels)
    ax.set_xlabel('F1 difference (percentage points)');ax.set_title('Modern references: descriptive pointwise 95% intervals only',pad=13)
    ax.grid(axis='x',alpha=.15);save(fig,out,'modern_descriptive_intervals',synthetic)

def ner_types(result,out,synthetic):
    types,support,values=entity_matrix(result)
    mask=np.repeat((np.asarray(support)==0)[:,None],len(MODEL_ORDER),axis=1)
    cmap=plt.get_cmap('viridis').copy();cmap.set_bad('#eeeeee')
    fig,ax=plt.subplots(figsize=(7.0,max(4.0,.17*len(types)+1.45)))
    fig.subplots_adjust(left=.23,right=.94,bottom=.22,top=.90)
    im=ax.imshow(np.ma.array(values*100,mask=mask),vmin=0,vmax=100,cmap=cmap,aspect='auto')
    ax.set_yticks(range(len(types)),[f'{t}  (n={n})' for t,n in zip(types,support)])
    ax.set_xticks(range(len(MODEL_ORDER)),[LABELS[m].replace(' (3-seed mean)','\nmean of 3 seeds') for m in MODEL_ORDER],rotation=40,ha='right')
    ax.set_title('Exact typed-span F1 by entity type; n = gold support',pad=12)
    fig.colorbar(im,ax=ax,fraction=.026,pad=.025,label='F1 (%)')
    fig.text(.23,.025,'Gray rows: no gold support; false positives remain in the count table.\nEncoder cells average three separate per-type F1 scores.',fontsize=7)
    save(fig,out,'ner_type_f1_support',synthetic)

def failure_rates(result,out,synthetic):
    models=MODEL_ORDER[2:];counts=[];denoms=[]
    for m in models:
        row=result['conditions']['ner_'+m]
        counts.append([row['failures'].get(k,0) for k,_ in FAILURES]);denoms.append(row['metrics']['items'])
    rates=np.asarray(counts)/np.asarray(denoms)[:,None]*100
    fig,ax=plt.subplots(figsize=(7.0,3.5));fig.subplots_adjust(left=.24,right=.95,bottom=.21,top=.82)
    im=ax.imshow(rates,vmin=0,vmax=100,cmap='Blues',aspect='auto')
    ax.set_yticks(range(len(models)),[LABELS[m] for m in models]);ax.set_xticks(range(len(FAILURES)),[l for _,l in FAILURES])
    for y in range(len(models)):
        for x in range(len(FAILURES)):
            ax.text(x,y,f'{counts[y][x]}/{denoms[y]}\n{rates[y,x]:.1f}%',ha='center',va='center',fontsize=7,color='white' if rates[y,x]>55 else '#202020')
    ax.set_title('NER output symptoms; categories overlap',pad=14)
    fig.colorbar(im,ax=ax,fraction=.026,pad=.025,label='Items (%)')
    fig.text(.24,.04,'Counts are item-level automatic parser/termination flags. They are not\nmutually exclusive, semantic judgments, or an additive failure total.',fontsize=7)
    save(fig,out,'ner_output_symptoms',synthetic)

def main_table(result,out,synthetic):
    prefix='SYNTHETIC_' if synthetic else ''
    lines=[r'\begin{table*}[t]',r'\centering\footnotesize',
           r'\caption{'+('SYNTHETIC FIXTURE. ' if synthetic else '')+r'Performance on 500 common test items per task. NER uses exact typed-span micro precision, recall and F1; SA uses fixed-binary macro metrics. Encoder values average three separate seed metrics, not ensemble predictions; SD is seed variability. Intervals are pointwise paired-cluster 95\% intervals conditional on fitted systems. Modern references are descriptive. All values except counts are percentages.}',
           r'\label{tab:final_main}',r'\begin{tabular}{llrrrrr}',r'\toprule',
           r'Task & System & Precision & Recall & F1 & 95\% interval & Seed SD\\',r'\midrule']
    for task in ['ner','sa']:
        for model in MODEL_ORDER:
            if task=='sa' and 'schema' in model:continue
            stat=saved_f1(result,task,model)
            if model.endswith('seedmean'):
                ids=[f'{task}_{model.split("_")[0]}_seed{s}' for s in SEEDS]
                sd=statistics.stdev(result['conditions'][c]['metrics']['f1'] for c in ids)*100
            else:ids=[f'{task}_{model}'];sd=None
            pk,rk=('precision','recall') if task=='ner' else ('macro_precision','macro_recall')
            p=statistics.mean(result['conditions'][c]['metrics'][pk] for c in ids)*100
            r=statistics.mean(result['conditions'][c]['metrics'][rk] for c in ids)*100
            lo,hi=[100*x for x in stat['ci95_pointwise']]
            sdtext='--' if sd is None else f'{sd:.2f}'
            lines.append(f'{task.upper()} & {LABELS[model]} & {p:.2f} & {r:.2f} & {stat["f1"]*100:.2f} & [{lo:.2f}, {hi:.2f}] & {sdtext}'+r'\\')
        if task=='ner':lines.append(r'\midrule')
    lines.extend([r'\bottomrule',r'\end{tabular}',r'\end{table*}'])
    (out/f'{prefix}main_results_table.tex').write_text('\n'.join(lines)+'\n',encoding='utf8')

def build(result,out,synthetic=False):
    validate(result,synthetic)
    out=Path(out)
    if out.exists():raise FileExistsError('Fresh output directory required')
    out.mkdir(parents=True);prepare_style()
    for function in [primary_forest,modern_forest,ner_types,failure_rates,main_table]:function(result,out,synthetic)
    return out

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results',type=Path,required=True);p.add_argument('--results-sha256',required=True)
    p.add_argument('--complete',type=Path);p.add_argument('--complete-sha256')
    p.add_argument('--output',type=Path,required=True);p.add_argument('--synthetic-fixture',action='store_true')
    a=p.parse_args()
    if digest(a.results)!=a.results_sha256:raise ValueError('RESULTS SHA mismatch')
    if not a.synthetic_fixture:
        if not a.complete or not a.complete_sha256 or digest(a.complete)!=a.complete_sha256:raise ValueError('Actual assembler COMPLETE receipt required')
        receipt=json.loads(a.complete.read_text(encoding='utf8'))
        if receipt.get('status')!='COMPLETE_RAW_RECONSTRUCTED_FIXED_ANALYSIS' or receipt['output_sha256'].get('RESULTS.json')!=a.results_sha256:
            raise ValueError('RESULTS not bound to completed fixed analysis')
    result=json.loads(a.results.read_text(encoding='utf8'))
    out=build(result,a.output,a.synthetic_fixture)
    manifest={'status':'SYNTHETIC_ONLY' if a.synthetic_fixture else 'PRESENTATION_FROM_VERIFIED_FIXED_ANALYSIS',
              'source_results_sha256':a.results_sha256,'source_complete_sha256':a.complete_sha256,
              'builder_sha256':digest(__file__),'matplotlib':matplotlib.__version__,'numpy':np.__version__,
              'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'new_statistics_computed':False,'per_type_encoder_cells':'arithmetic mean of three separate F1 values',
              'final_test_data_read_directly':False,'human_judgments_inferred':False,'visual_review_pending':True,
              'output_sha256':{x.name:digest(x) for x in out.iterdir() if x.is_file()}}
    (out/'FIGURE_MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf8')
    print(json.dumps({'status':manifest['status'],'output':str(out),'files':len(manifest['output_sha256'])}))

if __name__=='__main__':main()
