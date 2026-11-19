"""Presentation only: fixed tables and figures from verified saved statistics.

No inference, source test data, bootstrap, model selection, or rescoring.
Usage: python build_dg2027_assets.py --results RESULTS.json --output NEW_DIR
Matplotlib and ReportLab must be importable (the project's vendor path works).
"""
from pathlib import Path
import argparse
import hashlib
import json
import math
import os
import statistics

EXPECTED_SHA = '7b4b65a2743bddc4eac3ca6882fa16a40575df57d51fee62472c72967cc02b4c'
SEEDS = (13, 42, 2026)
ORDER = [
    ('xlmr_seedmean', 'XLM-R'), ('mbert_seedmean', 'mBERT'),
    ('qwen_ordinary4', 'Qwen ordinary'), ('qwen_schema4', 'Qwen schema'),
    ('sherkala_ordinary4', 'Sherkala ordinary'), ('sherkala_schema4', 'Sherkala schema'),
    ('gemma4_12b_ordinary4', 'Gemma 4 12B strict'),
    ('gpt61sol_ordinary4', 'GPT-6.1-Sol'),
]
PRIMARY = [
    ('Qwen_NER_schema4-minus-XLMR_seedmean', 'NER: Qwen schema\nminus XLM-R'),
    ('Sherkala_NER_schema4-minus-XLMR_seedmean', 'NER: Sherkala schema\nminus XLM-R'),
    ('Qwen_SA_ordinary4-minus-XLMR_seedmean', 'SA: Qwen ordinary\nminus XLM-R'),
    ('Sherkala_SA_ordinary4-minus-XLMR_seedmean', 'SA: Sherkala ordinary\nminus XLM-R'),
    ('Qwen_NER_schema4-minus-ordinary4', 'NER: Qwen\nschema minus ordinary'),
    ('Sherkala_NER_schema4-minus-ordinary4', 'NER: Sherkala\nschema minus ordinary'),
]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')


def table(caption, label, columns, header, rows, note):
    return '\n'.join([
        r'\begin{table}[t]', r'\centering',
        r'\caption{' + caption + '}', r'\label{' + label + '}',
        r'\begingroup\fontsize{8}{9.3}\selectfont',
        r'\setlength{\tabcolsep}{3.2pt}\renewcommand{\arraystretch}{1.08}',
        r'\begin{tabular}{@{}' + columns + '@{}}', r'\toprule',
        header + r'\\', r'\midrule', *[row + r'\\' for row in rows],
        r'\bottomrule', r'\end{tabular}', r'\par\smallskip',
        r'\begin{minipage}{\columnwidth}\raggedright\fontsize{8}{9.3}\selectfont',
        note, r'\end{minipage}\endgroup', r'\end{table}', '',
    ])


def build_tables(result, out):
    conditions = result['conditions']
    numeric = {'scale': 'percent except integer counts', 'ner': [], 'sa': [], 'schema': []}
    for task in ('ner', 'sa'):
        rows = []
        for model, label in ORDER:
            if task == 'sa' and 'schema' in model:
                continue
            if task == 'sa':
                label = label.replace(' ordinary', '').replace(' strict', '')
            ids = ([f'{task}_{model.split("_")[0]}_seed{s}' for s in SEEDS]
                   if model.endswith('seedmean') else [f'{task}_{model}'])
            f1 = statistics.mean(conditions[i]['metrics']['f1'] for i in ids) * 100
            assert math.isclose(f1, 100 * result['statistics'][task]['conditions'][f'{task}_{model}']['f1'], abs_tol=1e-12)
            record = {'system': label, 'condition_ids': ids, 'f1': f1}
            if task == 'ner':
                p, r = [statistics.mean(conditions[i]['metrics'][k] for i in ids) * 100
                        for k in ('precision', 'recall')]
                record.update(precision=p, recall=r, branch='strict_primary')
                rows.append(f'{label} & {p:.2f} & {r:.2f} & {f1:.2f}')
            else:
                class_f1 = []
                for cls in ('negative', 'positive'):
                    class_f1.append(statistics.mean(next(x['f1'] for x in conditions[i]['per_type_or_class']
                                                         if x['type'] == cls) for i in ids) * 100)
                invalid = {conditions[i]['metrics']['invalid'] for i in ids}
                if len(invalid) != 1:
                    raise ValueError('Invalid counts differ across seeds; do not average counts')
                invalid = invalid.pop()
                record.update(negative_f1=class_f1[0], positive_f1=class_f1[1], invalid=invalid)
                rows.append(f'{label} & {f1:.2f} & {class_f1[0]:.2f} & {class_f1[1]:.2f} & {invalid}')
            numeric[task].append(record)
            if task == 'ner' and model == 'gemma4_12b_ordinary4':
                s = conditions[ids[0]]['ner_fence_only_sensitivity']['fence_only_secondary']
                secondary = {'system': 'Gemma fence-only (secondary)', 'condition_ids': ids,
                             'branch': 'prespecified_fence_only_secondary',
                             **{k: 100 * s[k] for k in ('precision', 'recall', 'f1')}}
                numeric['ner'].append(secondary)
                rows.append(r'Gemma fence-only$^{\dagger}$ & ' + ' & '.join(f'{secondary[k]:.2f}' for k in ('precision', 'recall', 'f1')))
        if task == 'ner':
            caption = r'NER performance on 500 shared test sentences (\%).'
            note = ('Exact typed-span micro scores. Encoders: mean of three seed scores, not an ensemble. '
                    r'$^{\dagger}$Prespecified secondary analysis of the same Gemma answers; only one eligible whole-answer JSON fence is removed.')
            content = table(caption, 'tab:ner-results', 'lrrr', 'System & P & R & F1', rows, note)
        else:
            caption = r'SA performance on 500 shared test reviews (\%).'
            note = ('Gold labels: 94 negative and 406 positive. Encoders: mean of three seed scores; '
                    'each seed has zero invalid outputs. Invalid counts remain in the evaluation.')
            content = table(caption, 'tab:sa-results', 'lrrrr', 'System & Macro F1 & Neg. F1 & Pos. F1 & Invalid', rows, note)
        (out / f'{task}_results.tex').write_text(content, encoding='utf-8')
    rows = []
    for model, label in ORDER[2:6]:
        condition_id = 'ner_' + model
        v = conditions[condition_id]
        values = [v['failures'][key] for key in ('syntax_or_root_items', 'contract_items', 'alignment_items')]
        values += [v['metrics']['tp'], v['metrics']['fp']]
        numeric['schema'].append(dict(zip(('system', 'condition_id', 'syntax_or_root_items', 'contract_items', 'alignment_items', 'tp', 'fp'), [label, condition_id, *values])))
        rows.append(label + ' & ' + ' & '.join(map(str, values)))
    (out / 'schema_diagnostics.tex').write_text(table(
        'Matched NER output diagnostics on 500 sentences.', 'tab:schema-diagnostics', 'lrrrrr',
        'System & Syntax & Contract & Align. & TP & FP', rows,
        'Syntax includes JSON syntax/root failures. Syntax, contract, and alignment columns count affected items and may overlap. '
        'TP/FP count accepted typed spans; diagnostic counts add no scoring penalties.'), encoding='utf-8')
    assert [len(numeric[k]) for k in ('ner', 'sa', 'schema')] == [9, 6, 4]
    return numeric


def build_forest(result, out):
    os.environ.setdefault('MPLCONFIGDIR', str(out.parent / 'matplotlib_cache'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 7.5, 'axes.labelsize': 8,
                         'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5, 'pdf.fonttype': 42,
                         'axes.spines.top': False, 'axes.spines.right': False})
    saved = {row['comparison_id']: row for task in ('ner', 'sa')
             for row in result['statistics'][task]['paired_differences'] if row['family'] == 'primary_K6'}
    assert set(saved) == {key for key, _ in PRIMARY}
    fig, ax = plt.subplots(figsize=(3.45, 3.05))
    fig.subplots_adjust(left=.475, right=.975, bottom=.16, top=.91)
    records = []
    for i, (key, label) in enumerate(PRIMARY):
        row = saved[key]
        lo, hi = [100*x for x in row['ci95_pointwise']]
        alo, ahi = [100*x for x in row['ci_bonferroni_K6']]
        delta = 100 * row['delta_f1']
        y = 5-i
        ax.plot([alo, ahi], [y, y], color='#9aa9b6', lw=3.4, solid_capstyle='butt', zorder=2)
        ax.plot([lo, hi], [y, y], color='#172b3a', lw=1.1, solid_capstyle='butt', zorder=3)
        ax.plot(delta, y, 'o', color='#172b3a', ms=3.1, zorder=4)
        records.append({'comparison_id': key, 'label': label, 'delta_percentage_points': delta,
                        'ci95_percentage_points': [lo, hi], 'ci_bonferroni_K6_percentage_points': [alo, ahi]})
    ax.axvline(0, color='.4', lw=.7, ls='--', zorder=1)
    ax.set_xlim(-82, 22)
    ax.set_xticks([-80, -60, -40, -20, 0, 20])
    ax.set_yticks(range(5, -1, -1), [label for _, label in PRIMARY])
    ax.tick_params(axis='y', length=0, pad=4)
    ax.set_ylim(-.55, 5.55)
    ax.set_xlabel('F1 difference (percentage points)', labelpad=5)
    ax.spines['left'].set_visible(False)
    ax.grid(axis='x', alpha=.14, zorder=0)
    legend = [Line2D([0], [0], color='#172b3a', lw=1.1, label='Pointwise 95%'),
              Line2D([0], [0], color='#9aa9b6', lw=3.4, label='Bonferroni (K = 6)')]
    fig.legend(handles=legend, loc='upper center', bbox_to_anchor=(.5, 1), ncol=2,
               frameon=False, fontsize=7.5, handlelength=1.5, columnspacing=.8)
    fig.savefig(out / 'primary_paired_intervals.pdf', metadata={'Creator': 'Fixed-result presentation builder', 'CreationDate': None, 'ModDate': None})
    fig.savefig(out / 'primary_paired_intervals.png', dpi=220)
    plt.close(fig)
    return records, matplotlib.__version__


def build_workflow(out):
    from reportlab.pdfgen import canvas
    from reportlab.lib.colors import HexColor, white
    from reportlab.pdfbase.pdfmetrics import stringWidth, registerFont
    from reportlab.pdfbase.ttfonts import TTFont
    import matplotlib
    font_dir = Path(matplotlib.get_data_path()) / 'fonts' / 'ttf'
    registerFont(TTFont('DiagramSans', str(font_dir / 'DejaVuSans.ttf')))
    registerFont(TTFont('DiagramSansBold', str(font_dir / 'DejaVuSans-Bold.ttf')))
    width, height = 252, 210
    c = canvas.Canvas(str(out / 'methodology_workflow.pdf'), pagesize=(width, height), invariant=1,
                      initialFontName='DiagramSans')
    c.setTitle('Frozen evaluation workflow')
    blocks = [
        ('1  DATA AND PROTOCOL', ['KazNERD sentences / KazSAnDRA reviews', 'Training and dev decisions frozen before test'], 39),
        ('2  SHARED FINAL PANELS', ['500 NER sentences + 500 SA reviews', 'Identical task panels across evaluated systems'], 34),
        ('3  INFERENCE', ['Encoders: three independently trained seeds', 'LLMs: ordinary prompting', 'Qwen / Sherkala NER: matched schema decoding'], 45),
        ('4  SCORING AND UNCERTAINTY', ['Gold labels used only for scoring', 'NER: exact spans; SA: binary macro-F1', 'Strict primary / fence-only secondary', 'Paired intervals; six primary contrasts'], 49),
    ]
    top = height - 4
    for index, (title, lines, box_height) in enumerate(blocks):
        bottom = top - box_height
        c.setStrokeColor(HexColor('#384955')); c.setLineWidth(.65)
        c.setFillColor(white); c.rect(4, bottom, width-8, box_height, stroke=1, fill=1)
        c.setFillColor(HexColor('#172b3a')); c.setFont('DiagramSansBold', 8.3)
        c.drawCentredString(width/2, top-11, title)
        font_size = 7.8
        for j, line in enumerate(lines):
            assert stringWidth(line, 'DiagramSans', font_size) <= width-18, line
            c.setFont('DiagramSans', font_size)
            c.drawCentredString(width/2, top-22-j*8.4, line)
        if index < len(blocks)-1:
            c.setStrokeColor(HexColor('#384955')); c.setLineWidth(.75)
            c.line(width/2, bottom-1, width/2, bottom-9)
            path = c.beginPath(); path.moveTo(width/2-2.2, bottom-6.5); path.lineTo(width/2, bottom-9); path.lineTo(width/2+2.2, bottom-6.5); c.drawPath(path)
        top = bottom - 11
    c.showPage(); c.save()
    return {'width_pt': width, 'height_pt': height, 'blocks': [{'title': title, 'lines': lines} for title, lines, _ in blocks]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if sha(args.results) != EXPECTED_SHA:
        raise ValueError('RESULTS source SHA mismatch')
    result = json.loads(args.results.read_text(encoding='utf-8-sig'))
    assert result['status'] == 'COMPLETE_RAW_RECONSTRUCTED_FIXED_ANALYSIS'
    assert result['provenance_binding_verified'] and result['protocol_freeze_verified']
    assert len(result['conditions']) == 22
    assert all(c['metrics']['items'] == 500 for c in result['conditions'].values())
    if args.output.exists():
        raise FileExistsError('Fresh output directory required; previous renders are preserved')
    args.output.mkdir(parents=True)
    numeric = build_tables(result, args.output)
    intervals, mpl_version = build_forest(result, args.output)
    workflow = build_workflow(args.output)
    write_json(args.output / 'PRESENTATION_VALUES.json', {'source_sha256': EXPECTED_SHA,
               'tables': numeric, 'primary_paired_intervals': intervals, 'workflow': workflow})
    write_json(args.output / 'ASSET_MANIFEST.json', {
        'status': 'PRESENTATION_ONLY_FROM_UNCHANGED_FIXED_RESULTS',
        'results_sha256': EXPECTED_SHA, 'builder_sha256': sha(__file__),
        'new_inference_bootstrap_or_scoring': False, 'encoder_aggregation': 'arithmetic mean of three separate seed metrics',
        'matplotlib_version': mpl_version,
        'forest_dimensions_inches': [3.45, 3.05], 'workflow_dimensions_points': [252, 210],
        'table_font_points': 8, 'figure_min_font_points': 7.5,
        'output_sha256': {p.name: sha(p) for p in sorted(args.output.iterdir()) if p.is_file()},
        'visual_review': 'separate render review required',
    })
    print(json.dumps({'status': 'COMPLETE', 'output': str(args.output), 'tables': 3, 'figures': 2}))


if __name__ == '__main__':
    main()
