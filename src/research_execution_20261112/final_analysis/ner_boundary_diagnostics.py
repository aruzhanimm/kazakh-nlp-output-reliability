"""Descriptive exact-boundary diagnostics from the same retained NER spans.

No text normalization, overlap matching, output repair, gold change or new model
call. Untyped endpoints are sets. Type confusion is conditional on EXACT shared
boundaries with one predicted type. Multi-type boundaries remain explicit and
are excluded from this conditional confusion matrix, never chosen arbitrarily.
"""
from collections import Counter,defaultdict

POLICY={
    'status':'prespecified_descriptive_only',
    'input':'same accepted predicted spans and original gold as primary typed F1',
    'untyped_match':'exact (start,end) endpoint-set equality; types ignored and endpoints deduplicated',
    'type_confusion':'gold row, predicted column; exact shared boundaries with one predicted type only',
    'multi_type_prediction':'count boundary and distinct typed spans separately; exclude that boundary from type-confusion denominator',
    'gold':'strict original IOB2; at most one gold type per exact boundary',
    'missing_spurious':'missing/spurious exact boundaries; never greedy overlap or nearest-boundary matching',
    'inference':'descriptive counts only; no new confirmatory tests or confidence intervals',
    'interpretation':'output boundary detection and conditional type assignment under the fixed parser, not latent language comprehension',
}

def grouped(values,allowed):
    by=defaultdict(set);seen=set()
    for value in values:
        if len(value)!=3:raise ValueError('Typed span must contain start,end,type')
        start,end,kind=value
        if type(start)is not int or type(end)is not int or not 0<=start<end or kind not in allowed:raise ValueError('Invalid diagnostic span')
        if (start,end,kind) in seen:raise ValueError('Exact duplicate typed spans must be reconciled upstream')
        seen.add((start,end,kind));by[(start,end)].add(kind)
    return dict(by)

def analyze(rows,gold_by_id,allowed_types):
    """Micro totals over provided original items; no averaging per-item F1."""
    allowed=tuple(allowed_types)
    if not rows or not allowed or len(set(allowed))!=len(allowed):raise ValueError('Nonempty rows and unique ontology required')
    ids=[r['id'] for r in rows]
    if len(set(ids))!=len(ids) or set(ids)!=set(gold_by_id):raise ValueError('Exact diagnostic/gold item coverage required')
    counts=Counter();confusion=Counter();per_item=[];ambiguous=[]
    for row in rows:
        rid=row['id'];g=grouped(gold_by_id[rid],allowed);p=grouped(row['spans'],allowed)
        if any(len(types)!=1 for types in g.values()):raise ValueError('Original flat NER gold has conflicting types at the same boundary')
        gb,pb=set(g),set(p);matched=gb&pb;multi={b for b,types in p.items() if len(types)>1}
        eligible=matched-multi
        for boundary in eligible:
            confusion[(next(iter(g[boundary])),next(iter(p[boundary])))]+=1
        observed={
            'id':rid,'gold_boundaries':len(gb),'predicted_boundaries':len(pb),
            'matched_boundaries':len(matched),'missing_boundaries':len(gb-pb),'spurious_boundaries':len(pb-gb),
            'type_confusion_eligible_boundaries':len(eligible),
            'multi_type_predicted_boundaries':len(multi),
            'multi_type_predicted_typed_spans':sum(len(p[b]) for b in multi),
            'multi_type_on_gold_boundaries':len(multi&gb),
            'multi_type_on_spurious_boundaries':len(multi-gb),
        }
        per_item.append(observed);counts.update({k:v for k,v in observed.items() if k!='id'})
        for boundary in sorted(multi):
            ambiguous.append({'id':rid,'start':boundary[0],'end':boundary[1],
                              'predicted_types':sorted(p[boundary]),
                              'gold_type':next(iter(g[boundary])) if boundary in g else None})
    tp,fp,fn=counts['matched_boundaries'],counts['spurious_boundaries'],counts['missing_boundaries']
    correct=sum(confusion[t,t] for t in allowed);eligible=counts['type_confusion_eligible_boundaries']
    matrix=[[confusion[g,p] for p in allowed] for g in allowed]
    return {
        'policy':POLICY,'items':len(rows),'counts':dict(counts),
        'untyped_exact_boundary':{'tp':tp,'fp':fp,'fn':fn,'support':tp+fn,
            'precision':tp/(tp+fp) if tp+fp else 0.,'recall':tp/(tp+fn) if tp+fn else 0.,
            'f1':2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.},
        'conditional_type_assignment':{'eligible_boundaries':eligible,'correct_type':correct,'wrong_type':eligible-correct,
            'accuracy':correct/eligible if eligible else None,
            'excluded_multi_type_gold_boundaries':counts['multi_type_on_gold_boundaries']},
        'type_confusion':{'gold_row_types':list(allowed),'predicted_column_types':list(allowed),'matrix':matrix,
            'cells':[{'gold_type':g,'predicted_type':p,'count':confusion[g,p]} for g in allowed for p in allowed]},
        'multi_type_boundaries':ambiguous,'per_item_counts':per_item,
        'primary_typed_scores_changed':False,
    }
