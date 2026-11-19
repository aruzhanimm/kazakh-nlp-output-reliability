"""Rescore and audit this condition without loading a model or any test split."""
import csv
import math
import shutil
import statistics
from pathlib import Path
from pilot.common import ROOT,read_json,read_jsonl,write_json,file_hash,digest
from pilot.evaluate import summarize
from pilot.scoring import spans_bio
from .common import D,verify
from .main import RUN,dev_rows

def csv_rows(path,rows):
    if not rows:return
    with path.open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)

def audit():
    verify('benchmark_plan.lock.json');verify('main_plan.lock.json')
    budget=read_json(D/'main_budget.json');done=read_json(RUN/'completed.json') if (RUN/'completed.json').exists() else None
    rows=dev_rows();history=read_json(RUN/'learning_curve.json') if (RUN/'learning_curve.json').exists() else []
    run_hash=digest(read_json(RUN/'identity.json'));evaluations=[]
    for h in history:
        path=RUN/'evaluations'/f'epoch_{h["epoch"]:02d}';predictions=read_jsonl(path/'predictions.jsonl');metrics=read_json(path/'metrics.json')
        assert len(predictions)==11100
        assert all(p['run_hash']==run_hash and p['input_sha256']==digest(r) for p,r in zip(predictions,rows))
        rescored=summarize('ner',rows,predictions)
        for k in ['n','tp','fp','fn','precision','recall','f1','per_type','bio_repairs','runtime_failures','format_failures','invalid_candidate_fp']:
            assert rescored[k]==metrics[k],(h['epoch'],k)
        assert len(metrics['per_type'])==25 and metrics['f1']==h['dev_micro_f1']
        # Cross-check valid repaired BIO sequences with an independent established scorer.
        from seqeval.metrics import precision_score,recall_score,f1_score
        from seqeval.scheme import IOB2
        gold=[r['labels'] for r in rows]
        predicted=[spans_bio(p['parsed']['spans'],len(r['tokens'])) for p,r in zip(predictions,rows)]
        independent={name:fn(gold,predicted,mode='strict',scheme=IOB2,zero_division=0)
            for name,fn in [('precision',precision_score),('recall',recall_score),('f1',f1_score)]}
        assert metrics['invalid_candidate_fp']==0
        assert all(abs(independent[k]-metrics[k])<1e-12 for k in independent)
        write_json(path/'seqeval_crosscheck.json',{'package':'seqeval 1.2.2','mode':'strict IOB2 after disclosed deterministic BIO conversion','metrics':independent,'agreement_atol':1e-12})
        write_json(path/'metrics_rescored.json',rescored);evaluations.append({'epoch':h['epoch'],'metrics':metrics,'distributions':read_json(path/'distributions.json')})
    trace=read_jsonl(RUN/'training_committed.jsonl') if (RUN/'training_committed.jsonl').exists() else []
    coverage=[]
    if done:
        assert [s['update'] for s in trace]==list(range(1,done['updates']+1))
        for h in history:
            epoch=h['epoch'];steps=[s for s in trace if s['epoch']==epoch];ids=read_json(RUN/'orders'/f'epoch_{epoch:02d}.json')['feature_ids']
            assert len(steps)==5587 and len(ids)==len(set(ids))==89386
            assert sum(s['examples'] for s in steps)==89386 and sum(s['valid_labels'] for s in steps)==1034279
            assert steps[-1]['examples']==10 and all(s['examples']==16 for s in steps[:-1])
            for i,s in enumerate(steps):
                assert s['feature_ids_sha256']==digest(ids[i*16:(i+1)*16])
                assert math.isfinite(s['loss']) and math.isfinite(s['gradient_norm'])
            coverage.append({'epoch':epoch,'optimizer_updates':5587,'examples':89386,'valid_labels':1034279,'final_batch_examples':10,'final_batch_valid_labels':steps[-1]['valid_labels']})
        best=max(history,key=lambda h:h['dev_micro_f1'])
        assert done['best_epoch']==best['epoch'] and done['selected_checkpoint']==f'epoch_{best["epoch"]:02d}.pt'
        assert file_hash(RUN/'predictions.jsonl')==file_hash(RUN/'evaluations'/f'epoch_{best["epoch"]:02d}'/'predictions.jsonl')
        assert done['phase']=='complete' and len(history)<=5
    checkpoint_audit=[]
    for p in sorted((RUN/'checkpoints').glob('*.json')):
        if p.name=='latest.json':continue
        meta=read_json(p);payload=p.parent/meta['file'];assert meta['run_hash']==run_hash and file_hash(payload)==meta['sha256']
        checkpoint_audit.append(meta)
    assert len([x for x in checkpoint_audit if x['file'].startswith('rolling_')])<=2
    recovery=[]
    for p in sorted((RUN/'attempts').glob('*/recovery.json')):
        r=read_json(p)
        if r['restored']:
            assert r['expected']==r['actual'] and r['exact_state_signatures_verified']
            recovery.append(r)
    if done:
        assert any(r['updates']==1000 for r in recovery)
        second=read_jsonl(RUN/'attempts/2/training.jsonl');assert second[0]['update']==1001
    result={'run_hash':run_hash,'status':budget['status'],'complete':bool(done),'coverage':coverage,'epochs_rescored':len(evaluations),
        'checkpoints_verified':checkpoint_audit,'recovery':recovery,'source_locks_verified':True,'final_test_locked':True}
    write_json(D/'audit/main_verified.json',result)
    return budget,done,history,evaluations,trace,result

def report():
    budget,done,history,evaluations,trace,audited=audit();choice=read_json(D/'chosen_execution.json');bench=read_json(D/'benchmark_budget.json')
    normalization=read_json(D/'audit/normalization_fp32.json');pre=read_json(ROOT/'stage_preflight/resource_results.json')
    max_gradient=max(r['maximum_absolute_gradient_difference'] for r in normalization['results'])
    measured_features=sum(s['examples'] for s in trace);training_seconds=sum(s['seconds'] for s in trace)
    measured_rate=measured_features/training_seconds if training_seconds else None
    long_intervals=[{'update':s['update'],'epoch':s['epoch'],'recorded_seconds':s['seconds']} for s in trace if s['seconds']>60]
    reference_epochs=[];reference_steps=[]
    for h in history:
        steps=[s for s in trace if s['epoch']==h['epoch']]
        if len(steps)==5587 and not any(s['seconds']>60 for s in steps):
            reference_epochs.append(h['epoch']);reference_steps.extend(steps)
    reference_rate=sum(s['examples'] for s in reference_steps)/sum(s['seconds'] for s in reference_steps) if reference_steps else None
    write_json(D/'audit/timing_discontinuities.json',{'threshold_seconds':60,'intervals':long_intervals,
        'all_intervals_retained_in_raw_totals':True,'cause':'Not established; long wall-clock intervals are not evidence of sustained GPU computation.',
        'projection_reference_completed_epochs_without_intervals_above_threshold':reference_epochs})
    io_records=[read_json(p) for p in (RUN/'attempts').glob('*/checkpoint_*.json')]+[r for r in audited['checkpoints_verified'] if r['file'].startswith('epoch_')]
    save_seconds=statistics.mean(r['seconds_including_checksum'] for r in io_records) if io_records else 6.
    full_dev_seconds=statistics.mean(h['evaluation_wall_seconds_this_attempt'] for h in history) if history else 150.
    rate=reference_rate or choice['summary'][str(choice['microbatch'])]['median_features_per_second']
    # 27 periodic saves + 5 pre-eval saves + 5 retained epoch saves; startup/recovery/fingerprinting allowance.
    projected_xlmr_ner=(5*89386/rate+5*full_dev_seconds+37*save_seconds+120)/3600
    old_xlmr_ner=next(r for r in pre['completed'] if (r['model'],r['task'])==('xlmr','ner'))
    cross_stage_rate_ratio=old_xlmr_ner['features_per_second']/choice['summary']['2']['median_features_per_second']
    remaining=[]
    for old in pre['completed']:
        key=(old['model'],old['task']);is_current=key==('xlmr','ner');count=3 if is_current else 4
        base=projected_xlmr_ner if is_current else old['projected_five_epoch_run_hours']
        envelope=[.9*base,1.3*base] if is_current else list(old['projected_five_epoch_run_hours_range'])
        if key==('mbert','ner'):
            # Sensitivity allowance only: different stages do not isolate the cause of rate changes.
            envelope[1]=max(envelope[1],base*max(1.,cross_stage_rate_ratio)*1.3)
        remaining.append({'model':key[0],'task':key[1],'unstarted_conditions':count,'max5_hours_each':base,'planning_range_each':envelope,
            'basis':'completed full epochs without update intervals exceeding 60 s; full-dev evaluation' if reference_rate and is_current else 'short bounded benchmark' if is_current else 'previous 100-update calibration; not rebenchmarked under new NER normalization'})
    projected_remaining=sum(r['unstarted_conditions']*r['max5_hours_each'] for r in remaining)
    range_remaining=[sum(r['unstarted_conditions']*r['planning_range_each'][i] for r in remaining) for i in [0,1]]
    resources={'measured_features':measured_features,'measured_training_seconds':training_seconds,'measured_features_per_second':measured_rate,
        'main_wall_seconds':budget.get('elapsed_wall_seconds'),'checkpoint_save_including_checksum_seconds_mean':save_seconds,
        'timing_discontinuities_above60_seconds':long_intervals,'projection_reference_epochs':reference_epochs,'projection_reference_features_per_second':reference_rate,
        'full_dev_wall_seconds_mean':full_dev_seconds,'projected_max5_xlmr_ner_hours':projected_xlmr_ner,
        'unstarted_15_conditions':remaining,'remaining_15_projected_hours':projected_remaining,'remaining_15_planning_range_hours':range_remaining,
        'xlmr_micro2_cross_stage_rate_ratio_for_unmeasured_ner_sensitivity':cross_stage_rate_ratio,
        'disk_free_bytes_after_run':shutil.disk_usage(D).free,'projection_note':'Planning allowances, not statistical confidence intervals; no early-stop savings assumed.'}
    write_json(D/'resource_results.json',resources)
    csv_rows(D/'learning_curve.csv',history)
    per_type=[]
    for e in evaluations:
        per_type.extend({'epoch':e['epoch'],'type':t,**v} for t,v in e['metrics']['per_type'].items())
    csv_rows(D/'per_type_all_epochs.csv',per_type)
    if trace:
        smoothed=[]
        for i in range(0,len(trace),100):
            block=trace[i:i+100];smoothed.append({'update':block[-1]['update'],'epoch_fraction':block[-1]['update']/5587,
                'token_weighted_training_loss':sum(s['loss_sum'] for s in block)/sum(s['valid_labels'] for s in block)})
        csv_rows(D/'training_curve_100_updates.csv',smoothed)
    lines=['# First full-data encoder condition','',
        f"**Status: {budget['status']}.** Exactly one main condition: XLM-R base / NER / seed 13 / LR 1e-5. All earlier milestones are preserved. The remaining 15 encoder conditions are unstarted; final test inference remains locked.",'',
        '## Pre-main decisions and checks','',
        'Confirmed implementation difference: the former loop averaged microbatch mean losses; the main objective divides summed CE by all supervised first-subword labels in the complete optimizer batch. The actual final ten-example batch uses its actual label count. This pre-main amendment is documented in `METHOD_DECISIONS.md`; it does not establish the cause of any calibration score.',
        f'FP32/dropout-disabled variable-length tests passed for full and partial batches across microbatch sizes 2/4/8/full. Maximum absolute parameter-gradient difference: {max_gradient:.3g}. Ten CPU tests also cover atomic-save interruption/fallback, exact small-model resume, earliest-best/stopping rules, gold-independent alignment equivalence and nonfinite-logit detection. Existing scorer code is unchanged.','',
        f'Training-only benchmark: six fresh loads, 276 total optimizer updates including warmup/stress checks, {bench["spent_seconds"]/60:.2f} minutes of the 20-minute ceiling. Identical ordered training examples, normalized objective and optimizer/scheduler settings. All six completed; no development scores were collected.','',
        '| Microbatch × accumulation | Repeat rates (sentences/s) | Peak reserved GiB | Minimum observed free GiB | Eligible |',
        '|---|---:|---:|---:|---|']
    for m,x in choice['summary'].items():
        rs=x['results'];lines.append(f'| {m} × {16//int(m)} | '+ ' / '.join(f'{r["features_per_second"]:.2f}' for r in rs)+f' | {max(r["peak_gpu_reserved_bytes"] for r in rs)/2**30:.3f} | {min(r["minimum_free_gpu_bytes"] for r in rs)/2**30:.3f} | {x["eligible"]} |')
    lines += ['',f'Frozen choice: **{choice["microbatch"]} × {choice["gradient_accumulation"]}**, effective batch 16. It passed the predeclared repeat-stability, improvement and memory-headroom gates. Long-shape checks included the 16 longest distinct retained features and 16 copies of the longest 139-subword feature. This is resource selection, not a development-score comparison.',
        '', '## Main results','',
        'Initialization is the pinned pretrained XLM-R revision `e73636d4f797dec63c3081bb6ed5c7b0bb3f2089`, never calibration/benchmark weights. Each completed epoch covers all 89,386 retained sentences and 1,034,279 word labels, including all 25 entity types; evaluation covers all 11,100 retained dev sentences. Five-epoch linear schedule (27,935 steps), 1,676 warmup steps, BF16 autocast/FP32 parameters and CE, SDPA, gradient checkpointing, AdamW and clipping 1 remain fixed.','',
        '| Epoch | Token-weighted train loss | Dev exact micro-F1 % | Precision % | Recall % | BIO repairs | Predicted entities |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for h,e in zip(history,evaluations):
        lines.append(f'| {h["epoch"]} | {h["training_loss"]:.5f} | {100*h["dev_micro_f1"]:.3f} | {100*h["dev_precision"]:.3f} | {100*h["dev_recall"]:.3f} | {h["bio_repairs"]} | {e["distributions"]["entities"]} |')
    if done:
        chosen=next(e for e in evaluations if e['epoch']==done['best_epoch']);m=chosen['metrics'];dist=chosen['distributions']
        lines += ['',f'Selected **epoch {done["best_epoch"]}**, earliest exact maximum. Completed {len(history)} epochs / {done["updates"]:,} optimizer updates; early-stop rule triggered: {done["stop"]}. Selected checkpoint: `run_xlmr_ner_seed13_lr1e-5/checkpoints/{done["selected_checkpoint"]}`. No setting was changed in response to scores.',
            f'Selected exact counts: TP {m["tp"]:,}, FP {m["fp"]:,}, FN {m["fn"]:,}. Runtime failures {m["runtime_failures"]}; format failures {m["format_failures"]}; invalid-candidate FPs {m["invalid_candidate_fp"]}. BIO repairs {m["bio_repairs"]:,} are explicitly counted and use the inherited deterministic I→B conversion. All epoch predictions and all 25 per-type outcomes are retained.',
            f'Prediction distribution: {dist["words"]:,} word predictions; O={dist["word_tags"]["O"]:,}; {dist["entities"]:,} extracted entities. Mean maximum word probability {dist["mean_max_probability"]:.4f}; ten confidence bins and complete BIO/type distributions are in each epoch’s `distributions.json`. This confidence is descriptive, not calibrated probability evidence.','',
            '| Selected-epoch entity type | Gold support | Precision % | Recall % | Exact F1 % |','|---|---:|---:|---:|---:|']
        for t,v in m['per_type'].items():lines.append(f'| {t} | {v["support"]} | {100*v["precision"]:.2f} | {100*v["recall"]:.2f} | {100*v["f1"]:.2f} |')
    else:lines+=['','The run is incomplete/blocked. No partial development panel is represented as a full-panel score. Existing epoch results and safe checkpoints remain available. See `main_budget.json` and per-attempt failure/outcome records.']
    if trace:
        observed=done or read_json(RUN/'incomplete.json')
        lines+=['','## Runtime, recovery and preservation','',
            f'Recorded optimizer-update intervals total {training_seconds/3600:.3f} h, giving {measured_rate:.2f} sentences/s across all recorded intervals. Main operational wall time: {budget["elapsed_wall_seconds"]/3600:.3f} h of the eight-hour ceiling, including loads, checkpointing, full-dev evaluation and the planned fresh-process restart. Full-dev wall time averaged {full_dev_seconds/60:.2f} min; checkpoint save plus checksum averaged {save_seconds:.2f} s (fingerprinting is extra and included in operational wall time).',
            f'Long update intervals above 60 seconds: {len(long_intervals)}, totaling {sum(x["recorded_seconds"] for x in long_intervals)/3600:.3f} h. Their cause is not established. They remain in the raw traces and measured totals; they must not be described as sustained active GPU compute. Runtime projections instead use completed epochs {reference_epochs} without such intervals, or the pre-main benchmark if none qualify; reference throughput {rate:.2f} sentences/s. This explicit projection filter changes no scores, training settings or retained records.',
            f'Peak training allocated/reserved VRAM: {observed["peak_gpu_allocated_bytes"]/2**30:.3f}/{observed["peak_gpu_reserved_bytes"]/2**30:.3f} GiB on the measured RTX 5060 Ti. Training timings include finite-gradient checks and synchronized instrumentation; evaluation timings include finite-logit checks and confidence collection. They are operational, instrumented measurements.',
            'The real process restart restored update 1,000, verified matching model/optimizer/scheduler/RNG/progress signatures, then continued at update 1,001. This confirms a state round-trip and correct continuation; bitwise equivalence to a separate uninterrupted full GPU trajectory was not tested. CPU dropout-disabled interrupted/uninterrupted updates did match exactly.',
            f'All {len(audited["checkpoints_verified"])} retained full checkpoints passed SHA-256 verification: {len(history)} immutable epoch states plus two rolling recovery slots. Atomic payload/fsync/replace with checksum commit and fallback was tested by injected interruptions before and after replacement. Disk free after run: {resources["disk_free_bytes_after_run"]/2**30:.1f} GiB. No prior checkpoint was overwritten.',
            'Earlier-file preservation is audited separately in `audit/preservation_verified.json`. The temporary system-awake request ends with the launcher; no shutdown or persistent power-setting change is made.']
    lines+=['','## Remaining schedule and uncertainty','',
        f'The remaining **15 unstarted** maximum-five-epoch conditions project to **{projected_remaining:.1f} GPU-hours**, planning envelope **{range_remaining[0]:.1f}–{range_remaining[1]:.1f} h**. These are projections, not promised durations or confidence intervals. No early-stopping savings are assumed.','',
        '| Remaining family | Conditions | Projected max-five-epoch h each | Planning range h each |','|---|---:|---:|---:|']
    for r in remaining:lines.append(f'| {r["model"]} {r["task"]} | {r["unstarted_conditions"]} | {r["max5_hours_each"]:.2f} | {r["planning_range_each"][0]:.2f}–{r["planning_range_each"][1]:.2f} |')
    lines+=['','Proposed next stages, requiring a later instruction: finish the remaining three seed-13 NER LR-screen conditions; then four seed-13 SA LR-screen conditions; then eight selected-LR replications (seeds 42/2026). Keep all losing LR outcomes. Start only one bounded condition at a time, recheck disk and use a safe resumable ceiling for each. SA projected jobs can exceed eight hours and therefore require an explicitly planned split/budget; do not assume they fit this job’s ceiling.',
        'Only XLM-R NER now has full-epoch throughput measurements under the new objective/configuration. Other families retain the earlier 100-update point estimates and require their own longest-shape gates; the 8×2 speedup is not assumed transferable. The mBERT NER upper planning allowance additionally applies the observed old-calibration/new-benchmark XLM-R 2×8 rate ratio as a sensitivity scenario, not a measured mBERT slowdown or a causal estimate of normalization cost. Thermal/background load, rare SA lengths, full-data I/O and stopping behavior remain uncertain.',
        'Sherkala remains blocked by the preserved HTTP-401/GatedRepoError access record; no new access attempt or generation was made. Final test inference, paid APIs, noise, LLM fine-tuning and Qwen prompt changes remain unexecuted.',
        '','## Interpretation and artifacts','',
        'This is one exploratory full-data development condition, not the paper’s final experiment or evidence of novelty. Unequal label/pretraining budgets, repeated dev use, exact scoring/BIO repair, future bounded train-only demonstration sensitivity and unresolved contribution questions are documented in `METHOD_DECISIONS.md`.',
        'Every completed epoch was independently cross-checked against seqeval 1.2.2 strict IOB2 after the disclosed BIO conversion; precision, recall and exact micro-F1 agree within 1e-12. This does not replace the primary scorer or change failure accounting.',
        'Exact commands: `COMMANDS.md`. Curves: `learning_curve.csv`, `training_curve_100_updates.csv` and the run’s complete committed step trace. Per-type outcomes for every epoch: `per_type_all_epochs.csv`. Raw outputs, distributions and rescored metrics: `run_xlmr_ner_seed13_lr1e-5/evaluations/`. Frozen configuration/source: `main_config.json`, `main_plan.lock.json`, and the run’s `source/`.']
    (D/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print({'status':budget['status'],'epochs':len(history),'measured_features_per_second':measured_rate,'remaining_15_hours':projected_remaining})

if __name__=='__main__':report()
