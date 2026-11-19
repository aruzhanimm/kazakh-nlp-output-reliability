"""Isolated proposed NER evidence assembly; never imports a generation backend.

Production mention parser -> existing proposed whole-span owner merge -> strict
scoring. This is a component candidate, not an executable final inference runner.
"""
from pathlib import Path
import hashlib
import importlib.util
import json
import math

PROJECT = Path(__file__).resolve().parents[2]


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PARSER_PATH = PROJECT / 'interface_pilot/alignment.py'
MERGE_PATH = PROJECT / 'ner_evidence_20261111/analysis/strict_ner.py'
parser = _load('production_mention_parser', PARSER_PATH)
strict = _load('existing_proposed_strict_ner', MERGE_PATH)


class IncompleteRun(ValueError):
    """A missing/nonterminal record is not an empty model prediction."""


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(',', ':'), allow_nan=False).encode('utf-8')).hexdigest()


def text_hash(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def source_hashes():
    return {str(p.relative_to(PROJECT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [PARSER_PATH, MERGE_PATH, Path(__file__).resolve()]}


def contract_hash(allowed_types):
    return digest({'allowed_types': list(allowed_types), 'source_hashes': source_hashes(),
                   'policy': 'proposed_whole_span_owner_terminal_empty_completed_partial_keep_v1'})


def logits_failures(raw, processed, schema_constrained):
    """Consumes saved scalar summaries, not tensors. Intentional grammar -inf OK.

    Requires raw stats before grammar masking, processed stats after all masking.
    Unknown summaries fail closed rather than masquerading as a clean call.
    """
    keys = {'finite', 'nan', 'positive_inf', 'negative_inf', 'finite_min', 'finite_max'}
    for summary in [raw, processed]:
        if not keys <= summary.keys():
            raise ValueError('Incomplete logits summary')
        if any(type(summary[k]) is not int or summary[k] < 0
               for k in ['finite', 'nan', 'positive_inf', 'negative_inf']):
            raise ValueError('Invalid logits counters')
        if summary['finite']:
            if any(type(summary[k]) not in [int, float] or not math.isfinite(summary[k])
                   for k in ['finite_min', 'finite_max']) or summary['finite_min'] > summary['finite_max']:
                raise ValueError('Invalid finite-logits range')
        elif summary['finite_min'] is not None or summary['finite_max'] is not None:
            raise ValueError('Empty finite-logits range must be null')
    reasons = []
    if raw['nan'] or raw['positive_inf'] or raw['negative_inf'] or not raw['finite']:
        reasons.append('unexpected_raw_nonfinite')
    if raw['finite'] and raw['finite_min'] == raw['finite_max'] == 0:
        reasons.append('all_zero_raw_logits')
    if processed['nan'] or processed['positive_inf'] or not processed['finite']:
        reasons.append('invalid_processed_logits')
    if processed['negative_inf'] and not schema_constrained:
        reasons.append('unexpected_ordinary_negative_inf')
    return reasons


def make_request_manifest(item_id, tokens, windows, run_hash, condition, prepared_prompts, allowed_types):
    """CPU manifest helper. prepared_prompts must come from the native renderer.

    Tests use explicitly synthetic renderer outputs. Real tokenization/template
    correctness remains the backend/preflight's separate responsibility.
    """
    parser.text_offsets(tokens)
    bounds = [(w['start'], w['end']) for w in windows]
    strict.merge_window_spans(bounds, [[] for _ in windows], len(tokens))
    if len(windows) != len(prepared_prompts):
        raise ValueError('Missing prepared prompt')
    visible = {'id': item_id, 'tokens': tokens, 'windows': bounds}
    contract = contract_hash(allowed_types)
    manifest = []
    for wi, (window, prompt) in enumerate(zip(windows, prepared_prompts)):
        if window['tokens'] != tokens[window['start']:window['end']]:
            raise ValueError('Window text differs from original word slice')
        entry = {'id': item_id, 'window_index': wi, 'start': window['start'], 'end': window['end'],
                 'run_hash': run_hash, 'condition': condition, 'visible_input_sha256': digest(visible),
                 'contract_sha256': contract,
                 'window_tokens_sha256': digest(window['tokens']),
                 'messages_sha256': digest(prompt['messages']),
                 'rendered_prompt_sha256': text_hash(prompt['rendered_prompt']),
                 'input_token_ids_sha256': digest(prompt['input_token_ids'])}
        entry['request_identity'] = digest(entry)
        manifest.append(entry)
    return manifest


def _validate_attempts(record):
    attempts = record['attempts']
    if not isinstance(attempts, list) or len(attempts) > 3:
        raise ValueError('Invalid attempt ledger')
    if len({a['attempt_id'] for a in attempts}) != len(attempts):
        raise ValueError('Duplicate attempt ID')
    if any(a['status'] in ['started', 'unknown_timeout', 'pending'] for a in attempts):
        raise IncompleteRun('Ambiguous or unfinished attempt')
    # Prior attempts may only be confirmed transport failures before a response.
    for prior in attempts[:-1]:
        if prior['status'] != 'transport_failure' or prior['response_received'] is not False:
            raise ValueError('Retry after a response or non-transport failure')
    if record['status'] == 'completed':
        if not attempts or attempts[-1]['status'] != 'completed' or attempts[-1]['response_received'] is not True:
            raise ValueError('Completed response requires completed ledger')
    else:
        reason = record.get('terminal_reason')
        if not isinstance(reason, str) or not reason:
            raise ValueError('Explicit terminal reason required')
        if not attempts:
            if reason != 'input_context_failure':
                raise ValueError('Zero-attempt terminal allowed only for checked context overflow')
            context = record['context_budget']
            if not (type(context['reserved_output_tokens']) is int and context['reserved_output_tokens'] > 0 and
                    type(context['context_limit_tokens']) is int and context['context_limit_tokens'] > 0 and
                    len(record['input_token_ids']) + context['reserved_output_tokens'] > context['context_limit_tokens']):
                raise ValueError('Context failure is not supported by saved token count/budget')
        elif attempts[-1]['status'] not in ['runtime_failure', 'transport_failure']:
            raise ValueError('Contradictory terminal ledger')
        if attempts and attempts[-1]['status'] == 'transport_failure' and len(attempts) != 3:
            raise ValueError('Transport terminal before the registered three-attempt policy is exhausted')


def assemble_item(item_id, tokens, windows, manifest, records, allowed_types):
    """No gold argument. Complete records required for every frozen window.

    A terminal failure's recovered partial raw is retained in evidence but adds
    no spans. A completed response with invalid records/truncation retains its
    accepted spans and sets a failure flag. No retries or repairs occur here.
    """
    if not allowed_types or len(set(allowed_types)) != len(allowed_types):
        raise ValueError('Invalid allowed type list')
    if len(manifest) != len(windows):
        raise IncompleteRun('Incomplete request manifest')
    if not manifest:
        raise ValueError('Empty request manifest')
    if len({(m['run_hash'], m['condition']) for m in manifest}) != 1:
        raise ValueError('Mixed run/condition identities')
    contract = contract_hash(allowed_types)
    if any(m['contract_sha256'] != contract for m in manifest):
        raise ValueError('Parser/merger/ontology contract changed')
    if any(type(r.get('window_index')) is not int for r in records):
        raise ValueError('Invalid window identity')
    indexed = {r['window_index']: r for r in records}
    if len(indexed) != len(records):
        raise ValueError('Duplicate window record')
    if set(indexed) != set(range(len(windows))):
        raise IncompleteRun('Missing or unexpected window records')
    expected_visible = digest({'id': item_id, 'tokens': tokens,
                               'windows': [(w['start'], w['end']) for w in windows]})
    local, details = [], []
    for wi, (window, expected) in enumerate(zip(windows, manifest)):
        record = indexed[wi]
        if expected['id'] != item_id or expected['window_index'] != wi:
            raise ValueError('Wrong request manifest identity/order')
        if expected['visible_input_sha256'] != expected_visible:
            raise ValueError('Manifest belongs to different original input')
        if expected['start'] != window['start'] or expected['end'] != window['end'] or \
                window['tokens'] != tokens[window['start']:window['end']]:
            raise ValueError('Frozen window mismatch')
        if expected['window_tokens_sha256'] != digest(window['tokens']):
            raise ValueError('Window tokens hash mismatch')
        if digest({k: v for k, v in expected.items() if k != 'request_identity'}) != expected['request_identity']:
            raise ValueError('Corrupt request manifest hash')
        if any(record.get(k) != v for k, v in expected.items()):
            raise ValueError('Record provenance differs from frozen request')
        if record['status'] not in ['completed', 'terminal_failure']:
            raise IncompleteRun('Window has no explicit terminal outcome')
        if record['raw_sha256'] != text_hash(record['raw']):
            raise ValueError('Raw output integrity failure')
        if digest(record['messages']) != expected['messages_sha256'] or \
                text_hash(record['rendered_prompt']) != expected['rendered_prompt_sha256'] or \
                digest(record['input_token_ids']) != expected['input_token_ids_sha256']:
            raise ValueError('Saved prompt/token identity mismatch')
        if type(record['truncated']) is not bool or type(record['refusal']) is not bool:
            raise ValueError('Failure flags must be explicit booleans')
        _validate_attempts(record)
        terminal = record['status'] == 'terminal_failure'
        if not isinstance(record['execution_failure_reasons'], list):
            raise ValueError('Execution diagnostics must be explicit')
        if not terminal and (record.get('runtime_error') or record['execution_failure_reasons']):
            raise ValueError('Runtime failure must be explicit terminal_failure')
        if terminal:
            local.append(None)
            parsed = None
        else:
            parsed = parser.parse_mentions(record['raw'], window['tokens'], allowed_types)
            # Refusal is a completed model response, never eligible for retry.
            local.append([] if record['refusal'] else parsed['spans'])
        details.append({'window_index': wi, 'request_identity': expected['request_identity'],
                        'raw_sha256': record['raw_sha256'], 'status': record['status'],
                        'terminal_reason': record.get('terminal_reason'), 'parsed': parsed,
                        'truncated': record['truncated'], 'refusal': record['refusal']})
    merged = strict.merge_window_spans([(w['start'], w['end']) for w in windows], local, len(tokens))
    merge_conflicts = [r for r in merged['rejected'] if r['reason'] == 'overlap_first_valid_wins']
    parser_failed = any(d['parsed'] is not None and d['parsed']['failed'] for d in details)
    terminal_failed = bool(merged['failed_windows'])
    truncated = any(d['truncated'] for d in details)
    refused = any(d['refusal'] for d in details)
    failed = parser_failed or terminal_failed or truncated or refused or bool(merge_conflicts)
    provenance = []
    for span in merged['spans']:
        found = []
        for wi, d in enumerate(details):
            if local[wi] is None or d['refusal']:
                continue
            aligned = [x for x in d['parsed']['diagnostics'] if x['category'] == 'aligned']
            for ci, candidate in enumerate(local[wi]):
                global_span = (candidate[0] + windows[wi]['start'], candidate[1] + windows[wi]['start'], candidate[2])
                if global_span != span:
                    continue
                owner = max((min(span[0] - w['start'], w['end'] - span[1]), -j, j)
                            for j, w in enumerate(windows) if w['start'] <= span[0] and span[1] <= w['end'])[2]
                if owner == wi:
                    found.append({'span': span, 'window_index': wi, 'candidate_index': ci,
                                  'raw_record_index': aligned[ci]['record_index'],
                                  'request_identity': d['request_identity'], 'raw_sha256': d['raw_sha256']})
        if len(found) != 1:
            raise ValueError('Accepted span must have exactly one source candidate')
        provenance.extend(found)
    parser_extra = sum(d['parsed']['invalid_candidates'] for d in details if d['parsed'] is not None)
    return {'id': item_id, 'run_hash': manifest[0]['run_hash'], 'condition': manifest[0]['condition'],
            'n_words': len(tokens), 'visible_input_sha256': expected_visible, 'allowed_types': list(allowed_types),
            'spans': merged['spans'], 'failed': failed, 'parser_failed': parser_failed,
            'terminal_failed_windows': merged['failed_windows'],
            'partial_runtime_failure': terminal_failed and len(merged['failed_windows']) < len(windows),
            'truncated': truncated, 'refusal': refused, 'merge_alignment_failure': bool(merge_conflicts),
            'parser_invalid_candidate_extra_fp': parser_extra,
            'proposed_merge_overlap_extra_fp': len(merge_conflicts),
            'legacy_invalid_candidate_extra_fp': parser_extra + len(merge_conflicts),
            'merge_rejections': merged['rejected'], 'window_results': details, 'span_provenance': provenance}


def score_item(gold_labels, assembled):
    """Scoring boundary; full gold is introduced only after gold-free assembly."""
    if len(gold_labels) != assembled['n_words']:
        raise ValueError('Gold length differs from original item')
    for tag in gold_labels:
        if tag != 'O' and tag[2:] not in assembled['allowed_types']:
            raise ValueError('Gold type outside frozen ontology')
    gold, repairs = strict.bio_to_spans(gold_labels, repair_prediction=False)
    tp, fp, fn = strict.exact_counts(gold, assembled['spans'])
    return {**strict.prf(tp, fp, fn), 'gold_spans': gold,
            'exact_set_match': set(gold) == set(assembled['spans']),
            'failure_aware_exact': strict.failure_aware_exact(gold, assembled['spans'], assembled['failed']),
            'legacy_fp': fp + assembled['legacy_invalid_candidate_extra_fp']}


def assemble_panel(jobs, allowed_types, expected_item_ids):
    """Tiny CPU boundary to reject duplicate item IDs before any full-panel score."""
    ids = [j['item_id'] for j in jobs]
    if len(set(ids)) != len(ids):
        raise ValueError('Duplicate item IDs')
    if not expected_item_ids or len(set(expected_item_ids)) != len(expected_item_ids):
        raise ValueError('Invalid planned item ID manifest')
    if set(ids) != set(expected_item_ids):
        raise IncompleteRun('Missing or unexpected planned items')
    identities = {(m['run_hash'], m['condition']) for job in jobs for m in job['manifest']}
    if len(identities) != 1:
        raise ValueError('Panel mixes run/condition identities')
    by_id = {j['item_id']: j for j in jobs}
    return [assemble_item(**by_id[item_id], allowed_types=allowed_types) for item_id in expected_item_ids]
