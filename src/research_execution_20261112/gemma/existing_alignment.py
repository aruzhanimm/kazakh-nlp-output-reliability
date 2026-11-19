"""Gold-free exact mention alignment. No normalization, fuzzy matching, or repair."""
import json

def strict_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result: raise ValueError('Duplicate JSON object key: '+key)
        result[key] = value
    return result


def reject_constant(value):
    raise ValueError('Non-JSON numeric constant: '+value)


def text_offsets(tokens):
    if not tokens or any(not isinstance(t, str) or not t or any(c.isspace() for c in t) for t in tokens):
        raise ValueError('Original tokens must be nonempty and contain no whitespace')
    starts, ends, cursor = {}, {}, 0
    for i, word in enumerate(tokens):
        starts[cursor] = i
        cursor += len(word); ends[cursor] = i+1; cursor += 1
    return ' '.join(tokens), starts, ends


def occurrences(tokens, mention):
    """Enumerate literal matches that exactly cover whole original token spans."""
    text, starts, ends = text_offsets(tokens)
    if not isinstance(mention, str) or not mention: return [], []
    literal, aligned, offset = [], [], 0
    while True:
        begin = text.find(mention, offset)
        if begin < 0: break
        end = begin+len(mention); literal.append((begin, end))
        if begin in starts and end in ends:
            aligned.append((starts[begin], ends[end]))
        offset = begin+1  # Include overlapping literal occurrences deterministically.
    return aligned, literal


def parse_mentions(raw, tokens, allowed):
    result = {'spans': [], 'invalid_candidates': 0, 'failed': False, 'errors': [], 'diagnostics': []}
    try:
        obj = json.loads(raw, object_pairs_hook=strict_object, parse_constant=reject_constant)
        if not isinstance(obj, list): raise ValueError('Expected a JSON list')
    except (ValueError, TypeError) as error:
        result.update(failed=True, errors=[str(error)])
        result['diagnostics'].append({'category': 'format', 'reason': 'json_syntax_or_root'})
        return result
    seen = set()
    for i, record in enumerate(obj):
        signature = json.dumps(record, sort_keys=True, ensure_ascii=False)
        note = {'record_index': i, 'record': record}
        if signature in seen:
            result['diagnostics'].append({**note, 'category': 'duplicate', 'reason': 'exact_record_deduplicated'})
            continue
        seen.add(signature)
        reason = category = None; matches = []
        if not isinstance(record, dict): reason, category = 'record_not_object', 'format'
        else:
            mention = record.get('mention')
            if isinstance(mention, str) and mention:
                matches, literal = occurrences(tokens, mention)
                note.update(candidate_token_spans=matches, literal_occurrences=len(literal))
            if set(record) != {'mention', 'type', 'occurrence'}:
                if 'occurrence' not in record and len(matches) > 1:
                    reason, category = 'ambiguous_missing_occurrence', 'alignment'
                else: reason, category = 'wrong_schema_keys', 'format'
            elif not isinstance(mention, str) or not mention:
                reason, category = 'invalid_mention_string', 'format'
            elif not isinstance(record['type'], str) or record['type'] not in allowed:
                reason, category = 'invalid_entity_type', 'format'
            elif type(record['occurrence']) is not int or record['occurrence'] < 1:
                reason, category = 'invalid_occurrence_number', 'format'
            elif not matches:
                reason, category = ('boundary_mismatch' if literal else 'unmatched_string'), 'alignment'
            elif record['occurrence'] > len(matches):
                reason, category = 'occurrence_out_of_range', 'alignment'
            else:
                start, end = matches[record['occurrence']-1]
                if any(start < b and a < end for a, b, _ in result['spans']):
                    reason, category = 'overlap_first_valid_wins', 'alignment'
                else:
                    span = (start, end, record['type']); result['spans'].append(span)
                    result['diagnostics'].append({**note, 'category': 'aligned', 'token_span': span})
        if reason:
            result['invalid_candidates'] += 1; result['failed'] = True
            result['errors'].append({'invalid_record': record, 'reason': reason})
            result['diagnostics'].append({**note, 'category': category, 'reason': reason})
    return result


def demo_answer(tokens, spans):
    """Only callers with retained TRAINING labels may construct demonstrations."""
    answer = []
    for start, end, kind in spans:
        mention = ' '.join(tokens[start:end]); matches, _ = occurrences(tokens, mention)
        answer.append({'mention': mention, 'type': kind, 'occurrence': matches.index((start, end))+1})
    return answer
