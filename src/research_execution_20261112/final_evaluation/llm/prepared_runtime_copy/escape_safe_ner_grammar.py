"""Proposed, narrowly guarded correction for XGrammar 0.2.9 issue #800.

This module does not mutate a package, schema, parser, or stored predictions.
Admission requires separate actual CPU/tokenizer conformance evidence.
"""
import hashlib
import importlib.metadata
import json

SCHEMA_SHA256 = '080722300a679f49f9a8cadb30d2c5d84227854fffd8bcba08ab9831773ef46f'
ORIGINAL_EBNF_SHA256 = '9a1daac05ad90e9c2d522333e7a022ee08737959d7dcda49f44ab63f80e914c1'
COMPILE_SETTINGS = {'any_whitespace': True, 'strict_mode': True, 'any_order': False}
OLD_RULE = r'root_additional_prop_0_repeat ::= (([^\"\\\r\n]))'
NEW_RULE = r'root_additional_prop_0_repeat ::= (([^\"\\\x00-\x1f]) | ("\\" ([\"\\/bfnrt] | ("u" [0-9A-Fa-f]{4}))))'


def digest_text(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def patch_ebnf(original):
    """Fail closed unless the complete grammar is the locally audited version."""
    if digest_text(original) != ORIGINAL_EBNF_SHA256:
        raise ValueError('Original XGrammar EBNF differs from audited exact grammar')
    lines = original.splitlines(keepends=True)
    matches = [i for i, line in enumerate(lines) if line.rstrip('\r\n') == OLD_RULE]
    if len(matches) != 1:
        raise ValueError('Expected exactly one audited mention-character rule')
    index = matches[0]
    ending = lines[index][len(OLD_RULE):]
    lines[index] = NEW_RULE + ending
    patched = ''.join(lines)
    if sum(a != b for a, b in zip(original.splitlines(), patched.splitlines())) != 1:
        raise ValueError('Patch must change exactly one grammar rule')
    return patched


def compile_escape_safe_ner(compiler, schema, compile_settings):
    """Return (compiled, receipt); schema minLength=1 remains enforced.

    A JSON escaped unit is nonempty, so one or more legal units implements the
    only supported bound, minLength=1. This is not a general maxLength patch.
    """
    if importlib.metadata.version('xgrammar') != '0.2.9':
        raise ValueError('Correction requires exact audited XGrammar 0.2.9')
    actual_schema = digest_text(json.dumps(schema, ensure_ascii=False, sort_keys=True, separators=(',', ':')))
    if actual_schema != SCHEMA_SHA256 or compile_settings != COMPILE_SETTINGS:
        raise ValueError('Correction only supports the exact NER schema/settings')
    if list(schema['items']['properties']) != ['mention', 'type', 'occurrence']:
        raise ValueError('Exact ordered schema properties required')
    original = compiler.compile_json_schema(schema, **compile_settings)
    original_ebnf = str(original.grammar)
    corrected_ebnf = patch_ebnf(original_ebnf)
    corrected = compiler.compile_grammar(corrected_ebnf)
    receipt = {'implementation': 'xgrammar_0.2.9_ner_escape_rule_v1',
               'upstream_issue': 'https://github.com/mlc-ai/xgrammar/issues/800',
               'schema_sha256': actual_schema,
               'original_ebnf_sha256': digest_text(original_ebnf),
               'patched_input_ebnf_sha256': digest_text(corrected_ebnf),
               'compiled_ebnf_sha256': digest_text(str(corrected.grammar)),
               'old_rule': OLD_RULE, 'new_rule': NEW_RULE,
               'schema_minLength_one_preserved': True,
               'all_other_grammar_rules_unchanged_before_recompilation': True,
               'parser_or_metric_changed': False}
    return corrected, receipt
