"""Frozen-A source-to-common500 preparation; never implicitly authorizes test.

The metadata command reads only existing manifests/code. prepare requires an
external exact-hash Freeze A before any source bytes/IDs/gold are accessed.
"""
import argparse
import datetime
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[2]
sys.path.insert(0, str(PROJECT))
from audit_data import read_ner, read_sa, key as original_text_key
from pilot.inputs import owners
from pilot.scoring import bio_spans
from stage_preflight.views import common_windows, common_prefix

SELECTION = {'algorithm': 'sha256_compact_utf8_json_namespace_seed_task_id_then_lexical_id',
             'namespace': 'kazakh-final-v2', 'seed': 20261110, 'items_per_task': 500,
             'label_stratification': False, 'test_exclusions': 'none_keep_official_test_unchanged'}
VIEW_POLICY = {'encoder_max_tokens': 384, 'ner_max_words': 96, 'ner_overlap_words': 24,
               'sa_max_words': 128, 'cluster_text': 'complete_original_NFC_whitespace_collapse'}
MODEL_IDS = {'xlmr': ('FacebookAI/xlm-roberta-base', 'e73636d4f797dec63c3081bb6ed5c7b0bb3f2089'),
             'mbert': ('google-bert/bert-base-multilingual-cased', '3f076fdb1ab68d5b2880cb87a0886f315b8146f8')}


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8')


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for data in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(data)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as stream:
        stream.write(canonical(value) + b'\n')
        stream.flush()
        os.fsync(stream.fileno())


def sources():
    names = ['research_execution_20261112/final_evaluation/llm/prepare_common_panels.py',
             'audit_data.py', 'pilot/inputs.py', 'pilot/scoring.py', 'pilot/common.py',
             'pilot/models.py', 'stage_preflight/views.py', 'stage_preflight/common.py']
    return {name: sha(PROJECT / name) for name in names}


def metadata_draft():
    """Existing file manifest only: never opens an official test file."""
    manifest = read(PROJECT / 'data_manifest.json')
    source_files = {}
    for task in ('ner', 'sa'):
        row, = [x for x in manifest['files'] if x['task'] == task and x['split'] == 'test']
        source_files[task] = {'path': str(PROJECT / 'data/raw' / row['dataset'] / row['file_name']),
                              'sha256': row['sha256'], 'bytes': row['bytes'],
                              'repository': row['repo'], 'revision': row['commit']}
    return {'format': 'kazakh_final_freeze_a_v1', 'scope': 'final', 'status': 'DRAFT_NOT_AUTHORIZED',
            'final_test_access_authorized': False, 'decision_reference': None, 'frozen_utc': None,
            'selection': SELECTION, 'view_policy': VIEW_POLICY, 'preparation_sources': sources(),
            'source_manifest': {'path': str(PROJECT / 'data_manifest.json'), 'sha256': sha(PROJECT / 'data_manifest.json')},
            'source_files': source_files,
            'ner_labels': {'path': str(PROJECT / 'data/derived/ner_labels.json'), 'sha256': sha(PROJECT / 'data/derived/ner_labels.json')},
            'encoder_tokenizers': {name: {'model_id': value[0], 'revision': value[1], 'path': None, 'files': {}}
                                   for name, value in MODEL_IDS.items()},
            'packages': {'transformers': '4.57.6', 'tokenizers': '0.22.2'},
            'scientific_protocol': None, 'required_readiness_evidence': [],
            'metadata_only_test_files_not_opened': True}


def check_freeze_a(freeze):
    if freeze.get('format') != 'kazakh_final_freeze_a_v1' or freeze.get('scope') != 'final' or \
            freeze.get('status') != 'frozen_before_test_preparation' or freeze.get('final_test_access_authorized') is not True:
        raise PermissionError('Actual Freeze A is required before any final source access')
    if not freeze.get('decision_reference') or not freeze.get('frozen_utc'):
        raise PermissionError('Freeze decision and timestamp required')
    date = datetime.datetime.fromisoformat(freeze['frozen_utc'].replace('Z', '+00:00'))
    if date.tzinfo is None or date > datetime.datetime.now(datetime.timezone.utc):
        raise ValueError('Freeze timestamp must be aware and not future')
    if freeze.get('selection') != SELECTION or freeze.get('view_policy') != VIEW_POLICY or freeze.get('preparation_sources') != sources():
        raise ValueError('Selection/view/code differ from the frozen implementation')
    scientific = freeze.get('scientific_protocol')
    if not isinstance(scientific, dict) or not scientific.get('path') or not scientific.get('sha256'):
        raise ValueError('Scientific protocol file binding required')
    if sha(scientific['path']) != scientific['sha256']:
        raise ValueError('Scientific protocol hash mismatch')
    readiness = freeze.get('required_readiness_evidence')
    if not isinstance(readiness, list) or not readiness:
        raise ValueError('Actual dev/readiness evidence must be bound before test')
    for record in readiness:
        if sha(record['path']) != record['sha256']:
            raise ValueError('Readiness evidence hash mismatch')
    return True


def select_rows(rows, task, selection=SELECTION):
    ids = [r['id'] for r in rows]
    if len(set(ids)) != len(ids) or any(not i.startswith(task + ':test:') for i in ids):
        raise ValueError('Official test IDs invalid or duplicated')
    if len(rows) < selection['items_per_task']:
        raise ValueError('Official source smaller than frozen panel')
    def rank(row):
        value = json.dumps([selection['namespace'], selection['seed'], task, row['id']],
                           ensure_ascii=False, separators=(',', ':')).encode('utf-8')
        return hashlib.sha256(value).hexdigest(), row['id']
    return sorted(rows, key=rank)[:selection['items_per_task']]


def build_view(rows, task, toks, allowed_bio=None):
    """Pure view adapter; caller controls source access. Gold is returned separately."""
    items, gold = [], []
    if task not in ('ner', 'sa'): raise ValueError('Unsupported task')
    for row in rows:
        rid = row['id']
        original = ' '.join(row['tokens']) if task == 'ner' else row['text']
        item = {'id': rid, 'original_text': original, 'cluster_key': original_text_key(original), 'features': {}}
        if task == 'ner':
            if len(row['tokens']) != len(row['labels']) or (allowed_bio is not None and not set(row['labels']) <= set(allowed_bio)):
                raise ValueError('Original NER gold length/ontology differs; no repair or exclusion')
            bio_spans(row['labels'], repair=False)
            windows = common_windows(row['tokens'], toks, max_tokens=384, max_words=96, overlap=24)
            item.update(tokens=row['tokens'], windows=windows, owners=owners(windows, len(row['tokens'])))
            gold.append({'id': rid, 'tokens': row['tokens'], 'labels': row['labels']})
            for name, tok in toks.items():
                features = []
                for index, window in enumerate(windows):
                    encoded = tok(window['tokens'], is_split_into_words=True, truncation=False)
                    feature = {k: list(v) for k, v in encoded.items()}
                    feature.update(id=rid + ':w' + str(index), source_id=rid, window=index, word_ids=encoded.word_ids())
                    if len(feature['input_ids']) > 384 or set(x for x in feature['word_ids'] if x is not None) != set(range(len(window['tokens']))):
                        raise ValueError('NER tokenizer lost words or exceeded frozen context')
                    features.append(feature)
                item['features'][name] = features
        else:
            if row['label'] not in ('0', '1'): raise ValueError('Original binary SA label differs')
            visible, fallback = common_prefix(original, toks, max_tokens=384, max_words=128)
            item.update(text=original, visible_text=visible, capped=visible != original, character_fallback=fallback)
            gold.append({'id': rid, 'label': row['label']})
            for name, tok in toks.items():
                encoded = tok(visible, truncation=False)
                feature = {k: list(v) for k, v in encoded.items()}
                feature.update(id=rid, source_id=rid)
                if len(feature['input_ids']) > 384:
                    raise ValueError('SA context exceeds frozen shared prefix')
                item['features'][name] = [feature]
        if any(k in item for k in ('label', 'labels', 'gold', 'gold_spans')):
            raise AssertionError('Gold must not enter inference panel')
        items.append(item)
    panel = {'format': 'encoder_common_view_panel_v1', 'task': task, 'split': 'test',
             'expected_ids': [r['id'] for r in rows], 'items': items}
    return panel, {'task': task, 'split': 'test', 'rows': gold}


def prepare(freeze_path, freeze_hash, output):
    # No dataset/path stat, hash, open, or tokenizer load precedes this check.
    if sha(freeze_path) != freeze_hash:
        raise ValueError('Freeze A SHA mismatch')
    freeze = read(freeze_path)
    check_freeze_a(freeze)
    output = Path(output)
    if output.exists():
        raise FileExistsError('Use a fresh immutable panel output directory')
    manifest_spec = freeze['source_manifest']
    if sha(manifest_spec['path']) != manifest_spec['sha256']:
        raise ValueError('Source metadata manifest changed')
    manifest = read(manifest_spec['path'])
    for task, bound in freeze['source_files'].items():
        official, = [r for r in manifest['files'] if r['task'] == task and r['split'] == 'test']
        if bound['sha256'] != official['sha256'] or bound['bytes'] != official['bytes']:
            raise ValueError('Frozen source is not the existing official test file')
    if set(freeze['source_files']) != {'ner', 'sa'} or set(freeze['encoder_tokenizers']) != {'xlmr', 'mbert'}:
        raise ValueError('Both tasks and accepted encoder tokenizers required')
    label_spec = freeze['ner_labels']
    if sha(label_spec['path']) != label_spec['sha256']: raise ValueError('Original NER ontology metadata changed')
    labels = read(label_spec['path'])
    if len(labels['types']) != 25 or len(labels['bio']) != 51: raise ValueError('Original NER25/BIO51 ontology required')
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ['USE_TORCH'] = '0'
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    for package, expected in freeze['packages'].items():
        if importlib.metadata.version(package) != expected:
            raise ValueError('Tokenizer package differs from freeze')
    from transformers import AutoTokenizer
    toks = {}
    for name, spec in freeze['encoder_tokenizers'].items():
        if (spec['model_id'], spec['revision']) != MODEL_IDS[name] or not spec['files']:
            raise ValueError('Unpinned encoder tokenizer')
        folder = Path(spec['path'])
        for filename, expected in spec['files'].items():
            if Path(filename).name != filename or sha(folder / filename) != expected:
                raise ValueError('Tokenizer asset SHA mismatch')
        toks[name] = AutoTokenizer.from_pretrained(folder, local_files_only=True, use_fast=True, trust_remote_code=False)
        if not toks[name].is_fast:
            raise ValueError('Fast tokenizer required for exact word alignment')
    if toks['mbert'].do_lower_case is not False:
        raise ValueError('Cased mBERT tokenizer required')
    output.mkdir(parents=True, exist_ok=False)
    write_new(output / 'ACCESS_STARTED.json', {'freeze_a_sha256': freeze_hash,
        'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'purpose': 'deterministic preparation only'})
    summaries = {}
    for task in ('ner', 'sa'):
        bound = freeze['source_files'][task]
        path = Path(bound['path'])
        if path.stat().st_size != bound['bytes'] or sha(path) != bound['sha256']:
            raise ValueError('Official test source hash mismatch; no alternative source chosen')
        rows = (read_ner if task == 'ner' else read_sa)(path, 'test')
        chosen = select_rows(rows, task)
        panel, gold = build_view(chosen, task, toks, allowed_bio=labels['bio'] if task == 'ner' else None)
        panel['freeze_a_sha256'] = freeze_hash
        panel['source_sha256'] = bound['sha256']
        panel_file = output / (task + '_PANEL.json')
        gold_file = output / (task + '_GOLD.json')
        write_new(panel_file, panel)
        gold['panel_sha256'] = sha(panel_file)
        write_new(gold_file, gold)
        summaries[task] = {'path': str(panel_file.resolve()), 'sha256': sha(panel_file),
                           'ordered_ids_sha256': digest(panel['expected_ids']), 'items': len(chosen),
                           'gold': {'path': str(gold_file.resolve()), 'sha256': sha(gold_file)},
                           'official_rows': len(rows), 'selected_original_row_sha256': {r['id']: digest(r) for r in chosen},
                           'cluster_key_source': 'complete original text, never truncated prefix',
                           'windows': sum(len(r['windows']) for r in panel['items']) if task == 'ner' else len(chosen)}
    result = {'status': 'PREPARED_FOR_FREEZE_B_NOT_INFERENCE_AUTHORIZATION', 'freeze_a_sha256': freeze_hash,
              'panels': summaries, 'preparation_sources': sources(), 'model_calls': 0,
              'selection_used_labels': False, 'gold_in_inference_panels': False}
    write_new(output / 'PREPARATION_RECEIPT.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='command', required=True)
    draft = sub.add_parser('metadata-draft'); draft.add_argument('--output', required=True)
    prep = sub.add_parser('prepare'); prep.add_argument('--freeze-a', required=True)
    prep.add_argument('--freeze-a-sha256', required=True); prep.add_argument('--output', required=True)
    args = parser.parse_args()
    if args.command == 'metadata-draft':
        write_new(args.output, metadata_draft())
        print('DRAFT written; test files not opened')
    else:
        result = prepare(args.freeze_a, args.freeze_a_sha256, args.output)
        print(json.dumps({'status': result['status'], 'tasks': list(result['panels']), 'model_calls': 0}))
