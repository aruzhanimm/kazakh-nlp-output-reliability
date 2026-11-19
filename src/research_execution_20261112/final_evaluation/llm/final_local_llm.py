"""Separate final local-LLM adapter; original train/dev guards remain unchanged.

Uses the accepted HF runtime, immutable per-window ledger, mention parser and
NER merge. Gold never enters preparation or inference. All corpus/model access
is behind external freezes; this module's import only loads pure code.
"""
import argparse
import copy
import datetime
import hashlib
import importlib.util
import importlib.metadata
import json
import os
from pathlib import Path
import socket
import sys
import uuid

import prepare_common_panels as common
from prepare_common_panels import PROJECT, HERE, canonical, digest, sha, read, write_new
sys.path.insert(0, str(PROJECT / 'ner_runtime_20261111'))
sys.path.insert(0, str(PROJECT / 'ner_pipeline_20261111'))
import execution_runtime as execution
from storage import EvidenceStore, read_evidence, write_evidence, process_lock
from pilot.scoring import parse_sa
import native_inputs
sys.path.insert(0, str(HERE.parent))
import output_adapter

CONDITIONS = {f'{task}_{model}_{variant}': (task, model, variant == 'schema4')
              for task in ('ner', 'sa') for model in ('qwen', 'sherkala')
              for variant in (('ordinary4', 'schema4') if task == 'ner' else ('ordinary4',))}


def sources():
    names = ['research_execution_20261112/final_evaluation/llm/final_local_llm.py',
             'research_execution_20261112/final_evaluation/llm/prepare_common_panels.py',
             'ner_runtime_20261111/execution_runtime.py', 'ner_pipeline_20261111/storage.py',
             'ner_pipeline_20261111/native_inputs.py', 'interface_pilot/alignment.py',
             'ner_execution_prep_20261111/runner_review/evidence_assembly.py',
             'ner_evidence_20261111/analysis/strict_ner.py', 'pilot/scoring.py']
    names.append('research_execution_20261112/final_evaluation/output_adapter.py')
    return {name: sha(PROJECT / name) for name in names}


def bound_read(spec):
    if sha(spec['path']) != spec['sha256']:
        raise ValueError('Bound file SHA mismatch: ' + spec['path'])
    return read(spec['path'])


def load_hf(binding, preparation=False):
    root = Path(binding.get('preparation_runtime_path', binding['accepted_runtime']['path']) if preparation else binding['accepted_runtime']['path'])
    files = binding['accepted_runtime']['files']
    if not {'hf_backend.py', 'vendor/trace_schema.py', 'escape_safe_ner_grammar.py'} <= set(files):
        raise ValueError('Corrected accepted runtime files required')
    for name, expected in files.items():
        p = root / name
        if not p.resolve().is_relative_to(root.resolve()) or sha(p) != expected:
            raise ValueError('Accepted runtime source changed')
    sys.path.insert(0, str(root))
    spec = importlib.util.spec_from_file_location('accepted_final_local_hf', root / 'hf_backend.py')
    hf = importlib.util.module_from_spec(spec); spec.loader.exec_module(hf)
    if not hasattr(hf, 'compile_ner_grammar'):
        raise ValueError('Uncorrected historical grammar is not a final backend')
    return hf


def validate_binding(condition, binding, hf):
    if condition not in CONDITIONS:
        raise ValueError('Unknown fixed local final condition')
    task, model, constrained = CONDITIONS[condition]
    if (binding.get('task'), binding.get('model_key'), binding.get('shots')) != (task, model, 4):
        raise ValueError('Condition identity differs')
    config = binding['backend_config']
    expected_model = {'qwen': 'Qwen/Qwen2.5-7B-Instruct', 'sherkala': 'inception42/Llama-3.1-Sherkala-8B-Chat'}[model]
    if config['model_id'] != expected_model:
        raise ValueError('Condition model identity differs')
    if config['schema']['enabled'] != constrained:
        raise ValueError('Schema condition differs')
    validation = copy.deepcopy(config)
    if task == 'sa':
        if config['generation_template']['max_new_tokens'] != 32 or constrained:
            raise ValueError('Accepted SA cap32 ordinary generation required')
        validation['generation_template']['max_new_tokens'] = 768
    hf.validate_config(validation)
    evidence = binding['generation_evidence']
    value = evidence['template']
    if digest(value) != evidence['template_sha256'] or not evidence.get('source_file_sha256') or value != config['generation_template']:
        raise ValueError('Generation configuration differs from accepted evidence')
    prompt = binding['prompt_contract']
    if len(prompt['demonstrations']) != 4 or any(not d['id'].startswith(task + ':train:') for d in prompt['demonstrations']):
        raise ValueError('Exactly four frozen training demonstrations required')
    if hashlib.sha256(prompt['definition'].encode('utf-8')).hexdigest() != prompt['definition_source']['sha256']:
        raise ValueError('Definition text/source differs')
    return task, model


def sa_messages(text, demos, definition):
    messages = [{'role': 'system', 'content': definition}]
    for row in demos:
        if row['label'] not in ('0', '1'):
            raise ValueError('Frozen SA demonstration label must be original string0/1')
        messages += [{'role': 'user', 'content': 'Review (JSON string):\n' + json.dumps(row['visible_text'], ensure_ascii=False)},
                     {'role': 'assistant', 'content': json.dumps({'label': 'positive' if row['label'] == '1' else 'negative'},
                                                               ensure_ascii=False, separators=(',', ':'))}]
    messages.append({'role': 'user', 'content': 'Review (JSON string):\n' + json.dumps(text, ensure_ascii=False)})
    return messages


def request_manifest(item, task, run_hash, condition, prompts, allowed_types):
    """Portable final request identity; same parser/merge, POSIX source keys.

    The older development assembly's source-path key formatting depends on OS.
    Final requests therefore bind the shared portable output-adapter contract.
    """
    windows = item['windows'] if task == 'ner' else [{'start': 0, 'end': 1}]
    if len(windows) != len(prompts): raise ValueError('Missing native prompts')
    visible = {'id': item['id'], 'tokens': item['tokens'], 'windows': [(w['start'], w['end']) for w in windows]} if task == 'ner' else \
              {'id': item['id'], 'visible_text': item['visible_text']}
    contract = digest({'task': task, 'allowed_types': allowed_types if task == 'ner' else ['negative','positive'],
                       'sources': output_adapter.sources(), 'policy': 'final_shared_output_adapter_v1'})
    result = []
    for index, (window, prompt) in enumerate(zip(windows, prompts)):
        execution._validate_prompt(prompt)
        entry = {'id': item['id'], 'window_index': index, 'run_hash': run_hash, 'condition': condition,
                 'visible_input_sha256': digest(visible), 'contract_sha256': contract,
                 'messages_sha256': digest(prompt['messages']),
                 'rendered_prompt_sha256': execution.assembly.text_hash(prompt['rendered_prompt']),
                 'input_token_ids_sha256': digest(prompt['input_token_ids'])}
        if task == 'ner':
            if window['tokens'] != item['tokens'][window['start']:window['end']]: raise ValueError('Window token mismatch')
            entry.update(start=window['start'], end=window['end'], window_tokens_sha256=digest(window['tokens']))
        entry['request_identity'] = digest(entry); result.append(entry)
    return result


def prepare(freeze_path, freeze_sha, common_receipt, common_receipt_sha, condition, output):
    if sha(freeze_path) != freeze_sha:
        raise ValueError('Freeze A hash mismatch')
    freeze = read(freeze_path); common.check_freeze_a(freeze)
    bindings = bound_read(freeze['llm_bindings'])
    if condition not in freeze['llm_conditions'] or condition not in bindings['conditions']:
        raise PermissionError('Condition absent from Freeze A')
    binding = bindings['conditions'][condition]
    hf = load_hf(binding, preparation=True); task, _ = validate_binding(condition, binding, hf)
    if sha(common_receipt) != common_receipt_sha:
        raise ValueError('Common preparation receipt hash mismatch')
    receipt = read(common_receipt)
    if receipt['freeze_a_sha256'] != freeze_sha or receipt['status'] != 'PREPARED_FOR_FREEZE_B_NOT_INFERENCE_AUTHORIZATION':
        raise ValueError('Common panels belong to another Freeze A')
    panel = bound_read(receipt['panels'][task])
    if panel['freeze_a_sha256'] != freeze_sha or panel['task'] != task or panel['split'] != 'test' or \
            len(panel['expected_ids']) != 500 or [x['id'] for x in panel['items']] != panel['expected_ids']:
        raise ValueError('Common final panel identity/coverage differs')
    config = binding['backend_config']
    os.environ.update(CUDA_VISIBLE_DEVICES='', USE_TORCH='0', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    from transformers import AutoTokenizer
    tokenizer_path = binding.get('preparation_tokenizer_path', config['tokenizer_path'])
    hf.verify_files(tokenizer_path, config['tokenizer_files'])
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True, trust_remote_code=False, use_fast=True)
    header = {'format': 'kazakh_final_local_llm_v1', 'condition': condition, 'task': task,
              'freeze_a_sha256': freeze_sha, 'panel_sha256': receipt['panels'][task]['sha256'],
              'ordered_ids_sha256': digest(panel['expected_ids']), 'expected_ids': panel['expected_ids'],
              'binding': binding, 'binding_sha256': digest(binding), 'runtime_sources': sources(),
              'allowed_types': bindings['allowed_ner_types'], 'gold_present': False}
    run_hash = digest(header)
    items = []
    prompt_spec = binding['prompt_contract']
    for source in panel['items']:
        if any(k in source for k in ('label', 'labels', 'gold', 'gold_spans')):
            raise ValueError('Inference input contains query gold')
        item = {k: source[k] for k in ('id', 'original_text', 'cluster_key')}
        if task == 'ner':
            item.update(tokens=source['tokens'], windows=source['windows'])
            messages = [native_inputs.build_messages(w['tokens'], prompt_spec['definition'], prompt_spec['demonstrations'], header['allowed_types'])
                        for w in source['windows']]
        else:
            item['visible_text'] = source['visible_text']
            messages = [sa_messages(source['visible_text'], prompt_spec['demonstrations'], prompt_spec['definition'])]
        prompts = []
        for msg in messages:
            rendered = tokenizer.apply_chat_template(msg, chat_template=config['chat_template'], tokenize=False, add_generation_prompt=True)
            ids = tokenizer(rendered, add_special_tokens=False, truncation=False)['input_ids']
            prompts.append({'messages': msg, 'rendered_prompt': rendered, 'input_token_ids': ids})
        item['prepared_prompts'] = prompts
        item['manifest'] = request_manifest(item, task, run_hash, condition, prompts, header['allowed_types'])
        items.append(item)
    package = {**header, 'run_hash': run_hash, 'items': items,
               'request_count': sum(len(x['manifest']) for x in items)}
    write_new(output, package)
    return {'path': str(Path(output).resolve()), 'sha256': sha(output), 'condition': condition,
            'requests': package['request_count'], 'ordered_ids_sha256': header['ordered_ids_sha256']}


def load_final_package(freeze_path, freeze_sha, condition):
    if sha(freeze_path) != freeze_sha:
        raise ValueError('Freeze B digest mismatch')
    freeze = read(freeze_path)
    if freeze.get('scope') != 'final' or freeze.get('status') != 'frozen_for_final_inference' or \
            freeze.get('final_test_access_authorized') is not True or not freeze.get('decision_reference'):
        raise PermissionError('Actual final Freeze B required before package access')
    if condition not in freeze.get('llm_conditions', []) or freeze.get('llm_runtime_sources') != sources():
        raise ValueError('Condition/code differs from Freeze B')
    package = bound_read(freeze['llm_prepared_packages'][condition])
    if package['condition'] != condition or package['runtime_sources'] != sources() or package['binding_sha256'] != digest(package['binding']):
        raise ValueError('Prepared package binding differs')
    panel_spec = freeze['panels'][package['task']]
    if package['panel_sha256'] != panel_spec['sha256'] or package['ordered_ids_sha256'] != panel_spec['ordered_ids_sha256']:
        raise ValueError('LLM/encoder common panel differs')
    if package['freeze_a_sha256'] != freeze['freeze_a_sha256'] or len(package['expected_ids']) != 500 or \
            [x['id'] for x in package['items']] != package['expected_ids']:
        raise ValueError('Final panel/Freeze A/coverage mismatch')
    if len(set(package['expected_ids'])) != 500 or any(not x.startswith(package['task'] + ':test:') for x in package['expected_ids']):
        raise ValueError('Duplicate/out-of-scope final IDs')
    total = 0
    for item in package['items']:
        if any(k in item for k in ('label','labels','gold','gold_spans')) or len(item['manifest']) != len(item['prepared_prompts']):
            raise ValueError('Query gold or incomplete prompts in package')
        wanted = request_manifest(item, package['task'], package['run_hash'], condition, item['prepared_prompts'], package['allowed_types'])
        if wanted != item['manifest']: raise ValueError('Final request manifest differs')
        if package['task'] == 'sa' and len(item['manifest']) != 1:
            raise ValueError('SA requires one request')
        for manifest, prompt in zip(item['manifest'], item['prepared_prompts']):
            execution._validate_prompt(prompt)
            if manifest['id'] != item['id'] or manifest['condition'] != condition or \
                    manifest['request_identity'] != digest({k:v for k,v in manifest.items() if k != 'request_identity'}) or \
                    manifest['input_token_ids_sha256'] != digest(prompt['input_token_ids']) or \
                    manifest['messages_sha256'] != digest(prompt['messages']) or \
                    manifest['rendered_prompt_sha256'] != execution.assembly.text_hash(prompt['rendered_prompt']) or \
                    sum(m['role'] == 'assistant' for m in prompt['messages']) != 4:
                raise ValueError('Native final request identity differs')
            total += 1
    if total != package['request_count']: raise ValueError('Declared request count differs')
    return freeze, package


def parse_sa_record(record):
    return output_adapter.parse_final_sa(record['raw'], truncated=record['truncated'], refusal=record['refusal'],
        runtime_error=bool(record.get('runtime_error') or record.get('execution_failure_reasons')),
        terminal_failed=record['status'] == 'terminal_failure')


def generic_observation(hf, observation, prompt, config):
    if config['generation_template']['max_new_tokens'] == 768:
        return hf.validate_observation(observation, prompt, config)
    ids = observation['generated_token_ids']; cap = config['generation_template']['max_new_tokens']
    if not ids or len(ids) > cap or any(type(i) is not int or i < 0 for i in ids):
        raise ValueError('SA output token contract differs')
    if [x for step in observation['steps'] for x in step['selected_token_ids']] != ids:
        raise ValueError('SA trace/return token mismatch')
    actual = dict(observation['trace_metadata']['actual_generation_config'])
    if actual.pop('max_length') != len(prompt['input_token_ids']) + cap or actual != config['generation_template']:
        raise ValueError('SA effective generation differs')
    if observation['trace_metadata']['native_logits_processors'] != ['RepetitionPenaltyLogitsProcessor']:
        raise ValueError('SA logits processors differ')
    eos = config['generation_template']['eos_token_id']
    if len(ids) < cap and ids[-1] not in eos:
        raise ValueError('SA stopped before cap without native EOS')
    return len(ids) == cap and ids[-1] not in eos, hf.execution_failures(observation['steps'], False)


def check_admission(admission, freeze_sha, condition, package, output):
    now = datetime.datetime.now(datetime.timezone.utc)
    def aware(value):
        parsed = datetime.datetime.fromisoformat(value.replace('Z', '+00:00'))
        if parsed.tzinfo is None: raise ValueError('Admission timestamp must be aware')
        return parsed
    if admission.get('status') != 'approved_final_local_llm_inference' or admission.get('freeze_sha256') != freeze_sha or \
            admission.get('condition') != condition or admission.get('hostname') != socket.gethostname() or \
            not aware(admission['issued_utc']) <= now < aware(admission['expires_utc']):
        raise PermissionError('Matching current final resource admission required')
    if Path(admission['output_directory']).resolve() != Path(output).resolve() or \
            admission.get('gpu_uuid') != package['binding']['backend_config']['gpu_uuid'] or \
            admission.get('maximum_model_calls') != package['request_count']:
        raise PermissionError('Output/device/call budget differs')
    if Path(package['binding']['backend_config']['trace_dir']).resolve() != (Path(output) / 'traces').resolve():
        raise PermissionError('Trace location differs from frozen backend config')


class FinalBackend:
    mode = 'local_hf'

    def __init__(self, package, admission, freeze_sha, output):
        self.package = package; self.admission = admission; self.freeze_sha = freeze_sha; self.output = output
        self.config = package['binding']['backend_config']; self.calls = 0
        self.hf = load_hf(package['binding']); validate_binding(package['condition'], package['binding'], self.hf)
        check_admission(admission, freeze_sha, package['condition'], package, output)
        os.environ['CUDA_VISIBLE_DEVICES'] = self.config['gpu_uuid']; os.environ.pop('USE_TORCH', None)
        os.environ['NER_ISOLATED_PROCESS'] = '1'; os.environ['HF_HUB_OFFLINE'] = '1'; os.environ['TRANSFORMERS_OFFLINE'] = '1'
        self.hf.verify_gpu_exclusive(self.config['gpu_uuid'])
        for name, expected in self.config['packages'].items():
            if importlib.metadata.version(name) != expected:
                raise ValueError('Accepted runtime package version differs')
        self.files = {'model': self.hf.verify_files(self.config['model_path'], self.config['model_files'], model=True),
                      'tokenizer': self.hf.verify_files(self.config['tokenizer_path'], self.config['tokenizer_files'])}
        self.hf.verify_gpu_exclusive(self.config['gpu_uuid'])
        self.runtime = self.hf._HFRuntime(self.config)
        self.identity = {'kind': 'final_local_hf', 'backend_config_sha256': digest(self.config),
                         'gpu_uuid': self.config['gpu_uuid'], 'runtime_sources': package['binding']['accepted_runtime']['files'],
                         'grammar_implementation': self.runtime.grammar_receipt,
                         'freeze_b_sha256': freeze_sha}
        self.load_receipt = {'verified_files': self.files, 'runtime': self.runtime.load_receipt}

    def generate(self, request):
        check_admission(self.admission, self.freeze_sha, self.package['condition'], self.package, self.output)
        manifest, prompt = request['manifest'], request['prepared_prompt']
        if manifest['condition'] != self.package['condition'] or manifest['id'] not in self.package['expected_ids']:
            raise PermissionError('Request outside final admitted package')
        if self.calls >= self.admission['maximum_model_calls']:
            raise PermissionError('Final call ceiling reached')
        self.runtime.verify_input(prompt, self.config); self.runtime.check_resource(self.config['gpu_uuid'])
        trace = Path(self.output) / 'traces' / (manifest['request_identity'] + '.trace.jsonl')
        trace.parent.mkdir(parents=True, exist_ok=True)
        if trace.exists(): raise FileExistsError('Never overwrite an existing generation trace')
        self.calls += 1
        observed = self.runtime.generate(prompt, self.config, trace)
        reasons = list(observed.get('runtime_reasons', [])); truncated = False
        try:
            if not reasons: truncated, reasons = generic_observation(self.hf, observed, prompt, self.config)
        except Exception as error:
            reasons.append('post_generation_contract:' + type(error).__name__ + ': ' + str(error))
        return {'kind': 'runtime_failure' if reasons else 'completed', 'raw': observed.get('raw', ''),
                'generated_token_ids': observed.get('generated_token_ids', []), 'truncated': truncated, 'refusal': False,
                'runtime_error': '; '.join(reasons) if reasons else None, 'execution_failure_reasons': reasons,
                'metadata': {'backend_identity': self.identity, 'backend_config_sha256': digest(self.config),
                             'input_token_ids_sha256': digest(prompt['input_token_ids']),
                             'trace_path': str(trace), 'trace_sha256': sha(trace) if trace.exists() else None,
                             'observation': observed}}


def execute(freeze_path, freeze_sha, condition, admission_path, output):
    freeze, package = load_final_package(freeze_path, freeze_sha, condition)
    admission = read(admission_path); check_admission(admission, freeze_sha, condition, package, output)
    store = EvidenceStore(output)
    with process_lock(Path(output) / '.execution.lock'):
        store.bind({'freeze_sha256': freeze_sha, 'package_sha256': freeze['llm_prepared_packages'][condition]['sha256'],
                    'runtime_sources': sources(), 'local_transport_retries': 0})
        backend = None; predictions = []; total_attempts = 0
        config = package['binding']['backend_config']
        budget = {'context_limit_tokens': config['context_limit'], 'reserved_output_tokens': config['generation_template']['max_new_tokens']}
        class LazyBackend:
            mode = 'local_hf'
            identity = {'backend_config_sha256': digest(config), 'freeze_b_sha256': freeze_sha}
            def generate(self, request):
                nonlocal backend
                if backend is None:
                    backend = FinalBackend(package, admission, freeze_sha, output)
                    write_evidence(Path(output) / 'load_receipts' / (uuid.uuid4().hex + '.json'), backend.load_receipt)
                return backend.generate(request)
        lazy = LazyBackend()
        for item in package['items']:
            records = []
            for manifest, prompt in zip(item['manifest'], item['prepared_prompts']):
                record = execution._execute_window(store, manifest, prompt, budget, lazy, None)
                records.append(record)
                total_attempts += len(record['attempts'])
                if record.get('runtime_error') or record.get('execution_failure_reasons'):
                    raise execution.RuntimeWindowFailure('Saved final runtime failure; investigate before any further calls')
            if package['task'] == 'ner':
                adapted = [{**record, 'runtime_error': bool(record.get('runtime_error') or record.get('execution_failure_reasons')),
                            'raw_request_refs': [record['request_identity']]} for record in records]
                parsed = output_adapter.assemble_final_ner(item['id'], item['tokens'], item['windows'], adapted, package['allowed_types'])
            else:
                if len(records) != 1: raise ValueError('SA requires exactly one complete record')
                parsed = parse_sa_record(records[0])
            prediction = {'id': item['id'], 'condition': condition, 'panel_sha256': package['panel_sha256'],
                          'cluster_key': item['cluster_key'], **parsed,
                          'raw_request_refs': [m['request_identity'] for m in item['manifest']]}
            write_evidence(Path(output) / 'items' / (digest(item['id']) + '.json'), prediction)
            predictions.append(prediction)
        if [r['id'] for r in predictions] != package['expected_ids']:
            raise ValueError('Incomplete final prediction coverage')
        write_evidence(Path(output) / 'PREDICTIONS.json', {'condition': condition, 'panel_sha256': package['panel_sha256'],
                       'freeze_sha256': freeze_sha, 'rows': predictions})
        write_evidence(Path(output) / 'COMPLETE.json', {'status': 'complete', 'items': len(predictions),
                       'requests': package['request_count'], 'model_calls_total': total_attempts,
                       'gold_accessed_by_runner': False})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); sub = parser.add_subparsers(dest='command', required=True)
    prep = sub.add_parser('prepare'); prep.add_argument('--freeze-a', required=True); prep.add_argument('--freeze-a-sha256', required=True)
    prep.add_argument('--common-receipt', required=True); prep.add_argument('--common-receipt-sha256', required=True)
    prep.add_argument('--condition', choices=CONDITIONS, required=True); prep.add_argument('--output', required=True)
    run = sub.add_parser('execute'); run.add_argument('--freeze', required=True); run.add_argument('--freeze-sha256', required=True)
    run.add_argument('--condition', choices=CONDITIONS, required=True); run.add_argument('--admission', required=True); run.add_argument('--output', required=True)
    args = parser.parse_args()
    if args.command == 'prepare':
        print(json.dumps(prepare(args.freeze_a, args.freeze_a_sha256, args.common_receipt, args.common_receipt_sha256, args.condition, args.output)))
    else:
        execute(args.freeze, args.freeze_sha256, args.condition, args.admission, args.output)
