"""After actual Freeze A only: copy prepared gold-free inputs to a server ZIP.

Does not authorize inference, open original test sources, copy gold, load models,
or issue API requests. Writes draft master-B parts and explicit host path maps.
"""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import sys
import zipfile

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / 'llm'))
import final_local_llm as f
import encoder_runner as encoder


def check_package(package, condition, freeze_sha, panel_spec):
    if package.get('condition') != condition or package.get('freeze_a_sha256') != freeze_sha or \
            package.get('runtime_sources') != f.sources() or package.get('gold_present') is not False:
        raise ValueError('Native package does not match fixed final preparation')
    if package['binding_sha256'] != f.digest(package['binding']) or package['panel_sha256'] != panel_spec['sha256'] or \
            package['ordered_ids_sha256'] != panel_spec['ordered_ids_sha256']:
        raise ValueError('Native package/common panel binding differs')
    if len(package['expected_ids']) != 500 or len(set(package['expected_ids'])) != 500 or \
            f.digest(package['expected_ids']) != panel_spec['ordered_ids_sha256'] or \
            [row['id'] for row in package['items']] != package['expected_ids'] or \
            any(not rid.startswith(f.CONDITIONS[condition][0] + ':test:') for rid in package['expected_ids']):
        raise ValueError('Native package does not cover exactly500 ordered common items')
    if any(any(key in row for key in ('label', 'labels', 'gold', 'gold_spans')) for row in package['items']):
        raise ValueError('Query gold found in inference package')
    actual = sum(len(row['manifest']) for row in package['items'])
    if actual != package['request_count']: raise ValueError('Native request coverage mismatch')
    hf = f.load_hf(package['binding'], preparation=True)
    f.validate_binding(condition, package['binding'], hf)
    if package['binding']['accepted_runtime']['files']['escape_safe_ner_grammar.py'] != \
            '0ccb5e46a05dee7146e0de875da758532e74e1b5036a9dc292692fb9ce33c33c':
        raise ValueError('Accepted corrected grammar helper is not pinned')


def package_inputs(args):
    # This check occurs before reading receipt, panels, native packages, or registry.
    if f.sha(args.freeze_a) != args.freeze_a_sha256: raise ValueError('Freeze A file SHA differs')
    freeze = f.read(args.freeze_a); f.common.check_freeze_a(freeze)
    if args.output.exists() or args.output.with_suffix('.zip').exists():
        raise FileExistsError('Fresh no-overwrite payload destination required')
    server_root = PurePosixPath(args.server_input_root)
    if not server_root.is_absolute() or '..' in server_root.parts:
        raise ValueError('Explicit absolute server input destination required')
    if f.sha(args.common_receipt) != args.common_receipt_sha256: raise ValueError('Common preparation receipt changed')
    receipt = f.read(args.common_receipt)
    if receipt['freeze_a_sha256'] != args.freeze_a_sha256 or receipt['status'] != 'PREPARED_FOR_FREEZE_B_NOT_INFERENCE_AUTHORIZATION':
        raise ValueError('Common preparation is not bound to actual Freeze A')
    paths = {}; master = {'status': 'DRAFT_B_PARTS_NOT_INFERENCE_AUTHORIZATION', 'freeze_a_sha256': args.freeze_a_sha256,
        'panels': {}, 'gold': {}, 'scientific_protocol': freeze['scientific_protocol'],
        'llm_prepared_packages': {}, 'llm_conditions': sorted(f.CONDITIONS),
        'llm_runtime_sources': f.sources(), 'runtime_sources': encoder.sources()}
    pending = []
    for task in ('ner', 'sa'):
        spec = receipt['panels'][task]; source = Path(spec['path']); panel = f.bound_read(spec)
        if panel.get('split') != 'test' or panel.get('task') != task or panel.get('freeze_a_sha256') != args.freeze_a_sha256 or \
                len(panel['expected_ids']) != 500 or len(set(panel['expected_ids'])) != 500 or \
                [row['id'] for row in panel['items']] != panel['expected_ids'] or \
                f.digest(panel['expected_ids']) != spec['ordered_ids_sha256']:
            raise ValueError('Common panel provenance/membership differs')
        if any(any(key in item for key in ('label', 'labels', 'gold', 'gold_spans')) for item in panel['items']):
            raise ValueError('Query gold found in common inference panel')
        relative = 'common500/' + task + '_PANEL.json'; pending.append((source, relative))
        master['panels'][task] = {key: spec[key] for key in ('path', 'sha256', 'ordered_ids_sha256')}
        master['gold'][task] = dict(spec['gold'])  # metadata reference only; gold bytes never enter this ZIP
        paths['/panels/' + task + '/path'] = str(server_root / relative)
    if set(freeze['llm_conditions']) != set(f.CONDITIONS): raise ValueError('Actual Freeze A must fix all six local conditions')
    for condition in sorted(f.CONDITIONS):
        source = args.native_directory / (condition + '.json'); package = f.read(source)
        check_package(package, condition, args.freeze_a_sha256, master['panels'][f.CONDITIONS[condition][0]])
        relative = 'native/' + condition + '.json'; pending.append((source, relative))
        master['llm_prepared_packages'][condition] = {'path': str(source.resolve()), 'sha256': f.sha(source)}
        paths['/llm_prepared_packages/' + condition + '/path'] = str(server_root / relative)
    registry = f.read(args.selected_checkpoints); selected = encoder.validate_registry(registry)
    bindings = f.read(args.encoder_bindings)
    if bindings['selected_registry_sha256'] != f.sha(args.selected_checkpoints) or set(bindings['conditions']) != set(selected):
        raise ValueError('Encoder runtime bindings differ from the12 accepted selections')
    for key, source in [('selected_checkpoints', args.selected_checkpoints), ('runtime_bindings', args.encoder_bindings)]:
        relative = 'registries/' + ('SELECTED_CHECKPOINTS.json' if key == 'selected_checkpoints' else 'RUNTIME_BINDINGS.json')
        pending.append((source, relative)); master[key] = {'path': str(source.resolve()), 'sha256': f.sha(source)}
        paths['/' + key + '/path'] = str(server_root / relative)
    master['encoder_conditions'] = sorted(selected)
    args.output.mkdir(parents=True)
    records = []
    for source, relative in pending:
        data = source.read_bytes(); destination = args.output / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open('xb') as stream: stream.write(data)
        records.append({'relative_path': relative, 'local_source': str(source.resolve()),
            'server_destination': str(server_root / relative), 'bytes': len(data),
            'sha256': hashlib.sha256(data).hexdigest()})
    f.write_new(args.output / 'MASTER_B_PARTS_DRAFT.json', master)
    f.write_new(args.output / 'LOCAL_PATH_MAP.json', {'host': 'local', 'paths': {}})
    f.write_new(args.output / 'SERVER_PATH_MAP.json', {'host': 'server', 'paths': paths})
    manifest = {'status': 'PREPARED_INPUT_TRANSPORT_NOT_INFERENCE_AUTHORITY', 'freeze_a_sha256': args.freeze_a_sha256,
        'source_sha256': f.sha(__file__), 'files': records, 'gold_copied': False, 'model_calls': 0,
        'common_panel_ids_and_bytes_shared_across_hosts': True,
        'note': 'Root must combine the scientific decision/API/Gemma admissions with MASTER_B_PARTS_DRAFT and issue actual masterB, then explicit host projections.'}
    f.write_new(args.output / 'INPUT_TRANSPORT_MANIFEST.json', manifest)
    archive = args.output.with_suffix('.zip')
    with zipfile.ZipFile(archive, 'x', compression=zipfile.ZIP_DEFLATED) as zipped:
        for record in records: zipped.write(args.output / record['relative_path'], record['relative_path'])
        for name in ['MASTER_B_PARTS_DRAFT.json', 'LOCAL_PATH_MAP.json', 'SERVER_PATH_MAP.json', 'INPUT_TRANSPORT_MANIFEST.json']:
            zipped.write(args.output / name, name)
    with zipfile.ZipFile(archive) as zipped:
        if zipped.testzip() is not None: raise ValueError('Transport ZIP CRC mismatch')
        for record in records:
            if hashlib.sha256(zipped.read(record['relative_path'])).hexdigest() != record['sha256']:
                raise ValueError('Transport ZIP content mismatch')
    result = {'status': 'INPUT_TRANSPORT_READY_NO_INFERENCE', 'archive': str(archive.resolve()),
              'sha256': f.sha(archive), 'bytes': archive.stat().st_size, 'input_files': len(records),
              'master_b_parts': str(args.output / 'MASTER_B_PARTS_DRAFT.json'),
              'server_path_map': str(args.output / 'SERVER_PATH_MAP.json')}
    f.write_new(args.output.with_name(args.output.name + '_RECEIPT.json'), result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['freeze-a', 'common-receipt', 'native-directory', 'selected-checkpoints', 'encoder-bindings', 'output']:
        parser.add_argument('--' + name, required=True, type=Path)
    parser.add_argument('--freeze-a-sha256', required=True)
    parser.add_argument('--common-receipt-sha256', required=True)
    parser.add_argument('--server-input-root', required=True)
    package_inputs(parser.parse_args())
