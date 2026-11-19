"""Explicit host transport projections of one already-approved scientific Freeze B.

Reads only the master freeze and an explicit path map. Never opens panels, gold,
models, tokenizers, or prepared requests. File hashes and all scientific fields
remain identical. Actual destination files are verified by the inference runners.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path, PurePosixPath, PureWindowsPath
import re

PATTERNS = [r'/panels/(ner|sa)/path', r'/selected_checkpoints/path', r'/runtime_bindings/path',
            r'/llm_prepared_packages/(ner|sa)_(qwen|sherkala)_(ordinary4|schema4)/path',
            r'/gemma_prepared_packages/(ner|sa)_gemma_ordinary4/path']


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8')


def digest(value): return hashlib.sha256(canonical(value)).hexdigest()
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))


def allowed(pointer):
    return any(re.fullmatch(pattern, pointer) for pattern in PATTERNS)


def normalized_scientific(value, prefix=''):
    if isinstance(value, dict):
        return {key: ('<HOST_FILE_LOCATION>' if allowed(prefix + '/' + key)
                      else normalized_scientific(item, prefix + '/' + key)) for key, item in value.items()}
    if isinstance(value, list): return [normalized_scientific(item, prefix + '/' + str(i)) for i, item in enumerate(value)]
    return value


def project(master, master_sha, mapping):
    if master.get('scope') != 'final' or master.get('status') != 'frozen_for_final_inference' or \
            master.get('final_test_access_authorized') is not True or not master.get('decision_reference'):
        raise PermissionError('Only an externally approved actual scientific Freeze B can be projected')
    if 'host_projection' in master:
        raise ValueError('Project directly from the scientific master, never another projection')
    if set(mapping) != {'host', 'paths'} or mapping['host'] not in ('local', 'server') or not isinstance(mapping['paths'], dict):
        raise ValueError('Explicit local/server host and pointer→absolute-path map required')
    result = copy.deepcopy(master); changes = []
    for pointer, destination in mapping['paths'].items():
        if not allowed(pointer): raise ValueError('Non-transport field cannot be changed: ' + pointer)
        if not isinstance(destination, str) or not (PurePosixPath(destination).is_absolute() or PureWindowsPath(destination).is_absolute()):
            raise ValueError('Explicit absolute destination required')
        parts = pointer.split('/')[1:]; node = result
        for key in parts[:-1]: node = node[key]
        old = node[parts[-1]]
        if not isinstance(old, str) or not old:
            raise ValueError('Original bound path must exist')
        if not isinstance(node.get('sha256'), str) or len(node['sha256']) != 64:
            raise ValueError('Relocated file must retain an exact 64-character content SHA')
        node[parts[-1]] = destination
        changes.append({'pointer': pointer, 'original': old, 'destination': destination,
                        'unchanged_file_sha256': node['sha256']})
    scientific_sha = digest(normalized_scientific(master))
    if digest(normalized_scientific(result)) != scientific_sha:
        raise ValueError('Scientific payload changed during path projection')
    result['host_projection'] = {'format': 'kazakh_final_host_transport_v1', 'host': mapping['host'],
        'scientific_parent_sha256': master_sha, 'scientific_payload_sha256': scientific_sha,
        'path_map_sha256': digest(mapping), 'source_sha256': sha(__file__),
        'changes': changes, 'panels_or_requests_opened': False,
        'policy': 'Only host file locations changed; all referenced byte hashes, scientific settings, conditions, membership, and selection preserved.'}
    return result


def verify(master, master_sha, projected):
    info = projected['host_projection']
    if info['scientific_parent_sha256'] != master_sha or info['source_sha256'] != sha(__file__):
        raise ValueError('Master/source identity differs')
    mapping = {'host': info['host'], 'paths': {row['pointer']: row['destination'] for row in info['changes']}}
    if len(mapping['paths']) != len(info['changes']) or projected != project(master, master_sha, mapping):
        raise ValueError('Projection contains undocumented changes')
    return True


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--master', required=True, type=Path)
    parser.add_argument('--master-sha256', required=True)
    parser.add_argument('--path-map', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if sha(args.master) != args.master_sha256: raise ValueError('Scientific master bytes changed')
    master = read(args.master); projected = project(master, args.master_sha256, read(args.path_map))
    verify(master, args.master_sha256, projected)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('xb') as stream: stream.write(canonical(projected) + b'\n')
    print(json.dumps({'status': 'HOST_PROJECTION_CREATED_NO_DATA_READ', 'path': str(args.output),
                      'sha256': sha(args.output), 'scientific_parent_sha256': args.master_sha256}))
