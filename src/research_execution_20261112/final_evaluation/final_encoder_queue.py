"""Sequential final encoder queue with fresh per-condition resource admission.

Exactly8 local or4 server selected checkpoints, no training or checkpoint choice.
An actual Freeze B is checked before any common final panel is opened.
"""
import argparse
import datetime
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import encoder_runner as r

sys.path.insert(0, str(r.PROJECT / 'ner_pipeline_20261111'))
from storage import process_lock

GPU_LOCAL = 'GPU-fce273fb-c721-2acd-a940-f852f17ab380'
GPU_SERVER = 'GPU-8f767e03-8d0c-71f8-4d6d-cb3a88c92169'


def selected_conditions(freeze_path, freeze_sha, location):
    r.require(r.sha(freeze_path) == freeze_sha, 'Freeze B SHA mismatch')
    freeze = r.read(freeze_path)
    r.require(freeze.get('scope') == 'final', 'Final-only queue requires actual final scope')
    conditions = freeze.get('encoder_conditions', [])
    r.require(len(conditions) == len(set(conditions)) == 12, 'Exactly12 frozen encoder conditions required')
    r.check_gate(freeze, conditions[0])  # before opening any bound panel/registry
    selected = r.validate_registry(r.read(r.bound_file(freeze['selected_checkpoints'])))
    bindings = r.read(r.bound_file(freeze['runtime_bindings']))
    r.require(set(conditions) == set(selected) == set(bindings['conditions']), 'Encoder matrix/bindings differ')
    r.require(bindings['selected_registry_sha256'] == freeze['selected_checkpoints']['sha256'], 'Registry binding differs')
    chosen = sorted(c for c in conditions if bindings['conditions'][c]['location'] == location)
    r.require(len(chosen) == (8 if location == 'local' else 4), 'Expected8 local or4 server checkpoints')
    if location == 'server': r.require(all(c.startswith('sa_') for c in chosen), 'Only four accepted SA checkpoints belong on server')
    return chosen


def expiry(value):
    end = r.aware(value)
    r.require(r.now() < end <= r.now() + datetime.timedelta(hours=24), 'Explicit future expiry within24hours required')
    return end.isoformat()


def observe(gpu, location):
    result = subprocess.check_output(['nvidia-smi', '--query-gpu=uuid,name,memory.free,utilization.gpu', '--format=csv,noheader,nounits'], text=True)
    rows = [parts for parts in (line.split(',') for line in result.splitlines()) if parts[0].strip() == gpu]
    r.require(len(rows) == 1, 'Selected physical GPU missing')
    row = [part.strip() for part in rows[0]]
    r.require(int(row[2]) >= 8192 and int(row[3]) <= 10, 'GPU not sufficiently idle/free')
    r.require(('5060 Ti' in row[1]) if location == 'local' else ('A6000' in row[1]), 'Unexpected GPU hardware class')
    active = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader,nounits'], text=True)
    pids = [int(parts[1].strip()) for parts in (line.split(',') for line in active.splitlines()) if parts[0].strip() == gpu]
    names = r.windows_process_names(pids) if location == 'local' else {}
    policy = {'gpu_process_policy': 'windows_explicit_gui_baseline', 'gpu_gui_baseline': {str(k): v for k,v in names.items()}} if location == 'local' else {}
    r.check_gpu_processes(pids, names, policy)
    return {**policy, 'gpu_name': row[1], 'observed_gpu_free_MiB': int(row[2]),
            'observed_gpu_utilization_pct': int(row[3]), 'visible_pids': pids, 'utc': r.now().isoformat()}


def completed_reuse(output, run, panel):
    path = output / 'COMPLETE.json'
    if not path.exists(): return False
    complete = r.read(path); identity = r.read(output / 'RUN.json')
    r.require(all(identity.get(key) == value for key,value in run.items()), 'Completed encoder run identity differs')
    r.require(identity['run_sha256'] == r.digest({k:v for k,v in identity.items() if k != 'run_sha256'}), 'Run identity digest differs')
    r.require(complete['status'] == 'complete_predictions_pending_independent_scoring' and complete['n'] == 500 and
              complete['ordered_ids'] == panel['expected_ids'] and complete['model_signature_unchanged'] is True and
              complete['training_started'] is False and complete['optimizer_updates'] == 0, 'Incomplete or altered encoder completion')
    digests = []
    for item in panel['items']:
        record = r.read(output / 'items' / r.digest(item['id']) / 'OUTPUT.json')
        wanted = r.digest({k:v for k,v in record.items() if k != 'record_sha256'})
        r.require(record['id'] == item['id'] and record['input_sha256'] == r.digest(item) and
                  record['run_sha256'] == identity['run_sha256'] and record['record_sha256'] == wanted, 'Saved encoder evidence changed')
        digests.append(wanted)
    r.require(digests == complete['record_sha256'] and complete['run_sha256'] == identity['run_sha256'], 'Saved completion manifest differs')
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['verify', 'execute'])
    parser.add_argument('--freeze', required=True, type=Path); parser.add_argument('--freeze-sha256', required=True)
    parser.add_argument('--location', choices=['local', 'server'], required=True)
    parser.add_argument('--output-root', required=True, type=Path)
    parser.add_argument('--expires-utc', required=True)
    args = parser.parse_args(); end = expiry(args.expires_utc)
    conditions = selected_conditions(args.freeze, args.freeze_sha256, args.location)
    r.require((os.name == 'nt') == (args.location == 'local'), 'Queue location differs from actual host')
    gpu = GPU_LOCAL if args.location == 'local' else GPU_SERVER
    output_root = args.output_root.resolve()
    r.require(output_root.is_relative_to(r.HERE.resolve()) if args.location == 'local' else
              str(output_root).startswith('/home/jovyan/work/server_kazakh_final_'), 'Dedicated final output directory required')
    if args.action == 'verify':
        for condition in conditions: r.load_bound_inputs(args.freeze, args.freeze_sha256, condition)
        print(json.dumps({'status': 'PASS_FINAL_ENCODER_BINDINGS_NO_MODELS', 'conditions': conditions})); return
    with process_lock(output_root / ('.encoder_' + gpu + '.lock')):
        for condition in conditions:
            freeze, selected, binding, panel, run = r.load_bound_inputs(args.freeze, args.freeze_sha256, condition)
            r.require(len(panel['items']) == 500, 'Final encoder queue requires exactly500 common items')
            output = output_root / condition
            if completed_reuse(output, run, panel):
                print(json.dumps({'condition': condition, 'status': 'complete_reused_no_model_calls'}), flush=True); continue
            if output.exists() and any(output.iterdir()):
                raise RuntimeError('Existing incomplete encoder run requires explicit reconciliation; no automatic restart')
            observed = observe(gpu, args.location); current = r.now()
            admission = {'status': 'approved_encoder_inference', 'freeze_sha256': args.freeze_sha256, 'condition': condition,
                'hostname': socket.gethostname(), 'device': 'cuda:0', 'gpu_uuid': gpu, 'minimum_free_bytes': 8192 * 1024**2,
                'maximum_items': 500, 'exclusive_gpu_observed': True, 'output_directory': str(output),
                'issued_utc': current.isoformat(), 'expires_utc': end, 'hard_deadline_utc': end,
                'resource_observation': observed, 'controller_sha256': r.sha(__file__),
                'authorization': 'Actual Freeze B and user autonomous research authorization; no training or selection'}
            for key in ['gpu_process_policy', 'gpu_gui_baseline']:
                if key in observed: admission[key] = observed[key]
            r.check_admission(admission, run, output)
            stamp = current.strftime('%Y%m%dT%H%M%S%fZ')
            ap = output_root / 'admissions' / condition / (stamp + '.json'); r.write_new(ap, admission)
            log = output_root / 'logs' / (condition + '_' + stamp + '.log'); log.parent.mkdir(parents=True, exist_ok=True)
            command = [sys.executable, '-B', str(Path(r.__file__).resolve()), 'execute', '--freeze', str(args.freeze.resolve()),
                '--freeze-sha256', args.freeze_sha256, '--condition', condition, '--admission', str(ap), '--output', str(output)]
            env = os.environ.copy(); env.pop('USE_TORCH', None)
            env.update(CUDA_VISIBLE_DEVICES=gpu, HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', TOKENIZERS_PARALLELISM='false', PYTHONDONTWRITEBYTECODE='1')
            r.write_new(log.with_suffix('.launch.json'), {'command': command, 'gpu_uuid': gpu, 'utc': current.isoformat()})
            flags = subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
            with log.open('x', encoding='utf-8') as stream:
                process = subprocess.Popen(command, cwd=r.HERE, env=env, stdout=stream, stderr=subprocess.STDOUT, creationflags=flags)
                r.write_new(log.with_suffix('.pid.json'), {'pid': process.pid, 'utc': r.now().isoformat()})
                code = process.wait()
            r.write_new(log.with_suffix('.terminal.json'), {'returncode': code, 'utc': r.now().isoformat()})
            if code: raise RuntimeError('Encoder condition stopped; no retry or continuation: ' + condition)
            r.require(completed_reuse(output, run, panel), 'Successful process lacked complete verified output')
        receipt = {'status': 'complete', 'location': args.location, 'conditions': conditions,
                   'items_per_condition': 500, 'freeze_sha256': args.freeze_sha256}
        done = output_root / 'ENCODER_QUEUE_COMPLETE.json'
        if done.exists(): r.require(r.read(done) == receipt, 'Prior queue completion differs')
        else: r.write_new(done, receipt)
        print(json.dumps(receipt), flush=True)


if __name__ == '__main__': main()
