"""Explicit Freeze-B launch of two independent local-LLM queues, never training.

One process per condition; same immutable final runner and corrected HF runtime.
The desired completion time is not an implicit cancellation or coverage cutoff.
"""
import argparse
import datetime
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / 'llm'))
import final_local_llm as f

PYTHON = Path('/home/jovyan/work/server_qwen_resume_20261109T074536Z_9995683b/env/bin/python')
OVERLAY = Path('/home/jovyan/work/server_ner_followup_20261109T162529Z_0501bf1e/xgrammar_overlay')
GPUS = {'qwen': 'GPU-8f767e03-8d0c-71f8-4d6d-cb3a88c92169', 'sherkala': 'GPU-ee38694a-2f0c-8966-d4ad-e4e5f72086ae'}
GIB = 1024 ** 3


def now(): return datetime.datetime.now(datetime.timezone.utc)


def condition_order(model):
    return [f'ner_{model}_ordinary4', f'ner_{model}_schema4', f'sa_{model}_ordinary4']


def expiration(value):
    time = datetime.datetime.fromisoformat(value.replace('Z', '+00:00'))
    if time.tzinfo is None or not now() < time <= now() + datetime.timedelta(hours=24):
        raise ValueError('Explicit future resource expiry within24hours required')
    return time.isoformat()


def verify(args):
    expiry = expiration(args.expires_utc)
    packages = {}; root = args.output_root.resolve()
    if sys.platform != 'linux' or not str(root).startswith('/home/jovyan/work/server_kazakh_final_'):
        raise PermissionError('Dedicated Linux final research root required')
    for condition in f.CONDITIONS:
        freeze, package = f.load_final_package(args.freeze, args.freeze_sha256, condition)
        config = package['binding']['backend_config']; model = f.CONDITIONS[condition][1]
        if config['gpu_uuid'] != GPUS[model] or Path(config['trace_dir']).resolve() != root / condition / 'traces':
            raise PermissionError('Frozen model GPU or condition output directory differs')
        hf = f.load_hf(package['binding']); f.validate_binding(condition, package['binding'], hf)
        if not hasattr(hf, 'compile_ner_grammar') or package['binding']['accepted_runtime']['files']['escape_safe_ner_grammar.py'] != \
                '0ccb5e46a05dee7146e0de875da758532e74e1b5036a9dc292692fb9ce33c33c':
            raise ValueError('Accepted corrected grammar runtime required')
        packages[condition] = package
    if not PYTHON.is_file() or not OVERLAY.is_dir(): raise FileNotFoundError('Accepted isolated execution environment missing')
    return packages, expiry


def observe(args, packages, models, minimum_ram):
    memory = {line.split(':')[0]: int(line.split(':')[1].split()[0]) * 1024
              for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith(('MemTotal:', 'MemAvailable:'))}
    args.output_root.mkdir(parents=True, exist_ok=True)
    disk = shutil.disk_usage(args.output_root)._asdict()
    if memory['MemAvailable'] < minimum_ram or disk['free'] < 20 * GIB:
        raise RuntimeError('Insufficient available RAM/disk; no existing process modified')
    isolation = {}
    for model in models:
        package = packages[condition_order(model)[0]]
        isolation[model] = f.load_hf(package['binding']).verify_gpu_exclusive(GPUS[model])
    return {'utc': now().isoformat(), 'memory': memory, 'disk': disk,
            'minimum_available_bytes': minimum_ram, 'gpu_isolation': isolation}


def worker(args):
    packages, expiry = verify(args); model = args.model
    root = args.output_root.resolve()
    with f.process_lock(root / 'locks' / (GPUS[model] + '.lock')):
        for condition in condition_order(model):
            package = packages[condition]; output = root / condition
            # Reuse is handled by the original immutable per-request runner.
            # Completed output requires no GPU/model call, so its verification
            # may proceed even when the device was subsequently assigned elsewhere.
            complete = output / 'COMPLETE.json'
            observed = {'completed_store_revalidation_only': True} if complete.exists() else \
                observe(args, packages, [model], 28 * GIB)
            stamp = now().strftime('%Y%m%dT%H%M%S%fZ')
            admission = {'status': 'approved_final_local_llm_inference', 'freeze_sha256': args.freeze_sha256,
                'condition': condition, 'hostname': socket.gethostname(), 'issued_utc': now().isoformat(),
                'expires_utc': expiry, 'output_directory': str(output), 'gpu_uuid': GPUS[model],
                'maximum_model_calls': package['request_count'], 'resource_observation': observed,
                'authority': 'Actual scientific Freeze B and user autonomous research authorization'}
            f.check_admission(admission, args.freeze_sha256, condition, package, output)
            path = root / 'admissions' / condition / (stamp + '.json'); f.write_new(path, admission)
            log = root / 'logs' / (condition + '_' + stamp + '.log'); log.parent.mkdir(parents=True, exist_ok=True)
            command = [str(PYTHON), '-B', str(Path(f.__file__).resolve()), 'execute', '--freeze', str(args.freeze.resolve()),
                       '--freeze-sha256', args.freeze_sha256, '--condition', condition, '--admission', str(path), '--output', str(output)]
            env = os.environ.copy(); env.pop('USE_TORCH', None)
            env.update(CUDA_VISIBLE_DEVICES=GPUS[model], NER_ISOLATED_PROCESS='1', PYTHONPATH=str(OVERLAY),
                HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', HF_HUB_DISABLE_TELEMETRY='1',
                OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', TOKENIZERS_PARALLELISM='false', PYTHONDONTWRITEBYTECODE='1')
            f.write_new(log.with_suffix('.launch.json'), {'command': command, 'condition': condition,
                'freeze_sha256': args.freeze_sha256, 'controller_sha256': f.sha(__file__), 'gpu_uuid': GPUS[model]})
            with log.open('x', encoding='utf-8') as stream:
                process = subprocess.Popen(command, cwd=HERE, env=env, stdout=stream, stderr=subprocess.STDOUT)
                f.write_new(log.with_suffix('.pid.json'), {'pid': process.pid, 'utc': now().isoformat()})
                code = process.wait()
            f.write_new(log.with_suffix('.terminal.json'), {'returncode': code, 'utc': now().isoformat()})
            if code: raise RuntimeError('Final condition stopped; evidence preserved; no retry: ' + condition)
        f.write_evidence(root / (model.upper() + '_FINAL_COMPLETE.json'),
            {'model': model, 'conditions': condition_order(model), 'freeze_sha256': args.freeze_sha256})


def launch(args):
    packages, expiry = verify(args); root = args.output_root.resolve()
    with f.process_lock(root / 'locks/coordinator.lock'):
        observed = observe(args, packages, list(GPUS), 56 * GIB)
        stamp = now().strftime('%Y%m%dT%H%M%S%fZ'); folder = root / 'launches'
        f.write_new(folder / (stamp + '_resources.json'), observed)
        processes = []
        for model in GPUS:
            command = [str(PYTHON), '-B', str(Path(__file__).resolve()), 'worker', '--model', model,
                '--freeze', str(args.freeze.resolve()), '--freeze-sha256', args.freeze_sha256,
                '--output-root', str(root), '--expires-utc', expiry]
            log = folder / (stamp + '_' + model + '.log')
            with log.open('x', encoding='utf-8') as stream:
                process = subprocess.Popen(command, cwd=HERE, stdout=stream, stderr=subprocess.STDOUT)
            f.write_new(log.with_suffix('.pid.json'), {'pid': process.pid, 'command': command})
            processes.append((model, process, log))
        print(json.dumps({'status': 'TWO_FINAL_QUEUES_STARTED', 'pids': {m: p.pid for m,p,_ in processes}}), flush=True)
        terminals = [{'model': model, 'returncode': process.wait(), 'log': str(log)} for model, process, log in processes]
        f.write_new(folder / (stamp + '_terminal.json'), {'conditions': terminals})
        if any(row['returncode'] for row in terminals): raise RuntimeError('A final queue stopped; no automatic restart')
        f.write_evidence(root / 'LOCAL_LLM_FINAL_COMPLETE.json', {'status': 'complete', 'conditions': sorted(f.CONDITIONS),
                         'freeze_sha256': args.freeze_sha256, 'items_per_condition': 500})
        print('ALL_SIX_LOCAL_FINAL_CONDITIONS_COMPLETE', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['verify', 'launch', 'worker'])
    parser.add_argument('--freeze', required=True, type=Path); parser.add_argument('--freeze-sha256', required=True)
    parser.add_argument('--output-root', required=True, type=Path); parser.add_argument('--expires-utc', required=True)
    parser.add_argument('--model', choices=list(GPUS)); args = parser.parse_args()
    if args.action == 'verify': verify(args); print('PASS_FINAL_BINDINGS_NO_MODELS')
    elif args.action == 'launch': launch(args)
    elif not args.model: parser.error('worker requires --model')
    else: worker(args)
