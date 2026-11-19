"""NER train/dev orchestration with explicit bounded local-HF admission.

This module never imports torch, Transformers, an API client, or a dataset loader.
It is not a final-test runner and does not authorize real model calls. Source and
input identities bind every persisted response. Gold is accepted only separately
by the existing assembly.score_item after complete evidence assembly.
"""
from pathlib import Path
import datetime
import hashlib
import importlib.util
import re

import sys
import socket
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ner_pipeline_20261111"))
from storage import EvidenceStore, StoreCorruption, canonical, digest, read_evidence, write_evidence, process_lock

PROJECT = Path(__file__).resolve().parents[1]
ASSEMBLY_PATH = PROJECT / "ner_execution_prep_20261111/runner_review/evidence_assembly.py"
_spec = importlib.util.spec_from_file_location("ner_existing_evidence_assembly", ASSEMBLY_PATH)
assembly = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(assembly)
IncompleteRun = assembly.IncompleteRun


class RuntimeWindowFailure(IncompleteRun):
    """A persisted machinery failure stops before any later window is called."""


def admission_binding(admission):
    # Resource freshness can be renewed without changing the fixed scientific scope.
    # Original admission is separately archived by the panel runner.
    return {k: admission[k] for k in ("task", "split", "condition", "panel_sha256",
            "backend_config_sha256", "allowed_item_ids", "maximum_model_calls")}


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _sources():
    paths = [Path(__file__).resolve(), PROJECT / "ner_pipeline_20261111/storage.py", ASSEMBLY_PATH]
    return {p.relative_to(PROJECT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def _notify(hook, stage, details):
    if hook:
        hook(stage, details)


def _validate_prompt(prompt):
    if not isinstance(prompt["messages"], list) or not prompt["messages"]:
        raise ValueError("Messages must be a nonempty list")
    for message in prompt["messages"]:
        if set(message) != {"role", "content"} or message["role"] not in ("system", "user", "assistant") or not isinstance(message["content"], str):
            raise ValueError("Invalid native message")
    if not isinstance(prompt["rendered_prompt"], str):
        raise ValueError("Missing native render")
    ids = prompt["input_token_ids"]
    if not isinstance(ids, list) or not ids or any(type(x) is not int or x < 0 for x in ids):
        raise ValueError("Invalid native token IDs")
    canonical(prompt)


def _classify(response):
    """Validate a persisted backend return without parsing entity predictions."""
    if not isinstance(response, dict):
        raise IncompleteRun("Backend response contract is not an object; retained for inspection")
    kind = response.get("kind")
    if kind == "ambiguous_failure":
        raise IncompleteRun("Ambiguous backend exception; no automatic reissue permitted")
    if kind == "transport_failure":
        evidence = response.get("transport_evidence")
        if response.get("response_received") is not False or not isinstance(evidence, dict) or not isinstance(evidence.get("signal"), str) or not evidence["signal"]:
            raise IncompleteRun("Transport retry lacks explicit no-response evidence")
        if response.get("raw", "") != "" or response.get("generated_token_ids", []) != []:
            raise IncompleteRun("Transport failure contains received output; cannot retry")
        return "transport_failure"
    if kind not in ("completed", "runtime_failure"):
        raise IncompleteRun("Unknown saved backend status; cannot retry")
    if not isinstance(response.get("raw"), str):
        raise IncompleteRun("Missing raw response; cannot retry")
    ids = response.get("generated_token_ids")
    if not isinstance(ids, list) or any(type(x) is not int or x < 0 for x in ids):
        raise IncompleteRun("Missing/invalid generated token IDs; cannot retry")
    if any(type(response.get(k)) is not bool for k in ("truncated", "refusal")):
        raise IncompleteRun("Missing explicit termination/refusal flags")
    reasons = response.get("execution_failure_reasons")
    if not isinstance(reasons, list) or any(not isinstance(x, str) for x in reasons):
        raise IncompleteRun("Missing execution diagnostics")
    if kind == "completed" and (response.get("runtime_error") or reasons):
        raise IncompleteRun("Completed response contradicts execution failure flags")
    if kind == "runtime_failure" and not (response.get("runtime_error") or reasons):
        raise IncompleteRun("Runtime failure has no diagnostic reason")
    return kind


def _attempts(path, manifest):
    """Recover only unambiguous, contiguous persisted attempts (at most three)."""
    files = list((path / "attempts").glob("*.json")) if (path / "attempts").exists() else []
    named = [p for p in files if not p.name.startswith(".partial_")]
    if any(not re.fullmatch(r"000[1-3]\.(started|response|outcome)\.json", p.name) for p in named):
        raise StoreCorruption("Unexpected attempt file")
    numbers = sorted({int(p.name[:4]) for p in named})
    if numbers != list(range(1, len(numbers) + 1)):
        raise StoreCorruption("Attempt sequence has gaps")
    ledger, last_response = [], None
    for number in numbers:
        stem = path / "attempts" / f"{number:04d}"
        started_path = stem.with_suffix(".started.json")
        if not started_path.exists():
            raise StoreCorruption("Response/outcome lacks durable started record")
        started = read_evidence(started_path)
        attempt_id = manifest["request_identity"] + f":{number}"
        if started.get("request_identity") != manifest["request_identity"] or started.get("attempt_id") != attempt_id or started.get("number") != number:
            raise StoreCorruption("Attempt belongs to a different request")
        if ledger and ledger[-1]["status"] != "transport_failure":
            raise StoreCorruption("Attempt followed a completed/runtime response")
        response_path = stem.with_suffix(".response.json")
        if not response_path.exists():
            raise IncompleteRun("Started request lacks saved response; manual reconciliation required")
        captured = read_evidence(response_path)
        if captured.get("attempt_id") != attempt_id:
            raise StoreCorruption("Captured response belongs to another attempt")
        response = captured["backend_response"]
        status = _classify(response)
        request = read_evidence(path / "REQUEST.json")
        if request.get("backend_mode") == "local_hf":
            if status == "transport_failure":
                raise IncompleteRun("Local HF generation has no admitted transport retries")
            metadata = response.get("metadata", {})
            if metadata.get("input_token_ids_sha256") != manifest["input_token_ids_sha256"] or \
                    metadata.get("backend_config_sha256") != request["backend_identity"]["backend_config_sha256"]:
                raise IncompleteRun("Saved actual backend input/config identity does not match frozen request")
        outcome = {"attempt_id": attempt_id, "status": status,
                   "response_received": status != "transport_failure",
                   "started_utc": started["started_utc"], "captured_utc": captured["captured_utc"],
                   "response_payload_sha256": digest(captured)}
        write_evidence(stem.with_suffix(".outcome.json"), outcome)
        ledger.append(outcome)
        last_response = response
    return ledger, last_response


def _make_record(manifest, prompt, budget, ledger, response, overflow=False):
    terminal = overflow or (ledger and ledger[-1]["status"] != "completed")
    record = dict(manifest)
    record.update({k: prompt[k] for k in ("messages", "rendered_prompt", "input_token_ids")})
    record.update({"status": "terminal_failure" if terminal else "completed",
                   "raw": "" if response is None else response.get("raw", ""),
                   "generated_token_ids": [] if response is None else response.get("generated_token_ids", []),
                   "truncated": False if response is None else response.get("truncated", False),
                   "refusal": False if response is None else response.get("refusal", False),
                   "runtime_error": None if response is None else response.get("runtime_error"),
                   "execution_failure_reasons": [] if response is None else response.get("execution_failure_reasons", []),
                   "attempts": ledger, "context_budget": budget,
                   "backend_metadata": {} if response is None else response.get("metadata", {})})
    record["raw_sha256"] = assembly.text_hash(record["raw"])
    if terminal:
        record["terminal_reason"] = "input_context_failure" if overflow else ledger[-1]["status"]
    return record


def _execute_window(store, manifest, prompt, budget, backend, hook):
    path = store.request_dir(manifest["request_identity"])
    with process_lock(path / ".request.lock"):
        request = {"manifest": manifest, "prepared_prompt": prompt, "context_budget": budget,
                   "backend_mode": backend.mode, "backend_identity": backend.identity}
        write_evidence(path / "REQUEST.json", request)
        overflow = len(prompt["input_token_ids"]) + budget["reserved_output_tokens"] > budget["context_limit_tokens"]
        if overflow:
            if (path / "attempts").exists() and any((path / "attempts").iterdir()):
                raise StoreCorruption("Context-overflow request unexpectedly has attempts")
            record = _make_record(manifest, prompt, budget, [], None, overflow=True)
        else:
            ledger, response = _attempts(path, manifest)
            cached_path = path / "RECORD.json"
            if cached_path.exists():
                cached = read_evidence(cached_path)
                if not ledger or (ledger[-1]["status"] == "transport_failure" and len(ledger) != 3):
                    raise StoreCorruption("Final record lacks complete backing attempt evidence")
                recovered = _make_record(manifest, prompt, budget, ledger, response)
                if canonical(cached) != canonical(recovered):
                    raise StoreCorruption("Final record differs from underlying raw evidence")
                return cached
            # Existing record still gets fully reconstructed from underlying raw
            # evidence before reuse, so later corruption is not hidden by a cache.
            while not ledger or (ledger[-1]["status"] == "transport_failure" and len(ledger) < 3):
                number = len(ledger) + 1
                attempt_id = manifest["request_identity"] + f":{number}"
                stem = path / "attempts" / f"{number:04d}"
                started = {"request_identity": manifest["request_identity"], "attempt_id": attempt_id,
                           "number": number, "started_utc": utc()}
                write_evidence(stem.with_suffix(".started.json"), started)
                _notify(hook, "after_started_before_backend", {"request_path": str(path), "attempt_id": attempt_id})
                try:
                    returned = backend.generate(request)
                except Exception as exc:
                    # A generic exception is NOT evidence that no output existed.
                    # Its class/message are saved, then recovery stops. Do not log
                    # credentials: adapters must sanitize their exception messages.
                    returned = {"kind": "ambiguous_failure", "exception_type": type(exc).__name__,
                                "detail": str(exc)}
                captured = {"attempt_id": attempt_id, "captured_utc": utc(), "backend_response": returned}
                write_evidence(stem.with_suffix(".response.json"), captured)
                _notify(hook, "after_response_before_classification", {"request_path": str(path), "attempt_id": attempt_id})
                ledger, response = _attempts(path, manifest)
                _notify(hook, "after_outcome_before_next", {"request_path": str(path), "attempt_id": attempt_id})
            record = _make_record(manifest, prompt, budget, ledger, response)
        write_evidence(path / "RECORD.json", record)
        _notify(hook, "after_record_before_assembly", {"request_path": str(path)})
        return read_evidence(path / "RECORD.json")


def execute_prepared_item(*, store, item_id, tokens, windows, prepared_prompts,
                          allowed_types, run_identity, condition, backend,
                          context_budget=None, input_provenance=None, fault_hook=None, admission=None, panel_sha256=None):
    """Execute/recover synthetic fixtures or explicitly admitted local train/dev.

    The caller must validate the fixed panel before loading a real backend.
    Local HF permits no transport retry. Runtime faults preserve a window record
    then halt before later windows; no complete item score is published.
    """
    mode = getattr(backend, "mode", None)
    if mode not in ("synthetic", "local_hf"):
        raise PermissionError("Only synthetic or admitted local HF backends are supported")
    if mode == "local_hf":
        from admission import verify_execution_admission
        verify_execution_admission(admission, panel_sha256, condition, item_id, backend)
    if not isinstance(backend.identity, dict) or not backend.identity:
        raise ValueError("Explicit backend identity required")
    if not isinstance(run_identity, dict) or not run_identity:
        raise ValueError("Explicit experimental identity required")
    if not isinstance(item_id, str) or not item_id.startswith(("synthetic:", "ner:train:", "ner:valid:")):
        raise PermissionError("Only synthetic/train/dev item identities admitted; no final test")
    if not item_id.startswith("synthetic:") and not isinstance(input_provenance, dict):
        raise ValueError("Native train/dev inputs require their preparation provenance")
    if not isinstance(condition, str) or not condition:
        raise ValueError("Condition required")
    budget = context_budget or {"context_limit_tokens": 4096, "reserved_output_tokens": 768}
    if set(budget) != {"context_limit_tokens", "reserved_output_tokens"} or any(type(v) is not int or v <= 0 for v in budget.values()):
        raise ValueError("Positive context budget required")
    if not isinstance(store, EvidenceStore):
        raise TypeError("An explicit EvidenceStore is required")
    for prompt in prepared_prompts:
        _validate_prompt(prompt)
        if "context_budget" in prompt:
            supplied = prompt["context_budget"]
            if any(supplied.get(k) != v for k, v in budget.items()):
                raise ValueError("Prompt context budget differs from execution binding")
            count = len(prompt["input_token_ids"])
            if "input_tokens" in supplied and supplied["input_tokens"] != count:
                raise ValueError("Prompt input-token count is inconsistent")
            if "spare_tokens" in supplied and supplied["spare_tokens"] != budget["context_limit_tokens"] - budget["reserved_output_tokens"] - count:
                raise ValueError("Prompt spare-token count is inconsistent")
    binding = {"task": "ner", "mode": mode, "condition": condition,
               "experiment": run_identity, "backend": backend.identity, "sources": _sources(),
               "input_preparation_provenance": input_provenance,
               "assembly_contract_sha256": assembly.contract_hash(allowed_types),
               "context_budget": budget, "maximum_total_attempts": 3,
               "panel_sha256": panel_sha256,
               "admission_binding": admission_binding(admission) if mode == "local_hf" else None}
    run_hash = digest(binding)
    manifest = assembly.make_request_manifest(item_id, tokens, windows, run_hash, condition, prepared_prompts, allowed_types)
    # Validate structural coverage before any attempt; metadata binding blocks
    # resumed runs after source/config changes in an existing store directory.
    store.bind(binding)
    item_key = digest({"id": item_id})
    item_path = store.root / "items" / item_key
    assembled_path = item_path / "ASSEMBLED.json"
    if assembled_path.exists() and not (item_path / "MANIFEST.json").exists():
        raise StoreCorruption("Final item lacks its manifest")
    write_evidence(item_path / "MANIFEST.json",
                   {"id": item_id, "tokens": tokens, "windows": windows, "manifest": manifest})
    if assembled_path.exists():
        read_evidence(assembled_path)
        # A final item is evidence that all windows once reached final records.
        # Missing prerequisites must never be reinterpreted as unexecuted work.
        for entry in manifest:
            request_path = store.root / "requests" / entry["request_identity"]
            if not all((request_path / name).exists() for name in ("REQUEST.json", "RECORD.json")):
                raise StoreCorruption("Final item lacks backing window evidence")
    records = []
    for entry, prompt in zip(manifest, prepared_prompts):
        record = _execute_window(store, entry, prompt, budget, backend, fault_hook)
        records.append(record)
        if record.get('runtime_error') or record.get('execution_failure_reasons'):
            raise RuntimeWindowFailure('Persisted runtime window failure; item remains incomplete')
    _notify(fault_hook, "before_assembly", {"records": len(records)})
    assembled = assembly.assemble_item(item_id, tokens, windows, manifest, records, allowed_types)
    write_evidence(assembled_path, assembled)
    return {"manifest": manifest, "records": records, "assembled": assembled,
            "store_path": str(store.root), "run_hash": run_hash,
            "scientific_freeze": False, "real_model_execution_admitted": mode == "local_hf"}
