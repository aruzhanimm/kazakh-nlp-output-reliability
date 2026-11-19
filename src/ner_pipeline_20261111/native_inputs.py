"""NER-only, gold-free native input preparation. No model or network entry point.

This candidate component does not freeze scientific decisions or authorize inference.
The caller supplies definitions, training demonstrations, pinned tokenizers and template.
"""
from dataclasses import dataclass, asdict
from pathlib import Path
import ast
import hashlib
import importlib.util
import json
import os

PROJECT = Path(__file__).resolve().parents[1]
ALIGNMENT = PROJECT / "interface_pilot/alignment.py"


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def text_hash(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _alignment():
    spec = importlib.util.spec_from_file_location("ner_native_alignment", ALIGNMENT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class InputPreparationError(ValueError):
    """Preparation cannot be completed; never a silently omitted item/window."""


@dataclass(frozen=True)
class WindowPolicy:
    max_tokens: int = 384
    max_words: int = 96
    overlap: int = 24

    def check(self):
        if any(type(x) is not int for x in (self.max_tokens, self.max_words, self.overlap)):
            raise InputPreparationError("Window limits must be integers")
        if self.max_tokens < 1 or self.max_words < 1 or not 0 <= self.overlap < self.max_words:
            raise InputPreparationError("Invalid window policy")


@dataclass(frozen=True)
class NativeContract:
    model_id: str
    revision: str
    chat_template: str
    chat_template_sha256: str
    configured_template_sha256: str | None
    special_token_ids: dict
    tokenizer_files: dict
    tokenizer_folder: str


def verify_tokenizer_files(folder, expected_hashes):
    """Hash only explicitly named tokenizer assets, never weight files."""
    allowed = {"tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
               "added_tokens.json", "chat_template.jinja", "vocab.json", "vocab.txt",
               "merges.txt", "sentencepiece.bpe.model", "spiece.model", "config.json"}
    if not expected_hashes or not set(expected_hashes) <= allowed:
        raise InputPreparationError("Only explicitly pinned tokenizer files are permitted")
    folder = Path(folder).resolve()
    present = {name for name in allowed if (folder / name).is_file()}
    if present != set(expected_hashes):
        raise InputPreparationError("Every present loader-relevant tokenizer asset must be pinned")
    for name, expected in expected_hashes.items():
        path = folder / name
        if path.parent != folder or not path.is_file() or file_hash(path) != expected:
            raise InputPreparationError("Tokenizer file checksum mismatch: " + name)
    return {"folder": str(folder), "files": dict(expected_hashes)}


def load_pinned_tokenizer(folder, expected_hashes):
    """Offline tokenizer-only loader; caller must set offline/no-torch environment."""
    if os.environ.get("USE_TORCH") != "0" or os.environ.get("HF_HUB_OFFLINE") != "1" or \
            os.environ.get("TRANSFORMERS_OFFLINE") != "1":
        raise InputPreparationError("Explicit offline tokenizer-only environment required")
    identity = verify_tokenizer_files(folder, expected_hashes)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(identity["folder"], local_files_only=True,
                                             use_fast=True, trust_remote_code=False)
    if not tokenizer.is_fast:
        raise InputPreparationError("Fast tokenizer with word alignment required")
    return tokenizer, identity


def _check_contract(tokenizer, contract):
    if not isinstance(contract, NativeContract) or not contract.model_id or not contract.revision:
        raise InputPreparationError("Explicit native model contract required")
    if not contract.chat_template or text_hash(contract.chat_template) != contract.chat_template_sha256:
        raise InputPreparationError("Native template checksum mismatch")
    configured = tokenizer.chat_template
    if configured is not None and not isinstance(configured, str):
        raise InputPreparationError("Named template dictionaries require explicit future support")
    if (text_hash(configured) if configured is not None else None) != contract.configured_template_sha256:
        raise InputPreparationError("Configured tokenizer template differs from native contract")
    if not contract.tokenizer_files or not contract.tokenizer_folder:
        raise InputPreparationError("Pinned tokenizer identity required")
    # Files are verified during load; object-native special IDs are checked each preparation.
    # This component does not attest that an arbitrary caller-supplied object was loaded there.
    for token, expected in contract.special_token_ids.items():
        if type(expected) is not int or tokenizer.convert_tokens_to_ids(token) != expected:
            raise InputPreparationError("Native special token mismatch: " + token)


def _pinned_window_builder(source_path, expected_sha256):
    """Compile only the existing common_windows definition, avoiding mixed-task imports.

    This is trusted local project code pinned by the caller. No remote source is loaded.
    The function body is reused unchanged; no SA helper or module entry point is executed.
    """
    source_path = Path(source_path).resolve()
    if source_path != (PROJECT / "stage_preflight/views.py").resolve():
        raise InputPreparationError("Only the existing common-window source is accepted")
    if file_hash(source_path) != expected_sha256:
        raise InputPreparationError("Common-window source checksum mismatch")
    source = source_path.read_text(encoding="utf-8")
    matches = [node for node in ast.parse(source).body
               if isinstance(node, ast.FunctionDef) and node.name == "common_windows"]
    if len(matches) != 1 or matches[0].decorator_list:
        raise InputPreparationError("Unexpected common-window function definition")
    module = ast.Module(body=matches, type_ignores=[])
    namespace = {}
    exec(compile(module, str(source_path), "exec"), namespace)
    return namespace["common_windows"]


def build_messages(tokens, definitions, demonstrations, allowed_types):
    """Only demonstrations carry training spans; query tokens have no gold argument."""
    align = _alignment()
    text = align.text_offsets(tokens)[0]
    if not isinstance(definitions, str) or not definitions.strip():
        raise InputPreparationError("Nonempty explicit entity definitions required")
    if not allowed_types or len(set(allowed_types)) != len(allowed_types):
        raise InputPreparationError("Unique entity types required")
    messages = [{"role": "system", "content": definitions}]
    seen = set()
    for demo in demonstrations:
        if set(demo) != {"id", "tokens", "spans"}:
            raise InputPreparationError("Demonstrations require only id/tokens/spans")
        if not isinstance(demo["id"], str) or not demo["id"].startswith("ner:train:"):
            raise InputPreparationError("Only training demonstrations are accepted")
        if demo["id"] in seen:
            raise InputPreparationError("Duplicate training demonstration ID")
        seen.add(demo["id"])
        demo_text = align.text_offsets(demo["tokens"])[0]
        spans = demo["spans"]
        previous_end = 0
        for span in spans:
            if len(span) != 3:
                raise InputPreparationError("A demonstration span is start/end/type")
            start, end, kind = span
            if type(start) is not int or type(end) is not int or \
                    not previous_end <= start < end <= len(demo["tokens"]) or kind not in allowed_types:
                raise InputPreparationError("Invalid, unsorted or overlapping demonstration spans")
            previous_end = end
        answer = align.demo_answer(demo["tokens"], spans)
        compact = json.dumps(answer, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
        replay = align.parse_mentions(compact, demo["tokens"], allowed_types)
        if replay["failed"] or replay["spans"] != [tuple(s) for s in spans]:
            raise InputPreparationError("Training demonstration failed exact alignment roundtrip")
        messages.extend([{"role": "user", "content": "Kazakh text:\n" + demo_text},
                         {"role": "assistant", "content": compact}])
    messages.append({"role": "user", "content": "Kazakh text:\n" + text})
    return messages


def render_native(tokenizer, messages, contract, context_limit=4096, reserved_output_tokens=768):
    _check_contract(tokenizer, contract)
    if type(context_limit) is not int or type(reserved_output_tokens) is not int or \
            context_limit <= 0 or reserved_output_tokens <= 0:
        raise InputPreparationError("Positive integer context and output budgets required")
    rendered = tokenizer.apply_chat_template(messages, chat_template=contract.chat_template,
                                            tokenize=False, add_generation_prompt=True)
    token_ids = tokenizer(rendered, add_special_tokens=False, truncation=False)["input_ids"]
    template_ids = tokenizer.apply_chat_template(messages, chat_template=contract.chat_template,
                                                tokenize=True, add_generation_prompt=True,
                                                truncation=False)
    if token_ids != template_ids or not token_ids or any(type(x) is not int or x < 0 for x in token_ids):
        raise InputPreparationError("Native rendering and tokenization disagree")
    budget = {"input_tokens": len(token_ids), "context_limit_tokens": context_limit,
              "reserved_output_tokens": reserved_output_tokens,
              "spare_tokens": context_limit - reserved_output_tokens - len(token_ids)}
    return {"messages": messages, "rendered_prompt": rendered, "input_token_ids": token_ids,
            "messages_sha256": digest(messages), "rendered_prompt_sha256": text_hash(rendered),
            "input_token_ids_sha256": digest(token_ids), "context_budget": budget,
            "fits": budget["spare_tokens"] >= 0}


def prepare_native_item(item_id, tokens, *, encoder_tokenizers, llm_tokenizer, definitions,
                        demonstrations, allowed_types, native_contract, window_source_path,
                        window_source_sha256, window_policy=WindowPolicy(),
                        context_limit=4096, reserved_output_tokens=768):
    """Return every common window and native prompt, including explicit overflows.

    No dataset path or gold labels are accepted. This is local-native input preparation,
    not an API wire adapter. Full word coverage does not imply full long-entity coverage.
    """
    if not isinstance(item_id, str) or not item_id.startswith("ner:"):
        raise InputPreparationError("NER item ID required")
    _alignment().text_offsets(tokens)
    window_policy.check()
    if set(encoder_tokenizers) != {"xlmr", "mbert"}:
        raise InputPreparationError("Both fixed XLM-R and mBERT tokenizers are required")
    builder = _pinned_window_builder(window_source_path, window_source_sha256)
    try:
        windows = builder(tokens, encoder_tokenizers, **asdict(window_policy))
    except ValueError as error:
        raise InputPreparationError("Common windows cannot represent this item: " + str(error)) from error
    covered = set()
    for window in windows:
        if window["tokens"] != list(tokens[window["start"]:window["end"]]):
            raise InputPreparationError("Common-window text changed")
        covered.update(range(window["start"], window["end"]))
    if covered != set(range(len(tokens))):
        raise InputPreparationError("Common windows omitted original words")
    # Demonstrations must themselves fit one unchanged common window, as previously specified.
    for demo in demonstrations:
        if len(builder(demo["tokens"], encoder_tokenizers, **asdict(window_policy))) != 1:
            raise InputPreparationError("A demonstration does not fit one common window")
    prepared = [render_native(llm_tokenizer,
                             build_messages(w["tokens"], definitions, demonstrations, allowed_types),
                             native_contract, context_limit, reserved_output_tokens) for w in windows]
    provenance = {"model_id": native_contract.model_id, "revision": native_contract.revision,
                  "tokenizer_folder": native_contract.tokenizer_folder,
                  "tokenizer_files": native_contract.tokenizer_files,
                  "template_sha256": native_contract.chat_template_sha256,
                  "configured_template_sha256": native_contract.configured_template_sha256,
                  "special_token_ids": native_contract.special_token_ids,
                  "definitions_sha256": text_hash(definitions), "demonstrations_sha256": digest(demonstrations),
                  "demonstration_ids": [d["id"] for d in demonstrations],
                  "allowed_types_sha256": digest(list(allowed_types)),
                  "window_source_sha256": window_source_sha256, "window_policy": asdict(window_policy),
                  "alignment_source_sha256": file_hash(ALIGNMENT),
                  "adapter_source_sha256": file_hash(__file__),
                  "context_limit_tokens": context_limit, "reserved_output_tokens": reserved_output_tokens,
                  "scientific_freeze": False, "human_review_attested": False}
    return {"id": item_id, "tokens": list(tokens), "windows": windows, "prepared_prompts": prepared,
            "context_overflow_windows": [i for i, p in enumerate(prepared) if not p["fits"]],
            "provenance": provenance}
