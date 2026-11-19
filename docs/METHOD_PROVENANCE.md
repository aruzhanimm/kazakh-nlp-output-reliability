# Panel selection and protocol provenance

The final comparison uses one common panel of 500 items per task. The following details describe the executed primary evaluation. They do not change its predictions or metrics.

## Exact panel construction

[`PRIMARY_PROTOCOL.json`](../configs/PRIMARY_PROTOCOL.json) pins the official source files, their SHA256 values and the selection settings. The seed is **20261110**, the namespace is **`kazakh-final-v2`**, and the requested size is **500** for each task. There is no label stratification or exclusion from the official test population.

For each official stable ID, serialize the array `[namespace, seed, task, row_id]` with Python `json.dumps(..., ensure_ascii=False, separators=(',', ':'))`, encode the result as UTF-8, and compute its SHA256 hexadecimal digest. Sort ascending by `(digest, row_id)` and take the first 500 rows. The ID is the lexical tie-breaker. The seed is part of the hashed payload; selection does not depend on a pseudorandom-number-generator implementation.

NER IDs are `ner:test:<zero-based sentence index>` in the pinned CoNLL file. SA IDs are `sa:test:<custom_id>` from the pinned official CSV inside `05_pc_test.zip`. The candidate populations contain 11,307 NER sentences and 16,797 binary SA reviews. The released ID lists preserve the selected rank order.

The executed preparation entry point is [`prepare_common_panels.py`](../src/research_execution_20261112/final_evaluation/llm/prepare_common_panels.py). An older generic in-memory helper in `offline_kernel.py` retains a 2,000-item default; that default was not used to construct these final panels.

An independent standard-library check reconstructs the IDs from the pinned original sources and compares them with both released lists and evidence membership:

```sh
python src/verify_panel_selection.py --ner /path/to/IOB2_test.txt --sa /path/to/05_pc_test.zip
```

Both 500-item lists and their ordering were verified exactly. The [saved verification receipt](../specification/PANEL_SELECTION_VERIFICATION.json) records source hashes, population counts and the first serialized rank keys. No model, tokenizer, scoring or API call is involved in this check.

## What was fixed, and when

The internal Freeze A record is dated **12 November 2026, 12:55:37 UTC**. It binds the panel algorithm and sources, model conditions, prompts and demonstrations, decoding and output-processing policies, and the analysis specification. That specification includes the six primary contrasts and the descriptive fence-only policy. The preparation-access record follows at **12:55:55 UTC** and references Freeze A's SHA256. The downstream Freeze B records and final run records retain the hash links.

The [protocol timeline](../specification/PROTOCOL_TIMELINE.json) gives the exact recorded timestamps, source SHA256 values and verification scope. These are internally dated, hash-linked execution records, not an external preregistration. Here, *prespecified* means recorded before this final-panel preparation and inference. It does not mean that the official test files had never been inspected: the earlier corpus-integrity audit examined aggregate counts, BIO validity and overlap without model scores, as stated in the paper.

Public timeline entries are compact extracts from the archived records. Original manifests retain machine-specific paths and are preserved in the research archive; source hashes identify them. The release date of this explanatory document is not the protocol-freeze date.

## Model and checkpoint identities

- [`ENCODER_CONDITIONS.json`](../specification/ENCODER_CONDITIONS.json) records all 12 task/model/seed conditions, initialization and tokenizer revisions, optimizer settings, selected epochs, development scores and selected-checkpoint SHA256 values. The compact identities also appear in [`selected_checkpoint_hashes.json`](../configs/selected_checkpoint_hashes.json).
- [`LOCAL_LLM_CONDITIONS.json`](../specification/LOCAL_LLM_CONDITIONS.json) records Qwen and Sherkala revisions, tokenizer identities, decoding settings, native templates and actual runtime metadata.
- [`GEMMA_CONDITIONS.json`](../specification/GEMMA_CONDITIONS.json) records the pinned Gemma revision and decoding configuration, with the archived admission receipt's weight-file identity and package versions.
- [`API_CONDITIONS.json`](../specification/API_CONDITIONS.json) records the requested/returned model alias and conditions. No verified immutable API model snapshot is available; saved task answers remain reproducible inputs to scoring.

The source receipt identities are listed in [`SOURCE_HASHES.json`](../specification/SOURCE_HASHES.json). The November 19 identity audit rechecked the archived metadata and hash bindings; it did not reload multi-gigabyte model weights or repeat inference. Model weights are not included in this repository.
