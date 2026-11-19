Scientific specification for the primary Kazakh NER and sentiment comparison

This directory supplements the compact research project in ../. It describes the accepted 500-item panel per task and all 22 primary conditions. Every value was extracted from saved accepted configurations, immutable request packages or observed runtime receipts. Source hashes identify the original evidence; private absolute paths and operational notes are omitted.

Contents
PANEL_SELECTION_VERIFICATION.json: independent exact-ID reconstruction from pinned official source files.
PROTOCOL_TIMELINE.json: internally dated, hash-linked protocol records; not an external preregistration.
ENCODER_CONDITIONS.json: all 12 selected task/model/seed training configurations, development selections, checkpoint hashes and evaluation identities.
LOCAL_LLM_CONDITIONS.json: Qwen and Sherkala frozen ordinary/schema configurations, model and tokenizer revisions/hashes, native templates, exact generation settings and observed runtime environment.
SHARED_PROMPTS.json: exact four-shot system/example message prefixes, training row IDs, all 25 NER types and the query templates. The same prefixes were checked across all prompted systems. The query itself contains no gold labels.
GEMMA_CONDITIONS.json: frozen Gemma model revision, package versions and decoding policy, with recorded runtime-admission and weight-file identities. The appended receipt facts do not claim a new model loading or GPU check.
API_CONDITIONS.json: requested and returned API identity and task-specific settings. The API alias has no verified immutable snapshot.
ANALYSIS_POLICY.json: frozen metrics, six primary contrasts, descriptive contrasts, 50,000-draw paired bootstrap and fence-only policy.
TIMING_SCOPE.json: verified hardware and measurement boundaries for the Qwen/Sherkala NER timings reported in the manuscript.
SOURCE_HASHES.json: immutable identities of the source evidence from which these compact specifications were extracted.

Use ../configs/PRIMARY_PROTOCOL.json for dataset revisions, sample namespace/seed and common input policy; the adjacent *_primary500_ids.json files provide the exact ordered evaluation rows. Use ../analysis/data/ for saved aggregate results, separate seeds, per-type support and class-confusion counts. Follow ../docs/REPRODUCIBILITY.md to regenerate the included tables and figures without models or API calls.

Scope of reconstruction
This specification directory makes the implemented configurations inspectable. The adjacent ../evidence/ directory supplies released saved outputs and scoring records for independent rescoring, with its own scope and instructions. Trained weights, provider credentials and complete source datasets are not included. Repeating inference requires the pinned datasets/model access and a recreated runtime. Hashes alone do not supply external files. The table/figure workflow in ../ regenerates presentation outputs from completed analysis; the separate evidence workflow can recompute scores and optionally the paired bootstrap. No claim of bitwise reproduction from a mutable API alias is made.

The precise encoder initialization, optimizer, accumulation, scheduler horizon/warmup, stopping rule and selected epoch are specified per condition. Checkpoint hashes describe accepted primary runs; no expensive checkpoint deserialization was performed during the preparation of this compact supplement.
