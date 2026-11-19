# Research source coverage

The source tree combines the accepted local training and analysis code with the implementations preserved in the final server export. The checksummed final-run records identify the evaluated versions. Existing source files are retained without adapting their historical paths or bypassing runtime checks.

| Stage | Entry point under `src/` |
|---|---|
| NER encoder training | `ner_training_queue01/` |
| Sentiment encoder training | `sa_training_queue01/` |
| Encoder evaluation | `research_execution_20261112/final_evaluation/encoder_runner.py` |
| Qwen and Sherkala evaluation | `research_execution_20261112/final_evaluation/llm/final_local_llm.py` |
| Accepted generation backend and escape-safe grammar | `research_execution_20261112/final_evaluation/llm/prepared_runtime_copy/` |
| Gemma final evaluation and native output handling | `research_execution_20261112/gemma/run_final_gemma.py` and `native_adapter.py` |
| API evaluation | `research_execution_20261112/api/` |
| Statistical analysis | `research_execution_20261112/final_analysis/` |

[`SERVER_SOURCE_IDENTITIES.json`](../configs/SERVER_SOURCE_IDENTITIES.json) records the source hashes checked against the accepted final freeze and the verified server export. The Gemma training-only pilot helper and input templates are included because its final runner binds those exact inputs and implementation files. They are not additional test conditions. The accepted Hugging Face backend requires its adjacent grammar helper and `vendor/trace_schema.py`; keep their relative layout intact.

## What can be reproduced from this repository

| Operation | Included inputs | Additional requirements |
|---|---|---|
| Regenerate tables and figures | Aggregate results, CSV exports, builders and expected outputs | Pinned plotting dependencies |
| Inspect scoring and inference rules | Source, schemas, prompt templates and source identities | None for reading the implementation |
| Re-score released records | `evidence/` scoring records and rescoring utility, plus the original scoring source | Follow the evidence documentation; NumPy is needed for the optional paired bootstrap |
| Repeat training or inference | Historical source, selected scientific configurations and fixed identities | Datasets, selected weights where applicable, model access, runtime admission records and machine-specific paths |

Weights, full model caches, complete execution traces and original request ledgers are outside this compact distribution. The released evidence supports independent score checking within its documented scope. The repository is not a full server backup or a one-command recreation of the original environment. Runtime checks must be preserved when preparing a new environment; any changed model, configuration or data should be recorded as a new run.
