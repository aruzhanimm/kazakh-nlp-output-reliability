# Experimental design

The study compares task-specific multilingual encoders with four-shot language models on Kazakh named entity recognition and sentiment analysis.

| Component | Specification |
|---|---|
| NER data | KazNERD, original 25-type ontology and released split |
| Sentiment data | KazSAnDRA, released binary-polarity mapping |
| Evaluation panels | 500 deterministically selected shared test items per task |
| Encoders | XLM-R and mBERT; seeds 13, 42, 2026; selection from development evidence |
| Prompted models | Qwen2.5-7B-Instruct, Llama-3.1-Sherkala-8B-Chat, Gemma4-12B-it and GPT-6.1-Sol; four demonstrations |
| Primary NER metric | Exact-span, exact-type entity micro-F1 |
| Primary sentiment metric | Macro-F1 over the two fixed classes |
| Paired uncertainty | Text-cluster bootstrap with 50,000 draws |
| Comparisons | Six primary contrasts; four separately identified descriptive reference-model contrasts |

The matched NER schema comparison changes the generation constraint while holding the corresponding model, examples, inputs and parsing rules fixed. Format validity, schema validity and text alignment are reported separately from entity quality. Rejected candidates do not incur invented extra penalties in standard entity F1.

The prespecified fence-only sensitivity analysis removes a single outer Markdown fence from otherwise unchanged responses. Its results accompany, rather than replace, strict primary parsing. No additional generations are involved.

The main NER panel has 578 gold entities across 23 observed types. Per-type support is included with the results. Encoder means summarize three separate seed scores; they are not an ensemble or pooled count estimate. No aggregate score combines NER and sentiment.

See [`configs/`](../configs/) for fixed sample IDs and source identities, and [`analysis/data/`](../analysis/data/) for detailed results.
