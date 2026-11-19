# Per-item evaluation evidence

This directory contains the fixed 500-item panel for each task and predictions for
all 22 primary conditions: 12 NER conditions and 10 sentiment conditions. The
6,000 encoder predictions and 5,000 saved language-model task answers are sufficient
to independently recompute the reported primary metrics. The four encoder means
are means of three separate seed scores, not ensembles.

From the supplement root, run with Python 3.10 or newer:

```sh
python -B evidence/rescore.py --output ../rescored_metrics.json
```

The default command uses only the Python standard library. It verifies file and
source-code hashes, reconstructs encoder NER spans from preserved word-level BIO
predictions, reconstructs encoder SA labels from saved logits, and reparses each
LLM task answer with the original parser. It then recomputes exact-span NER metrics,
binary SA metrics and confusions, encoder seed means/SD, output diagnostics, and
the prespecified fence-only sensitivity. Every prediction must match the saved
primary prediction before its score is checked. Expected aggregates are used only
as comparison targets, not as the source of recalculated scores.

To also regenerate the 50,000 paired bootstrap draws and verify all reported
intervals, install NumPy and add `--bootstrap`:

```sh
python -B evidence/rescore.py --bootstrap --output ../rescored_with_intervals.json
```

The original analysis used NumPy 2.5.4. This optional step retains the original
seed, cluster ordering, corpus metrics, three-seed averaging and six-comparison
family. It does not add new inferential comparisons. The script refuses to
overwrite an output file. Omitting `--output` prints a compact validation summary.

## Included records

- `ner_items.jsonl`: source IDs, original KazNERD tokens, original gold BIO labels,
  exact gold spans and complete-text group hashes.
- `sa_items.jsonl`: source IDs, binary gold labels, original/visible text hashes
  and prefix metadata. Review texts are obtained separately from the source release.
- `predictions/*.jsonl`: one file per condition in the fixed panel order. Encoder
  rows preserve BIO tags or binary logits. LLM rows preserve the original task-answer
  string, failure flags, accepted prediction and hashes.
- `EXPECTED_METRICS.json`: comparisons extracted from the completed fixed analysis.
- `MANIFEST.json`: file hashes, unchanged scoring-source hashes and artifact identities.

An LLM task answer means the saved text at the task-parser boundary. The API HTTP
envelope and Gemma's native special-token wrapper are not redistributed here.
No task-answer text has been repaired, rewritten or generated for this supplement.
The fence-only analysis creates a separate derived view and leaves the saved answer
unchanged. This package reproduces parsing and scoring; it does not regenerate
model outputs, replay GPU logits, or claim access to an immutable API snapshot.

No credentials, HTTP headers, private human reviews, chat history, model weights,
development generations or the later rare-type follow-up are included.

## Inspecting SA source text

Metrics can be reproduced without downloading any dataset. For source-text review,
obtain the pinned KazSAnDRA `05_pc_test.zip` from the link in
[DATA_ATTRIBUTION.md](DATA_ATTRIBUTION.md), then run:

```sh
python -B evidence/restore_sa_text.py --source-zip /path/to/05_pc_test.zip --output ../sa_panel_text.jsonl
```

The script checks the source ZIP hash, all 500 original-text hashes, exact visible
prefix hashes and class labels before writing a new local file. It makes no network
requests. The recovered texts remain subject to the dataset's original terms.
