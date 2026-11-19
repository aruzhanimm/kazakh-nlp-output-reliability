# Third-party materials

Dataset and model creators retain their respective rights. The repository does not redistribute complete datasets or model weights.

- **KazNERD:** source repository and pinned revision are recorded in [`src/data_manifest.json`](src/data_manifest.json). Cite the [original dataset paper](https://aclanthology.org/2022.lrec-1.44/).
- **KazSAnDRA:** the same manifest records the [original dataset repository](https://github.com/IS2AI/KazSAnDRA), revision and file hashes.
- **Models:** XLM-R, mBERT, Qwen, Sherkala, Gemma and the API reference remain subject to their respective licenses and service terms.
- **Dataset excerpts:** the fixed demonstrations and illustrative examples document the evaluation procedure; their use remains subject to the original dataset terms.

## Selected KazNERD evidence

The released NER scoring evidence contains selected original KazNERD tokens and gold labels, together with the evaluated systems' saved predictions or responses. KazNERD is distributed under **Creative Commons Attribution 4.0 International (CC BY 4.0)**. Credit belongs to Rustem Yeshpanov, Yerbolat Khassanov and Huseyin Atakan Varol. Cite [KazNERD: Kazakh Named Entity Recognition Dataset](https://aclanthology.org/2022.lrec-1.44/) and consult the [dataset repository](https://github.com/IS2AI/KazNERD) and [license terms](https://creativecommons.org/licenses/by/4.0/).

The distributed dataset material is a selected subset in an analysis-oriented representation. The original token strings and gold annotations are retained; model predictions and diagnostic fields are separate research outputs and are not part of the dataset's gold labels. Dataset revisions and evaluation row IDs are recorded in `configs/`.

## Sentiment evidence

Complete original KazSAnDRA review texts are excluded. The released sentiment records support score checking without redistributing the full reviews; source text reconstruction requires obtaining the pinned dataset under its original terms. Fixed training demonstrations are retained to document the experiment.

No blanket license is assigned to third-party materials. Author-created research materials remain under the author's control; a public code license has not yet been selected. Publication of this repository does not grant a new license over datasets, model weights or other third-party material.
