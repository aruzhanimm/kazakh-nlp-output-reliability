# Dataset attribution and scope

## KazNERD

The 500 sentences, original tokens and gold BIO annotations in `ner_items.jsonl`
are an unchanged subset of **KazNERD**, by Rustem Yeshpanov, Yerbolat Khassanov
and Huseyin Atakan Varol (ISSAI), published as *KazNERD: Kazakh Named Entity
Recognition Dataset*, LREC 2022, pp. 417–426.

- [Dataset paper](https://aclanthology.org/2022.lrec-1.44/)
- [Original repository](https://github.com/IS2AI/KazNERD)
- [CC BY 4.0 license stated by the repository](https://github.com/IS2AI/KazNERD/blob/main/LICENSE.md)
- [Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/)
- Source revision: `bd4333d0f5952b9fafb2ef2ac2fefa0ad3c0333f`.
- Source file: `KazNERD/IOB2_test.txt`.
- Source SHA256: `359a1f61f6a21280c6c47b4fde9a5adcd07411d62015aa6e32d940c51f163b7a`.

Changes consist of selecting the frozen 500 source rows and serializing them as
JSON Lines with token-index spans and provenance hashes. Text and annotations are
unchanged. Predictions are research outputs and are distinguished from the original
annotations. This use does not imply endorsement by the dataset creators.

## KazSAnDRA

The sentiment evaluation uses **KazSAnDRA**, by Rustem Yeshpanov and Huseyin Atakan
Varol, *KazSAnDRA: Kazakh Sentiment Analysis Dataset of Reviews and Attitudes*,
LREC-COLING 2024, pp. 9657–9667.

- [Dataset paper](https://aclanthology.org/2024.lrec-main.844/)
- [Original repository](https://github.com/IS2AI/KazSAnDRA)
- [Pinned original test ZIP](https://raw.githubusercontent.com/IS2AI/KazSAnDRA/d769cf7f7bd62b59f88915508d4bc062310b54d6/dataset/05_pc_test.zip)
- Source revision: `d769cf7f7bd62b59f88915508d4bc062310b54d6`.
- Source SHA256: `cec729d4fd5dda6e1d73c3ce9d6e5feed01b24be79b8135bfb0a204e3f1e6451`.

This evidence directory does not redistribute the 500 review texts. It provides
source identifiers, binary class labels, text hashes and predictions; exact input
texts can be restored locally from the pinned source ZIP. No blanket license is
assigned here to the source reviews. Dataset and model creators retain their rights.

The generated task answers and derived predictions document the experiments.
NER answers may contain short source mentions from the attributed KazNERD subset.
The relevant model/service terms continue to apply; no model weights are included.
