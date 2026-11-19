"""One-time preparation adapter; never imported by a training worker."""
import json,shutil
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parent.parent; Q=ROOT/'sa_training_queue01'
assert not (Q/'BUDGET.json').exists()
start=datetime(2026,11,10,9,25,15,tzinfo=timezone.utc).timestamp()
(Q/'BUDGET.json').write_text(json.dumps({'start_unix':start,'preparation_deadline_unix':start+5400,'deadline_unix':start+72*3600,'per_condition_seconds':28800,'report_reserve_seconds':900,'shutdown_enabled':False,'final_test_locked':True},indent=2))
for name in ['__init__.py','common.py','checkpoints.py','reader.py','restore.py','worker.py','core.py','monitor.py']:
    p=ROOT/'ner_training_queue01'/name
    text=p.read_text(encoding='utf-8') if p.exists() else ''
    text=text.replace('ner_training_queue01','sa_training_queue01').replace('NER_CONDITION_DIR','SA_CONDITION_DIR')
    if name=='core.py':
        text=text.replace('from encoder_main01.core import update,collator,labels','from .objective import update,collator\n\ndef labels():return ["negative","positive"]')
        text=text.replace('_ner_','_sa_').replace("f'ner:{split}:'","f'sa:{split}:'").replace('AutoModelForTokenClassification','AutoModelForSequenceClassification')
    if name=='worker.py':
        text=text.replace('from encoder_main01.control import','from .control import').replace("epoch_valid_labels","epoch_examples").replace("training_first_subword_labels","training_examples").replace("values['valid_labels']","values['examples']")
        text=text.replace("metrics['f1']","metrics['macro_f1']").replace("'dev_micro_f1'","'dev_macro_f1'")
        text=text.replace("'dev_precision':metrics['precision'],'dev_recall':metrics['recall'],\n                'bio_repairs':metrics['bio_repairs'],","'dev_accuracy':metrics['accuracy'],'dev_negative_recall':metrics['per_class']['0']['recall'],\n                'dev_positive_recall':metrics['per_class']['1']['recall'],")
        text=text.replace("'training_valid_labels'","'training_examples'")
        text=text.replace("result={'microbatch':microbatch", "result={'microbatch':microbatch")
        text=text.replace("longest=sorted(range(len(data)),key=lambda i:(-len(data[i]['input_ids']),i))[:16]", "longest=read_json(D.parent/'FIXTURE.json')['longest_indices'];assert indices[:400]==read_json(D.parent/'FIXTURE.json')['ordered_indices']")
    (Q/name).write_text(text,encoding='utf-8')
text=(ROOT/'encoder_main01/control.py').read_text().replace('epoch_valid_labels','epoch_examples')
(Q/'control.py').write_text(text,encoding='utf-8')
shutil.copy2(Path('C:/Users/Aru/Documents/Codex/2026-11-09/new-chat/outputs/sa_training_20261110/INPUT_AUDIT.json'),Q/'INHERITED_INPUT_AUDIT.json')
