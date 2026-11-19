"""Preparation-only deterministic adapters; old sources are read, never written."""
import ast,difflib
from pathlib import Path
from pilot.common import ROOT
Q=ROOT/'ner_training_queue01';OLD=ROOT/'encoder_lr_screen02_recovery01'
def replace(s,a,b):
    assert a in s,a
    return s.replace(a,b)
def emit(name,s):
    ast.parse(s);(Q/name).write_text(s,encoding='utf-8')

s=(OLD/'checkpoints.py').read_text(encoding='utf-8')
# Retention runs only after live-state nonmutation and independent publication passed.
s=replace(s,"    append(Path(directory)/'save_events.jsonl'","    retain(directory)\n    append(Path(directory)/'save_events.jsonl'")
original=(ROOT/'encoder_lr_screen02/checkpoints.py').read_text(encoding='utf-8')
retention=original[original.index('def retain('):original.index('\ndef latest(')]
retention=replace(retention,"for meta in keep:verify_payload(directory/meta['file'],meta)","for meta in keep:\n        # Previously independent-verified survivors, stable bytes required now.\n        proof=read_json(directory/f'generation_{meta[\"sequence\"]:06d}.independent.json')\n        assert proof['accepted'] and proof['sha256_after']==meta['sha256']\n        assert file_hash(directory/meta['file'])==meta['sha256']")
emit('checkpoints.py',s+'\nfrom encoder_lr_screen02.checkpoints import durable_json\n'+retention)

s=(OLD/'reader.py').read_text(encoding='utf-8')
s=replace(s,'from .common import COMPONENTS,utc,durable','from .common import COMPONENTS,utc,durable,config')
s=replace(s,"assert p['updates']==p['epoch']*5587+math.ceil(p['cursor']/16)","c=config();assert p['updates']==p['epoch']*c['updates_per_epoch']+math.ceil(p['cursor']/c['effective_batch'])")
s=replace(s,"0<=p['cursor']<=89386","0<=p['cursor']<=c['feature_counts']['train']")
s=replace(s,"assert set(params)==set(state['optimizer']['state'])","assert set(params)==set(state['optimizer']['state']) if p['updates'] else not state['optimizer']['state']")
s=replace(s,"'unchanged_historical_structural_rules_applied_to_same_loaded_state':True","'historical_structural_rules_parameterized_for_feature_count_and_zero_update':True")
emit('reader.py',s)

s=(OLD/'restore.py').read_text(encoding='utf-8')
s=replace(s,'from .common import D,INPUT,PIN,COMPONENTS,utc,durable','from .common import D,COMPONENTS,utc,durable,config')
s=replace(s,'from .streams import fingerprint_stream,compare_streams','from encoder_lr_screen02_recovery01.streams import fingerprint_stream,compare_streams')
s=replace(s,"assert path.resolve()==INPUT.resolve() or path.resolve().is_relative_to(D.resolve()),'Unapproved recovery source'\n    if path.resolve()==INPUT.resolve():assert file_hash(path)==PIN==meta['sha256']","assert path.resolve().is_relative_to(D.resolve()),'Unapproved recovery source'\n    assert meta['run_hash']==config()['run_hash'],'Incompatible condition'")
s=replace(s,"sched.base_lrs==[3e-5]","sched.base_lrs==[config()['learning_rate']]")
emit('restore.py',s)

# Keep exact prediction/evaluation logic; model-specific features and explicit token_type_ids.
s=(ROOT/'encoder_main01/main.py').read_text(encoding='utf-8')
start=s.index('def dev_rows():');end=s.index('\ndef prepare():')
body=s[s.index('def predict('):s.index('\ndef run(deadline,attempt):')]
body=replace(body,"for k in ['input_ids','attention_mask']","for k in ['input_ids','attention_mask','token_type_ids'] if k in feature")
header='''import time
from collections import Counter
from datetime import datetime,timezone
from pilot.common import ROOT,read_json,read_jsonl,write_json,write_jsonl,file_hash,digest
from pilot.evaluate import summarize
from pilot.inputs import first_word_predictions
from pilot.scoring import bio_spans
from encoder_main01.control import status
from .core import labels
from .common import RUN
'''
emit('evaluation.py',header+'\n'+s[start:end]+'\n'+body)

s=(OLD/'fresh_reload.py').read_text(encoding='utf-8')
s=replace(s,'from encoder_main01.core import load_data','from .core import load_data,initialize')
s=replace(s,'utc,durable,initialize,verify_plan','utc,durable,verify_plan')
s=replace(s,'import encoder_main01.main as evaluation','from . import evaluation')
emit('fresh_reload.py',s)

diff=[]
for name in ['checkpoints.py','reader.py','restore.py','fresh_reload.py']:
    diff.extend(difflib.unified_diff((OLD/name).read_text(encoding='utf-8').splitlines(True),(Q/name).read_text(encoding='utf-8').splitlines(True),fromfile='encoder_lr_screen02_recovery01/'+name,tofile='ner_training_queue01/'+name))
diff.extend(difflib.unified_diff((ROOT/'encoder_main01/main.py').read_text(encoding='utf-8').splitlines(True),(Q/'evaluation.py').read_text(encoding='utf-8').splitlines(True),fromfile='encoder_main01/main.py',tofile='ner_training_queue01/evaluation.py'))
(Q/'ADAPTER_DIFF.patch').write_text(''.join(diff),encoding='utf-8')
