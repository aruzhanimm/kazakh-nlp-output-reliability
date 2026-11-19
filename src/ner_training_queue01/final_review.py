"""Terminal report supplement. Saved artifacts only; never loads a model or dataset test."""
import csv,json,zipfile,hashlib,difflib
from pathlib import Path
from datetime import datetime
from pilot.common import ROOT,read_json,file_hash
from .common import Q,utc,durable
from .report import optional,csv_out

def read_csv(path):
    if not path.exists():return []
    with path.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))

def main():
    q=read_json(Q/'QUEUE.json');final=read_json(Q/'FINAL_STATUS.json')
    assert q['status'] in ['blocked','partial','verified_complete']
    rows=[];curves=[];types=[];details=[]
    baseline=read_json(Q/'BASELINE_RESOURCES.json')
    for e,r in zip(q['conditions'],final['conditions']):
        x=dict(r);d=Path(e['directory']) if e.get('directory') else None
        if r['condition']=='xlmr_lr1e-5_seed13':
            x.update(update_wall_seconds=baseline['total_recorded_update_seconds_excluding_gap_including_replay'],peak_allocated_bytes=baseline['peak_allocated_bytes'],
                time_scope='historical original+continuation+935 replay update intervals; entire S3-correlated interval excluded; includes repeated935')
            curve_path=ROOT/'checkpoint_recovery01/learning_curve.csv';type_path=ROOT/'encoder_cont01/per_type_all_epochs.csv'
        elif r['condition']=='xlmr_lr3e-5_seed13_recovery':
            s=optional(ROOT/'encoder_lr_screen02_recovery01/resource_results.json')
            x.update(update_wall_seconds=s.get('C_and_continuation_synchronized_update_seconds'),time_scope='current recovery C+continuation; inherited first1000 and separate R validation excluded')
            curve_path=ROOT/'encoder_lr_screen02_recovery01/learning_curve.csv';type_path=ROOT/'encoder_lr_screen02_recovery01/per_type_all_epochs.csv'
        else:
            s=optional(d/'RESOURCES.json') if d else {}
            x.update(update_wall_seconds=s.get('training_synchronized_update_wall_seconds'),time_scope='fresh condition synchronized update intervals')
            curve_path=d/'learning_curve.csv' if d else Q/'nonexistent';type_path=d/'per_type_all_epochs.csv' if d else Q/'nonexistent'
            ended=[datetime.fromisoformat(v['ended_utc']).timestamp() for v in e.get('stages',{}).values() if 'ended_utc' in v]
            if ended and 'start_unix' in e:x['condition_wall_seconds']=max(ended)-e['start_unix']
        rows.append(x)
        curves += [{'condition':r['condition'],**v} for v in read_csv(curve_path)]
        types += [{'condition':r['condition'],**v} for v in read_csv(type_path)]
        for name,stage in e.get('stages',{}).items():details.append({'condition':e['id'],'stage':name,**stage})
    csv_out(Q/'ALL_CONDITIONS.csv',rows);csv_out(Q/'ALL_LEARNING_CURVES.csv',curves);csv_out(Q/'ALL_PER_TYPE_RESULTS.csv',types)
    durable(Q/'RUNTIME_STAGE_BREAKDOWN.json',{'utc':utc(),'stages':details,'baseline':baseline,'scope':'wall stages include initialization/readers; synchronized update totals do not isolate GPU kernel time'})
    def show(x):return '—' if x is None else str(x)
    table='\n'.join('| '+' | '.join(show(x.get(k)) for k in ['model','lr','seed','selected_epoch','full_dev_f1','update_wall_seconds','peak_allocated_bytes','checkpoint','status','verification'])+' |' for x in rows)
    ag='\n'.join('| '+' | '.join(show(x.get(k)) for k in ['model','selected_lr','seed13','seed42','seed2026','mean','sample_sd'])+' |' for x in final['seed_aggregates'])
    text=f'''# Итог последовательной NER-очереди

Статус: **{final['status']}**. Историческая причина прежних checkpoint-инцидентов: **unresolved**.
Текущие проверки оценивают новые сохранения и исполнения; они не доказывают причину старого сбоя.
Конкретный блокер, если есть: `{final.get('failure',{}).get('error','нет') if final.get('failure') else 'нет'}`.

| Модель | LR | Seed | Выбранная эпоха | Full-dev F1 | Update wall, s | Peak allocated, bytes | Checkpoint | Статус | Fresh verification |
|---|---:|---:|---:|---:|---:|---:|---|---|---|
{table}

Временная колонка — сумма инструментированных синхронизированных интервалов обновления,
не чистое время GPU kernels. Область измерения каждого запуска явно указана в ALL_CONDITIONS.csv.
Для исторического1e-5 исключён целый интервал, сопоставленный с S3 sleep; включён повтор935updates
при восстановлении. Для recovery3e-5 отдельно учтены C+continuation, R-проверка, checkpoint,
evaluation и загрузки; унаследованные первые1000updates не выдаются за вновь измеренные.
Новые условия имеют отдельные wall-бюджеты от подготовки до проверки. RUNTIME_STAGE_BREAKDOWN.json
и RESOURCES.json разделяют этапы и показывают оставшийся overhead; паузы не объявлены compute.

| Модель | Выбранный LR | Seed13 | Seed42 | Seed2026 | Среднее | Выборочное SD |
|---|---:|---:|---:|---:|---:|---:|
{ag}

Среднее/SD вычисляются только для трёх проверенных seeds13/42/2026. Лучший seed не выбирается.
Проигравший LR сохранён. Правило LR неизменно: лучший full-dev exact-span/type micro-F1 каждого
seed13условия, точная ничья оставляет1e-5. Панель200 используется лишь для описательного отчёта
из сохранённых full-dev predictions, без новой inference и без выбора по её результату.
LR-screen использует один seed13; дополнительные seeds проверяют только выбранный LR.
Поэтому SD этой тройки не оценивает вариативность проигравшего LR и не доказывает значимость
различия между LR. Это вариативность обучения, а не неопределённость по выборке предложений.

Все достигнутые эпохи сохранены в ALL_LEARNING_CURVES.csv, per-type результаты в ALL_PER_TYPE_RESULTS.csv.
Подробные predictions, tags, spans, confidences и distributions остаются локально у своих запусков.
Завершённая эпоха сама по себе не считается проверенным завершением условия. В таблице нет нулевых
метрик вместо отсутствующих измерений. Если очередь остановилась, pending условия не запускались.

Подготовка проверила обе полные токенизации, общий видимый текст, first-subword ownership,
нормировку loss, закреплённые revisions и применение seed к инициализации/dropout/порядку данных.
Каждый новый запуск начинается от pretrained, с новой task head и optimizer/scheduler.
Actual save/readback и fresh GPU restore предшествуют основному обучению; новые checkpoints
принимаются после независимого reader. Плановый restart на1000updates проверяет все пять
компонентов. Для выбранного checkpoint требуется точное совпадение tags/spans/confidences/metrics
на11100dev в свежем процессе. Старые ожидаемые сигнатуры и неудачные аудиты не изменены.

Один controller-only дефект обнаружен до запуска новых условий: журнал отчёта recovery мог
меняться после sealing manifest. Adapter queue_v2 размещает его вне recovery. Был перезапущен
только наблюдатель без дочерних процессов; исходные launcher и training PID проверены живыми
до/после. Исходный queue.py сохранён. Отдельная ошибка индекса в ad-hoc тесте adapter описана
в CONTROLLER_V2_TEST.json; исправлен сам тест, реальных state checks не повторяли.

Научные ограничения: многократно использованный dev является exploratory; три seeds дают
ограниченную оценку вариативности, особенно у редких типов. Энкодеры и zero/few-shot LLM получают
неравные объёмы task labels. Новизна вклада, pretraining contamination и финальное test-сравнение
этой очередью не решаются. Слабые результаты не исключаются и протокол ради scores не меняется.

Master Plan прочитан как контекст и не редактировался. Серверные задачи не затронуты.
SA/LLM/новые LR или seeds не запускались. Final test остаётся закрытым, shutdown отключён.

Точная команда действующего контроллера:

```powershell
Set-Location -LiteralPath 'C:\\Users\\Aru\\Desktop\\рисерч ниц ласт\\kazakh_nlp_starter'
.\\.venv\\Scripts\\python.exe -B -X utf8 -m ner_training_queue01.queue_v2
```

Это запись фактического запуска, не команда повторять завершённые или blocked условия.
QUEUE.json содержит команды дочерних этапов, PID/create-time, конфигурации и неизменные deadlines.
При новом входе active owner не дублируется; terminal queue не запускается повторно.
Исходники: SOURCE_DIFF.patch плюс adapter в REVIEW_SOURCE_DIFF.patch; отдельные source locks
связывают обе версии. Компактный review_bundle_complete.zip не содержит весов и полных streams.
'''
    (Q/'FINAL_REPORT.md').write_text(text,encoding='utf-8')
    # The terminal report script itself is reviewable, without altering frozen execution sources.
    extra=''.join(''.join(difflib.unified_diff([],p.read_text(encoding='utf-8').splitlines(True),fromfile='/dev/null',tofile='ner_training_queue01/'+p.name)) for p in [Path(__file__),Q/'monitor.py'])
    (Q/'FINAL_SOURCE_DIFF.patch').write_text((Q/'REVIEW_SOURCE_DIFF.patch').read_text(encoding='utf-8')+extra,encoding='utf-8')
    authoritative={}
    excluded={'AUTHORITATIVE_ARTIFACT_MANIFEST.json','COMPLETE_BUNDLE_VERIFICATION.json','review_bundle_complete.zip'}
    for p in Q.rglob('*'):
        if not p.is_file() or p.name in excluded or any(x in p.parts for x in ['runtime','items','__pycache__']):continue
        authoritative[p.relative_to(Q).as_posix()]={'bytes':p.stat().st_size,'sha256':file_hash(p)}
    durable(Q/'AUTHORITATIVE_ARTIFACT_MANIFEST.json',{'utc':utc(),'files':authoritative,
        'scope':'post-queue hashes, including retained local weights and aggregate predictions; duplicate item caches and runtime caches excluded',
        'hash_is_not_commit_acceptance':'Uncommitted/failed payloads remain unaccepted regardless of being inventoried; acceptance requires generation receipts and all audits.'})
    additions=[p for p in Q.rglob('*') if p.is_file() and (p.suffix=='.log' or p.name=='queue.jsonl') and p.stat().st_size<2*2**20 and 'runtime' not in p.parts]
    additions += [Q/n for n in ['FINAL_REPORT.md','ALL_CONDITIONS.csv','ALL_LEARNING_CURVES.csv','ALL_PER_TYPE_RESULTS.csv','RUNTIME_STAGE_BREAKDOWN.json','BASELINE_RESOURCES.json','FINAL_SOURCE_DIFF.patch','final_review.py','AUTHORITATIVE_ARTIFACT_MANIFEST.json'] if (Q/n).exists()]
    # Old scientific controls are incorporated by copies, never rewritten.
    controls=[]
    for folder in ['checkpoint_recovery01','encoder_lr_screen02_recovery01']:
        for name in ['REPORT.md','FINAL_STATUS.json','resource_results.json','learning_curve.csv','per_type_all_epochs.csv','per_type_epoch05.csv','BUNDLE_VERIFICATION.json']:
            p=ROOT/folder/name
            if p.exists():controls.append(p)
    target=Q/'review_bundle_complete.zip';members={}
    with zipfile.ZipFile(Q/'review_bundle.zip') as old,zipfile.ZipFile(target,'x',zipfile.ZIP_DEFLATED) as z:
        for name in old.namelist():data=old.read(name);z.writestr(name,data);members[name]=hashlib.sha256(data).hexdigest()
        for p in additions+controls:
            name=p.relative_to(Q).as_posix() if p.is_relative_to(Q) else 'previous_controls/'+p.relative_to(ROOT).as_posix()
            if name in members:continue
            z.write(p,name);members[name]=file_hash(p)
        z.writestr('COMPLETE_MEMBER_HASHES.json',json.dumps(members,indent=2))
    with zipfile.ZipFile(target) as z:
        assert z.testzip() is None
        for n,h in members.items():assert hashlib.sha256(z.read(n)).hexdigest()==h
    durable(Q/'COMPLETE_BUNDLE_VERIFICATION.json',{'utc':utc(),'sha256':file_hash(target),'bytes':target.stat().st_size,'members':len(members),
        'all_member_hashes_verified':True,'weights_or_full_streams_included':False})

if __name__=='__main__':main()
