"""Local presentation reproduction; no network, corpus, raw answers, or models."""
from pathlib import Path
import hashlib, json, subprocess, sys

ROOT = Path(__file__).resolve().parent
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()

if __name__ == '__main__':
    b = json.loads((ROOT/'REPRODUCTION_BINDINGS.json').read_text())
    for path, key in [('data/RESULTS.json','results_sha256'),('data/COMPLETE.json','complete_sha256'),('code/build_figures.py','builder_sha256')]:
        if sha(ROOT/path) != b[key]:
            raise ValueError('Changed bound file: '+path)
    complete = json.loads((ROOT/'data/COMPLETE.json').read_text())
    for p in (ROOT/'data').glob('TABLE_*.csv'):
        if sha(p) != complete['output_sha256'][p.name]:
            raise ValueError('Changed original detailed table: '+p.name)
    output = ROOT/'regenerated'
    if output.exists():
        raise FileExistsError('Move the previous regenerated folder before another run')
    subprocess.run([sys.executable, str(ROOT/'code/build_figures.py'),
                    '--results',str(ROOT/'data/RESULTS.json'),'--results-sha256',b['results_sha256'],
                    '--complete',str(ROOT/'data/COMPLETE.json'),'--complete-sha256',b['complete_sha256'],
                    '--output',str(output)], check=True)
    generated = output/'main_results_table.tex'
    if generated.read_bytes() != (ROOT/'expected/main_results_table.tex').read_bytes():
        raise ValueError('Regenerated main table differs')
    if 'evidence_table_builder_sha256' in b:
        for name,key in [('code/build_evidence_tables.py','evidence_table_builder_sha256'),('code/build_illustration_table.py','illustration_builder_sha256'),('data/illustrative_case_excerpt.json','illustrative_excerpt_sha256')]:
            if sha(ROOT/name)!=b[key]:raise ValueError('Changed bound file: '+name)
        tables=output/'tables'
        subprocess.run([sys.executable,str(ROOT/'code/build_evidence_tables.py'),
                        '--results',str(ROOT/'data/RESULTS.json'),'--results-sha256',b['results_sha256'],
                        '--output',str(tables)],check=True)
        subprocess.run([sys.executable,str(ROOT/'code/build_illustration_table.py'),
                        '--cases',str(ROOT/'data/illustrative_case_excerpt.json'),
                        '--cases-sha256',b['illustrative_excerpt_sha256'],
                        '--output',str(tables/'illustrative_cases.tex')],check=True)
        for p in (ROOT/'expected/tables').glob('*.tex'):
            if p.read_bytes()!=(tables/p.name).read_bytes():raise ValueError('Table differs: '+p.name)
        print('Six computed supplementary tables and four-case illustrative table reproduced exactly.')
    if 'conference_table_builder_sha256' in b:
        script=ROOT/'code/build_conference_tables.py'
        if sha(script)!=b['conference_table_builder_sha256']:raise ValueError('Conference table builder changed')
        conference=output/'conference'
        subprocess.run([sys.executable,str(script),'--results',str(ROOT/'data/RESULTS.json'),
                        '--results-sha256',b['results_sha256'],'--output',str(conference)],check=True)
        for p in (ROOT/'expected/conference').glob('*.tex'):
            if p.read_bytes()!=(conference/p.name).read_bytes():raise ValueError('Conference table differs: '+p.name)
        print('Two conference tables reproduced exactly; ten generated LaTeX tables match in total.')
    if 'dg2027_builder_sha256' in b:
        script = ROOT/'code/build_dg2027_assets.py'
        if sha(script) != b['dg2027_builder_sha256']:
            raise ValueError('Compact presentation builder changed')
        expected = ROOT/'expected/dg2027'
        if sha(expected/'PRESENTATION_VALUES.json') != b['dg2027_presentation_values_sha256']:
            raise ValueError('Compact presentation numeric source changed')
        dg2027 = output/'dg2027'
        subprocess.run([sys.executable, str(script), '--results', str(ROOT/'data/RESULTS.json'),
                        '--output', str(dg2027)], check=True)
        for name in ('ner_results.tex', 'sa_results.tex', 'schema_diagnostics.tex', 'PRESENTATION_VALUES.json'):
            if (expected/name).read_bytes() != (dg2027/name).read_bytes():
                raise ValueError('Compact presentation differs: '+name)
        print('Three additional compact tables and numeric figure values reproduced exactly; thirteen LaTeX tables match in total.')
    print('Main LaTeX table reproduced exactly; original detailed CSV hashes verified. Four figure families regenerated.')
    print('PDF creation timestamps/font rendering can prevent byte-identical image files; inspect their saved numeric source and rendered appearance.')
