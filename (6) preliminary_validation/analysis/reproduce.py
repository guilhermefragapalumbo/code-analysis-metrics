"""
Recompute the preliminary-validation results from the archived LLM outputs.

Steps
  1. Flatten every raw LLM output in ../results/raw into one long table
     (output/scores_long.csv): one row per project, state, file, run, stage
     and compliance check.
  2. Rebuild the per-task scores of each project state exactly as the
     original pipeline did (mean over the scanned files, zeros included) and
     check them against the archived ../results/task_scores_by_state.
  3. Recompute Table 3 (Food Allergy, five runs) from the raw runs, the
     "before" column of Table 5 (House Price, five runs) from the archived
     per-run means, and Tables 4 and 6 (metrics) with
     ../../(5) metrics/implementation.
  4. Compare every cell with manuscript_values.json and write
     output/REPRODUCIBILITY_REPORT.md.

Standard library only. Usage:
    python reproduce.py
"""
import csv
import json
import re
import statistics
import sys
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / 'results'
OUT = HERE / 'output'
sys.path.insert(0, str(HERE.parent.parent / '(5) metrics' / 'implementation'))
import metrics  # noqa: E402

def write_text(path, text):
    """Write UTF-8 text with LF line endings on every platform."""
    with open(path, 'w', encoding='utf-8', newline='\n') as f:
        f.write(text)


STAGES = [
    'data_ingestion_and_data_versioning', 'data_preprocessing', 'model_training_and_tuning',
    'model_evaluation', 'model_versioning', 'model_deployment', 'monitoring_and_maintenance',
]
SCANNED_FILES = {
    'food_allergy/before/main': 'Projeto/Algoritms/main.py',
    'food_allergy/after/main': 'Projeto/Algoritms/main.py',
    'house_price/after/Script1_TPRecurso_G4': 'Script1_TPRecurso_G4.py',
    'house_price/after/Script2_TPRecurso_G4': 'Script2_TPRecurso_G4.py',
}
# Files whose seven-stage pass fed the metric tables (Tables 4 and 6). For House Price "before"
# only the per-task scores were archived, so there is nothing to rebuild them from.
METRIC_INPUTS = {
    ('food_allergy', 'before'): ['food_allergy/before/main'],
    ('food_allergy', 'after'): ['food_allergy/after/main'],
    ('house_price', 'before'): [],
    ('house_price', 'after'): ['house_price/after/Script1_TPRecurso_G4', 'house_price/after/Script2_TPRecurso_G4'],
}
RUN_RE = re.compile(r'^(?P<stage>[a-z_]+?)(?:_run(?P<run>[1-5]))?\.json$')
TOL = 0.011


def half_up(x, nd=0):
    q = Decimal(1).scaleb(-nd)
    return float(Decimal(str(x)).quantize(q, rounding=ROUND_HALF_UP))


# ---------------------------------------------------------------- step 1

def load_raw():
    rows = []
    for folder, scanned in SCANNED_FILES.items():
        project, state, _ = folder.split('/')
        for path in sorted((RESULTS / 'raw' / folder).glob('*.json')):
            m = RUN_RE.match(path.name)
            if not m or m.group('stage') not in STAGES:
                continue
            run = m.group('run') or 'full'
            for item in json.loads(path.read_text(encoding='utf-8')):
                score = int(item['Score'])
                rows.append({
                    'project': project, 'state': state, 'file': scanned, 'run': run,
                    'stage': m.group('stage'), 'task': item['Task Name'],
                    'compliance_check': item['Compliance Check'], 'score': score,
                    'applicable': int(score != 0),
                    'source': f'results/raw/{folder}/{path.name}',
                })
    return rows


def write_csv(path, rows, fields=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = fields or list(rows[0])
    with path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields, lineterminator='\n')
        w.writeheader()
        w.writerows(rows)


# ---------------------------------------------------------------- step 2

def rebuild_task_scores(rows, project, state):
    folders = {f'results/raw/{f}/' for f in METRIC_INPUTS[(project, state)]}
    per_task = {}
    for r in rows:
        if r['run'] == 'full' and any(r['source'].startswith(f) for f in folders):
            per_task.setdefault(r['task'], []).append(r['score'])
    return {t: statistics.mean(v) for t, v in per_task.items()}


# ---------------------------------------------------------------- step 3

def runs_table(rows, project, state):
    """Mean and population SD over repeated runs 1-5 of the training stage of main.py."""
    per_task = {}
    for r in rows:
        if (r['project'], r['state'], r['file'], r['stage']) == (project, state, 'Projeto/Algoritms/main.py', 'model_training_and_tuning') \
                and r['run'] in {'1', '2', '3', '4', '5'}:
            per_task.setdefault(r['task'], []).append(r['score'])
    return {t: (statistics.mean(v), statistics.pstdev(v), v) for t, v in per_task.items()}


def house_before_runs():
    folder = RESULTS / 'house_price_before_training_runs_aggregated'
    runs = [{d['Task Name']: d['Mean Score'] for d in json.loads(p.read_text(encoding='utf-8'))}
            for p in sorted(folder.glob('run*.json'))]
    tasks = [t for run in runs for t in run]
    # a task missing from a run was scored 0 in both files and dropped by the old aggregation
    per_task = {t: [run.get(t, 0.0) for run in runs] for t in dict.fromkeys(tasks)}
    return {t: (statistics.mean(v), statistics.pstdev(v), v) for t, v in per_task.items()}


# ---------------------------------------------------------------- step 4

def compare_runs(recomputed, published):
    lines, ok = [], 0
    lines.append('| Task | Runs 1-5 | Recomputed mean (SD) | Reported | Mean matches | SD matches |')
    lines.append('|---|---|---|---|---|---|')
    yes = lambda m: 'yes' if m else '**no**'
    for task, (pub_mean, pub_sd) in published.items():
        if task not in recomputed:
            lines.append(f'| {task} | - | not available | {pub_mean} ({pub_sd}) | - | - |')
            continue
        mean, sd, runs = recomputed[task]
        mean_ok = half_up(mean) == pub_mean
        sd_ok = abs(half_up(sd, 2) - pub_sd) <= TOL
        ok += mean_ok + sd_ok
        runs_txt = ', '.join(f'{x:g}' for x in runs)
        lines.append(f'| {task} | {runs_txt} | {mean:.2f} ({sd:.2f}) | {pub_mean} ({pub_sd}) | {yes(mean_ok)} | {yes(sd_ok)} |')
    return lines, ok, 2 * len(published)


def compare_metrics(before, after, published):
    lines, ok = [], 0
    lines.append('| Metric | Recomputed before | Reported before | Recomputed after | Reported after | Match |')
    lines.append('|---|---|---|---|---|---|')
    for name, (pub_b, pub_a) in published.items():
        b, a = before[name], after[name]
        mb, ma = abs(b - pub_b) <= TOL, abs(a - pub_a) <= TOL
        ok += mb + ma
        flag = 'yes' if mb and ma else '**no** (' + ', '.join(s for s, m in (('before', mb), ('after', ma)) if not m) + ')'
        lines.append(f'| {name} | {b} | {pub_b} | {a} | {pub_a} | {flag} |')
    return lines, ok, 2 * len(published)


def main():
    published = json.loads((HERE / 'manuscript_values.json').read_text(encoding='utf-8'))
    rows = load_raw()
    write_csv(OUT / 'scores_long.csv', rows)

    report = ['# Reproducibility report', '',
              'Generated by `reproduce.py` from the archived LLM outputs in `../results`. '
              'Do not edit by hand.', '']

    # step 2: per-state task scores
    report += ['## Per-task scores per project state', '',
               'Rebuilt from the raw outputs (mean over scanned files, zeros included) and compared with the '
               'archived `results/task_scores_by_state/*.csv` that the metric script consumed.', '',
               '| State | Tasks | Identical to archive |', '|---|---|---|']
    task_scores = {}
    for (project, state) in METRIC_INPUTS:
        archived = metrics.load_task_scores(RESULTS / 'task_scores_by_state' / f'{project}_{state}.csv')
        if METRIC_INPUTS[(project, state)]:
            rebuilt = rebuild_task_scores(rows, project, state)
            same = set(rebuilt) == set(archived) and all(abs(rebuilt[t] - archived[t]) < 1e-9 for t in archived)
            diff = [t for t in archived if abs(rebuilt.get(t, -1) - archived[t]) >= 1e-9]
            status = 'yes' if same else 'no: ' + ', '.join(diff)
        else:
            status = 'archived per-task scores only (no per-file outputs)'
        report.append(f'| {project} {state} | {len(archived)} | {status} |')
        task_scores[(project, state)] = archived
        write_csv(OUT / f'task_scores_{project}_{state}.csv',
                  [{'task': t, 'score': round(v, 4)} for t, v in archived.items()])
    report.append('')

    summary = []

    def metric_section(number, key, project, title):
        before = metrics.compute_all(task_scores[(project, 'before')])
        after = metrics.compute_all(task_scores[(project, 'after')])
        write_csv(OUT / f'metrics_{project}.csv',
                  [{'metric': k, 'before_pct': before[k], 'after_pct': after[k]} for k in before])
        lines, ok, n = compare_metrics(before, after, published[key])
        summary.append((f'Table {number}', ok, n))
        return [f'## Table {number} - {title}', ''] + lines + ['']

    # Table 3
    t3 = {s: runs_table(rows, 'food_allergy', s) for s in ('before', 'after')}
    report += ['## Table 3 - Food Allergy, Model Training and Tuning, five runs', '',
               'Mean of runs 1-5 rounded half-up to an integer; SD is the population SD.', '']
    for state in ('before', 'after'):
        lines, ok, n = compare_runs(t3[state], published['table3_food_allergy_training'][state])
        report += [f'### {state.capitalize()}', ''] + lines + ['']
        summary.append((f'Table 3 {state}', ok, n))

    # Table 4
    report += metric_section(4, 'table4_food_allergy_metrics', 'food_allergy', 'Food Allergy metrics')

    # Table 5
    t5b = house_before_runs()
    report += ['## Table 5 - House Price, Model Training and Tuning, five runs', '',
               'House Price was evaluated on two files. For each run, the value of a check is the mean of its '
               'scores in the two files, excluding zeros, which is why values such as 3.5 occur. The mean '
               '(rounded half-up to an integer) and the population SD are computed over these five per-run '
               'values.', '']
    lines, ok, n = compare_runs(t5b, published['table5_house_price_training']['before'])
    report += ['### Before', ''] + lines + ['']
    summary.append(('Table 5 before', ok, n))
    summary.append(('Table 5 after', None, 'not recomputable: run outputs not archived'))
    report += ['### After', '',
               'Reported as originally obtained; the outputs of these runs were not archived and cannot be '
               'recomputed. See ../README.md, section "Known limitations".', '']

    # Table 6
    report += metric_section(6, 'table6_house_price_metrics', 'house_price', 'House Price metrics')

    head = ['## Summary', '', '| Table | Cells reproduced |', '|---|---|']
    cell = lambda ok, n: n if ok is None else f'{ok} / {n}'
    head += [f'| {name} | {cell(ok, n)} |' for name, ok, n in summary]
    report[4:4] = head + ['']
    write_text(OUT / 'REPRODUCIBILITY_REPORT.md', '\n'.join(report) + '\n')

    print(f'{len(rows)} rows -> output/scores_long.csv')
    for name, ok, n in summary:
        print(f'  {name}: {cell(ok, n)}')


if __name__ == '__main__':
    main()
