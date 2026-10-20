"""
Metric implementations used for the preliminary validation (Tables 4 and 6).

Standard-library port of original/CMC_new_metrics.py, which produced the
published values. The port keeps the original semantics exactly, including
its zero handling, which differs between metrics:

  * "nonzero" metrics drop tasks whose score is 0 (not applicable) first;
  * "all" metrics keep zeros (e.g. Data Integrity Score uses min() over all).

Input:  a mapping {task name: score in [0, 5]} (one aggregated score per task).
Output: each metric's value in its native unit, plus the percentage reported
        in the article (value / 5 * 100, or points / max points * 100).

Run `python metrics.py <task_scores.csv>` to print all metrics for one state.
"""
import csv
import json
import statistics
import sys
from math import prod
from pathlib import Path

DEFINITIONS = json.loads((Path(__file__).resolve().parent / 'implemented_metrics.json').read_text(encoding='utf-8'))


def _scores(task_scores, tasks, nonzero):
    vals = [task_scores[t] for t in tasks if t in task_scores]
    return [v for v in vals if v != 0] if nonzero else vals


def _std(vals):
    # pandas Series.std(): sample standard deviation, NaN for a single value
    return statistics.stdev(vals) if len(vals) > 1 else None


def compute(defn, task_scores):
    """Return the metric value in its native unit, or None where the original yields NaN/raises."""
    rule = defn['aggregation']
    tasks = defn['inputs']
    v = _scores(task_scores, tasks, defn['zero_handling'] == 'nonzero')

    if rule == 'mean':
        return statistics.mean(v) if v else 0.0
    if rule == 'points':
        # sum of non-zero scores over 5 points per non-zero task
        return (sum(v), 5.0 * len(v)) if v else (0.0, 5.0 * len(tasks))
    if rule == 'min':
        return min(v) if v else 0.0
    if rule == 'median':
        return round(statistics.median(v), 2) if v else 0.0
    if rule == 'median_minus_std':
        if not v:
            return 0.0
        s = _std(v)
        return None if s is None else statistics.median(v) - s
    if rule == 'mean_times_saturation':
        return statistics.mean(v) * (sum(1 for x in v if x == 5) / len(v)) * 5 if v else 0.0
    if rule == 'mean_minus_gap_to_max':
        return statistics.mean(v) - (5 - min(v)) if v else 0.0
    if rule == 'product':
        if defn.get('min_inputs') and len(v) < defn['min_inputs']:
            return 0.0
        return float(prod(v)) if v else 0.0
    if rule == 'weighted_sum':
        w = dict(zip(tasks, defn['weights']))
        present = [t for t in tasks if t in task_scores and task_scores[t] != 0]
        return sum(task_scores[t] * w[t] for t in present) if present else 0.0
    if rule == 'equal_weight_pair':
        # original: 0.5 * first + 0.5 * second row; raises with fewer than two
        return 0.5 * v[0] + 0.5 * v[1] if len(v) >= 2 else (0.0 if not v else None)
    if rule == 'min_max_ratio':
        return (min(v) / max(v) if max(v) != 0 else 0) * 5 if v else 0.0
    if rule == 'proportion_below':
        return sum(1 for x in v if x < defn['threshold']) / len(v) * 5 if v else 0.0
    if rule == 'proportion_at_least':
        return sum(1 for x in v if x >= defn['threshold']) / len(v) * 5 if v else 0.0
    if rule == 'std':
        return _std(v) if v else 0.0
    if rule == 'pre_post_balance':
        pre = sum(task_scores.get(t, 0) for t in defn['inputs_pre'])
        post = sum(task_scores.get(t, 0) for t in defn['inputs_post'])
        diff = pre - post
        return diff / max(abs(diff), 1)
    raise ValueError(f'unknown aggregation {rule!r}')


def as_percent(defn, value):
    """Convert a native value to the percentage used in the article.

    The original prints 0-5 and points values rounded to 2 decimals, and the
    article's percentages were derived from those printed values, so the same
    rounding is applied before converting.
    """
    if value is None:
        return None
    if defn['aggregation'] == 'points':
        total, max_points = value
        return round(round(total, 2) / max_points * 100, 2)
    if defn['native_unit'] == 'score_0_5':
        return round(round(value, 2) / 5 * 100, 2)
    return round(value / 5 * 100, 2)


def compute_all(task_scores):
    out = {}
    for d in DEFINITIONS:
        value = compute(d, task_scores)
        out[d['name']] = as_percent(d, value)
    return out


def load_task_scores(path):
    with open(path, encoding='utf-8') as f:
        return {r['Task Name']: float(r['Score']) for r in csv.DictReader(f)}


if __name__ == '__main__':
    for name, pct in compute_all(load_task_scores(sys.argv[1])).items():
        print(f'{name:50s} {pct}')
