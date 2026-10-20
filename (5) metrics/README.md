# (5) Metrics

Metrics aggregate the 0–5 compliance-check scores into indicators of ethical compliance, per lifecycle stage and per
ethical dimension. This folder contains the 41 metrics defined for the framework and computed in the preliminary
validation; Tables 4 and 6 of the article report 23 of them.

| File | Content |
|---|---|
| [`metrics.md`](metrics.md) | All 41 metrics in one table: rule, handling of not-applicable scores, and input tasks |
| `implementation/implemented_metrics.json` | The same definitions in machine-readable form (below) |
| `implementation/original/CMC_new_metrics.py` | The script that produced the values in Tables 4 and 6 (requires pandas) |
| `implementation/metrics.py` | Standard-library version of the same functions, used by `(6) preliminary_validation/analysis/reproduce.py`. It gives the same result as the original for all 41 metrics on the four project states of the validation. |

## How a metric is computed

1. Each compliance check has a score from 0 to 5 per project state; 0 means *not applicable*.
2. A metric takes a fixed set of input tasks and applies its rule (for example mean, minimum, median, weighted sum,
   proportion above a threshold or standard deviation).
3. Most metrics first drop the inputs scored 0, so that checks that do not apply to a project do not lower its score; a
   few keep them. This is stated per metric.
4. The result is reported as a percentage: a 0–5 value as value / 5 × 100, a points-based value as points / maximum
   points × 100.

Example:

| Metric | Rule | Inputs |
|---|---|---|
| Data Integrity Score | minimum | Data Quality Checks, Schema Validation |
| Auditability Index | mean | Metadata Management, Data Versioning, Logging and Debugging, Model Registry Management, Experiment Tracking |

## Fields of `implemented_metrics.json`

| Field | Meaning |
|---|---|
| `id` | Function name in `CMC_new_metrics.py` |
| `name` | Metric name as printed by the original script |
| `reported_in_article_as` | Name used in Tables 4 and 6, or `null` if the metric is not reported in the article |
| `inputs` | Task names the metric is computed from (read from the source code) |
| `aggregation` | Rule applied to the inputs, e.g. `mean`, `min`, `median`, `points`, `weighted_sum`, `product`, `std`, `proportion_at_least` |
| `zero_handling` | `nonzero`: tasks scored 0 (not applicable) are dropped first. `all`: zeros are kept. |
| `native_unit` | `percentage`, `score_0_5` or `points`, as printed by the original. The article reports all of them as percentages. |
| `weights`, `threshold`, `min_inputs`, `inputs_pre`/`inputs_post` | Rule parameters, where applicable |
| `note` | Behaviour of the original code worth knowing when re-analysing (e.g. weights that are not renormalised) |

## Computing the metrics

For one project state:

```bash
cd "(5) metrics/implementation"
python metrics.py "../../(6) preliminary_validation/results/task_scores_by_state/food_allergy_before.csv"
```

For all four project states of the validation, with a comparison against the article:
[`(6) preliminary_validation/analysis/reproduce.py`](../(6)%20preliminary_validation/analysis/reproduce.py).
