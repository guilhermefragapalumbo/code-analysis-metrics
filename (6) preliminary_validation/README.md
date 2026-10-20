# (6) Preliminary Validation

This folder contains everything behind the preliminary validation in the article (Tables 3–6): the evaluation
script and prompts exactly as used, every archived LLM output, the per-task scores that fed the metrics, and a
script that recomputes the published tables from those files.

```
(6) preliminary_validation/
├── evaluation/
│   ├── code_analysis.py                  evaluation script, archived as run
│   ├── compliance_checks/                the 69 compliance checks the script read (one file per stage)
│   ├── prompts/                          system prompt, output instruction, one fully instantiated user prompt
│   └── export_prompts.py                 regenerates prompts/ from code_analysis.py (no API call)
├── results/
│   ├── raw/<project>/<state>/<file>/     one JSON per lifecycle stage, as returned by the LLM
│   ├── task_scores_by_state/             per-task scores per project state, as fed to the metric script
│   └── house_price_before_training_runs_aggregated/
└── analysis/
    ├── reproduce.py                      recomputes Tables 3–6 and compares them with the article
    ├── manuscript_values.json            the published values, transcribed for comparison
    └── output/                           scores_long.csv, recomputed tables, REPRODUCIBILITY_REPORT.md
```

**Quick start.** Python 3.8 or later, standard library only:

```bash
cd "(6) preliminary_validation/analysis"
python reproduce.py
```

The script writes `output/REPRODUCIBILITY_REPORT.md`, which compares every table cell with the published value.

---

## 1. Evaluation protocol

### 1.1 Model and inference parameters

| Item | Value |
|---|---|
| API | OpenAI Chat Completions, through the legacy `openai.ChatCompletion.create` interface (`openai` Python package < 1.0) |
| Model | Selected by the script's `--model` argument, whose default is `gpt-4`. The value passed in each run was not logged, so the exact model snapshot cannot be stated. |
| `temperature` | 0.3 |
| `top_p`, `max_tokens`, `seed`, penalties | Not set (API defaults). No seed was used. |
| Messages | One `system` message (`prompts/system_prompt.txt`) and one `user` message per call |
| Evaluation dates | September – October 2025 |

### 1.2 Prompts

- **System prompt:** [`evaluation/prompts/system_prompt.txt`](evaluation/prompts/system_prompt.txt). It defines the task and the 0–5
  scoring rubric given to the model:

  | Score | Rubric text given to the model |
  |---|---|
  | 0 | The task IS NOT applicable to the context of the problem and code. |
  | 1 | The task IS applicable to the context of the code but is not implemented at all. |
  | 2 | The task IS applicable to the context of the code but is barely present and/or lacks key components. |
  | 3 | The task IS applicable to the context of the code and somewhat implemented, but is missing some required elements. |
  | 4 | The task IS applicable to the context of the code and mostly implemented, but is with minor deficiencies. |
  | 5 | The task IS applicable to the context of the code and is fully implemented with no need for improvement. |

- **User prompt:** built by `build_user_prompt()` in `code_analysis.py`: the full source file, then the compliance checks of one
  stage (`- Task: … | Check: … | Description: …`), then the output instruction
  ([`output_format_instruction.txt`](evaluation/prompts/output_format_instruction.txt)). A complete example with all variables
  filled in, exactly as sent, is
  [`example_user_prompt_food_allergy_before_main_training.txt`](evaluation/prompts/example_user_prompt_food_allergy_before_main_training.txt).

### 1.3 What the model receives per call

- **One source file and one lifecycle stage per call.** With `--stage all`, the script iterates over the files and, for each
  file, over the seven stages, so each file receives seven calls.
- **The whole file**, read as text, with no truncation, chunking or preprocessing. Nothing about other files of the project
  is included; each file is assessed on its own.
- The script opens files with the platform default encoding. The evaluations ran on Windows (cp1252), so non-ASCII
  characters in the source, such as Portuguese accents in comments and labels, reached the model mis-decoded
  (e.g. `ObservaÃ§Ãµes` for `Observações`). The example prompt reproduces this.
- **All checks of the stage in one prompt**, in the order of the stage file in `evaluation/compliance_checks/`.
- Only `.py` files were evaluated. The evaluated files of each project are listed in section 3 and in
  [`../ML Projects/README.md`](../ML%20Projects/README.md).

### 1.4 Output format, parsing and failures

- The model is asked for a JSON array with one object per check:
  `{"Task Name", "Compliance Check", "Score" (0–5), "Reasoning", "Recommendation"}`.
- The response is parsed with `json.loads` as returned. There is no repair, no retry and no post-hoc change to scores.
- If a response does not parse, the script prints it and saves nothing for that file and stage, which would appear as a
  missing stage file. None of the archived passes is missing a stage, so no response was lost to parse errors in the
  data published here.
- A score of 0 means "not applicable", as defined in the rubric. Zeros are kept in the raw outputs and excluded from the
  metrics as described in section 2.

### 1.5 Repeated runs

To estimate run-to-run variability (Tables 3 and 5), the **Model Training and Tuning** stage was evaluated in five runs
with identical inputs (the same prompt, file and parameters). Nothing was changed between runs, so the differences come
from sampling alone. The runs are stored as `model_training_and_tuning_run1.json` … `_run5.json`. The un-suffixed file of
each stage belongs to the seven-stage pass whose scores feed the metrics (Tables 4 and 6).

### 1.6 Compliance checks used

`evaluation/compliance_checks/` holds the definitions as read by the script. They match
[`(3) compliance_check_development/compliance_list.json`](../(3)%20compliance_check_development/compliance_list.json) except
for one name: the check of *Hyperparameter Initialization* was called *Default Hyperparameter Setup* when the evaluations
were run and *Hyperparameter Setup* in the published list. Its description is identical.

---

## 2. From scores to metrics

1. **Per-task score per project state.** For each task, the mean of the scores over the files scanned in that state, zeros
   included (`results/task_scores_by_state/`). For the Food Allergy project this is `main.py` only (section 3).
2. **Metrics.** Each metric applies its rule to these per-task scores. Most metrics first drop tasks scored 0 (not
   applicable), and a few keep them; the rule, zero handling and inputs of each metric are listed in
   [`(5) metrics/implementation/implemented_metrics.json`](../(5)%20metrics/implementation/implemented_metrics.json).
3. **Percentages.** Values on a 0–5 scale are reported as value / 5 × 100, and points-based metrics as points / maximum
   points × 100.

The metric code is in [`(5) metrics/implementation`](../(5)%20metrics/implementation): `metrics.py`, a
standard-library version used by `reproduce.py`, and `original/CMC_new_metrics.py`, the script that produced the published
values. On the four archived project states the two give identical results for all 41 metrics.

---

## 3. What was evaluated

| Project | State | Evaluated files | Model Training and Tuning runs (Tables 3 and 5) | Metrics (Tables 4 and 6) |
|---|---|---|---|---|
| Food Allergy | before | `main.py` | ×5 | `main.py` |
| Food Allergy | after | `main.py` | ×5 | `main.py` |
| House Price | before | `Script1_TPRecurso_G4.py`, `Script2_TPRecurso_G4.py` | ×5 | both scripts |
| House Price | after | `Script1_TPRecurso_G4.py`, `Script2_TPRecurso_G4.py` | ×5 | both scripts |

---

## 4. Data for re-analysis

`analysis/output/scores_long.csv` contains every archived score, one row per combination:

| Column | Meaning |
|---|---|
| `project` | `food_allergy` or `house_price` |
| `state` | `before` or `after` the interventions |
| `file` | scanned file, relative to the project folder in `ML Projects/` |
| `run` | `full` for the seven-stage pass; `1`–`5` for the five Model Training and Tuning runs |
| `stage` | lifecycle stage |
| `task`, `compliance_check` | as returned by the model |
| `score` | 0–5 |
| `applicable` | 1 if `score` > 0; 0 if the model scored the check as not applicable |
| `source` | the raw JSON file the row comes from |

Two caveats when using it:

- `task` and `compliance_check` are copied from the model's output. In a few rows the model shortened or renamed them
  (e.g. *Data Drift Monitoring*). Use the 69 names in `evaluation/compliance_checks/` as the reference list.
- House Price *before* has no rows: for this state only the per-task scores
  (`results/task_scores_by_state/house_price_before.csv`) and the per-run means of Table 5 were archived (section 5).

The definitions of the 41 metrics are in [`(5) metrics`](../(5)%20metrics): `metrics.md` as a table and
`implementation/implemented_metrics.json` in machine-readable form.

---

## 5. Known limitations

- **Table 5 (House Price, five runs), *after* column.** The outputs of these runs were not retained, so this column is
  reported as originally obtained and cannot be recomputed. The evaluation prompt has since been revised, so re-running
  the five runs today would not reproduce the conditions of the original evaluation and could give different results.
- **Table 5, *before* column.** House Price was evaluated on two files. For each run, the archived value of each check is
  its mean over the two files, excluding zeros (`results/house_price_before_training_runs_aggregated/`), which is why
  non-integer values such as 3.5 occur. The reported mean and SD are computed across these five per-run values.
- **House Price *before*: per-task scores only.** Table 6 *before* is recomputed from the archived per-task scores
  (`results/task_scores_by_state/house_price_before.csv`); the per-file outputs behind them were not retained.
- **Model identity.** See section 1.1.
- Cells that do not reproduce are listed in `analysis/output/REPRODUCIBILITY_REPORT.md`.
