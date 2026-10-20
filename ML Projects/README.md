# ML Projects

The two student projects used in the preliminary validation (article, Section "Preliminary Validation"), each in its
original state (`_BEFORE`) and after the LLM-guided interventions (`_AFTER`).

| Folder | Project in the article |
|---|---|
| `FOOD_ALLERGY_ANALYSIS_BEFORE`, `FOOD_ALLERGY_ANALYSIS_AFTER` | Food Allergy Analysis |
| `HOUSE_PRICE_FORECAST_BEFORE`, `HOUSE_PRICE_FORECAST_AFTER` | House Price Forecast |

## Files that were evaluated

Each file was evaluated on its own (see
[`(6) preliminary_validation/README.md`](../(6)%20preliminary_validation/README.md)).

| Project | State | Evaluated files |
|---|---|---|
| Food Allergy | before | `Projeto/Algoritms/main.py` |
| Food Allergy | after | `Projeto/Algoritms/main.py` |
| House Price | before | `Script1_TPRecurso_G4.py`, `Script2_TPRecurso_G4.py` |
| House Price | after | `Script1_TPRecurso_G4.py`, `Script2_TPRecurso_G4.py` |

The other files of the projects are not part of the results reported in the article. The evaluated files in this folder
are byte-identical to the files that were evaluated.

## Data

Both states of a project use the same dataset, so it is stored once, in the `_BEFORE` folder.

- Food Allergy: `FOOD_ALLERGY_ANALYSIS_BEFORE/Projeto/Algoritms/food-allergy-analysis-Zenodo.csv`, the public,
  de-identified food-allergy dataset distributed on Zenodo, as used by the students.
- House Price: `HOUSE_PRICE_FORECAST_BEFORE/DataFrame_AfterLR.csv`, the dataset as prepared by the students.
