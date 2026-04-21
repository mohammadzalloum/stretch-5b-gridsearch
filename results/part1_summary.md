# Part 1 — GridSearchCV Summary

## Best model
- Best params: `{'max_depth': 5, 'min_samples_split': 10, 'n_estimators': 200}`
- Best inner CV F1: **0.502**
- Hold-out test F1: **0.466**
- Heatmap fixed at `min_samples_split = 10`

## One-standard-error rule
- Score threshold: **0.483**
- Chosen simpler config within one standard error:
  - `max_depth = 3`
  - `min_samples_split = 10`
  - `n_estimators = 50`
  - `mean_test_score = 0.490`

## Plateau evidence
- Number of configs within 0.005 of best: **7**

## Interpretation
The strongest-performing region is defined more by `max_depth = 5` than by a single unique configuration, which suggests a sweet spot and mild plateau. The detailed results and generalization gaps are saved in CSV form for inspection.
