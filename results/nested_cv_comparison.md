# Nested Cross-Validation Comparison

| Metric | Random Forest | Decision Tree |
|---|---:|---:|
| Inner best_score_ (mean across 5 outer folds) | 0.499 | 0.476 |
| Outer nested CV score (mean across 5 outer folds) | 0.486 | 0.463 |
| Gap (inner - outer) | 0.013 | 0.013 |
| Mean absolute gap | 0.022 | 0.026 |
| Inner score std | 0.005 | 0.013 |
| Outer score std | 0.027 | 0.021 |
| Gap std | 0.031 | 0.032 |

## Best-parameter frequency

### Random Forest
- `{"max_depth": 3, "min_samples_split": 2, "n_estimators": 100}` → 1 fold(s)
- `{"max_depth": 5, "min_samples_split": 10, "n_estimators": 200}` → 1 fold(s)
- `{"max_depth": 5, "min_samples_split": 10, "n_estimators": 50}` → 1 fold(s)
- `{"max_depth": 5, "min_samples_split": 2, "n_estimators": 200}` → 1 fold(s)
- `{"max_depth": 5, "min_samples_split": 5, "n_estimators": 50}` → 1 fold(s)

### Decision Tree
- `{"max_depth": 3, "min_samples_split": 2}` → 4 fold(s)
- `{"max_depth": 5, "min_samples_split": 10}` → 1 fold(s)
