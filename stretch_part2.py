import os
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold, GridSearchCV


NUMERIC_FEATURES = [
    "tenure",
    "monthly_charges",
    "total_charges",
    "num_support_calls",
    "senior_citizen",
    "has_partner",
    "has_dependents",
    "contract_months",
]


def load_full_data(filepath="data/telecom_churn.csv"):
    df = pd.read_csv(filepath)
    X = df[NUMERIC_FEATURES].copy()
    y = df["churned"].copy()
    return X, y


def make_rf_grid_search(inner_random_state=42, model_random_state=42):
    param_grid = {
        "n_estimators": [50, 100, 200],
        "max_depth": [3, 5, 10, 20, None],
        "min_samples_split": [2, 5, 10],
    }

    inner_cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=inner_random_state,
    )

    estimator = RandomForestClassifier(
        class_weight="balanced",
        random_state=model_random_state,
        n_jobs=1,
    )

    search = GridSearchCV(
        estimator=estimator,
        param_grid=param_grid,
        scoring="f1",
        cv=inner_cv,
        n_jobs=-1,
        refit=True,
        return_train_score=False,
    )
    return search


def make_dt_grid_search(inner_random_state=42, model_random_state=42):
    param_grid = {
        "max_depth": [3, 5, 10, 20, None],
        "min_samples_split": [2, 5, 10],
    }

    inner_cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=inner_random_state,
    )

    estimator = DecisionTreeClassifier(
        class_weight="balanced",
        random_state=model_random_state,
    )

    search = GridSearchCV(
        estimator=estimator,
        param_grid=param_grid,
        scoring="f1",
        cv=inner_cv,
        n_jobs=-1,
        refit=True,
        return_train_score=False,
    )
    return search


def run_nested_cv(X, y, model_family, outer_random_state=123, inner_random_state=42):
    outer_cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=outer_random_state,
    )

    fold_rows = []

    for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(X, y), start=1):
        X_train_fold = X.iloc[train_idx]
        X_test_fold = X.iloc[test_idx]
        y_train_fold = y.iloc[train_idx]
        y_test_fold = y.iloc[test_idx]

        if model_family == "Random Forest":
            search = make_rf_grid_search(
                inner_random_state=inner_random_state,
                model_random_state=42,
            )
        elif model_family == "Decision Tree":
            search = make_dt_grid_search(
                inner_random_state=inner_random_state,
                model_random_state=42,
            )
        else:
            raise ValueError(f"Unknown model_family: {model_family}")

        search.fit(X_train_fold, y_train_fold)

        best_model = search.best_estimator_
        y_pred_outer = best_model.predict(X_test_fold)
        outer_f1 = f1_score(y_test_fold, y_pred_outer, zero_division=0)

        inner_best = search.best_score_
        gap = inner_best - outer_f1
        abs_gap = abs(gap)

        row = {
            "model_family": model_family,
            "outer_fold": fold_idx,
            "inner_best_score": float(inner_best),
            "outer_test_f1": float(outer_f1),
            "gap": float(gap),
            "abs_gap": float(abs_gap),
            "best_params": json.dumps(search.best_params_, sort_keys=True),
        }
        fold_rows.append(row)

        print(
            f"[{model_family} | fold {fold_idx}] "
            f"inner_best={inner_best:.3f} | outer_f1={outer_f1:.3f} | gap={gap:.3f}"
        )
        print(f"  best_params={search.best_params_}")

    return pd.DataFrame(fold_rows)


def summarize_nested_results(all_fold_results):
    summary = (
        all_fold_results
        .groupby("model_family", as_index=False)
        .agg(
            inner_best_score_mean=("inner_best_score", "mean"),
            outer_nested_score_mean=("outer_test_f1", "mean"),
            gap_mean=("gap", "mean"),
            abs_gap_mean=("abs_gap", "mean"),
            inner_best_score_std=("inner_best_score", "std"),
            outer_nested_score_std=("outer_test_f1", "std"),
            gap_std=("gap", "std"),
        )
    )

    order = {"Random Forest": 0, "Decision Tree": 1}
    summary["sort_order"] = summary["model_family"].map(order)
    summary = summary.sort_values("sort_order").drop(columns="sort_order")

    return summary


def best_params_frequency(all_fold_results):
    freq = (
        all_fold_results
        .groupby(["model_family", "best_params"], as_index=False)
        .size()
        .rename(columns={"size": "count"})
        .sort_values(["model_family", "count"], ascending=[True, False])
        .reset_index(drop=True)
    )
    return freq


def plot_nested_mean_scores(summary_df, output_path="results/nested_cv_mean_scores.png"):
    labels = summary_df["model_family"].tolist()
    inner_scores = summary_df["inner_best_score_mean"].tolist()
    outer_scores = summary_df["outer_nested_score_mean"].tolist()

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 6))
    bars1 = ax.bar(x - width / 2, inner_scores, width, label="Inner best_score_ mean")
    bars2 = ax.bar(x + width / 2, outer_scores, width, label="Outer nested mean")

    ax.set_title("Nested CV: Inner vs Outer Mean F1")
    ax.set_ylabel("F1 score")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, max(inner_scores + outer_scores) + 0.08)
    ax.legend()

    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height + 0.003,
                f"{height:.3f}",
                ha="center",
                va="bottom",
                fontsize=10,
            )

    fig.tight_layout()
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_fold_scores(all_fold_results, output_path="results/nested_cv_fold_scores.png"):
    fig, ax = plt.subplots(figsize=(9, 6))

    for model_family in ["Random Forest", "Decision Tree"]:
        subset = all_fold_results[all_fold_results["model_family"] == model_family].copy()
        subset = subset.sort_values("outer_fold")

        ax.plot(
            subset["outer_fold"],
            subset["inner_best_score"],
            marker="o",
            linestyle="-",
            label=f"{model_family} inner",
        )
        ax.plot(
            subset["outer_fold"],
            subset["outer_test_f1"],
            marker="o",
            linestyle="--",
            label=f"{model_family} outer",
        )

    ax.set_title("Nested CV Fold-by-Fold Scores")
    ax.set_xlabel("Outer fold")
    ax.set_ylabel("F1 score")
    ax.set_xticks(sorted(all_fold_results["outer_fold"].unique()))
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def save_markdown_comparison(summary_df, freq_df, output_path="results/nested_cv_comparison.md"):
    summary_idx = summary_df.set_index("model_family")

    rf = summary_idx.loc["Random Forest"]
    dt = summary_idx.loc["Decision Tree"]

    rf_freq = freq_df[freq_df["model_family"] == "Random Forest"]
    dt_freq = freq_df[freq_df["model_family"] == "Decision Tree"]

    rf_freq_md = "\n".join(
        [f"- `{row.best_params}` → {row.count} fold(s)" for row in rf_freq.itertuples()]
    ) or "- None"

    dt_freq_md = "\n".join(
        [f"- `{row.best_params}` → {row.count} fold(s)" for row in dt_freq.itertuples()]
    ) or "- None"

    md = f"""# Nested Cross-Validation Comparison

| Metric | Random Forest | Decision Tree |
|---|---:|---:|
| Inner best_score_ (mean across 5 outer folds) | {rf['inner_best_score_mean']:.3f} | {dt['inner_best_score_mean']:.3f} |
| Outer nested CV score (mean across 5 outer folds) | {rf['outer_nested_score_mean']:.3f} | {dt['outer_nested_score_mean']:.3f} |
| Gap (inner - outer) | {rf['gap_mean']:.3f} | {dt['gap_mean']:.3f} |
| Mean absolute gap | {rf['abs_gap_mean']:.3f} | {dt['abs_gap_mean']:.3f} |
| Inner score std | {rf['inner_best_score_std']:.3f} | {dt['inner_best_score_std']:.3f} |
| Outer score std | {rf['outer_nested_score_std']:.3f} | {dt['outer_nested_score_std']:.3f} |
| Gap std | {rf['gap_std']:.3f} | {dt['gap_std']:.3f} |

## Best-parameter frequency

### Random Forest
{rf_freq_md}

### Decision Tree
{dt_freq_md}
"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(md)


def main():
    os.makedirs("results", exist_ok=True)

    X, y = load_full_data()
    print(f"Full dataset size: {len(X)} rows")
    print(f"Overall churn rate: {y.mean():.2%}")

    print("\n--- Running nested CV: Random Forest ---")
    rf_fold_results = run_nested_cv(
        X,
        y,
        model_family="Random Forest",
        outer_random_state=123,
        inner_random_state=42,
    )

    print("\n--- Running nested CV: Decision Tree ---")
    dt_fold_results = run_nested_cv(
        X,
        y,
        model_family="Decision Tree",
        outer_random_state=123,
        inner_random_state=42,
    )

    all_fold_results = pd.concat([rf_fold_results, dt_fold_results], ignore_index=True)
    summary_df = summarize_nested_results(all_fold_results)
    freq_df = best_params_frequency(all_fold_results)

    all_fold_results.to_csv("results/nested_cv_fold_results.csv", index=False)
    summary_df.to_csv("results/nested_cv_summary.csv", index=False)
    freq_df.to_csv("results/nested_cv_best_params_frequency.csv", index=False)

    plot_nested_mean_scores(summary_df, "results/nested_cv_mean_scores.png")
    plot_fold_scores(all_fold_results, "results/nested_cv_fold_scores.png")
    save_markdown_comparison(summary_df, freq_df, "results/nested_cv_comparison.md")

    print("\n--- Part 2: Nested CV Summary ---")
    print(summary_df.to_string(index=False))

    print("\n--- Best params frequency ---")
    print(freq_df.to_string(index=False))

    print("\nSaved files:")
    print("  - results/nested_cv_fold_results.csv")
    print("  - results/nested_cv_summary.csv")
    print("  - results/nested_cv_best_params_frequency.csv")
    print("  - results/nested_cv_mean_scores.png")
    print("  - results/nested_cv_fold_scores.png")
    print("  - results/nested_cv_comparison.md")


if __name__ == "__main__":
    main()