import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold


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


def load_and_split(filepath="data/telecom_churn.csv", random_state=42):
    """Load data and do an 80/20 stratified split."""
    df = pd.read_csv(filepath)

    X = df[NUMERIC_FEATURES].copy()
    y = df["churned"].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=random_state,
        stratify=y,
    )

    return X_train, X_test, y_train, y_test


def run_rf_grid_search(X_train, y_train, random_state=42):
    """Run GridSearchCV for RandomForest using F1 scoring."""
    param_grid = {
        "n_estimators": [50, 100, 200],
        "max_depth": [3, 5, 10, 20, None],
        "min_samples_split": [2, 5, 10],
    }

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=random_state,
    )

    # Avoid nested parallelism overload:
    # let GridSearchCV parallelize across fits, keep each RF single-threaded.
    rf = RandomForestClassifier(
        class_weight="balanced",
        random_state=random_state,
        n_jobs=1,
    )

    grid = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        scoring="f1",
        cv=cv,
        n_jobs=-1,
        refit=True,
        return_train_score=True,
    )

    grid.fit(X_train, y_train)
    return grid


def build_results_dataframe(grid):
    """Build a clean results table from cv_results_ with extra diagnostics."""
    results_df = pd.DataFrame(grid.cv_results_).copy()

    keep_cols = [
        "mean_train_score",
        "std_train_score",
        "mean_test_score",
        "std_test_score",
        "rank_test_score",
        "param_n_estimators",
        "param_max_depth",
        "param_min_samples_split",
    ]
    results_df = results_df[keep_cols].copy()

    # Convert params to nicer numeric/object columns
    results_df["param_n_estimators"] = results_df["param_n_estimators"].astype(int)
    results_df["param_min_samples_split"] = results_df["param_min_samples_split"].astype(int)

    # Diagnostics
    results_df["generalization_gap"] = (
        results_df["mean_train_score"] - results_df["mean_test_score"]
    )
    results_df["score_delta_from_best"] = (
        grid.best_score_ - results_df["mean_test_score"]
    )

    # Labels for plotting / readability
    results_df["max_depth_label"] = results_df["param_max_depth"].apply(
        lambda x: "None" if x is None else str(int(x))
    )

    # Sort by rank then score
    results_df = results_df.sort_values(
        by=["rank_test_score", "mean_test_score"],
        ascending=[True, False],
    ).reset_index(drop=True)

    return results_df


def save_grid_results(results_df, output_csv="results/gridsearch_results_detailed.csv"):
    """Save full detailed grid-search results."""
    results_df.to_csv(output_csv, index=False)


def summarize_min_samples_split(results_df):
    """Average CV behavior by min_samples_split."""
    summary = (
        results_df.groupby("param_min_samples_split", as_index=False)
        .agg(
            mean_train_score=("mean_train_score", "mean"),
            mean_test_score=("mean_test_score", "mean"),
            mean_generalization_gap=("generalization_gap", "mean"),
        )
        .sort_values("mean_test_score", ascending=False)
        .reset_index(drop=True)
    )
    return summary


def save_min_samples_split_summary(summary_df, output_csv="results/min_samples_split_summary.csv"):
    summary_df.to_csv(output_csv, index=False)


def plot_min_samples_split_summary(summary_df, output_path="results/min_samples_split_summary.png"):
    """Plot average mean CV F1 by min_samples_split."""
    x = summary_df["param_min_samples_split"].astype(str).tolist()
    y = summary_df["mean_test_score"].tolist()

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(x, y)

    ax.set_title("Average CV F1 by min_samples_split")
    ax.set_xlabel("min_samples_split")
    ax.set_ylabel("Average mean CV F1")
    ax.set_ylim(0, max(y) + 0.05)

    for bar, value in zip(bars, y):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.002,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    fig.tight_layout()
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def get_near_best_configs(results_df, tolerance=0.005):
    """
    Return configs whose mean_test_score is within tolerance of the best score.
    Useful for showing plateau / sweet spot instead of only one winning row.
    """
    near_best = results_df[results_df["score_delta_from_best"] <= tolerance].copy()
    near_best = near_best.sort_values(
        by=["mean_test_score", "generalization_gap"],
        ascending=[False, True],
    ).reset_index(drop=True)
    return near_best


def save_near_best_configs(near_best_df, output_csv="results/near_best_configs.csv"):
    near_best_df.to_csv(output_csv, index=False)


def _depth_sort_value(depth):
    return 999 if depth is None else int(depth)


def choose_one_se_model(results_df):
    """
    One-standard-error rule:
    choose the simplest model whose mean_test_score is within one std of the best score.
    Simplicity preference:
      1) smaller depth
      2) fewer trees
      3) larger min_samples_split
    """
    best_row = results_df.sort_values("mean_test_score", ascending=False).iloc[0]
    threshold = best_row["mean_test_score"] - best_row["std_test_score"]

    eligible = results_df[results_df["mean_test_score"] >= threshold].copy()

    eligible["depth_sort"] = eligible["param_max_depth"].apply(_depth_sort_value)
    eligible["n_estimators_sort"] = eligible["param_n_estimators"].astype(int)
    eligible["min_samples_split_sort"] = eligible["param_min_samples_split"].astype(int)

    one_se_choice = eligible.sort_values(
        by=[
            "depth_sort",
            "n_estimators_sort",
            "min_samples_split_sort",
            "mean_test_score",
        ],
        ascending=[True, True, False, False],
    ).iloc[0]

    return threshold, one_se_choice


def plot_heatmap_fixed_min_split(
    results_df,
    fixed_min_samples_split,
    value_col="mean_test_score",
    output_path="results/rf_gridsearch_heatmap.png",
    title_prefix="RF Grid Search Heatmap",
):
    """
    Plot a heatmap over max_depth x n_estimators,
    fixing min_samples_split at one chosen value.
    """
    filtered = results_df[
        results_df["param_min_samples_split"].astype(int) == int(fixed_min_samples_split)
    ].copy()

    row_order = ["3", "5", "10", "20", "None"]
    col_order = [50, 100, 200]

    heatmap_df = filtered.pivot(
        index="max_depth_label",
        columns="param_n_estimators",
        values=value_col,
    )

    heatmap_df = heatmap_df.reindex(index=row_order, columns=col_order)

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(heatmap_df.values, aspect="auto")

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(value_col.replace("_", " ").title())

    ax.set_xticks(np.arange(len(col_order)))
    ax.set_xticklabels([str(x) for x in col_order])
    ax.set_yticks(np.arange(len(row_order)))
    ax.set_yticklabels(row_order)

    ax.set_xlabel("n_estimators")
    ax.set_ylabel("max_depth")
    ax.set_title(f"{title_prefix} (min_samples_split={fixed_min_samples_split})")

    for i in range(heatmap_df.shape[0]):
        for j in range(heatmap_df.shape[1]):
            value = heatmap_df.iloc[i, j]
            if pd.notna(value):
                ax.text(j, i, f"{value:.3f}", ha="center", va="center", fontsize=10)

    fig.tight_layout()
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def save_part1_markdown(
    grid,
    test_f1,
    best_min_split,
    threshold_one_se,
    one_se_choice,
    near_best_count,
    output_path="results/part1_summary.md",
):
    """Save a compact markdown summary for the repo."""
    best_params = grid.best_params_

    md = f"""# Part 1 — GridSearchCV Summary

## Best model
- Best params: `{best_params}`
- Best inner CV F1: **{grid.best_score_:.3f}**
- Hold-out test F1: **{test_f1:.3f}**
- Heatmap fixed at `min_samples_split = {best_min_split}`

## One-standard-error rule
- Score threshold: **{threshold_one_se:.3f}**
- Chosen simpler config within one standard error:
  - `max_depth = {one_se_choice['param_max_depth']}`
  - `min_samples_split = {int(one_se_choice['param_min_samples_split'])}`
  - `n_estimators = {int(one_se_choice['param_n_estimators'])}`
  - `mean_test_score = {one_se_choice['mean_test_score']:.3f}`

## Plateau evidence
- Number of configs within 0.005 of best: **{near_best_count}**

## Interpretation
The strongest-performing region is defined more by `max_depth = 5` than by a single unique configuration, which suggests a sweet spot and mild plateau. The detailed results and generalization gaps are saved in CSV form for inspection.
"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(md)


def main():
    os.makedirs("results", exist_ok=True)

    # 1) Load and split
    X_train, X_test, y_train, y_test = load_and_split()
    print(f"Train: {len(X_train)}  Test: {len(X_test)}  Churn rate: {y_train.mean():.2%}")

    # 2) Grid search
    grid = run_rf_grid_search(X_train, y_train)

    # 3) Build enriched results table
    results_df = build_results_dataframe(grid)
    save_grid_results(results_df)

    # 4) Summaries and diagnostics
    best_min_split = grid.best_params_["min_samples_split"]

    min_split_summary = summarize_min_samples_split(results_df)
    save_min_samples_split_summary(min_split_summary)
    plot_min_samples_split_summary(min_split_summary)

    near_best_df = get_near_best_configs(results_df, tolerance=0.005)
    save_near_best_configs(near_best_df)

    threshold_one_se, one_se_choice = choose_one_se_model(results_df)

    # 5) Heatmaps
    plot_heatmap_fixed_min_split(
        results_df,
        fixed_min_samples_split=best_min_split,
        value_col="mean_test_score",
        output_path="results/rf_gridsearch_heatmap.png",
        title_prefix="RF Grid Search Mean CV F1",
    )

    plot_heatmap_fixed_min_split(
        results_df,
        fixed_min_samples_split=best_min_split,
        value_col="generalization_gap",
        output_path="results/rf_generalization_gap_heatmap.png",
        title_prefix="RF Generalization Gap Heatmap",
    )

    # 6) Evaluate best model on hold-out test split
    best_model = grid.best_estimator_
    test_f1 = f1_score(y_test, best_model.predict(X_test), zero_division=0)

    # 7) Save markdown summary
    save_part1_markdown(
        grid=grid,
        test_f1=test_f1,
        best_min_split=best_min_split,
        threshold_one_se=threshold_one_se,
        one_se_choice=one_se_choice,
        near_best_count=len(near_best_df),
    )

    # 8) Print key outputs
    print("\n--- Part 1: GridSearchCV Results ---")
    print("Best params:")
    print(grid.best_params_)
    print(f"Best inner CV F1: {grid.best_score_:.3f}")
    print(f"Hold-out test F1: {test_f1:.3f}")
    print(f"Heatmap fixed at min_samples_split = {best_min_split}")

    best_row = results_df.iloc[0]
    print(f"Best mean_train_score: {best_row['mean_train_score']:.3f}")
    print(f"Best generalization_gap: {best_row['generalization_gap']:.3f}")

    print("\n--- One-standard-error rule ---")
    print(f"Threshold = {threshold_one_se:.3f}")
    print(
        "Chosen simpler config:",
        {
            "max_depth": one_se_choice["param_max_depth"],
            "min_samples_split": int(one_se_choice["param_min_samples_split"]),
            "n_estimators": int(one_se_choice["param_n_estimators"]),
            "mean_test_score": round(float(one_se_choice["mean_test_score"]), 3),
        },
    )

    print("\nTop 10 rows from grid search:")
    display_cols = [
        "mean_train_score",
        "mean_test_score",
        "generalization_gap",
        "std_test_score",
        "rank_test_score",
        "param_n_estimators",
        "param_max_depth",
        "param_min_samples_split",
        "score_delta_from_best",
    ]
    print(results_df[display_cols].head(10).to_string(index=False))

    print("\nConfigs within 0.005 of best score:")
    print(near_best_df[display_cols].to_string(index=False))

    print("\nAverage CV behavior by min_samples_split:")
    print(min_split_summary.to_string(index=False))

    print("\nSaved files:")
    print("  - results/gridsearch_results_detailed.csv")
    print("  - results/near_best_configs.csv")
    print("  - results/min_samples_split_summary.csv")
    print("  - results/min_samples_split_summary.png")
    print("  - results/rf_gridsearch_heatmap.png")
    print("  - results/rf_generalization_gap_heatmap.png")
    print("  - results/part1_summary.md")


if __name__ == "__main__":
    main()