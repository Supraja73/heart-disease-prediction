"""
eda.py
------
Exploratory Data Analysis (EDA) module.

Contains functions for:
  - Printing dataset info / statistics
  - Checking missing values
  - Plotting distributions, count plots, and correlation heatmap

All plots are saved to the 'notebooks/' folder so you can view them later.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Consistent style for all plots
sns.set_theme(style="whitegrid", palette="Set2")
PLOT_DIR = "notebooks"


def explore_data(df: pd.DataFrame) -> None:
    """
    Print basic information about the dataset.

    Parameters
    ----------
    df : pd.DataFrame
        The raw (or cleaned) dataframe.
    """
    print("\n" + "=" * 60)
    print("  DATASET OVERVIEW")
    print("=" * 60)

    print(f"\n Shape : {df.shape[0]} rows × {df.shape[1]} columns")
    print("\n First 5 rows:")
    print(df.head().to_string())

    print("\n\n Data types:")
    print(df.dtypes.to_string())

    print("\n\n Statistical summary:")
    print(df.describe().round(2).to_string())


def check_missing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Identify and report missing values.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame
        DataFrame showing count and percentage of missing values per column.
    """
    print("\n" + "=" * 60)
    print("  MISSING VALUE ANALYSIS")
    print("=" * 60)

    missing_count = df.isnull().sum()
    missing_pct   = (missing_count / len(df) * 100).round(2)

    missing_df = pd.DataFrame({
        "Missing Count": missing_count,
        "Missing %":     missing_pct,
    })
    missing_df = missing_df[missing_df["Missing Count"] > 0]

    if missing_df.empty:
        print("\n  No missing values found — dataset is complete!\n")
    else:
        print(f"\n  Columns with missing values:\n{missing_df.to_string()}\n")

    return missing_df


def plot_target_distribution(df: pd.DataFrame) -> None:
    """
    Bar chart showing how many patients have / don't have heart disease.
    """
    os.makedirs(PLOT_DIR, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Target Variable Distribution", fontsize=15, fontweight="bold")

    # Count plot
    sns.countplot(x="target", hue="target", data=df, palette=["#2ecc71", "#e74c3c"], ax=axes[0], legend=False)
    axes[0].set_title("Count of Patients")
    axes[0].set_xlabel("Target (0 = No Disease, 1 = Disease)")
    axes[0].set_ylabel("Number of Patients")
    for bar in axes[0].patches:
        axes[0].annotate(
            f'{int(bar.get_height())}',
            (bar.get_x() + bar.get_width() / 2, bar.get_height()),
            ha="center", va="bottom", fontsize=12
        )

    # Pie chart
    target_counts = df["target"].value_counts()
    axes[1].pie(
        target_counts,
        labels=["No Disease", "Heart Disease"],
        autopct="%1.1f%%",
        colors=["#2ecc71", "#e74c3c"],
        startangle=140,
        textprops={"fontsize": 12},
    )
    axes[1].set_title("Proportion of Patients")

    plt.tight_layout()
    path = os.path.join(PLOT_DIR, "target_distribution.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Saved → {path}")


def plot_feature_distributions(df: pd.DataFrame) -> None:
    """
    Histogram for every numeric feature to understand their spread.
    """
    os.makedirs(PLOT_DIR, exist_ok=True)

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    if "target" in numeric_cols:
        numeric_cols.remove("target")

    n_cols = 3
    n_rows = (len(numeric_cols) + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, n_rows * 4))
    fig.suptitle("Feature Distributions", fontsize=15, fontweight="bold", y=1.01)
    axes = axes.flatten()

    for i, col in enumerate(numeric_cols):
        sns.histplot(df[col], kde=True, color="#3498db", ax=axes[i])
        axes[i].set_title(col)
        axes[i].set_xlabel("")

    # Hide any unused subplot panels
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    path = os.path.join(PLOT_DIR, "feature_distributions.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Saved → {path}")


def plot_correlation_heatmap(df: pd.DataFrame) -> None:
    """
    Correlation heatmap — shows how strongly features are related to each other
    and to the target variable. High positive correlation → dark red;
    high negative correlation → dark blue.
    """
    os.makedirs(PLOT_DIR, exist_ok=True)

    corr = df.corr(numeric_only=True)

    plt.figure(figsize=(12, 9))
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        linewidths=0.5,
        square=True,
        cbar_kws={"shrink": 0.8},
    )
    plt.title("Feature Correlation Heatmap", fontsize=15, fontweight="bold", pad=12)
    plt.tight_layout()

    path = os.path.join(PLOT_DIR, "correlation_heatmap.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Saved → {path}")


def plot_feature_vs_target(df: pd.DataFrame) -> None:
    """
    Box plots showing each numeric feature split by the target label.
    This helps identify features that differ most between disease / no-disease groups.
    """
    os.makedirs(PLOT_DIR, exist_ok=True)

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    if "target" in numeric_cols:
        numeric_cols.remove("target")

    n_cols = 3
    n_rows = (len(numeric_cols) + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, n_rows * 4))
    fig.suptitle("Feature vs Target (Box Plots)", fontsize=15, fontweight="bold", y=1.01)
    axes = axes.flatten()

    for i, col in enumerate(numeric_cols):
        sns.boxplot(
            x="target",
            y=col,
            hue="target",
            data=df,
            palette=["#2ecc71", "#e74c3c"],
            ax=axes[i],
            legend=False,
        )
        axes[i].set_title(col)
        axes[i].set_xlabel("Target (0=No Disease, 1=Disease)")

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    path = os.path.join(PLOT_DIR, "feature_vs_target.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Saved → {path}")
