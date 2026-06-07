"""
train.py
--------
Model training, evaluation, and comparison module.

Models trained:
  1. Logistic Regression – linear baseline; fast, interpretable.
  2. Decision Tree       – non-linear; easy to visualise.
  3. Random Forest       – ensemble of trees; usually best performer.

Evaluation metrics (for binary classification):
  - Accuracy  : (TP + TN) / Total — overall correctness
  - Precision : TP / (TP + FP)   — how often a positive prediction is correct
  - Recall    : TP / (TP + FN)   — how many actual positives were caught
  - F1 Score  : harmonic mean of Precision & Recall — balance of both
  - Confusion Matrix: full breakdown of TP, TN, FP, FN
"""

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)


MODEL_DIR = "models"


# ──────────────────────────────────────────────────────────────────────────────
# 1. Build models
# ──────────────────────────────────────────────────────────────────────────────

def build_models() -> dict:
    """
    Instantiate the three classifiers with sensible hyper-parameters.

    Returns
    -------
    dict : {model_name: sklearn_estimator}
    """
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,        # give it enough iterations to converge
            random_state=42,
            C=1.0,                # regularisation strength (higher = less regularisation)
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=5,          # limit depth to prevent overfitting
            random_state=42,
            min_samples_split=10, # need at least 10 samples to split a node
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,     # number of trees in the forest
            max_depth=6,
            random_state=42,
            min_samples_split=10,
            n_jobs=-1,            # use all available CPU cores
        ),
    }
    return models


# ──────────────────────────────────────────────────────────────────────────────
# 2. Train + evaluate
# ──────────────────────────────────────────────────────────────────────────────

def evaluate_model(model, X_test, y_test, model_name: str) -> dict:
    """
    Compute all evaluation metrics for a trained model.

    Parameters
    ----------
    model      : trained sklearn estimator
    X_test     : test features
    y_test     : true labels
    model_name : string label for printing

    Returns
    -------
    dict of metrics
    """
    y_pred = model.predict(X_test)

    metrics = {
        "Model":     model_name,
        "Accuracy":  round(accuracy_score(y_test, y_pred),  4),
        "Precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "Recall":    round(recall_score(y_test, y_pred,    zero_division=0), 4),
        "F1 Score":  round(f1_score(y_test, y_pred,        zero_division=0), 4),
    }
    return metrics


def train_and_evaluate(
    models: dict,
    X_train, X_test,
    y_train, y_test,
) -> pd.DataFrame:
    """
    Train every model and collect their evaluation metrics.

    Parameters
    ----------
    models  : dict from build_models()
    X_train, X_test : feature arrays
    y_train, y_test : label arrays

    Returns
    -------
    results_df : pd.DataFrame – one row per model with all metrics
    """
    results = []

    for name, model in models.items():
        print(f"\n[TRAIN] Training {name} …")
        model.fit(X_train, y_train)

        metrics = evaluate_model(model, X_test, y_test, name)
        results.append(metrics)

        # Print the detailed classification report
        y_pred = model.predict(X_test)
        print(f"\n  Classification Report — {name}:")
        print(classification_report(y_test, y_pred, target_names=["No Disease", "Heart Disease"]))

    results_df = pd.DataFrame(results).set_index("Model")
    return results_df


# ──────────────────────────────────────────────────────────────────────────────
# 3. Visualise results
# ──────────────────────────────────────────────────────────────────────────────

def plot_confusion_matrices(models: dict, X_test, y_test, plot_dir: str = "notebooks") -> None:
    """
    Plot a side-by-side confusion matrix for each model.

    A confusion matrix shows:
      - True Positives  (top-left for each class)
      - False Positives / Negatives
      Darker cells = more samples in that cell.
    """
    os.makedirs(plot_dir, exist_ok=True)
    n = len(models)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    if n == 1:
        axes = [axes]

    fig.suptitle("Confusion Matrices", fontsize=15, fontweight="bold")

    for ax, (name, model) in zip(axes, models.items()):
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["No Disease", "Disease"],
            yticklabels=["No Disease", "Disease"],
            ax=ax,
            linewidths=0.5,
        )
        ax.set_title(name)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")

    plt.tight_layout()
    path = os.path.join(plot_dir, "confusion_matrices.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Saved → {path}")


def plot_model_comparison(results_df: pd.DataFrame, plot_dir: str = "notebooks") -> None:
    """
    Grouped bar chart comparing all models across all metrics.
    Makes it easy to spot which model wins in each dimension.
    """
    os.makedirs(plot_dir, exist_ok=True)

    ax = results_df.plot(
        kind="bar",
        figsize=(12, 6),
        colormap="Set2",
        edgecolor="white",
        rot=0,
    )
    ax.set_title("Model Comparison — All Metrics", fontsize=15, fontweight="bold")
    ax.set_xlabel("Model")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.1)
    ax.legend(loc="lower right")
    ax.axhline(y=0.9, color="grey", linestyle="--", linewidth=0.8, label="0.90 threshold")

    # Annotate bars with their values
    for container in ax.containers:
        ax.bar_label(container, fmt="%.2f", fontsize=8, padding=2)

    plt.tight_layout()
    path = os.path.join(plot_dir, "model_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Saved → {path}")


def plot_feature_importance(model, feature_names: list, plot_dir: str = "notebooks") -> None:
    """
    Bar chart of Random Forest feature importances.
    Shows which features the model relies on the most.
    """
    if not hasattr(model, "feature_importances_"):
        return

    os.makedirs(plot_dir, exist_ok=True)

    importances = pd.Series(model.feature_importances_, index=feature_names)
    importances = importances.sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(9, 6))
    importances.plot(kind="barh", color="#3498db", edgecolor="white", ax=ax)
    ax.set_title("Random Forest — Feature Importances", fontsize=13, fontweight="bold")
    ax.set_xlabel("Importance Score")
    plt.tight_layout()

    path = os.path.join(plot_dir, "feature_importance.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Saved → {path}")


# ──────────────────────────────────────────────────────────────────────────────
# 4. Select and save best model
# ──────────────────────────────────────────────────────────────────────────────

def select_best_model(results_df: pd.DataFrame) -> str:
    """
    Select the best model based on F1 Score.
    (F1 is preferred over accuracy for medical tasks because it balances
     Precision and Recall — missing a real patient (false negative) is costly.)

    Parameters
    ----------
    results_df : DataFrame from train_and_evaluate()

    Returns
    -------
    str : name of the best model
    """
    best_model_name = results_df["F1 Score"].idxmax()
    print(f"\n[RESULT] Best model (highest F1 Score): {best_model_name}")
    print(f"         F1 Score = {results_df.loc[best_model_name, 'F1 Score']:.4f}")
    return best_model_name


def save_model(model, scaler, filename: str = "best_model.pkl") -> str:
    """
    Persist the trained model AND the scaler together using pickle.
    Saving both ensures that when we later make a prediction, we can
    apply the exact same scaling that was used during training.

    Parameters
    ----------
    model    : trained sklearn estimator
    scaler   : fitted StandardScaler
    filename : output filename inside the 'models/' directory

    Returns
    -------
    str : full path to the saved file
    """
    os.makedirs(MODEL_DIR, exist_ok=True)
    path = os.path.join(MODEL_DIR, filename)

    bundle = {"model": model, "scaler": scaler}
    with open(path, "wb") as f:
        pickle.dump(bundle, f)

    print(f"[SAVE] Model + scaler saved → {path}")
    return path


def load_model(filename: str = "best_model.pkl"):
    """
    Load the saved model bundle from disk.

    Returns
    -------
    (model, scaler) tuple
    """
    path = os.path.join(MODEL_DIR, filename)
    with open(path, "rb") as f:
        bundle = pickle.load(f)
    return bundle["model"], bundle["scaler"]
