"""
app.py
------
Main entry-point for the Heart Disease Prediction ML pipeline.

Run this script to:
  1. Load and explore the dataset
  2. Clean and pre-process the data
  3. Train three ML models and evaluate them
  4. Print a comparison table and select the best model
  5. Save the best model to disk
  6. Demonstrate a sample prediction

Usage:
    python app.py
"""

import sys
import os

# Make sure the src package is importable regardless of working directory
sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd

from src.data_loader   import load_data
from src.eda           import (
    explore_data,
    check_missing,
    plot_target_distribution,
    plot_feature_distributions,
    plot_correlation_heatmap,
    plot_feature_vs_target,
)
from src.preprocessing import (
    clean_data,
    split_features_target,
    get_train_test_split,
    scale_features,
)
from src.train import (
    build_models,
    train_and_evaluate,
    plot_confusion_matrices,
    plot_model_comparison,
    plot_feature_importance,
    select_best_model,
    save_model,
)
from src.predict import predict_from_input, FEATURE_NAMES


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 1 — Load Data
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "█" * 60)
print("  STEP 1 — Data Loading")
print("█" * 60)

df_raw = load_data(data_dir="data")
print(f"  Loaded {df_raw.shape[0]} rows × {df_raw.shape[1]} columns")


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 2 — Exploratory Data Analysis
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "█" * 60)
print("  STEP 2 — Exploratory Data Analysis")
print("█" * 60)

explore_data(df_raw)
check_missing(df_raw)

print("\n[EDA] Generating visualisation plots …")
plot_target_distribution(df_raw)
plot_feature_distributions(df_raw)
plot_correlation_heatmap(df_raw)
plot_feature_vs_target(df_raw)
print("[EDA] All plots saved to 'notebooks/' folder.")


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 3 — Data Cleaning & Pre-processing
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "█" * 60)
print("  STEP 3 — Data Cleaning & Pre-processing")
print("█" * 60)

df_clean = clean_data(df_raw.copy())

# Separate features and target
X, y = split_features_target(df_clean)

# 80 / 20 train-test split
X_train, X_test, y_train, y_test = get_train_test_split(X, y)

# Standardise features
X_train_sc, X_test_sc, scaler = scale_features(X_train, X_test)

print(f"\n  Feature matrix shape : X_train={X_train_sc.shape}  X_test={X_test_sc.shape}")
print(f"  Class distribution   : train={y_train.value_counts().to_dict()}")


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 4 — Model Training & Evaluation
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "█" * 60)
print("  STEP 4 — Model Training & Evaluation")
print("█" * 60)

models = build_models()
results_df = train_and_evaluate(models, X_train_sc, X_test_sc, y_train, y_test)

# Plot confusion matrices for all models
plot_confusion_matrices(models, X_test_sc, y_test)

# Plot grouped bar chart comparison
plot_model_comparison(results_df)

# Plot Random Forest feature importances
rf_model = models["Random Forest"]
plot_feature_importance(rf_model, feature_names=FEATURE_NAMES)


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 5 — Model Comparison Table
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "█" * 60)
print("  STEP 5 — Model Comparison Table")
print("█" * 60)

print("\n" + results_df.to_string())
print()

best_name = select_best_model(results_df)
best_model = models[best_name]


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 6 — Save Best Model
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "█" * 60)
print("  STEP 6 — Saving Best Model")
print("█" * 60)

save_model(best_model, scaler, filename="best_model.pkl")


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 7 — Sample Prediction (demo with hardcoded values)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "█" * 60)
print("  STEP 7 — Sample Prediction (demo patient)")
print("█" * 60)

# Example patient data — a 58-year-old male with several risk factors
sample_patient = {
    "age":      58,
    "sex":      1,       # Male
    "cp":       0,       # Typical angina
    "trestbps": 140,     # Slightly high blood pressure
    "chol":     289,     # Elevated cholesterol
    "fbs":      0,
    "restecg":  0,
    "thalach":  140,
    "exang":    1,       # Exercise-induced angina present
    "oldpeak":  1.8,
    "slope":    1,
    "ca":       2,       # 2 major vessels colored
    "thal":     3,       # Reversible defect
}

result = predict_from_input(sample_patient, best_model, scaler)

print(f"\n  Patient data : {sample_patient}")
print(f"\n  Prediction   : {result['label']}")
print(f"  Confidence   : {result['confidence']}%")
print(f"  Probabilities: {result['probabilities']}")


print("\n" + "█" * 60)
print("  PIPELINE COMPLETE")
print("  • Plots saved in:  notebooks/")
print("  • Model saved in:  models/best_model.pkl")
print("  • Run Streamlit app: streamlit run streamlit_app.py")
print("█" * 60 + "\n")
