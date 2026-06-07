"""
preprocessing.py
----------------
Data cleaning and preparation pipeline.

Steps performed:
  1. Handle missing values (median imputation for numeric columns)
  2. Remove duplicate rows
  3. Encode categorical-like integer columns (optional one-hot)
  4. Scale numeric features using StandardScaler
  5. Split into train / test sets

Why StandardScaler?
  Logistic Regression converges faster and performs better when features
  are on the same scale. Tree-based models (Decision Tree, Random Forest)
  are scale-invariant, but scaling doesn't hurt them either.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the raw dataset.

    Actions:
      - Replace '?' strings with NaN (UCI dataset quirk)
      - Convert all columns to numeric where possible
      - Fill missing values with the column median
      - Drop duplicate rows

    Parameters
    ----------
    df : pd.DataFrame
        Raw dataframe straight from data_loader.

    Returns
    -------
    pd.DataFrame
        Cleaned dataframe.
    """
    print("\n[PREPROCESSING] Starting data cleaning …")

    # Step 1 – Replace '?' with NaN (UCI files sometimes have this)
    df = df.replace("?", np.nan)

    # Step 2 – Force all columns to numeric (coerce errors → NaN)
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    before_rows = len(df)

    # Step 3 – Impute missing values with the MEDIAN of each column
    #   Why median? It's robust to outliers (unlike mean).
    for col in df.columns:
        if df[col].isnull().any():
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            print(f"  [FILL] '{col}' → filled with median {median_val:.2f}")

    # Step 4 – Drop exact duplicate rows
    df.drop_duplicates(inplace=True)
    after_rows = len(df)
    print(f"  [DEDUP] Removed {before_rows - after_rows} duplicate row(s). Rows remaining: {after_rows}")

    # Step 5 – Convert the target to integer labels {0, 1}
    #   The UCI file sometimes has values 0-4; convert anything > 0 → 1
    if df["target"].max() > 1:
        df["target"] = (df["target"] > 0).astype(int)
        print("  [TARGET] Converted multi-class target → binary (0/1)")

    print("[PREPROCESSING] Cleaning complete.\n")
    return df.reset_index(drop=True)


def split_features_target(df: pd.DataFrame):
    """
    Separate features (X) from the target column (y).

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    X : pd.DataFrame   – feature matrix
    y : pd.Series      – target vector
    """
    X = df.drop(columns=["target"])
    y = df["target"]
    return X, y


def get_train_test_split(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.20,
    random_state: int = 42,
):
    """
    Split data into training and testing sets.

    We use 80% for training and 20% for testing.
    stratify=y ensures both splits have the same class balance.

    Parameters
    ----------
    X            : feature matrix
    y            : target vector
    test_size    : fraction of data for testing (default 0.20)
    random_state : seed for reproducibility

    Returns
    -------
    X_train, X_test, y_train, y_test
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,          # keep class proportions equal in both splits
    )
    print(f"[SPLIT] Train: {len(X_train)} samples | Test: {len(X_test)} samples")
    return X_train, X_test, y_train, y_test


def scale_features(X_train: pd.DataFrame, X_test: pd.DataFrame):
    """
    Standardise features to zero mean and unit variance.

    IMPORTANT: We fit the scaler ONLY on the training data, then apply
    (transform) it to both train and test sets. This prevents 'data leakage'
    — i.e. the test set statistics must NOT influence the scaler.

    Parameters
    ----------
    X_train : training features
    X_test  : test features

    Returns
    -------
    X_train_scaled : np.ndarray
    X_test_scaled  : np.ndarray
    scaler         : fitted StandardScaler (needed later for predictions)
    """
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)   # fit + transform on train
    X_test_scaled  = scaler.transform(X_test)         # transform only on test

    print("[SCALE] Features standardised with StandardScaler (mean=0, std=1)")
    return X_train_scaled, X_test_scaled, scaler
