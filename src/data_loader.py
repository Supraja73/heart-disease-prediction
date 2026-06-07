"""
data_loader.py
--------------
Handles downloading and loading the Heart Disease dataset.
We use the UCI Heart Disease dataset (Cleveland subset), which is publicly
available and widely used for ML benchmarking.

Dataset columns (14 features used):
  age, sex, cp, trestbps, chol, fbs, restecg,
  thalach, exang, oldpeak, slope, ca, thal, target
"""

import pandas as pd
import numpy as np
import os


# Column names as defined by the UCI repository
COLUMN_NAMES = [
    "age",       # Age of the patient
    "sex",       # Sex (1 = male, 0 = female)
    "cp",        # Chest pain type (0-3)
    "trestbps",  # Resting blood pressure (mm Hg)
    "chol",      # Serum cholesterol (mg/dl)
    "fbs",       # Fasting blood sugar > 120 mg/dl (1 = true, 0 = false)
    "restecg",   # Resting ECG results (0-2)
    "thalach",   # Maximum heart rate achieved
    "exang",     # Exercise-induced angina (1 = yes, 0 = no)
    "oldpeak",   # ST depression induced by exercise
    "slope",     # Slope of the peak exercise ST segment (0-2)
    "ca",        # Number of major vessels colored by fluoroscopy (0-3)
    "thal",      # Thalassemia (1 = normal, 2 = fixed defect, 3 = reversible defect)
    "target",    # Diagnosis (1 = heart disease, 0 = no heart disease)
]

# Multiple mirror URLs to try in order (UCI blocks direct scraping; GitHub mirrors work)
MIRROR_URLS = [
    "https://raw.githubusercontent.com/sharmaroshan/Heart-UCI-Dataset/master/heart.csv",
    "https://raw.githubusercontent.com/BindiChen/machine-learning/main/data-analysis/003-heart-disease-EDA/heart.csv",
]


def load_data(data_dir: str = "data") -> pd.DataFrame:
    """
    Load the Heart Disease dataset.

    Strategy:
      1. If a local CSV already exists in data_dir, load from disk.
      2. Otherwise, try to download from UCI; if that fails (no internet),
         generate a realistic synthetic dataset so the code always runs.

    Parameters
    ----------
    data_dir : str
        Directory where the CSV will be saved / loaded from.

    Returns
    -------
    pd.DataFrame
        Raw dataframe with proper column names.
    """
    os.makedirs(data_dir, exist_ok=True)
    local_path = os.path.join(data_dir, "heart.csv")

    # ── 1. Load from local file if it already exists ──────────────────────
    if os.path.exists(local_path):
        print(f"[INFO] Loading dataset from local file: {local_path}")
        df = pd.read_csv(local_path)
        return df

    # ── 2. Try downloading from mirror URLs ───────────────────────────────────
    for url in MIRROR_URLS:
        try:
            print(f"[INFO] Trying: {url}")
            df = pd.read_csv(url, header=0)
            # Standardise column names (some mirrors use different names)
            df.columns = [c.lower().strip() for c in df.columns]
            # Rename 'condition' or 'num' to 'target' if present
            for alt in ["condition", "num", "output"]:
                if alt in df.columns and "target" not in df.columns:
                    df.rename(columns={alt: "target"}, inplace=True)
            # Keep only the 14 expected columns if they exist
            if all(c in df.columns for c in COLUMN_NAMES):
                df = df[COLUMN_NAMES]
            df.to_csv(local_path, index=False)
            print(f"[INFO] Dataset saved to {local_path}")
            return df
        except Exception as exc:
            print(f"[WARN] Mirror failed ({exc}). Trying next …")

    print("[INFO] All mirrors failed. Generating synthetic dataset …")

    # ── 3. Synthetic fallback ──────────────────────────────────────────────
    df = _generate_synthetic_data(n=303, seed=42)
    df.to_csv(local_path, index=False)
    print(f"[INFO] Synthetic dataset saved to {local_path}")
    return df


# ──────────────────────────────────────────────────────────────────────────────
# Internal helper
# ──────────────────────────────────────────────────────────────────────────────

def _generate_synthetic_data(n: int = 303, seed: int = 42) -> pd.DataFrame:
    """
    Generate a realistic synthetic Heart Disease dataset that mirrors the
    statistical properties of the original UCI Cleveland data.

    This is ONLY used when the real dataset cannot be downloaded.
    The target is built using a weighted logistic-like rule so that
    realistic 'healthy' profiles score low and 'at-risk' profiles score high.
    """
    rng = np.random.default_rng(seed)

    age      = rng.integers(29, 78, size=n)
    sex      = rng.choice([0, 1], size=n, p=[0.32, 0.68])
    cp       = rng.choice([0, 1, 2, 3], size=n, p=[0.47, 0.17, 0.28, 0.08])
    trestbps = rng.normal(131, 17, size=n).clip(94, 200).astype(int)
    chol     = rng.normal(246, 52, size=n).clip(126, 564).astype(int)
    fbs      = rng.choice([0, 1], size=n, p=[0.85, 0.15])
    restecg  = rng.choice([0, 1, 2], size=n, p=[0.50, 0.48, 0.02])
    thalach  = rng.normal(149, 23, size=n).clip(71, 202).astype(int)
    exang    = rng.choice([0, 1], size=n, p=[0.67, 0.33])
    oldpeak  = rng.exponential(1.0, size=n).clip(0, 6.2).round(1)
    slope    = rng.choice([0, 1, 2], size=n, p=[0.07, 0.47, 0.46])
    ca       = rng.choice([0, 1, 2, 3], size=n, p=[0.58, 0.22, 0.13, 0.07])
    thal     = rng.choice([1, 2, 3], size=n, p=[0.06, 0.54, 0.40])

    # Build target with a realistic multi-factor scoring rule
    # (mirrors known clinical risk factors for heart disease)
    score = (
        (age > 55).astype(float) * 0.20          # older age = higher risk
        + (sex == 1).astype(float) * 0.10        # male = slightly higher risk
        + (cp == 0).astype(float) * 0.30         # typical angina = high risk
        + (exang == 1).astype(float) * 0.20      # exercise angina = risk
        + (ca > 0).astype(float) * 0.25          # blocked vessels = risk
        + (thal == 3).astype(float) * 0.15       # reversible defect = risk
        + (oldpeak > 2).astype(float) * 0.10     # high ST depression = risk
        + (slope == 0).astype(float) * 0.10      # downsloping = risk
        - (cp == 2).astype(float) * 0.15         # non-anginal pain = lower risk
        - (thalach > 160).astype(float) * 0.10   # high max HR = lower risk
        + rng.uniform(-0.1, 0.1, size=n)         # small random noise
    )
    # Convert continuous score to binary label at the 0.5 midpoint
    target = (score > 0.50).astype(int)

    df = pd.DataFrame({
        "age": age, "sex": sex, "cp": cp, "trestbps": trestbps,
        "chol": chol, "fbs": fbs, "restecg": restecg, "thalach": thalach,
        "exang": exang, "oldpeak": oldpeak, "slope": slope,
        "ca": ca, "thal": thal, "target": target,
    })
    return df
