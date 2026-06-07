"""
predict.py
----------
Prediction utility module.

Provides:
  - predict_from_input()  → make a prediction from a dict of feature values
  - interactive_predict() → command-line interface for manual user input
"""

import numpy as np
import pandas as pd


# Feature definitions: (name, description, valid range / options)
FEATURE_INFO = [
    ("age",      "Age (years)",                                      (29, 77)),
    ("sex",      "Sex (1=Male, 0=Female)",                           (0, 1)),
    ("cp",       "Chest Pain Type (0=Typical, 1=Atypical, 2=Non-anginal, 3=Asymptomatic)", (0, 3)),
    ("trestbps", "Resting Blood Pressure (mm Hg)",                   (94, 200)),
    ("chol",     "Serum Cholesterol (mg/dl)",                        (126, 564)),
    ("fbs",      "Fasting Blood Sugar > 120 mg/dl (1=Yes, 0=No)",   (0, 1)),
    ("restecg",  "Resting ECG (0=Normal, 1=ST-T wave abnormality, 2=LVH)", (0, 2)),
    ("thalach",  "Max Heart Rate Achieved",                          (71, 202)),
    ("exang",    "Exercise-Induced Angina (1=Yes, 0=No)",            (0, 1)),
    ("oldpeak",  "ST Depression Induced by Exercise",                (0.0, 6.2)),
    ("slope",    "Slope of Peak Exercise ST Segment (0=Down, 1=Flat, 2=Up)", (0, 2)),
    ("ca",       "Number of Major Vessels Colored by Fluoroscopy (0-3)", (0, 3)),
    ("thal",     "Thalassemia (1=Normal, 2=Fixed defect, 3=Reversible defect)", (1, 3)),
]

FEATURE_NAMES = [fi[0] for fi in FEATURE_INFO]


def predict_from_input(input_dict: dict, model, scaler) -> dict:
    """
    Run a prediction for a single patient given their feature values.

    Parameters
    ----------
    input_dict : dict mapping feature name → value
                 e.g. {"age": 55, "sex": 1, "cp": 0, ...}
    model      : trained sklearn classifier
    scaler     : fitted StandardScaler

    Returns
    -------
    dict with keys:
      "prediction"  : 0 or 1
      "label"       : "Heart Disease Detected" or "No Heart Disease"
      "confidence"  : probability of the predicted class (0.0–1.0)
      "probabilities": dict {0: prob_no_disease, 1: prob_disease}
    """
    # Build a single-row DataFrame in the correct column order
    row = pd.DataFrame([input_dict], columns=FEATURE_NAMES)

    # Apply the same scaling that was used during training
    row_scaled = scaler.transform(row)

    # Get the hard prediction (0 or 1)
    prediction = int(model.predict(row_scaled)[0])

    # Get probability estimates (most sklearn classifiers support this)
    proba = model.predict_proba(row_scaled)[0]          # [prob_class_0, prob_class_1]
    confidence = float(proba[prediction])

    label = "❤️  Heart Disease Detected" if prediction == 1 else "✅  No Heart Disease"

    return {
        "prediction":    prediction,
        "label":         label,
        "confidence":    round(confidence * 100, 2),    # as a percentage
        "probabilities": {
            "No Heart Disease":    round(float(proba[0]) * 100, 2),
            "Heart Disease":       round(float(proba[1]) * 100, 2),
        },
    }


def interactive_predict(model, scaler) -> None:
    """
    Command-line interface that prompts the user to enter feature values
    one by one, then prints the prediction result.

    Usage:
        from src.train import load_model
        from src.predict import interactive_predict
        model, scaler = load_model()
        interactive_predict(model, scaler)
    """
    print("\n" + "=" * 60)
    print("  HEART DISEASE PREDICTION — Manual Input")
    print("=" * 60)
    print("  Please enter the patient's details below.\n")

    input_dict = {}

    for name, description, value_range in FEATURE_INFO:
        lo, hi = value_range
        while True:
            try:
                raw = input(f"  {description} [{lo} – {hi}]: ").strip()
                value = float(raw)
                if lo <= value <= hi:
                    input_dict[name] = value
                    break
                else:
                    print(f"    ⚠  Please enter a value between {lo} and {hi}.")
            except ValueError:
                print("    ⚠  Invalid input. Please enter a number.")

    # Run the prediction
    result = predict_from_input(input_dict, model, scaler)

    print("\n" + "=" * 60)
    print("  PREDICTION RESULT")
    print("=" * 60)
    print(f"\n  Result     : {result['label']}")
    print(f"  Confidence : {result['confidence']}%")
    print(f"\n  Probability Breakdown:")
    for cls, pct in result["probabilities"].items():
        print(f"    {cls}: {pct}%")
    print("\n" + "=" * 60)

    if result["prediction"] == 1:
        print("  ⚠  Recommendation: Please consult a cardiologist.")
    else:
        print("  ✅  Keep up a healthy lifestyle!")
    print("=" * 60 + "\n")
