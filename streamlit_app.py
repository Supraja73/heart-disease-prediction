"""
streamlit_app.py
----------------
Streamlit web application for Heart Disease Prediction.

Provides a user-friendly interface where users can:
  - Enter patient details using sliders and dropdowns
  - Click "Predict" to get an instant result
  - View the confidence score and probability breakdown

Run with:
    streamlit run streamlit_app.py
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
import streamlit as st

# ── Make sure our src package is importable ───────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))
from src.predict import predict_from_input, FEATURE_NAMES
from src.train   import load_model, save_model, build_models
from src.data_loader   import load_data
from src.preprocessing import clean_data, split_features_target, get_train_test_split, scale_features
from src.train         import train_and_evaluate, select_best_model


# ── Page configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Heart Disease Predictor",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Space Grotesk', sans-serif;
    }
    .main { background: #0f0f1a; }
    .block-container { padding: 2rem 3rem; }

    .hero-title {
        font-size: 2.8rem;
        font-weight: 700;
        background: linear-gradient(135deg, #ff6b6b, #ffa07a);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0;
    }
    .hero-sub {
        font-size: 1.05rem;
        color: #aaa;
        margin-top: 0.3rem;
        margin-bottom: 2rem;
    }

    .metric-card {
        background: #1a1a2e;
        border: 1px solid #2a2a4a;
        border-radius: 14px;
        padding: 1.2rem 1.5rem;
        text-align: center;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #ff6b6b;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #888;
        margin-top: 0.2rem;
    }

    .result-positive {
        background: linear-gradient(135deg, #3d0000, #660000);
        border: 2px solid #ff4444;
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
        animation: pulse 2s infinite;
    }
    .result-negative {
        background: linear-gradient(135deg, #003d00, #006600);
        border: 2px solid #44ff44;
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
    }
    .result-text {
        font-size: 1.8rem;
        font-weight: 700;
        color: #fff;
    }
    .confidence-text {
        font-size: 1rem;
        color: rgba(255,255,255,0.75);
        margin-top: 0.5rem;
    }

    @keyframes pulse {
        0%   { box-shadow: 0 0 0 0 rgba(255, 68, 68, 0.4); }
        70%  { box-shadow: 0 0 0 12px rgba(255, 68, 68, 0); }
        100% { box-shadow: 0 0 0 0 rgba(255, 68, 68, 0); }
    }

    div[data-testid="stButton"] > button {
        width: 100%;
        background: linear-gradient(135deg, #ff6b6b, #ff4500);
        color: white;
        font-size: 1.1rem;
        font-weight: 700;
        padding: 0.9rem 0;
        border-radius: 12px;
        border: none;
        cursor: pointer;
        transition: transform 0.1s, box-shadow 0.1s;
    }
    div[data-testid="stButton"] > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 24px rgba(255,107,107,0.35);
    }

    .section-header {
        font-size: 1.1rem;
        font-weight: 600;
        color: #ff9999;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        border-bottom: 1px solid #2a2a4a;
        padding-bottom: 0.5rem;
        margin-bottom: 1rem;
    }
    .disclaimer {
        background: #1a1a2e;
        border-left: 4px solid #ff6b6b;
        padding: 1rem 1.5rem;
        border-radius: 0 10px 10px 0;
        color: #aaa;
        font-size: 0.85rem;
        margin-top: 2rem;
    }
</style>
""", unsafe_allow_html=True)


# ── Load / train model ─────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model …")
def get_model():
    """Load the saved model; train it fresh if not found."""
    model_path = os.path.join("models", "best_model.pkl")
    if os.path.exists(model_path):
        return load_model("best_model.pkl")

    # No saved model → run the full pipeline silently
    df   = load_data("data")
    df   = clean_data(df)
    X, y = split_features_target(df)
    X_tr, X_te, y_tr, y_te = get_train_test_split(X, y)
    X_tr_sc, X_te_sc, scaler = scale_features(X_tr, X_te)
    models   = build_models()
    results  = train_and_evaluate(models, X_tr_sc, X_te_sc, y_tr, y_te)
    best_name = select_best_model(results)
    best_model = models[best_name]
    save_model(best_model, scaler)
    return best_model, scaler


model, scaler = get_model()


# ══════════════════════════════════════════════════════════════════════════════
#  UI Layout
# ══════════════════════════════════════════════════════════════════════════════

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown('<p class="hero-title">🫀 Heart Disease Predictor</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="hero-sub">Enter patient details to assess heart disease risk using AI</p>',
    unsafe_allow_html=True,
)

left_col, right_col = st.columns([3, 2], gap="large")

# ── Left: Input form ──────────────────────────────────────────────────────────
with left_col:

    st.markdown('<p class="section-header">👤 Patient Demographics</p>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        age = st.slider("Age", min_value=20, max_value=80, value=52)
    with c2:
        sex = st.selectbox("Sex", options=[1, 0], format_func=lambda x: "Male" if x == 1 else "Female")

    st.markdown('<p class="section-header">💓 Cardiac Indicators</p>', unsafe_allow_html=True)
    c3, c4 = st.columns(2)
    with c3:
        cp = st.selectbox(
            "Chest Pain Type",
            options=[0, 1, 2, 3],
            format_func=lambda x: {
                0: "Typical Angina",
                1: "Atypical Angina",
                2: "Non-anginal Pain",
                3: "Asymptomatic",
            }[x],
        )
        trestbps = st.slider("Resting Blood Pressure (mm Hg)", 80, 200, 130)
        chol     = st.slider("Serum Cholesterol (mg/dl)", 100, 600, 246)
    with c4:
        thalach  = st.slider("Max Heart Rate Achieved", 60, 220, 149)
        oldpeak  = st.slider("ST Depression (Oldpeak)", 0.0, 7.0, 1.0, step=0.1)
        exang    = st.selectbox(
            "Exercise-Induced Angina",
            options=[0, 1],
            format_func=lambda x: "Yes" if x == 1 else "No",
        )

    st.markdown('<p class="section-header">🔬 Clinical Tests</p>', unsafe_allow_html=True)
    c5, c6 = st.columns(2)
    with c5:
        fbs = st.selectbox(
            "Fasting Blood Sugar > 120 mg/dl",
            options=[0, 1],
            format_func=lambda x: "True" if x == 1 else "False",
        )
        restecg = st.selectbox(
            "Resting ECG Results",
            options=[0, 1, 2],
            format_func=lambda x: {
                0: "Normal",
                1: "ST-T wave abnormality",
                2: "Left ventricular hypertrophy",
            }[x],
        )
    with c6:
        slope = st.selectbox(
            "Slope of Peak Exercise ST",
            options=[0, 1, 2],
            format_func=lambda x: {0: "Downsloping", 1: "Flat", 2: "Upsloping"}[x],
        )
        ca = st.selectbox(
            "Major Vessels Colored by Fluoroscopy",
            options=[0, 1, 2, 3],
            format_func=lambda x: f"{x} vessel{'s' if x != 1 else ''}",
        )
        thal = st.selectbox(
            "Thalassemia",
            options=[1, 2, 3],
            format_func=lambda x: {
                1: "Normal",
                2: "Fixed Defect",
                3: "Reversible Defect",
            }[x],
        )

    predict_btn = st.button("🔍  Run Prediction", use_container_width=True)

# ── Right: Results ────────────────────────────────────────────────────────────
with right_col:
    st.markdown('<p class="section-header">📊 Prediction Result</p>', unsafe_allow_html=True)

    if predict_btn:
        patient_data = {
            "age": age, "sex": sex, "cp": cp,
            "trestbps": trestbps, "chol": chol, "fbs": fbs,
            "restecg": restecg, "thalach": thalach, "exang": exang,
            "oldpeak": oldpeak, "slope": slope, "ca": ca, "thal": thal,
        }

        result = predict_from_input(patient_data, model, scaler)

        if result["prediction"] == 1:
            st.markdown(f"""
            <div class="result-positive">
                <div class="result-text">❤️‍🔥 Heart Disease Detected</div>
                <div class="confidence-text">Confidence: {result['confidence']}%</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-negative">
                <div class="result-text">✅ No Heart Disease</div>
                <div class="confidence-text">Confidence: {result['confidence']}%</div>
            </div>
            """, unsafe_allow_html=True)

        # Probability breakdown
        st.markdown("#### Probability Breakdown")
        proba = result["probabilities"]

        col_a, col_b = st.columns(2)
        with col_a:
            st.metric("No Disease", f"{proba['No Heart Disease']}%")
        with col_b:
            st.metric("Heart Disease", f"{proba['Heart Disease']}%")

        # Progress bars
        st.progress(proba["No Heart Disease"] / 100, text="No Disease")
        st.progress(proba["Heart Disease"]    / 100, text="Heart Disease")

        # Feature summary
        st.markdown("#### Patient Summary")
        summary = pd.DataFrame({
            "Feature": list(patient_data.keys()),
            "Value":   list(patient_data.values()),
        })
        st.dataframe(summary, use_container_width=True, hide_index=True)

        if result["prediction"] == 1:
            st.warning(
                "⚠️ This result suggests elevated risk. "
                "Please consult a qualified cardiologist immediately.",
                icon="🏥",
            )
        else:
            st.success(
                "✅ No significant risk detected. "
                "Continue regular health check-ups and maintain a healthy lifestyle.",
                icon="💚",
            )

    else:
        st.info("👈 Fill in the patient details and click **Run Prediction**.")

        # Show quick reference
        st.markdown("#### Quick Reference — Normal Ranges")
        reference = pd.DataFrame({
            "Parameter":       ["Blood Pressure", "Cholesterol", "Max Heart Rate", "Fasting Blood Sugar"],
            "Normal Range":    ["< 120/80 mm Hg", "< 200 mg/dl", "Depends on age", "< 100 mg/dl"],
        })
        st.dataframe(reference, use_container_width=True, hide_index=True)


# ── Disclaimer ────────────────────────────────────────────────────────────────
st.markdown("""
<div class="disclaimer">
  ⚕️ <strong>Medical Disclaimer:</strong> This tool is for educational purposes only and is NOT
  a substitute for professional medical advice, diagnosis, or treatment. Always seek the
  guidance of your physician or other qualified health provider.
</div>
""", unsafe_allow_html=True)

# ── Sidebar — About ───────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🫀 About")
    st.markdown("""
    This application uses machine learning to predict the likelihood of
    heart disease based on 13 clinical features from the
    **UCI Heart Disease Dataset (Cleveland)**.

    **Models Trained:**
    - Logistic Regression
    - Decision Tree
    - Random Forest *(usually best)*

    **Evaluation Metrics:**
    - Accuracy · Precision · Recall · F1 Score

    **Tech Stack:**
    - Python · scikit-learn · pandas · Streamlit

    ---
    Made for educational purposes.
    """)

    st.markdown("## 📁 Project Structure")
    st.code("""
heart-disease-prediction/
├── data/
├── notebooks/
├── models/
├── src/
│   ├── data_loader.py
│   ├── eda.py
│   ├── preprocessing.py
│   ├── train.py
│   └── predict.py
├── app.py
├── streamlit_app.py
├── requirements.txt
└── README.md
    """, language="")
