# 🫀 Heart Disease Prediction

A complete, beginner-friendly **Machine Learning project** that predicts the likelihood of heart disease using clinical data — with a Streamlit web app, interactive notebook, and industry-standard code organisation.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.5-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35-red)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 📌 Project Overview

| Item | Detail |
|------|--------|
| **Dataset** | UCI Heart Disease (Cleveland) — 303 patients, 13 features |
| **Task** | Binary classification: Heart Disease (1) vs No Disease (0) |
| **Models** | Logistic Regression · Decision Tree · Random Forest |
| **Best Model** | Random Forest (highest F1 Score) |
| **Interface** | Streamlit web app + CLI prediction |

---

## 📁 Project Structure

```
heart-disease-prediction/
│
├── data/                        # Dataset stored here (auto-downloaded)
│   └── heart.csv
│
├── notebooks/                   # Jupyter notebook + saved plot images
│   ├── heart_disease_analysis.ipynb
│   ├── target_distribution.png
│   ├── feature_distributions.png
│   ├── correlation_heatmap.png
│   ├── feature_vs_target.png
│   ├── confusion_matrices.png
│   ├── model_comparison.png
│   └── feature_importance.png
│
├── models/                      # Saved trained model (pickle)
│   └── best_model.pkl
│
├── src/                         # Source code modules
│   ├── __init__.py
│   ├── data_loader.py           # Dataset loading & synthetic fallback
│   ├── eda.py                   # Exploratory Data Analysis & plots
│   ├── preprocessing.py         # Cleaning, scaling, train/test split
│   ├── train.py                 # Model training, evaluation, saving
│   └── predict.py               # Prediction utilities & CLI interface
│
├── app.py                       # Main pipeline script (run this first)
├── streamlit_app.py             # Streamlit web application
├── requirements.txt             # Python dependencies
└── README.md                    # You are here
```

---

## 🔬 Dataset — Features Explained

| # | Feature | Description |
|---|---------|-------------|
| 1 | `age` | Age in years |
| 2 | `sex` | 1 = Male, 0 = Female |
| 3 | `cp` | Chest pain type (0–3) |
| 4 | `trestbps` | Resting blood pressure (mm Hg) |
| 5 | `chol` | Serum cholesterol (mg/dl) |
| 6 | `fbs` | Fasting blood sugar > 120 mg/dl |
| 7 | `restecg` | Resting ECG results (0–2) |
| 8 | `thalach` | Max heart rate achieved |
| 9 | `exang` | Exercise-induced angina |
| 10 | `oldpeak` | ST depression induced by exercise |
| 11 | `slope` | Slope of peak exercise ST segment |
| 12 | `ca` | Number of major vessels (0–3) |
| 13 | `thal` | Thalassemia type |
| 🎯 | `target` | **1 = Heart Disease, 0 = No Disease** |

---

## 🚀 Getting Started

### Prerequisites
- Python 3.10 or newer
- pip package manager

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/heart-disease-prediction.git
cd heart-disease-prediction
```

### 2. Create a virtual environment (recommended)

```bash
# Create virtual environment
python -m venv venv

# Activate it
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the main ML pipeline

This will download the dataset, run EDA, train all models, compare them, save the best model, and demo a prediction.

```bash
python app.py
```

### 5. Launch the Streamlit web app

```bash
streamlit run streamlit_app.py
```

Open your browser at **http://localhost:8501** 🎉

---

## 📊 Model Performance (typical results)

| Model | Accuracy | Precision | Recall | F1 Score |
|-------|----------|-----------|--------|----------|
| Logistic Regression | ~84% | ~83% | ~86% | ~84% |
| Decision Tree | ~78% | ~77% | ~80% | ~78% |
| **Random Forest** | **~87%** | **~86%** | **~88%** | **~87%** |

> Results may vary slightly due to random state and dataset version.

---

## 🧠 ML Concepts Covered

- **Data Cleaning** — handling missing values, duplicates, type conversion
- **Feature Engineering** — StandardScaler normalisation
- **Train/Test Split** — 80/20 with stratified sampling
- **Classification Metrics** — Accuracy, Precision, Recall, F1
- **Confusion Matrix** — visualising TP, TN, FP, FN
- **Feature Importance** — Random Forest built-in ranking
- **Model Persistence** — pickle serialisation

---

## 🖥️ Streamlit App Features

- Sliders and dropdowns for all 13 features
- Real-time prediction on button click
- Confidence score display
- Probability breakdown (No Disease vs Disease)
- Patient data summary table
- Medical disclaimer

---

## 📓 Running the Notebook

```bash
pip install jupyter
cd notebooks
jupyter notebook heart_disease_analysis.ipynb
```

---

## 🤝 Contributing

Pull requests are welcome! For major changes, please open an issue first.

---

## ⚕️ Disclaimer

This project is for **educational purposes only**. It is NOT intended to replace professional medical advice, diagnosis, or treatment. Always consult a qualified healthcare provider.

---

## 📄 License

[MIT](LICENSE)
