# 💳 Credit Card Fraud Detection

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)](https://python.org)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter)](https://jupyter.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-f7931e?logo=scikitlearn)](https://scikit-learn.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An end-to-end machine learning project for detecting fraudulent credit card transactions using ensemble models, SMOTE oversampling, threshold tuning, and SHAP explainability.

---

## 📌 Problem Statement

Credit card fraud is a critical challenge for financial institutions. This project builds a binary classifier to identify fraudulent transactions from a highly imbalanced dataset where only **0.17%** of transactions are fraudulent. Standard accuracy metrics are misleading here — the focus is on **Precision-Recall AUC** and **F1-Score**.

---

## 📂 Dataset

- **Source:** [Kaggle — Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Records:** 284,807 transactions (September 2013, European cardholders)
- **Features:** 30 total — `Time`, `Amount`, and `V1`–`V28` (PCA-anonymised)
- **Target:** `Class` — `0` = Legitimate, `1` = Fraud
- **Imbalance:** Only 492 fraud cases (0.172%)

> ⚠️ `creditcard.csv` is not included in this repo due to its size (~150 MB). Download it from Kaggle and place it in the project root.

---

## 🛠️ Project Structure

```
credit-card-fraud-detection/
│
├── ccfd.ipynb              # Main Jupyter Notebook (full pipeline)
├── creditcard.csv          # Dataset (download from Kaggle — not in repo)
├── models/                 # Saved models (auto-created on run)
│   ├── best_model.pkl
│   ├── scaler.pkl
│   └── model_info.json
├── README.md
├── requirements.txt
└── .gitignore
```

---

## 🔬 Methodology

### 1. Exploratory Data Analysis
- Class distribution, transaction amount & time analysis
- Correlation heatmap and top features correlated with fraud

### 2. Feature Engineering
- Extracted `Hour` from the `Time` column to capture daily patterns
- Dropped raw `Time` and `Amount` — replaced by scaled versions

### 3. Train / Test Split
- 80/20 stratified split preserving fraud ratio

### 4. Feature Scaling
- `StandardScaler` fit **only on training data** to prevent data leakage

### 5. Class Imbalance — SMOTE
- Applied SMOTE exclusively on the training set to generate synthetic fraud samples

### 6. Models Trained

| Model | Type |
|-------|------|
| Logistic Regression | Linear baseline |
| Random Forest | Bagging ensemble |
| XGBoost | Gradient boosting |
| LightGBM | Fast gradient boosting |

### 7. Evaluation & Threshold Tuning
- Models compared by **PR-AUC** (best metric for imbalanced fraud data)
- Decision threshold optimised for maximum F1-Score

### 8. Explainability — SHAP
- SHAP TreeExplainer used to visualise global and local feature importance

### 9. Model Persistence
- Best model and scaler saved via `joblib` for production inference

---

## 📊 Results

| Model | ROC-AUC | PR-AUC | Notes |
|-------|---------|--------|-------|
| Logistic Regression | ~0.97 | ~0.72 | Fast, interpretable baseline |
| Random Forest | ~0.98 | ~0.85 | Strong ensemble performance |
| XGBoost | ~0.98 | ~0.87 | Excellent imbalanced handling |
| **LightGBM** | **~0.98** | **~0.88** | **Best overall — fastest training** |

> Results may vary slightly by environment. Run the notebook for exact numbers.

---

## ⚙️ Setup & Installation

### Prerequisites
- Python 3.8+
- Jupyter Notebook or JupyterLab

### Install dependencies

```bash
pip install -r requirements.txt
```

### Run the notebook

```bash
jupyter notebook ccfd.ipynb
```

---

## 📦 Requirements

```
pandas
numpy
matplotlib
seaborn
scikit-learn
imbalanced-learn
xgboost
lightgbm
shap
joblib
```

Install all at once:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn xgboost lightgbm shap joblib
```

---

## 🚀 Inference — Predict on New Data

After running the notebook, load the saved model for predictions:

```python
import joblib, json
import pandas as pd

model  = joblib.load("models/best_model.pkl")
scaler = joblib.load("models/scaler.pkl")

with open("models/model_info.json") as f:
    info = json.load(f)

# Your new transactions (must include V1-V28 + Hour)
X_new = pd.DataFrame([...], columns=info["features"])
X_scaled = scaler.transform(X_new)
probs = model.predict_proba(X_scaled)[:, 1]
preds = (probs >= info["best_threshold"]).astype(int)
```

---

## 🧠 Key Learnings

- **Accuracy is a misleading metric** for highly imbalanced datasets — a model predicting "legit" every time achieves 99.83% accuracy but catches 0 frauds
- **PR-AUC** is the correct metric to optimise for fraud detection
- **SMOTE must only be applied on training data** — applying it before splitting leaks information
- **Threshold tuning** recovers additional true positives at the cost of acceptable false positives
- **SHAP** reveals that V14, V17, and V12 are the most predictive fraud indicators

---

## 📄 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

## 🙋 Author

**Rajneesh Babu**  
GitHub: [@rajneeshbabu](https://github.com/rajneeshbabu)

---

## ⭐ Acknowledgements

- Dataset: [ULB Machine Learning Group](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) via Kaggle
- SHAP: [slundberg/shap](https://github.com/slundberg/shap)
