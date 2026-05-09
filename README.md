# 💳 Credit Card Fraud Detection

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-EE4C2C?logo=pytorch)](https://pytorch.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-f7931e?logo=scikitlearn)](https://scikit-learn.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An end-to-end machine learning project for detecting fraudulent credit card transactions. Combines a tuned ensemble model (LightGBM / XGBoost / Random Forest) with a PyTorch autoencoder for anomaly detection — with full SHAP explainability and support for Apple Silicon GPU (MPS).

---

## 📌 Problem Statement

Credit card fraud is a critical challenge for financial institutions. This project builds a binary classifier on a highly imbalanced dataset where only **0.17%** of transactions are fraudulent. Standard accuracy is meaningless here — the project focuses on **Precision-Recall AUC** and **F1-Score**.

---

## 📂 Dataset

- **Source:** [Kaggle — Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Records:** 284,807 transactions (September 2013, European cardholders)
- **Features:** 30 total — `Time`, `Amount`, and `V1`–`V28` (PCA-anonymised)
- **Target:** `Class` — `0` = Legitimate, `1` = Fraud
- **Imbalance:** Only 492 fraud cases (0.172%)

> ⚠️ `creditcard.csv` is not included in this repo (~150 MB). Download it from Kaggle and place it in the project root before running the notebook.

---

## 🗂️ Project Structure

```
credit-card-fraud-detection/
│
├── ccfd.ipynb          # Main notebook — full pipeline (run this)
├── train.py            # Standalone training script (alternative to notebook)
├── model_arch.py       # Shared PyTorch model class (used by train.py)
├── requirements.txt    # Python dependencies
├── README.md
├── .gitignore
│
├── creditcard.csv      # Dataset — download from Kaggle, not in repo
└── models/             # Auto-generated after running the notebook / train.py
    ├── best_model.pkl      # Tuned best classifier (joblib)
    ├── scaler.pkl          # Fitted StandardScaler
    ├── autoencoder.pt      # PyTorch autoencoder weights
    └── model_info.json     # Metadata: thresholds, metrics, feature names
```

---

## 🔬 Pipeline (inside `ccfd.ipynb`)

### 1. Data Loading & EDA
- Class distribution, transaction amount & time analysis
- Correlation heatmap — top features correlated with fraud
- KDE plots for PCA features (V1–V10) by class

### 2. Feature Engineering & Preprocessing
- Extract `Hour` from `Time` column (daily fraud patterns)
- Drop raw `Time` and `Amount`
- **Stratified 80/20 train/test split**
- `StandardScaler` fit **only on training data** (no leakage)
- **SMOTE** applied **only on training data** (no leakage)

### 3. Model Comparison
Four candidates trained on the SMOTE-balanced training set and evaluated on the held-out test set:

| Model | Type |
|-------|------|
| Logistic Regression | Linear baseline |
| Random Forest | Bagging ensemble |
| XGBoost | Gradient boosting |
| LightGBM | Fast gradient boosting |

Best model selected by **PR-AUC on test set**.

### 4. Hyperparameter Tuning — RandomizedSearchCV
- `RandomizedSearchCV` samples **40 random combinations** from a large param grid
- Scoring metric: `average_precision` (PR-AUC)
- Final model retrained on full SMOTE training set with best params

### 5. Threshold Tuning & Evaluation
- Decision threshold optimised on Precision-Recall curve for **maximum F1-Score**
- Confusion matrix + full classification report

### 6. SHAP Explainability
- `TreeExplainer` global feature importance
- Beeswarm plot — feature impact direction on fraud prediction

### 7. PyTorch Autoencoder — Anomaly Detection
- Encoder–Decoder trained **on legitimate transactions only**
- Fraud = high reconstruction error (out-of-distribution)
- Training on **Apple Silicon GPU (MPS)** / NVIDIA CUDA / CPU — auto-detected
- Architecture: `input → 64 → 32 → 16 (bottleneck) → 32 → 64 → input`
- Early stopping + `ReduceLROnPlateau` scheduler

### 8. Ensemble
- `ensemble_score = 0.6 × supervised_prob + 0.4 × ae_score_normalised`
- Separate threshold tuning for the ensemble score

### 9. Save Artefacts
- `best_model.pkl`, `scaler.pkl`, `autoencoder.pt`, `model_info.json`

### 10. Summary
- Comparison table and bar chart across all models and ensemble
- Key takeaways

---

## 📊 Results

| Model | ROC-AUC | PR-AUC | Notes |
|-------|---------|--------|-------|
| Logistic Regression | ~0.97 | ~0.72 | Fast interpretable baseline |
| Random Forest | ~0.98 | ~0.85 | Strong ensemble |
| XGBoost | ~0.98 | ~0.87 | Excellent imbalanced handling |
| **LightGBM (tuned)** | **~0.98** | **~0.88+** | **Best — after RandomizedSearchCV** |
| Autoencoder (PyTorch) | — | ~0.40 | Anomaly detection only |
| **Ensemble** | **~0.98** | **~0.88+** | **Supervised + Autoencoder** |

> Run the notebook for exact numbers. Results depend on RandomizedSearchCV's random sampling.

---

## ⚙️ Setup & Installation

### 1. Clone the repo

```bash
git clone https://github.com/rajneeshbabu/credit-card-fraud-detection.git
cd credit-card-fraud-detection
```

### 2. Install PyTorch (Mac GPU support)

```bash
pip install torch torchvision torchaudio
```

### 3. Install remaining dependencies

```bash
pip install -r requirements.txt
```

### 4. Download the dataset

Download `creditcard.csv` from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) and place it in the project root.

---

## 🚀 Run the Notebook

```bash
jupyter notebook ccfd.ipynb
```

Run all cells top to bottom. The notebook will:
1. Compare all four models and automatically pick the best by PR-AUC
2. Run RandomizedSearchCV tuning on the winner
3. Train the PyTorch autoencoder on your GPU (MPS / CUDA / CPU)
4. Evaluate the ensemble and save all artefacts to `models/`

**Expected runtime:** ~10–20 minutes depending on hardware.

### Alternative — standalone training script

```bash
python train.py
```

Produces the same `models/` artefacts without Jupyter.

---

## 🧠 Key Learnings

- **Accuracy is misleading** for imbalanced data — a model predicting "legit" every time achieves 99.83% accuracy but catches 0 frauds
- **PR-AUC is the right metric** for fraud detection
- **SMOTE must be applied only on training data** — applying before splitting leaks information
- **StandardScaler must be fit only on training data** — same reason
- **Threshold tuning** recovers true positives missed by the default 0.5 cut-off
- **RandomizedSearchCV** is far more efficient than GridSearch for large parameter spaces
- **Autoencoder anomaly detection** is complementary to supervised learning — it catches out-of-distribution fraud the classifier may miss
- **SHAP** confirms that V14, V17, and V12 are the most predictive fraud indicators

---

## 📄 License

This project is licensed under the MIT License.

---

## 🙋 Author

**Rajneesh Babu**
GitHub: [@rajneeshbabu](https://github.com/rajneeshbabu)

---

## ⭐ Acknowledgements

- Dataset: [ULB Machine Learning Group](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) via Kaggle
- SHAP: [slundberg/shap](https://github.com/slundberg/shap)
- PyTorch: [pytorch.org](https://pytorch.org)
