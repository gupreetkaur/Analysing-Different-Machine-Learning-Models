# Model Evaluation Results

This folder contains comprehensive outputs from evaluating six machine learning classifiers across four distinct datasets. All models were trained and tested using a stratified 80/20 train-test split with `random_state=42` to ensure reproducibility and fair comparison.

---

## 📊 Executive Summary

### Performance Overview by Dataset

| Dataset | Best Model | Accuracy | Precision | Recall | F1-Score | Full Details |
|---------|-----------|----------|-----------|--------|----------|--------------|
| **Digit Recognizer** | XGBoost | **97.39%** | 97.40% | 97.39% | 97.39% | [Table](accuracy_tables/Digit_Recognizer_accuracy.csv) \| [Chart](graphs/Digit_Recognizer_accuracy.png) |
| **Titanic Survival** | Decision Tree | **82.12%** | - | - | - | [Table](accuracy_tables/Titanic_accuracy.csv) \| [Chart](graphs/Titanic_accuracy.png) |
| **Wine Quality** | Random Forest | **62.50%** | - | - | - | [Table](accuracy_tables/Wine_Quality_accuracy.csv) \| [Chart](graphs/Wine_Quality_accuracy.png) |
| **Bank Marketing** | XGBoost | **91.36%** | - | - | - | [Table](accuracy_tables/Bank_Marketing_accuracy.csv) \| [Chart](graphs/Bank_Marketing_accuracy.png) |

---

## 📁 Folder Structure & Contents

### `accuracy_tables/` — Model Performance Comparison
**Purpose:** Numerical comparison of all six models for each dataset

**Files:** 
- `Digit_Recognizer_accuracy.csv` — 6 models evaluated on digit classification (0-9)
- `Titanic_accuracy.csv` — 6 models evaluated on binary passenger survival prediction
- `Wine_Quality_accuracy.csv` — 6 models evaluated on wine quality rating prediction
- `Bank_Marketing_accuracy.csv` — 6 models evaluated on customer marketing response

**Content:** Each CSV contains:
- **Model name** — Algorithm used
- **Accuracy** — Percentage of correct predictions on test set
- **Precision** — True positives / (true positives + false positives)
- **Recall** — True positives / (true positives + false negatives)
- **F1-Score** — Harmonic mean of precision and recall

**Example usage:** Use these tables to compare model performance side-by-side for decision-making.

---

### `confusion_matrices/` — Classification Visual Analysis
**Purpose:** Visual representation of correct and incorrect predictions for each model-dataset combination

**Files:** 24 PNG images (6 models × 4 datasets)
- Pattern: `{Dataset}_{Model}.png`
- Example: `Digit_Recognizer_xgboost.png`, `Bank_Marketing_decision_tree.png`

**What the matrices show:**
- **Diagonal cells** (darker) = Correct predictions (True Positives/Negatives)
- **Off-diagonal cells** (lighter) = Misclassifications (False Positives/Negatives)
- **Row labels** = True class labels
- **Column labels** = Predicted class labels

**Insights by dataset:**
- **Digit Recognizer:** 10×10 matrix (digits 0-9); XGBoost shows minimal confusion
- **Titanic:** 2×2 matrix (survived/not survived)
- **Wine Quality:** Multi-class matrix
- **Bank Marketing:** Binary classification matrix

**Usage:** Identify which classes are most commonly confused and where improvements are needed.

---

### `graphs/` — Accuracy Comparison Charts
**Purpose:** Visual comparison of model performance on each dataset

**Files:** 4 PNG images (one per dataset)
- `Digit_Recognizer_accuracy.png`
- `Titanic_accuracy.png`
- `Wine_Quality_accuracy.png`
- `Bank_Marketing_accuracy.png`

**Chart Type:** Bar charts showing accuracy scores for all 6 models

**Key observations:**
- **Digit Recognizer:** XGBoost achieves highest accuracy (97.39%); Random Forest also performs well
- **Titanic:** More variation in model performance; Decision Tree outperforms others
- **Wine Quality:** Moderate performance across all models (~60-62% best)
- **Bank Marketing:** XGBoost leads (91.36%); strong performance overall

**Usage:** Quick visual reference for model comparison and communication.

---

### `reports/` — Detailed Classification Reports
**Purpose:** Per-class performance metrics for detailed analysis

**Files:** 24 text files (6 models × 4 datasets)
- Pattern: `{Dataset}_{Model}.txt`
- Example: `Digit_Recognizer_xgboost.txt`

**Report content:**
- **Precision per class** — How often the model correctly predicted that class
- **Recall per class** — How often the model found instances of that class
- **F1-score per class** — Balanced metric combining precision and recall
- **Support** — Number of true instances of each class in test set
- **Macro average** — Unweighted mean across all classes
- **Weighted average** — Mean weighted by support (useful for imbalanced datasets)

**Example interpretation:**
```
Class 0 (Digit '0'):
  Precision: 0.98 → When model predicted '0', it was correct 98% of the time
  Recall: 0.99 → Model found 99% of all '0' digits
  F1-score: 0.98 → Strong balanced performance
  Support: 827 → 827 test samples were digit '0'
```

**Usage:** Identify which classes/categories are hardest to predict and where models struggle.

---

### Root Directory — Timestamped JSON Results
**Purpose:** Raw, structured results from individual model runs for reproducibility and logging

**Files:** 24 JSON files (6 models × 4 datasets)
- Pattern: `{Dataset}_{Model}_{Timestamp}.json`
- Example: `Digit_Recognizer_xgboost_20260814_174345.json`

**JSON Structure:**
```json
{
  "dataset": "Digit_Recognizer",
  "model_name": "xgboost",
  "timestamp": "2026-08-14 17:43:45.483006",
  "metrics": {
    "accuracy": 0.9739285714285715,
    "precision": 0.9740112081640679,
    "recall": 0.9739285714285715,
    "f1": 0.9739334871545342,
    "confusion_matrix": [[...], [...], ...],
    "classification_report": {
      "0": { "precision": ..., "recall": ..., "f1-score": ..., "support": ... },
      "1": { ... },
      ...
      "accuracy": ...,
      "macro avg": { ... },
      "weighted avg": { ... }
    }
  }
}
```

**Key fields:**
- `dataset` — Name of the dataset
- `model_name` — ML algorithm used
- `timestamp` — Exact date/time of evaluation
- `metrics.accuracy` — Overall accuracy as decimal (e.g., 0.9739 = 97.39%)
- `metrics.confusion_matrix` — 2D array showing true vs. predicted labels
- `metrics.classification_report` — Detailed per-class metrics

**Usage:** 
- Programmatically load results for further analysis
- Verify exact metrics and reproduce results
- Track historical model performance over time
- Integrate with visualization or reporting pipelines

---

## 🧠 Models Evaluated

All six classifiers were evaluated consistently:

1. **Logistic Regression** — Linear classifier; fast, interpretable, baseline model
2. **Decision Tree** — Tree-based; good for non-linear patterns, interpretable
3. **Random Forest** — Ensemble of decision trees; reduces overfitting, robust
4. **K-Nearest Neighbours (KNN)** — Instance-based; simple, non-parametric
5. **Support Vector Machine (SVM)** — Kernel-based; effective in high-dimensional spaces
6. **XGBoost** — Gradient boosting ensemble; often achieves highest accuracy, computationally intensive

---

## 📈 Key Findings & Recommendations

### By Dataset:

**Digit Recognizer (Best: XGBoost 97.39%)**
- Excellent performance across all models
- XGBoost is the recommended choice
- Confusion mainly between visually similar digits (e.g., 4 and 9)

**Titanic Survival (Best: Decision Tree 82.12%)**
- Decision Tree significantly outperforms others
- Moderate accuracy; room for feature engineering
- Binary classification problem

**Wine Quality (Best: Random Forest 62.50%)**
- Lowest accuracy across all datasets
- Suggests limited predictive features or inherent difficulty
- Consider collecting more data or engineering additional features

**Bank Marketing (Best: XGBoost 91.36%)**
- Very strong performance; excellent for production
- XGBoost is recommended
- Good for customer targeting campaigns

### General Recommendations:
- **For production:** Use XGBoost (Digit Recognizer, Bank Marketing) or Decision Tree (Titanic)
- **For interpretability:** Use Decision Tree or Logistic Regression
- **For speed:** Use Logistic Regression or Decision Tree
- **For accuracy:** Use XGBoost or Random Forest

---

## 🔍 How to Use These Results

1. **Quick comparison:** Check the accuracy tables and graphs
2. **Detailed analysis:** Review confusion matrices for error patterns
3. **Per-class performance:** Examine the reports for specific classes
4. **Reproducibility:** Load the JSON files to verify exact metrics
5. **Integration:** Use JSON files to programmatically load results into dashboards or reports

---

## 📋 Methodology

**Data Split:** 80% training, 20% testing (stratified)
**Random State:** 42 (ensures reproducibility)
**Evaluation Metrics:** 
- Accuracy (overall correctness)
- Precision (false positive rate)
- Recall (false negative rate)
- F1-Score (harmonic mean of precision and recall)

**Timestamp Format:** `YYYY-MM-DD HH:MM:SS.MMMMMM` (UTC)

---

## 📝 Notes

- All models used the same train-test split for fair comparison
- Timestamps reflect when each model was evaluated
- JSON files preserve complete evaluation data for auditing
- Confusion matrices help identify systematic prediction errors
- Per-class reports enable targeted model improvements

---
