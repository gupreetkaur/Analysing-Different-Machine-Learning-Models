# Model Evaluation Results

This folder contains the outputs produced when the six classifiers were evaluated on the four datasets. Each model uses the same stratified 80/20 train-test split with `random_state=42` for a fair comparison.

## Best accuracy by dataset

| Dataset | Best model | Accuracy | Full comparison |
| --- | --- | ---: | --- |
| Digit Recognizer | XGBoost | 97.39% | [accuracy table](accuracy_tables/Digit_Recognizer_accuracy.csv) |
| Titanic | Decision Tree | 82.12% | [accuracy table](accuracy_tables/Titanic_accuracy.csv) |
| Wine Quality | Random Forest | 62.50% | [accuracy table](accuracy_tables/Wine_Quality_accuracy.csv) |
| Bank Marketing | XGBoost | 91.36% | [accuracy table](accuracy_tables/Bank_Marketing_accuracy.csv) |

## Folder guide

| Location | Contents |
| --- | --- |
| `accuracy_tables/` | CSV comparison of accuracy, weighted precision, recall, and F1-score for all six models. |
| `confusion_matrices/` | PNG confusion-matrix charts for every dataset and model combination. |
| `graphs/` | Accuracy-comparison charts for each dataset. |
| `reports/` | Text classification reports with per-class precision, recall, F1-score, and support. |
| Root of `Results/` | Timestamped JSON files containing the structured metrics for individual runs. |

The evaluated models are Logistic Regression, Decision Tree, Random Forest, K-Nearest Neighbours, Support Vector Machine, and XGBoost.
