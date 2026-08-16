# Analysing Different Machine Learning Models

This project compares six classification algorithms on four datasets through a shared, reproducible workflow. Each dataset has its own Jupyter notebook for exploration and training, while reusable modules handle preprocessing, evaluation, visualisation, and saving results.

The repository includes 20-row CSV previews. Download the complete data from the Kaggle sources below before running the notebooks.

## Current implementation

- Added a shared pipeline that trains Logistic Regression, Decision Tree, Random Forest, KNN, SVM, and XGBoost with a consistent interface.
- Added stratified 80/20 train-test splits with `random_state=42` for reproducible and comparable evaluation.
- Added feature scaling fitted only on the training data for Logistic Regression, KNN, and SVM.
- Added weighted accuracy, precision, recall, F1-score, confusion matrices, and classification reports for every model run.
- Added automatic saving of JSON metrics, text reports, accuracy tables, confusion-matrix images, and model-comparison charts under `Results/`.

## Project goal

The project answers a practical question: **which classification model performs best for each dataset when every candidate is evaluated on the same held-out test split?** It provides a consistent comparison across image-like numeric data, small tabular data, multiclass quality data, and a large marketing dataset.

## Datasets and preparation

| Dataset | Target | Kaggle source | Preparation performed |
| --- | --- | --- | --- |
| Digit Recognizer | `label` | [Digit Recognizer](https://www.kaggle.com/competitions/digit-recognizer) | Classifies digits 0-9 from 784 grayscale pixel features. |
| Titanic | `Survived` | [Titanic - Machine Learning from Disaster](https://www.kaggle.com/competitions/titanic) | Fills missing `Age` and `Embarked` values, removes `Name`, `Ticket`, and `Cabin`, then one-hot encodes `Sex` and `Embarked`. |
| Wine Quality | `quality` | [winequality-red.csv](https://www.kaggle.com/datasets/uciml/red-wine-quality-cortez-et-al-2009) | Removes duplicate rows before training. |
| Bank Marketing | `y` | [Bank Marketing Data Set](https://www.kaggle.com/datasets/joseguzman/bank-marketing) | Removes duplicate rows, one-hot encodes categorical predictors, and maps `no`/`yes` to `0`/`1`. |

## Notebooks

| Dataset | Prediction task | Notebook |
| --- | --- | --- |
| Digit Recognizer | Handwritten digit classification | [digit_recognizer.ipynb](Notebooks/digit_recognizer.ipynb) |
| Titanic | Passenger survival | [titanic.ipynb](Notebooks/titanic.ipynb) |
| Wine Quality | Wine-quality class | [wine_quality.ipynb](Notebooks/wine_quality.ipynb) |
| Bank Marketing | Term-deposit subscription | [bank_marketing.ipynb](Notebooks/bank_marketing.ipynb) |

## Best model by dataset

The following results are from the same reproducible 80/20 held-out test approach. Accuracy identifies the best model in this project; use the linked CSV files to also compare weighted precision, recall, and F1.

| Dataset | Best model | Accuracy | Why it performed best here | Detailed metrics |
| --- | --- | ---: | --- | --- |
| Digit Recognizer | XGBoost | 97.39% | Boosted trees capture complex, non-linear relationships among the 784 pixel features. | [CSV](Results/accuracy_tables/Digit_Recognizer_accuracy.csv) |
| Titanic | Decision Tree | 82.12% | A single tree captured useful threshold and interaction rules in the small, prepared tabular dataset. | [CSV](Results/accuracy_tables/Titanic_accuracy.csv) |
| Wine Quality | Random Forest | 62.50% | Combining many trees handled non-linear interactions among the physicochemical measurements better than one model. | [CSV](Results/accuracy_tables/Wine_Quality_accuracy.csv) |
| Bank Marketing | XGBoost | 91.36% | Boosting modelled the non-linear patterns in the encoded customer and campaign features most effectively. | [CSV](Results/accuracy_tables/Bank_Marketing_accuracy.csv) |

These explanations are interpretations of the observed test results, not guarantees that the same model will always win. Retune and validate again when the data, features, or objective changes.

## Model comparison at a glance

| Dataset | Logistic Regression | Decision Tree | Random Forest | KNN | SVM | XGBoost |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Digit Recognizer | 90.21% | 85.51% | 96.40% | 93.75% | 95.71% | **97.39%** |
| Titanic | 80.45% | **82.12%** | 81.56% | 81.56% | 81.01% | 80.45% |
| Wine Quality | 58.09% | 51.47% | **62.50%** | 56.25% | 59.19% | 60.66% |
| Bank Marketing | 90.93% | 89.18% | 91.17% | 89.85% | 90.82% | **91.36%** |

## Models

- Logistic Regression
- Decision Tree
- Random Forest
- K-Nearest Neighbors (KNN)
- Support Vector Machine (SVM)
- XGBoost

Logistic Regression, KNN, and SVM use standardised features. Decision Tree, Random Forest, and XGBoost train on the original feature values.

## Generated outputs

Each notebook saves the following artifacts in `Results/`:

- `accuracy_tables/`: one CSV comparison table per dataset
- `confusion_matrices/`: one PNG confusion matrix for each model and dataset
- `graphs/`: model-accuracy comparison charts
- `reports/`: text classification reports
- root `Results/` directory: timestamped JSON files containing metrics, confusion matrices, and structured classification reports

## Project structure

```text
.
|-- Data/                         # Source CSV files
|-- Models/
|   `-- pipeline.py               # Model selection, training, and evaluation
|-- Notebooks/                    # Dataset-specific analysis notebooks
|-- Results/
|   |-- accuracy_tables/          # CSV metric summaries
|   |-- confusion_matrices/       # Per-model confusion-matrix plots
|   |-- graphs/                   # Accuracy comparison plots
|   `-- reports/                  # Classification reports
`-- utils/                        # Preprocessing, metrics, plots, and logging
```

## Installation

Use Python 3.10 or newer, then install the dependencies:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost jupyter
```

## Run a notebook

From the project root:

```bash
jupyter notebook
```

Open a notebook from `Notebooks/` and run its cells in order. If running from an IDE, set the notebook working directory to `Notebooks` so the relative paths to `../Data` and `../Results` resolve correctly.

## Reproducibility

Train-test splits and applicable model configurations use `random_state=42`. Run a notebook from top to bottom to regenerate its saved outputs.
