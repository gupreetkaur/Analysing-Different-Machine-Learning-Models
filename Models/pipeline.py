from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import classification_report

from utils.metrics import evaluate_model
from utils.visualization import plot_confusion_matrix
from utils.logger import (save_results, save_classification_report)

# Return a reproducible classifier for the requested experiment name.
def get_model(model_name):
    models = {

        "logistic_regression": LogisticRegression(max_iter=5000, random_state=42),
        "decision_tree": DecisionTreeClassifier(random_state=42),
        "random_forest": RandomForestClassifier(random_state=42),
        "knn": KNeighborsClassifier(),
        "svm": SVC(),
        "xgboost": XGBClassifier(
            random_state=42,
            eval_metric="logloss")

    }

    return models[model_name]

def run_pipeline(
    X_train,
    X_test,
    X_train_scaled,
    X_test_scaled,
    Y_train,
    Y_test,
    model_name,
    dataset_name
):

    # Use the same model instance throughout training, prediction, and reporting.
    model = get_model(model_name)

    scaled_models = [
        "logistic_regression",
        "knn",
        "svm"
    ]

    # Distance- and margin-based models are sensitive to feature magnitude;
    # tree-based models are trained on the original feature values.
    if model_name in scaled_models:
        train_data = X_train_scaled
        test_data = X_test_scaled
    else:
        train_data = X_train
        test_data = X_test

    # XGBoost requires class labels starting from zero, so retain an encoder to
    # translate predictions back to the dataset's original label values.
    if model_name == "xgboost":

        encoder = LabelEncoder()

        Y_train_encoded = encoder.fit_transform(Y_train)
        Y_test_encoded = encoder.transform(Y_test)

        model.fit(
            train_data,
            Y_train_encoded
        )

        Y_pred_encoded = model.predict(test_data)

        # Convert predictions back to original labels
        Y_pred = encoder.inverse_transform(
            Y_pred_encoded.astype(int)
        )

    else:

        model.fit(
            train_data,
            Y_train
        )

        Y_pred = model.predict(test_data)

    # Evaluate with original labels so every model is compared on the same target.
    metrics = evaluate_model(
        Y_test,
        Y_pred
    )

    plot_confusion_matrix(
        Y_test,
        Y_pred,
        model_name,
        dataset_name
    )

    # Produce a human-readable report in addition to the structured metrics.
    report = classification_report(
        Y_test,
        Y_pred,
        zero_division=0
    )

    print(f"\nClassification Report - {model_name}")
    print(report)

    save_classification_report(
        dataset_name,
        model_name,
        report
    )

    save_results(
        dataset_name,
        model_name,
        metrics
    )

    return metrics
