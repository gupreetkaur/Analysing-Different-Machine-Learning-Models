from sklearn.metrics import(
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report)

def evaluate_model(Y_true, Y_pred):
    # Weighted averages account for imbalanced classes while retaining every class.
    return{
        'accuracy' : accuracy_score(Y_true, Y_pred),

        'precision' : precision_score(Y_true, Y_pred, average='weighted', zero_division=0),

        'recall' : recall_score(Y_true, Y_pred, average='weighted', zero_division=0),

        'f1' : f1_score(Y_true, Y_pred, average='weighted', zero_division=0),

        # Convert NumPy arrays to lists so the metrics can be stored as JSON.
        'confusion_matrix' : confusion_matrix(Y_true, Y_pred).tolist(),

        'classification_report' : classification_report(Y_true, Y_pred, output_dict = True, zero_division=0)
    }
