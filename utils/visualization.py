import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(Y_true, Y_pred, model_name, dataset_name):

    # Keep each model's confusion matrix as a separate, high-resolution artifact.
    save_dir = "../Results/confusion_matrices"
    os.makedirs("../Results/confusion_matrices", exist_ok=True)

    # Generate confusion matrix
    cm = confusion_matrix(Y_true, Y_pred)

    plt.figure(figsize=(8,6))

    # Plot heatmap
    sns.heatmap(cm, annot=True, fmt="g", cmap="Blues")

    plt.title(f"{dataset_name}_{model_name} Confusion Matrix")

    plt.ylabel('True label')

    plt.xlabel('Predicted label')

    filename = os.path.join(save_dir, f"{dataset_name}_{model_name}.png")
    print("Saving confusion matrix to:", filename)

    plt.savefig(filename, dpi=300, bbox_inches="tight")

    plt.show()

    plt.close()


def plot_accuracy_graph(dataset_name, all_results):

    save_dir = "../Results/graphs"
    os.makedirs("../Results/graphs", exist_ok=True)

    # Extract the summary fields expected in the accumulated experiment results.
    models = [r["Model"] for r in all_results]
    accuracy = [r["Accuracy"] for r in all_results]

    plt.figure(figsize=(8,6))

    plt.bar(models, accuracy)

    plt.ylabel("Accuracy")

    plt.title(f"{dataset_name} Model Comparison")

    # Rotate labels to keep longer model names legible in the saved chart.
    plt.xticks(rotation=20)

    plt.tight_layout()

    filename = os.path.join(
        save_dir,
        f"{dataset_name}_accuracy.png"
    )

    plt.savefig(filename, dpi=300, bbox_inches="tight")

    plt.show()

    plt.close()
