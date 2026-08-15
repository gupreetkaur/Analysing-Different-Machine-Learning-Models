import json
import os
import pandas as pd
from datetime import datetime

def save_results(dataset_name, model_name, metrics):

    # Create the output folder once so each experiment can be persisted independently.
    os.makedirs("../Results", exist_ok=True)

    filename = (
        f"../Results/" 
        f"{dataset_name}_" 
        f"{model_name}_" 
        f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )

    print("Saving JSON to:", filename)

    data = {
        "dataset" : dataset_name,
        "model_name" : model_name,
        "timestamp" : str(datetime.now()),
        "metrics" : metrics

    }

    # Save metrics as JSON so downstream tools can load them without parsing text.
    with open(filename, "w") as f:
        json.dump(data, f, indent=4)


def save_accuracy_table(dataset_name, all_results):

    os.makedirs("../Results/accuracy_tables", exist_ok=True)

    # One row per model makes the comparison easy to inspect or reuse elsewhere.
    df = pd.DataFrame(all_results)

    df.to_csv(
        f"../Results/accuracy_tables/{dataset_name}_accuracy.csv",
        index=False
    )

def save_classification_report(dataset_name, model_name, report):

    os.makedirs("../Results/reports", exist_ok=True)

    # Keep the detailed text report alongside the machine-readable metric file.
    with open(
            f"../Results/reports/{dataset_name}_{model_name}.txt",
        "w"
    ) as f:

        f.write(report)
