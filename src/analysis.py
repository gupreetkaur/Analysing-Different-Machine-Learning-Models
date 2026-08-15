"""Small CLI entrypoint for quick experiments.
Usage: python -m src.analysis --demo
"""
import argparse
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


def demo():
    data = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=42)
    clf = RandomForestClassifier(n_estimators=50, random_state=42)
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"Iris demo accuracy: {acc:.4f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--demo", action="store_true", help="Run the small sklearn demo on Iris")
    args = p.parse_args()
    if args.demo:
        demo()
    else:
        p.print_help()


if __name__ == "__main__":
    main()
