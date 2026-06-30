#!/usr/bin/env python3
"""
src/qubo_project/model.py
Phase 3 & Phase 4: Learning, Prediction, and Test Data Classification.
"""

import argparse
import os
import sys
import time
import json
import joblib
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, confusion_matrix


def get_classifier(classifier_name: str, seed: int):
    """Maps the configuration string to an instantiated model."""
    if classifier_name == "random_forest":
        return RandomForestClassifier(random_state=seed)
    elif classifier_name == "logistic_regression":
        return LogisticRegression(max_iter=1000, random_state=seed)
    elif classifier_name == "decision_tree":
        return DecisionTreeClassifier(random_state=seed)
    else:
        raise ValueError(
            f"Unknown classifier: {classifier_name}. Use 'random_forest', 'logistic_regression' or 'decision_tree'.")


def train(classifier: str, reducedTrain_csv: str, target_column: str,
          model_path: str, metrics_json: str, seed: int = 42):
    """Executes the training sequence matching Page 9/11 definitions."""
    start_input_time = time.time()
    df = pd.read_csv(reducedTrain_csv)
    dataset_input_time = time.time() - start_input_time

    X_train = df.drop(columns=[target_column])
    y_train = df[target_column]

    n_samples, n_features = X_train.shape

    target_1_count = int(np.sum(y_train == 1))
    target_1_percentage = round(float((target_1_count / n_samples) * 100), 2)

    clf = get_classifier(classifier, seed)

    start_train_time = time.time()
    clf.fit(X_train, y_train)
    training_time = time.time() - start_train_time

    os.makedirs(os.path.dirname(os.path.abspath(model_path)), exist_ok=True)
    joblib.dump(clf, model_path)

    train_metrics = {
        "classifier": classifier,
        "seed": seed,
        "training_dataset": os.path.basename(reducedTrain_csv),
        "target_column": target_column,
        "model_path": os.path.basename(model_path),
        "n_samples": n_samples,
        "n_features": n_features,
        "target_1_percentage": target_1_percentage,
        "dataset_input_time": round(dataset_input_time, 2),
        "training_time": round(training_time, 2)
    }

    os.makedirs(os.path.dirname(os.path.abspath(metrics_json)), exist_ok=True)
    with open(metrics_json, 'w') as f:
        json.dump(train_metrics, f, indent=4)


def predict(reduced_Test_csv: str, target_column: str, model_path: str,
            predictions_csv: str, classif_stats_json: str):
    """Executes the prediction sequence matching Phase 4 strict CSV layout (row_n,target,prediction,score)."""
    df = pd.read_csv(reduced_Test_csv)
    X_test = df.drop(columns=[target_column])
    y_true = df[target_column]

    n_samples = len(df)
    target_1_count = int(np.sum(y_true == 1))
    target_1_percentage = float((target_1_count / n_samples) * 100)

    clf = joblib.load(model_path)
    classifier_name = type(clf).__name__
    if "RandomForest" in classifier_name:
        clf_label = "random_forest"
    elif "LogisticRegression" in classifier_name:
        clf_label = "logistic_regression"
    else:
        clf_label = "decision_tree"

    # Inference predictions and probabilities
    y_pred = clf.predict(X_test)

    try:
        y_scores = clf.predict_proba(X_test)[:, 1]
    except AttributeError:
        # Fallback if the classifier doesn't support predict_proba natively
        y_scores = y_pred.astype(float)

    # STRICT PHASE 4 CSV FORMAT: row_n, target, prediction, score
    output_df = pd.DataFrame({
        "row_n": df.index,
        "target": y_true,
        "prediction": y_pred,
        "score": np.round(y_scores, 2)
    })

    os.makedirs(os.path.dirname(os.path.abspath(predictions_csv)), exist_ok=True)
    output_df.to_csv(predictions_csv, index=False)

    # Compute evaluation metrics
    acc = float(accuracy_score(y_true, y_pred))
    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, labels=[0, 1], zero_division=0)

    try:
        roc_auc = float(roc_auc_score(y_true, y_scores))
    except:
        roc_auc = float(roc_auc_score(y_true, y_pred))

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    stats_data = {
        "classifier": clf_label,
        "n_samples": n_samples,
        "target_1_count": target_1_count,
        "target_1_percentage": float(round(target_1_percentage, 2)),
        "accuracy": acc,
        "class_0": {
            "precision": float(precision[0]),
            "recall": float(recall[0]),
            "f1": float(f1[0]),
            "support": int(support[0])
        },
        "class_1": {
            "precision": float(precision[1]),
            "recall": float(recall[1]),
            "f1": float(f1[1]),
            "support": int(support[1])
        },
        "roc_auc": roc_auc,
        "confusion_matrix": {
            "labels": [0, 1],
            "matrix": cm.tolist()
        }
    }

    os.makedirs(os.path.dirname(os.path.abspath(classif_stats_json)), exist_ok=True)
    with open(classif_stats_json, 'w') as f:
        json.dump(stats_data, f, indent=4)


def main():
    parser = argparse.ArgumentParser(description="Phase 3 and Phase 4 Operational Router")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Subparser for training (Phase 3)
    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--classifier", default="random_forest")
    train_parser.add_argument("--in-reduced", required=True)
    train_parser.add_argument("--target", required=True)
    train_parser.add_argument("--out-model", required=True)
    train_parser.add_argument("--out-metrics", required=True)
    train_parser.add_argument("--seed", type=int, default=42)

    # Subparser for prediction (Phase 4)
    predict_parser = subparsers.add_parser("predict")
    predict_parser.add_argument("--input-testset", required=True)
    predict_parser.add_argument("--target", required=True)
    predict_parser.add_argument("--model", required=True)
    predict_parser.add_argument("--out-predictions", required=True)
    predict_parser.add_argument("--out-stats", required=True)

    args = parser.parse_args()

    if args.command == "train":
        train(
            classifier=args.classifier,
            reducedTrain_csv=args.in_reduced,
            target_column=args.target,
            model_path=args.out_model,
            metrics_json=args.out_metrics,
            seed=args.seed
        )
    elif args.command == "predict":
        predict(
            reduced_Test_csv=args.input_testset,
            target_column=args.target,
            model_path=args.model,
            predictions_csv=args.out_predictions,
            classif_stats_json=args.out_stats
        )


if __name__ == "__main__":
    main()