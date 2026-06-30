import os
import json
import pandas as pd
import numpy as np


def test_training_outputs():
    """Verifies the correct generation of the serialized model and training metrics."""
    assert os.path.exists("outputs/model.joblib"), "Model .joblib file not found in the expected path!"
    assert os.path.exists("outputs/training_metrics.json"), "Missing training_metrics.json file!"

    with open("outputs/training_metrics.json", "r") as f:
        metrics = json.load(f)

    assert "classifier" in metrics
    assert "training_time" in metrics
    assert "n_features" in metrics


def test_phase4_prediction_dataframe_structure():
    """Strictly verifies Phase 4 CSV column layout and constraints (row_n, target, prediction, score)."""
    predictions_path = "outputs/predictions.csv"
    assert os.path.exists(predictions_path), "Missing Phase 4 predictions.csv file!"

    df = pd.read_csv(predictions_path)

    # Mandatory columns required by Phase 4 specification
    expected_columns = ["row_n", "target", "prediction", "score"]
    assert list(df.columns) == expected_columns, f"CSV columns do not strictly match Phase 4 layout: {df.columns}"

    # Assert correct types and values range
    assert df["target"].isin([0, 1]).all(), "Target column contains invalid non-binary values!"
    assert df["prediction"].isin([0, 1]).all(), "Prediction column contains invalid non-binary values!"
    assert df["score"].between(0.0, 1.0).all(), "Score probability values are out of bounds [0, 1]!"


def test_phase4_classification_stats_json():
    """Verifies the structural integrity of the classification statistics reporting."""
    stats_path = "outputs/classification_stats.json"
    assert os.path.exists(stats_path), "Missing classification_stats.json file!"

    with open(stats_path, "r") as f:
        stats = json.load(f)

    # Required statistical keys from Section 11.3 / Phase 4
    required_keys = ["classifier", "n_samples", "target_1_percentage", "accuracy", "class_0", "class_1", "roc_auc",
                     "confusion_matrix"]
    for key in required_keys:
        assert key in stats, f"Mandatory tracking key '{key}' is missing from classification_stats.json!"

    # Deep verification of nested structures
    for class_key in ["class_0", "class_1"]:
        for metric in ["precision", "recall", "f1", "support"]:
            assert metric in stats[class_key], f"Metric '{metric}' missing inside '{class_key}' configuration!"

    assert "matrix" in stats["confusion_matrix"], "Missing standard confusion matrix array!"