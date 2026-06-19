import os
import json
import math
import pandas as pd
import numpy as np

# Standard paths configuration aligned with new outputs layout
INPUT_NORM = "outputs/normalized.csv"
OUT_TRAIN = "outputs/training_reduced.csv"
OUT_TEST = "outputs/test_reduced.csv"
OUT_LOG = "outputs/optimizations.csv"
OUT_SUMMARY = "outputs/feature_selection_result.json"


def test_output_files_exist():
    """Verify that all required output targets have been successfully generated."""
    assert os.path.exists(OUT_TRAIN), "Reduced training CSV file not found"
    assert os.path.exists(OUT_TEST), "Reduced test CSV file not found"
    assert os.path.exists(OUT_LOG), "Optimization steps historical CSV log file not found"
    assert os.path.exists(OUT_SUMMARY), "Summary JSON statistics file not found"


def test_feature_reduction_proportions():
    """Verify that the number of selected features matches the target threshold within the allowed tolerance."""
    orig_norm = pd.read_csv(INPUT_NORM)
    reduced_train = pd.read_csv(OUT_TRAIN)

    m_orig = len(orig_norm.columns) - 1
    target_k = int(round(0.20 * m_orig))
    k_effective = len(reduced_train.columns) - 1

    assert abs(k_effective - target_k) <= 1, \
        f"Selected features count ({k_effective}) is out of tolerance boundaries for target K ({target_k})"


def test_row_preservation_and_split():
    """Verify sequential 70/30 row splitting and check total row count preservation."""
    orig_norm = pd.read_csv(INPUT_NORM)
    reduced_train = pd.read_csv(OUT_TRAIN)
    reduced_test = pd.read_csv(OUT_TEST)

    total_rows_orig = len(orig_norm)
    assert len(reduced_train) + len(reduced_test) == total_rows_orig, \
        "Total number of rows after splitting does not match the original input dataset layout"

    expected_train_size = int(math.floor(total_rows_orig * 0.70))
    assert len(reduced_train) == expected_train_size, \
        f"Reduced train dataset size ({len(reduced_train)}) does not follow the expected 70% threshold ({expected_train_size})"


def test_target_column_position_and_values():
    """Verify that the target column resides in the last column position and its sequence is unaltered."""
    orig_norm = pd.read_csv(INPUT_NORM)["target"].reset_index(drop=True)
    reduced_train = pd.read_csv(OUT_TRAIN)
    reduced_test = pd.read_csv(OUT_TEST)

    assert reduced_train.columns[-1] == "target", "Target column is not positioned last in the reduced train file"
    assert reduced_test.columns[-1] == "target", "Target column is not positioned last in the reduced test file"

    combined_target = pd.concat([reduced_train["target"], reduced_test["target"]], ignore_index=True)
    pd.testing.assert_series_equal(
        orig_norm, combined_target, check_names=False,
        obj="The sequential order or values of the target column have been altered"
    )


def test_json_metrics_structure():
    """Verify that all mandatory summary metrics from the screenshot example are present."""
    with open(OUT_SUMMARY, "r") as f:
        summary = json.load(f)

    mandatory_keys = [
        "n_features", "target_ratio", "target_k", "allowance", "n_selected",
        "alpha", "selected_vector", "selected_feature_names", "algorithm",
        "seed", "alpha_computations", "percTest", "training_dataset_size",
        "test_dataset_size", "q_matrix_creation_time", "mean_optimization_time",
        "std_dev_optimization_time"
    ]

    for key in mandatory_keys:
        assert key in summary, f"Mandatory reporting field '{key}' from specifications is missing from the JSON file"