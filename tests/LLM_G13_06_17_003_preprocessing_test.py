import os
import json
import pandas as pd
import numpy as np

# Standardized project paths matching Section 12 outputs structure
ORIG = "data/input_dataset.csv"
NORMALIZED = "outputs/normalized.csv"
JSON_STATS = "outputs/preprocessing_result.json"


def test_output_files_exist():
    """Verify that all required output files have been successfully generated."""
    assert os.path.exists(NORMALIZED), "Normalized CSV file not found in outputs directory"
    assert os.path.exists(JSON_STATS), "JSON metrics summary file not found in outputs directory"


def test_json_structure_and_keys():
    """Verify that the summary metrics structure matches the expected evaluation format keys."""
    with open(JSON_STATS, "r") as f:
        data = json.load(f)

    mandatory_keys = [
        "n_input_features", "n_kept_features", "dataset_size",
        "dataset_input_time", "dataset_processing_time", "dropped_feature_names"
    ]
    for key in mandatory_keys:
        assert key in data, f"Mandatory key '{key}' is missing from the JSON metrics file"


def test_target_preserved_exactly():
    """Verify the positional integrity and sequential alignment of the target column."""
    orig_df = pd.read_csv(ORIG)
    norm_df = pd.read_csv(NORMALIZED)

    # Target column must be explicitly positioned as the very last column
    assert norm_df.columns[-1] == "target", "Target column is not located in the final position"

    # Ensure that target values sequence remains unshuffled
    pd.testing.assert_series_equal(
        orig_df["target"].reset_index(drop=True),
        norm_df["target"].reset_index(drop=True),
        check_names=False,
        obj="Target row ordering or content values have been corrupted"
    )


def test_normalization_properties():
    """Verify that z-score constraints are met globally (mean ~0, std ~1)."""
    norm_df = pd.read_csv(NORMALIZED)
    features_df = norm_df.drop(columns=["target"])

    for col in features_df.columns:
        mean = features_df[col].mean()
        std = features_df[col].std(ddof=0)

        assert abs(mean) < 1e-2, f"Column '{col}' mean deviates significantly from 0 (mean={mean})"
        # If the column is not constant, check unit variance convergence
        if std > 1e-4:
            assert abs(std - 1.0) < 1e-2, f"Column '{col}' standard deviation deviates from 1 (std={std})"