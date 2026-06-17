import os
import pandas as pd
import numpy as np

# Percorsi standard del progetto
ORIG = "data/input_dataset.csv"
TRAIN = "outputs/preprocessed_train.csv"
TEST = "outputs/preprocessed_test.csv"


def test_output_files_exist():
    """Verifica che i file preprocessati siano stati generati."""
    assert os.path.exists(TRAIN), "Train CSV non trovato"
    assert os.path.exists(TEST), "Test CSV non trovato"


def test_row_counts_and_no_loss():
    """Verifica che non ci sia perdita di righe."""
    orig = pd.read_csv(ORIG)
    train = pd.read_csv(TRAIN)
    test = pd.read_csv(TEST)

    assert len(train) + len(test) == len(orig), \
        "Il numero di righe train+test non coincide con l'originale"


def test_target_preserved_and_ordered():
    """Verifica che la colonna target sia preservata e nello stesso ordine."""
    orig = pd.read_csv(ORIG)["target"].reset_index(drop=True)
    train = pd.read_csv(TRAIN)["target"]
    test = pd.read_csv(TEST)["target"]

    out = pd.concat([train, test], ignore_index=True)

    pd.testing.assert_series_equal(
        orig, out, check_names=False,
        obj="La colonna target non è preservata correttamente"
    )


def test_normalization_mean_std():
    """Verifica che le feature normalizzate abbiano media ~0 e std ~1."""
    train = pd.read_csv(TRAIN)
    test = pd.read_csv(TEST)

    # Unisco train e test per verificare la normalizzazione globale
    combined = pd.concat(
        [train.drop(columns=["target"]), test.drop(columns=["target"])],
        ignore_index=True
    )

    for col in combined.columns:
        mean = combined[col].mean()
        std = combined[col].std(ddof=0)  # std popolazione

        assert abs(mean) < 1e-2, f"Mean non ~0 per colonna {col}: {mean}"
        assert abs(std - 1) < 1e-2, f"Std non ~1 per colonna {col}: {std}"
