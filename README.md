# README

Codice da eseguire per il preprocessing:
```
python3 src/qubo_project/LLM_G13_06_17_002_preprocessing.py \
  --input data/input_dataset.csv \
  --target target \
  --out-data outputs/normalized.csv \
  --out-json outputs/preprocessing_result.json \
  --min-perc-valid 0.05
```

Codice da eseguire per il feature selection:
```
python3 src/qubo_project/LLM_G13_06_19_001_feature_selection.py \
  --in-normalized outputs/normalized.csv \
  --out-train outputs/training_reduced.csv \
  --out-test outputs/test_reduced.csv \
  --out-optimizations outputs/optimizations.csv \
  --out-json outputs/feature_selection_result.json \
  --target target \
  --perc-selected 0.20 \
  --allowance 1 \
  --perc-test 0.30 \
  --seed 42 \
  --alpha-computations 10
```

Codice per eseguire test automatici:
```
pytest -q
```