# README

Codice da eseguire per il preprocessing:
```
python3 src/qubo_project/LLM_G13_06_17_002_preprocessing.py \
  --input_csv data/input_dataset.csv \
  --target_col target \
  --zero_threshold 0.95 \
  --test_percentage 0.2 \
  --output_train outputs/preprocessed_train.csv \
  --output_test outputs/preprocessed_test.csv \
  --chunksize 100000 \
  --verbos
```

Codice per eseguire test automatici:
```
pytest -q
```