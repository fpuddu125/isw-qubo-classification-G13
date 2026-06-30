# README

# Binary classification with QUBO feature reduction

This project is about developing an application to classify data records from a dataset. 
It uses binary classification, which means assigning each data sample to one of two categories, such as "low risk" versus "high risk."

## Project Structure

```text
isw-qubo-classification-G13/
├── data/                      
├── outputs/                   
├── src/
│   └── qubo_project/
│       ├── __init__.py
│       ├── gui.py             
│       ├── LLM_G13_06_17_002_preprocessing.py
│       ├── LLM_G13_06_19_001_feature_selection.py
│       └── LLM_G13_06_29_001_model.py
└── README.md
```

## Phase 1: Preprocessing
```text
This step loads the input dataset, separates 
the target column, and removes columns with too 
many missing or zero values based on a set 
threshold. It then applies a z-score 
normalization to all remaining features and 
splits the data into training and test sets using 
a clean cut at a specific percentage.
```

## Phase 2: Feature Reduction via QUBO
```text
This phase converts feature selection into a QUBO
optimization problem to pick a target percentage
of features within a given tolerance. It 
searches for the best parameter α, runs an 
optimization algorithm to generate a binary 
selection vector, and outputs a reduced training 
dataset saved as a .csv file.
```

## Phase 3: Classifier Learning
```text
This phase implements three binary classification
algorithms (including Random Forest) and trains
the user-selected model using only the 
QUBO-reduced features. The trained model is 
then saved as a `.joblib` file alongside key
training statistics such as execution time.
```

## Phase 4: Test Data Classification
```text
This final phase applies the trained classifier 
to the test dataset using the exact same 
QUBO-selected features. It generates a `.csv` 
file containing the predictions and prediction 
scores for each record, while saving overall 
performance evaluation statistics like Precision, 
Recall, F1-score, ROC-AUC, and the confusion 
matrix.
```

## GUI
```text
This phase develops an interactive Python web 
application using **Streamlit** to act as a 
central control panel. It allows users to easily 
choose datasets, trigger each core pipeline step 
(Preprocessing, QUBO Selection, Training, and 
Prediction), catch input errors gracefully, and 
display or save the main output metrics directly 
on screen.
```

## How to run the application
These are the commands to write in the terminal to perform the various phases

Preprocessing:
```text
python3 src/qubo_project/LLM_G13_06_17_002_preprocessing.py \
  --input data/sample_test_dataset.csv \
  --target target \
  --out-data outputs/normalized.csv \
  --out-json outputs/preprocessing_result.json \
  --min-perc-valid 0.06
```

Feature Reduction via QUBO:
```text
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

Classifier Learning:
```text
python3 src/qubo_project/LLM_G13_06_29_001_model.py train \
  --classifier random_forest \
  --in-reduced outputs/training_reduced.csv \
  --target target \
  --out-model outputs/model.joblib \
  --out-metrics outputs/training_metrics.json \
  --seed 42
```

Test Data Classification:
```text
python3 src/qubo_project/LLM_G13_06_29_001_model.py predict \
  --input-testset outputs/test_reduced.csv \
  --target target \
  --model outputs/model.joblib \
  --out-predictions outputs/predictions.csv \
  --out-stats outputs/classification_stats.json
```

GUI:
```text
streamlit run src/qubo_project/gui.py 
```

Automatic test:
```text
streamlit run src/qubo_project/!!!DA MODIFICAREgui.py
```