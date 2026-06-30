import os
import sys
import json
import streamlit as st

# Automatically handle directory paths to find internal pipeline modules smoothly
current_dir = os.path.dirname(os.path.abspath(__file__))  # src/qubo_project
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))  # src
root_dir = os.path.abspath(os.path.join(parent_dir, ".."))  # project root

if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

# Importing using the exact filenames present in your qubo_project folder
try:
    from qubo_project.LLM_G13_06_17_002_preprocessing import fit_normalize
    from qubo_project.LLM_G13_06_19_001_feature_selection import select_features
    from qubo_project.LLM_G13_06_29_001_model import train, predict
except ImportError as e:
    st.error(f"Critical Error loading pipeline modules: {e}")

# Page configuration
st.set_page_config(page_title="QUBO Project Pipeline", page_icon="📊", layout="wide")

st.title("Binary classification with QUBO feature reduction")

# Navigation tabs mapping exactly to the 4 pipeline phases
tab1, tab2, tab3, tab4 = st.tabs([
    "1. Preprocessing",
    "2. Feature Selection",
    "3. Model Training",
    "4. Test Prediction"
])

# Global state setup for user logging feedback
if "log_history" not in st.session_state:
    st.session_state.log_history = ["[SYSTEM] App started. Ready to execute the QUBO pipeline."]


def log(message):
    st.session_state.log_history.append(message)


# Ensure folders exist so listdir doesn't crash
os.makedirs("data", exist_ok=True)
os.makedirs("outputs", exist_ok=True)


# Helper function to list CSV files in a specific directory
def get_csv_files(directory):
    files = [f for f in os.listdir(directory) if f.endswith('.csv')]
    return files if files else ["No CSV files found"]


# ==========================================
# TAB 1: PREPROCESSING
# ==========================================
with tab1:
    st.header("Step 1: Preprocessing")

    data_files = get_csv_files("data")
    selected_data_file = st.selectbox("Select Input Dataset from 'data/' folder:", data_files)
    input_file = os.path.join("data", selected_data_file) if selected_data_file != "No CSV files found" else ""

    target_column = st.text_input("Target Column Name:", value="target", key="target_p")
    min_perc_valid = st.number_input("Validity Threshold (minPercValid):", min_value=0.0, max_value=1.0, value=0.05,
                                     step=0.01)

    if st.button("Run Preprocessing Step", key="btn_p"):
        if not input_file or selected_data_file == "No CSV files found":
            st.error("Please add a valid CSV dataset inside the 'data/' folder first.")
        else:
            try:
                log("[INFO] Starting preprocessing step...")
                fit_normalize(
                    input_csv=input_file,
                    target_column=target_column,
                    normalized_csv="outputs/normalized.csv",
                    outInitalRes_json="outputs/preprocessing_result.json",
                    minPercValid=float(min_perc_valid)
                )
                log("[SUCCESS] Preprocessing completed! Output saved to outputs/normalized.csv")
                st.success("Preprocessing completed successfully!")
            except Exception as e:
                log(f"[ERROR] Preprocessing failed: {str(e)}")
                st.error(f"Execution Error: {str(e)}")

# ==========================================
# TAB 2: FEATURE SELECTION
# ==========================================
with tab2:
    st.header("Step 2: QUBO Feature Selection")

    output_files_tab2 = get_csv_files("outputs")
    default_idx_tab2 = output_files_tab2.index("normalized.csv") if "normalized.csv" in output_files_tab2 else 0
    selected_norm_file = st.selectbox("Select Normalized Dataset from 'outputs/' folder:", output_files_tab2,
                                      index=default_idx_tab2)
    norm_file = os.path.join("outputs", selected_norm_file) if selected_norm_file != "No CSV files found" else ""

    perc_selected = st.number_input("Feature Selection Ratio (percSelected):", min_value=0.01, max_value=1.0,
                                    value=0.20, step=0.05)
    allowance = st.number_input("Constraint Tolerance (allowance):", min_value=0, value=1, step=1)

    if st.button("Run QUBO Feature Selection", key="btn_f"):
        if not norm_file or selected_norm_file == "No CSV files found":
            st.error("Normalized dataset not found. Please run the Preprocessing step first.")
        else:
            try:
                log("[INFO] Running QUBO matrix formulation and selection optimization...")
                select_features(
                    normalized_csv=norm_file,
                    reducedTrain_csv="outputs/training_reduced.csv",
                    reducedTest_csv="outputs/test_reduced.csv",
                    output_ottim_csv="outputs/optimizations.csv",
                    output_json="outputs/feature_selection_result.json",
                    target_column=target_column,
                    percSelected=float(perc_selected),
                    allowance=int(allowance),
                    seed=42
                )
                log("[SUCCESS] Feature selection complete. Reduced datasets saved in outputs/")
                st.success("QUBO Feature Selection finished successfully!")
            except Exception as e:
                log(f"[ERROR] Feature Selection failed: {str(e)}")
                st.error(f"Execution Error: {str(e)}")

# ==========================================
# TAB 3: TRAINING
# ==========================================
with tab3:
    st.header("Step 3: Model Training")

    classifier_choice = st.selectbox("Classifier Architecture:",
                                     ["random_forest", "logistic_regression", "decision_tree"])

    output_files_tab3 = get_csv_files("outputs")
    default_idx_tab3 = output_files_tab3.index(
        "training_reduced.csv") if "training_reduced.csv" in output_files_tab3 else 0
    selected_train_file = st.selectbox("Select Reduced Training Dataset from 'outputs/' folder:", output_files_tab3,
                                       index=default_idx_tab3)
    train_reduced_file = os.path.join("outputs",
                                      selected_train_file) if selected_train_file != "No CSV files found" else ""

    if st.button("Train Selected Model", key="btn_t"):
        if not train_reduced_file or selected_train_file == "No CSV files found":
            st.error("Reduced training file not found. Please run Feature Selection first.")
        else:
            try:
                log(f"[INFO] Training {classifier_choice} model...")
                train(
                    classifier=classifier_choice,
                    reducedTrain_csv=train_reduced_file,
                    target_column=target_column,
                    model_path="outputs/model.joblib",
                    metrics_json="outputs/training_metrics.json",
                    seed=42
                )
                log("[SUCCESS] Model trained and saved to outputs/model.joblib")
                st.success("Model training successfully completed!")
            except Exception as e:
                log(f"[ERROR] Model training failed: {str(e)}")
                st.error(f"Execution Error: {str(e)}")

# ==========================================
# TAB 4: PREDICTION
# ==========================================
with tab4:
    st.header("Step 4: Test Dataset Prediction")

    output_files_tab4 = get_csv_files("outputs")
    default_idx_tab4 = output_files_tab4.index("test_reduced.csv") if "test_reduced.csv" in output_files_tab4 else 0
    selected_test_file = st.selectbox("Select Reduced Test Dataset from 'outputs/' folder:", output_files_tab4,
                                      index=default_idx_tab4)
    test_reduced_file = os.path.join("outputs",
                                     selected_test_file) if selected_test_file != "No CSV files found" else ""

    if st.button("Generate Test Predictions", key="btn_pred"):
        if not test_reduced_file or selected_test_file == "No CSV files found" or not os.path.exists(
                "outputs/model.joblib"):
            st.error("Missing test data or trained model file in outputs/ directory.")
        else:
            try:
                log("[INFO] Generating predictions on the test dataset...")
                predict(
                    reduced_Test_csv=test_reduced_file,
                    target_column=target_column,
                    model_path="outputs/model.joblib",
                    predictions_csv="outputs/predictions.csv",
                    classif_stats_json="outputs/classification_stats.json"
                )
                log("[SUCCESS] Inference complete. Mapped outputs saved to outputs/predictions.csv")
                st.success("Predictions generated successfully!")
            except Exception as e:
                log(f"[ERROR] Inference step failed: {str(e)}")
                st.error(f"Execution Error: {str(e)}")

# ==========================================
# REQUIREMENT #6: DISPLAY / VIEW MAIN OUTPUTS
# ==========================================
st.markdown("---")

# Compact Metrics Inspector using Expanders
with st.expander("Outputs"):
    c1, c2, c3 = st.columns(3)
    with c1:
        if os.path.exists("outputs/preprocessing_result.json"):
            with open("outputs/preprocessing_result.json", "r") as f:
                st.json({"Preprocessing": json.load(f)})
    with c2:
        if os.path.exists("outputs/training_metrics.json"):
            with open("outputs/training_metrics.json", "r") as f:
                st.json({"Training": json.load(f)})
    with c3:
        if os.path.exists("outputs/classification_stats.json"):
            with open("outputs/classification_stats.json", "r") as f:
                st.json({"Prediction": json.load(f)})

# Compact System Logs
with st.expander("View Activity Log"):
    st.text_area("System Logs", value="\n".join(st.session_state.log_history), height=100, label_visibility="collapsed")