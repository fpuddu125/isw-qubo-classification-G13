#!/usr/bin/env python3
"""
src/qubo_project/feature_selection.py
Phase 2: Feature reduction via QUBO optimization using Spearman correlation.
Completely aligned with the professor's mandatory function templates and CLI specifications.
"""

import argparse
import os
import sys
import time
import json
import math
import pandas as pd
import numpy as np


def compute_spearman_correlations(df, target_col):
    """
    Computes Spearman absolute correlations between features and target, and among feature pairs.
    """
    features_df = df.drop(columns=[target_col])
    rho_V = features_df.corrwith(df[target_col], method='spearman').abs().fillna(0).values
    rho_features = features_df.corr(method='spearman').abs().fillna(0).values
    return rho_V, rho_features, list(features_df.columns)


def calculate_energy(state, Q):
    """
    Computes the QUBO objective function energy: x^T * Q * x
    """
    return float(state.T @ Q @ state)


def simulated_annealing_qubo(Q, seed=42, steps=1000, T_start=10.0, T_end=0.01):
    """
    Native Simulated Annealing solver to minimize the standard QUBO energy matrix.
    """
    rng = np.random.default_rng(seed)
    n = Q.shape[0]

    current_state = rng.choice([0, 1], size=n, p=[0.8, 0.2])
    current_energy = calculate_energy(current_state, Q)

    best_state = current_state.copy()
    best_energy = current_energy

    T = T_start
    decay = (T_end / T_start) ** (1.0 / steps)

    for step in range(steps):
        idx = rng.integers(0, n)

        next_state = current_state.copy()
        next_state[idx] = 1 - next_state[idx]

        next_energy = calculate_energy(next_state, Q)
        delta_energy = next_energy - current_energy

        if delta_energy < 0 or rng.random() < math.exp(-delta_energy / T):
            current_state = next_state
            current_energy = next_energy

            if current_energy < best_energy:
                best_energy = current_energy
                best_state = current_state.copy()

        T *= decay

    return best_state, best_energy


def select_features(normalized_csv: str, reducedTrain_csv: str, reducedTest_csv: str,
                    output_ottim_csv: str, output_json: str, target_column: str,
                    percTest: float = 0.30, percSelected: float = 0.20, allowance: int = 1,
                    seed: int = 42, alpha_computations: int = 100):
    """
    MANDATORY function interface required by Section 11.2 of the specification rules.
    """
    start_time = time.time()

    # Load data
    df = pd.read_csv(normalized_csv)
    n_total_samples = len(df)

    t0_q = time.time()
    rho_V, rho_features, feature_names = compute_spearman_correlations(df, target_column)
    q_matrix_creation_time = time.time() - t0_q

    m = len(feature_names)
    target_k = int(round(percSelected * m))
    min_k = target_k - allowance
    max_k = target_k + allowance

    optimization_history = []
    optimization_times = []

    alpha_low = 0.0
    alpha_high = 1.0
    best_alpha = 0.5
    best_vector = np.zeros(m, dtype=int)
    found_valid = False
    current_seed = seed

    for comp_idx in range(alpha_computations):
        alpha = (alpha_low + alpha_high) / 2.0

        # Building the QUBO penalty matrix
        Q = np.zeros((m, m))
        np.fill_diagonal(Q, -alpha * rho_V)
        for i in range(m):
            for j in range(m):
                if i != j:
                    Q[i, j] = (1.0 - alpha) * rho_features[i, j]

        t0_opt = time.time()
        x_opt, cost_opt = simulated_annealing_qubo(Q, seed=current_seed, steps=1000)
        opt_time = time.time() - t0_opt

        n_selected = int(np.sum(x_opt))
        optimization_times.append(opt_time)

        optimization_history.append({
            'alpha': alpha,
            'time': opt_time,
            'n_features': n_selected,
            'cost': cost_opt
        })

        if min_k <= n_selected <= max_k:
            best_alpha = alpha
            best_vector = x_opt
            found_valid = True
            break
        elif n_selected < min_k:
            alpha_low = alpha
        else:
            alpha_high = alpha

        current_seed += 1

    if not found_valid and optimization_history:
        closest_opt = min(optimization_history, key=lambda x: abs(x['n_features'] - target_k))
        best_alpha = closest_opt['alpha']

        Q = np.zeros((m, m))
        np.fill_diagonal(Q, -best_alpha * rho_V)
        for i in range(m):
            for j in range(m):
                if i != j:
                    Q[i, j] = (1.0 - best_alpha) * rho_features[i, j]
        best_vector, _ = simulated_annealing_qubo(Q, seed=seed, steps=1000)

    selected_features = [feature_names[i] for i in range(m) if best_vector[i] == 1]
    n_selected_final = len(selected_features)

    # Save output optimization history (sorted by alpha as requested)
    log_df = pd.DataFrame(optimization_history).sort_values(by='alpha')
    os.makedirs(os.path.dirname(os.path.abspath(output_ottim_csv)), exist_ok=True)
    log_df.to_csv(output_ottim_csv, index=False)

    # Dataset split logic (Deterministic sequential split)
    m_train_size = int(math.floor(n_total_samples * (1.0 - percTest)))
    cols_to_keep = selected_features + [target_column]
    reduced_df = df[cols_to_keep]

    train_reduced = reduced_df.iloc[:m_train_size].reset_index(drop=True)
    test_reduced = reduced_df.iloc[m_train_size:].reset_index(drop=True)

    os.makedirs(os.path.dirname(os.path.abspath(reducedTrain_csv)), exist_ok=True)
    train_reduced.to_csv(reducedTrain_csv, index=False)

    os.makedirs(os.path.dirname(os.path.abspath(reducedTest_csv)), exist_ok=True)
    test_reduced.to_csv(reducedTest_csv, index=False)

    # JSON output matching exact names from screenshot examples
    mean_opt_time = float(np.mean(optimization_times)) if optimization_times else 0.0
    std_opt_time = float(np.std(optimization_times)) if optimization_times else 0.0

    summary_data = {
        "n_features": m,
        "target_ratio": float(percSelected),
        "target_k": target_k,
        "allowance": allowance,
        "n_selected": n_selected_final,
        "alpha": float(best_alpha),
        "selected_vector": best_vector.tolist(),
        "selected_feature_names": selected_features,
        "algorithm": "simulated_annealing",
        "seed": seed,
        "alpha_computations": len(optimization_history),
        "percTest": percTest,
        "training_dataset_size": len(train_reduced),
        "test_dataset_size": len(test_reduced),
        "q_matrix_creation_time": float(q_matrix_creation_time),
        "mean_optimization_time": mean_opt_time,
        "std_dev_optimization_time": std_opt_time
    }

    os.makedirs(os.path.dirname(os.path.abspath(output_json)), exist_ok=True)
    with open(output_json, 'w') as f:
        json.dump(summary_data, f, indent=4)


def main():
    """
    Mandatory CLI parsing mapped strictly to screenshot parameters.
    """
    p = argparse.ArgumentParser(description="Mandatory CLI format for Phase 2")
    p.add_argument("--in-normalized", required=True, help="Path to input normalized CSV")
    p.add_argument("--out-train", required=True, help="Path to output reduced train CSV")
    p.add_argument("--out-test", required=True, help="Path to output reduced test CSV")
    p.add_argument("--out-optimizations", required=True, help="Path to output log CSV")
    p.add_argument("--out-json", required=True, help="Path to output metrics JSON")
    p.add_argument("--target", required=True, help="Name of the target column")
    p.add_argument("--perc-selected", type=float, default=0.20, help="Percentage of features to select")
    p.add_argument("--allowance", type=int, default=1, help="Allowance threshold")
    p.add_argument("--perc-test", type=float, default=0.30, help="Test set ratio percentage")
    p.add_argument("--seed", type=int, default=42, help="Random seed repeatability")
    p.add_argument("--alpha-computations", type=int, default=10, help="Max alpha tuning loops")

    args = p.parse_args()

    select_features(
        normalized_csv=args.in_normalized,
        reducedTrain_csv=args.out_train,
        reducedTest_csv=args.out_test,
        output_ottim_csv=args.out_optimizations,
        output_json=args.out_json,
        target_column=args.target,
        percTest=args.perc_test,
        percSelected=args.perc_selected,
        allowance=args.allowance,
        seed=args.seed,
        alpha_computations=args.alpha_computations
    )

if __name__ == "__main__":
    main()