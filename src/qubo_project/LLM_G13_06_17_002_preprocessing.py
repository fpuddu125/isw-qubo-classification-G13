#!/usr/bin/env python3
"""
LLM_G13_06_17_002_preprocessing.py
Fase 1 preprocessing for ISW QUBO classification project.

Usage example (from repo root):
python src/qubo_project/LLM_G13_06_17_002_preprocessing.py \
  --input_csv data/input_dataset.csv \
  --target_col target \
  --zero_threshold 0.95 \
  --test_percentage 0.2 \
  --output_train outputs/preprocessed_train.csv \
  --output_test outputs/preprocessed_test.csv \
  --chunksize 100000 \
  --verbose
"""

import argparse
import os
import sys
import math
import pandas as pd
import numpy as np

def parse_args():
    p = argparse.ArgumentParser(description="ISW Project Phase 1 preprocessing (streaming)")
    p.add_argument("--input_csv", required=True, help="Path to input CSV (full dataset)")
    p.add_argument("--target_col", required=True, help="Name of the target column (0/1)")
    p.add_argument("--zero_threshold", type=float, default=0.95,
                   help="Remove columns with fraction of (NaN or zero) > this threshold (default 0.95)")
    p.add_argument("--test_percentage", type=float, default=0.2,
                   help="Fraction of dataset to reserve for test (e.g., 0.2). Must be in [0,1).")
    p.add_argument("--output_train", default="outputs/preprocessed_train.csv", help="Output train CSV path")
    p.add_argument("--output_test", default="outputs/preprocessed_test.csv", help="Output test CSV path")
    p.add_argument("--chunksize", type=int, default=100000, help="Rows per chunk for streaming (default 100000)")
    p.add_argument("--verbose", action="store_true", help="Verbose logging")
    return p.parse_args()

def ensure_output_dirs(paths):
    for p in paths:
        d = os.path.dirname(p)
        if d and not os.path.exists(d):
            os.makedirs(d, exist_ok=True)

def first_pass_stats(input_csv, target_col, chunksize, verbose):
    """
    First pass: determine total rows, per-column counts of NaN and zeros,
    and detect columns that cannot be coerced to numeric.
    Returns:
      total_rows, columns (list), nan_zero_counts (dict col->count), non_numeric_cols (set)
    """
    reader = pd.read_csv(input_csv, chunksize=chunksize, low_memory=False)
    total_rows = 0
    nan_zero_counts = None
    non_numeric_cols = set()
    columns = None

    for i, chunk in enumerate(reader):
        if columns is None:
            columns = list(chunk.columns)
            if target_col not in columns:
                raise ValueError(f"Target column '{target_col}' not found in CSV columns: {columns}")
            # initialize counts dict
            nan_zero_counts = {c: 0 for c in columns if c != target_col}

        # count rows
        n_chunk = len(chunk)
        total_rows += n_chunk

        # For feature columns only (exclude target)
        features = chunk.drop(columns=[target_col])

        # Try to coerce to numeric; mark columns that become all NaN after coercion as non-numeric candidates
        coerced = features.apply(pd.to_numeric, errors='coerce')

        # Count NaN or exactly zero per column
        for col in coerced.columns:
            # NaN count
            nan_count = coerced[col].isna().sum()
            # zero count (exact zero)
            zero_count = (coerced[col] == 0).sum()
            nan_zero_counts[col] += int(nan_count + zero_count)

            # If entire column is non-numeric in this chunk (all NaN after coercion), mark it as non-numeric candidate
            # We'll decide final non-numeric after scanning all chunks
            if coerced[col].notna().sum() == 0:
                non_numeric_cols.add(col)
            else:
                # if column had numeric values in this chunk, ensure it's not considered non-numeric
                if col in non_numeric_cols:
                    non_numeric_cols.discard(col)

        if verbose and (i % 10 == 0):
            print(f"[first_pass] processed chunks: {i+1}, total_rows so far: {total_rows}")

    return total_rows, columns, nan_zero_counts, non_numeric_cols

def decide_columns_to_keep(columns, target_col, nan_zero_counts, total_rows, zero_threshold, non_numeric_cols, verbose):
    """
    Decide which columns to drop based on threshold and non-numeric detection.
    Returns list of kept feature columns (numeric), and list of dropped columns.
    """
    dropped = []
    kept = []
    for c in columns:
        if c == target_col:
            continue
        frac = nan_zero_counts.get(c, 0) / total_rows
        if frac > zero_threshold:
            dropped.append(c)
            if verbose:
                print(f"[drop] column {c} dropped for sparsity: {nan_zero_counts.get(c,0)}/{total_rows} ({frac:.4f}) > {zero_threshold}")
        elif c in non_numeric_cols:
            dropped.append(c)
            if verbose:
                print(f"[drop] column {c} dropped for non-numeric values")
        else:
            kept.append(c)
    return kept, dropped

def compute_mean_std(input_csv, target_col, kept_cols, chunksize, verbose):
    """
    Two-pass streaming computation of mean and std for kept_cols using sums and sums of squares.
    Returns dicts: mean[col], std[col]
    """
    sums = {c: 0.0 for c in kept_cols}
    sumsqs = {c: 0.0 for c in kept_cols}
    counts = {c: 0 for c in kept_cols}
    reader = pd.read_csv(input_csv, usecols=kept_cols + [target_col], chunksize=chunksize, low_memory=False)

    for i, chunk in enumerate(reader):
        features = chunk[kept_cols].apply(pd.to_numeric, errors='coerce')
        # For each column, consider only non-NaN values in sums/counts
        for c in kept_cols:
            col = features[c]
            valid = col.dropna()
            if len(valid) == 0:
                continue
            vals = valid.values.astype(float)
            sums[c] += vals.sum()
            sumsqs[c] += (vals * vals).sum()
            counts[c] += len(vals)
        if verbose and (i % 10 == 0):
            print(f"[meanstd] processed chunks: {i+1}")

    means = {}
    stds = {}
    for c in kept_cols:
        n = counts[c]
        if n == 0:
            means[c] = 0.0
            stds[c] = 1.0
            if verbose:
                print(f"[meanstd] column {c} has no valid values; default mean=0 std=1")
            continue
        mean = sums[c] / n
        # variance = (sum(x^2)/n) - mean^2 ; use sample std? spec asks unit std (population or sample not specified) — use population std (denominator n)
        var = (sumsqs[c] / n) - (mean * mean)
        var = max(var, 0.0)
        if var == 0:
            std = 0.0
        else:
            std = math.sqrt(var)

        means[c] = mean
        stds[c] = std
    return means, stds

def transform_and_split(input_csv, target_col, kept_cols, means, stds, total_rows, test_percentage,
                        output_train, output_test, chunksize, verbose):
    """
    Third pass: read in chunks, normalize kept_cols using means/stds, reattach target, and write rows to train/test CSVs
    using a clean cut at M = floor(n*(1-test_percentage)).
    """
    M = int(math.floor(total_rows * (1.0 - test_percentage)))
    if verbose:
        print(f"[split] total_rows={total_rows}, test_percentage={test_percentage}, M(train size)={M}")

    # Prepare output files (write header on first write)
    train_written = False
    test_written = False

    reader = pd.read_csv(input_csv, usecols=kept_cols + [target_col], chunksize=chunksize, low_memory=False)
    row_offset = 0
    for i, chunk in enumerate(reader):
        n_chunk = len(chunk)
        # Normalize features
        features = chunk[kept_cols].apply(pd.to_numeric, errors='coerce')
        for c in kept_cols:
            features[c] = (features[c] - means[c]) / stds[c]
        # Reattach target (keep as-is)
        out_chunk = pd.concat([features, chunk[[target_col]].reset_index(drop=True)], axis=1)

        # Determine which rows go to train/test
        start = row_offset
        end = row_offset + n_chunk  # exclusive
        # train rows in this chunk: indices where idx < M
        train_mask = (np.arange(start, end) < M)
        if train_mask.any():
            train_rows = out_chunk.iloc[train_mask.nonzero()[0]].reset_index(drop=True)
            if not train_written:
                train_rows.to_csv(output_train, index=False, mode='w')
                train_written = True
            else:
                train_rows.to_csv(output_train, index=False, mode='a', header=False)
        # test rows
        test_mask = (np.arange(start, end) >= M)
        if test_mask.any():
            test_rows = out_chunk.iloc[test_mask.nonzero()[0]].reset_index(drop=True)
            if not test_written:
                test_rows.to_csv(output_test, index=False, mode='w')
                test_written = True
            else:
                test_rows.to_csv(output_test, index=False, mode='a', header=False)

        row_offset += n_chunk
        if verbose and (i % 10 == 0):
            print(f"[transform] processed chunks: {i+1}, rows processed: {row_offset}/{total_rows}")

    if verbose:
        print(f"[done] Written train rows: {min(M, row_offset)}; test rows: {max(0, row_offset - M)}")

def main():
    args = parse_args()
    if not (0.0 <= args.test_percentage < 1.0):
        raise ValueError("test_percentage must be in [0.0, 1.0)")

    ensure_output_dirs([args.output_train, args.output_test])

    if args.verbose:
        print("Starting first pass to collect sparsity and row count...")

    total_rows, columns, nan_zero_counts, non_numeric_cols = first_pass_stats(
        args.input_csv, args.target_col, args.chunksize, args.verbose)

    if args.verbose:
        print(f"Total rows detected: {total_rows}")
        print(f"Detected non-numeric columns (candidates): {sorted(list(non_numeric_cols))}")

    kept_cols, dropped_cols = decide_columns_to_keep(
        columns, args.target_col, nan_zero_counts, total_rows, args.zero_threshold, non_numeric_cols, args.verbose)

    if args.verbose:
        print(f"Kept numeric feature columns: {len(kept_cols)}")
        print(f"Dropped columns: {dropped_cols}")

    if len(kept_cols) == 0:
        print("Warning: no numeric features remain after cleaning. Exiting.")
        sys.exit(1)

    if args.verbose:
        print("Computing means and stds (streaming)...")

    means, stds = compute_mean_std(args.input_csv, args.target_col, kept_cols, args.chunksize, args.verbose)

    zero_std_cols = [c for c in kept_cols if stds[c] == 0.0]

    if zero_std_cols:
        if args.verbose:
            print(f"[drop] Removing constant columns (std=0): {zero_std_cols}")

        kept_cols = [c for c in kept_cols if c not in zero_std_cols]

        means, stds = compute_mean_std(
            args.input_csv, args.target_col, kept_cols, args.chunksize, args.verbose
        )

    if args.verbose:
        print("Applying normalization and splitting dataset (streaming)...")

    transform_and_split(args.input_csv, args.target_col, kept_cols, means, stds, total_rows,
                        args.test_percentage, args.output_train, args.output_test, args.chunksize, args.verbose)

    if args.verbose:
        print("Preprocessing completed successfully.")

if __name__ == "__main__":
    main()
