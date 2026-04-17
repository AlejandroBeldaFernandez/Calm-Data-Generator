"""
Tutorial 10: Advanced Drift Scenarios
=====================================

This tutorial demonstrates advanced usage of the DriftInjector to simulate
complex data drift scenarios, including conditional drift, gradient drift,
and correlation matrix changes.
"""

import pandas as pd
import numpy as np
import os
import shutil
from calm_data_generator.generators.drift import DriftInjector

# Setup output directory
OUTPUT_DIR = "tutorial_drifts_output"
if os.path.exists(OUTPUT_DIR):
    shutil.rmtree(OUTPUT_DIR)
os.makedirs(OUTPUT_DIR)

# ============================================================
# 1. Prepare Baseline Data
# ============================================================

np.random.seed(100)
n_samples = 1000
data = pd.DataFrame(
    {
        "timestamp": pd.date_range("2024-01-01", periods=n_samples, freq="H"),
        "feature_A": np.random.normal(10, 2, n_samples),
        "feature_B": np.random.normal(50, 10, n_samples),
        "category": np.random.choice(["X", "Y"], n_samples),
        "target": np.random.randint(0, 2, n_samples),
    }
)

injector = DriftInjector(
    output_dir=OUTPUT_DIR, time_col="timestamp", target_column="target"
)

print("Baseline data created.")

# ============================================================
# 2. Gradual Drift (Smooth Transition)
# ============================================================

print("\n--- Injecting Gradual Drift ---")

# Shift Feature A mean by +5 gradually between index 300 and 700
drifted_gradual = injector.inject_feature_drift_gradual(
    df=data.copy(),
    feature_cols=["feature_A"],
    drift_magnitude=0.5,  # 50% shift relative to mean or scale
    drift_type="shift",
    start_index=300,
    center=500,
    width=400,
    profile="sigmoid",
    # auto_report argument removed as it's not supported in method, uses init setting
)

print(
    f"Mean feature_A (first 100): {drifted_gradual['feature_A'].iloc[:100].mean():.2f}"
)
print(
    f"Mean feature_A (last 100):  {drifted_gradual['feature_A'].iloc[-100:].mean():.2f}"
)

# ============================================================
# 3. Conditional Drift (Subpopulation Shift)
# ============================================================

print("\n--- Injecting Conditional Drift ---")

# Only drift Feature B for rows where Category == 'X'
# Simulate a sensor failure only in region 'X'
drifted_conditional = injector.inject_conditional_drift(
    df=data.copy(),
    feature_cols=["feature_B"],
    conditions=[{"column": "category", "operator": "==", "value": "X"}],
    drift_type="add_value",
    drift_magnitude=20.0,
    drift_value=20.0,  # Add 20 to the value
    start_index=500,
)

# Contrast X vs Y after drift
mask_x_post = (drifted_conditional["category"] == "X") & (
    drifted_conditional.index >= 500
)
mask_y_post = (drifted_conditional["category"] == "Y") & (
    drifted_conditional.index >= 500
)

print(
    f"Mean Feature B (Category X, post-drift): {drifted_conditional.loc[mask_x_post, 'feature_B'].mean():.2f}"
)
print(
    f"Mean Feature B (Category Y, post-drift): {drifted_conditional.loc[mask_y_post, 'feature_B'].mean():.2f}"
)

# ============================================================
# 4. Correlation Matrix Drift
# ============================================================

print("\n--- Injecting Correlation Field Drift ---")

# Force Feature A and Feature B to become highly correlated (0.9)
# Currently they are independent (random normal)
target_corr = np.array([[1.0, 0.95], [0.95, 1.0]])

try:
    drifted_corr = injector.inject_correlation_matrix_drift(
        df=data.copy(),
        feature_cols=["feature_A", "feature_B"],
        target_correlation_matrix=target_corr,
    )

    # Check correlation
    orig_corr = data[["feature_A", "feature_B"]].corr().iloc[0, 1]
    new_corr = drifted_corr[["feature_A", "feature_B"]].corr().iloc[0, 1]

    print(f"Original Correlation: {orig_corr:.3f}")
    print(f"New Correlation:      {new_corr:.3f}")

except ImportError:
    print("Skipping Correlation Drift (numpy/scipy issue or missing dependency)")
except Exception as e:
    print(f"Drift failed: {e}")


# ============================================================
# 5. Label Shift (Prior Probability Shift)
# ============================================================

print("\n--- Injecting Label Shift ---")

# Change class balance from ~50/50 to 10% Class 0, 90% Class 1
drifted_label = injector.inject_label_shift(
    df=data.copy(),
    target_col="target",
    target_distribution={0: 0.1, 1: 0.9},
)

print("Original Distribution:")
print(data["target"].value_counts(normalize=True))
print("Drifted Distribution:")
print(drifted_label["target"].value_counts(normalize=True))

# ============================================================
# 6. Functional Drift — magnitude driven by another column
# ============================================================

print("\n--- Functional Drift: noise scales with feature_A value ---")

# Sensor noise that grows exponentially with the signal level
drifted_functional = injector.inject_functional_drift(
    df=data.copy(),
    target_cols=["feature_B"],
    driver_col="feature_A",
    magnitude_func="exponential",
    magnitude_params={"scale": 0.01, "rate": 0.05},
    drift_type="additive",
)

print(
    f"Mean |delta| feature_B (low feature_A rows):  "
    f"{(drifted_functional['feature_B'] - data['feature_B']).abs()[data['feature_A'] < data['feature_A'].median()].mean():.4f}"
)
print(
    f"Mean |delta| feature_B (high feature_A rows): "
    f"{(drifted_functional['feature_B'] - data['feature_B']).abs()[data['feature_A'] >= data['feature_A'].median()].mean():.4f}"
)

# ============================================================
# 7. Causal Cascade — DAG-based propagation
# ============================================================

print("\n--- Causal Cascade: temperature → feature_A → feature_B ---")

# Define a causal chain: temperature change ripples through the system
dag = {
    "feature_A": [],
    "feature_B": [
        {"parent": "feature_A", "func": "linear", "params": {"slope": 0.8}}
    ],
}

drifted_cascade = injector.inject_causal_cascade(
    df=data.copy(),
    dag_config=dag,
    trigger_col="feature_A",
    trigger_delta=3.0,        # +3 units to feature_A
    start_index=500,          # Only affect rows 500 onwards
)

print(f"Delta feature_A (row 600): {drifted_cascade['feature_A'].iloc[600] - data['feature_A'].iloc[600]:.2f}")
print(f"Delta feature_B (row 600): {drifted_cascade['feature_B'].iloc[600] - data['feature_B'].iloc[600]:.2f}")
print(f"(feature_B should be ~0.8 × delta_feature_A = {0.8 * 3.0:.2f})")

print("\nAdvanced Drift tutorial completed!")
