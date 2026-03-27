"""
Tests for Pilar 5 features:
  - DriftInjector.inject_functional_drift
  - DriftInjector.inject_causal_cascade
  - ScenarioInjector.evolve_features with type="driven_by"
  - propagate_numeric_drift free function (backward compat with old _propagate_numeric_drift)
"""

import numpy as np
import pandas as pd
import pytest

from calm_data_generator.generators.drift.DriftInjector import DriftInjector
from calm_data_generator.generators.dynamics.ScenarioInjector import ScenarioInjector
from calm_data_generator.generators.utils.propagation import apply_func, propagate_numeric_drift


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_df(n=100, seed=42):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "temperature": rng.uniform(20, 50, n),
        "sensor":      rng.uniform(0, 1, n),
        "pressure":    rng.uniform(1, 5, n),
    })


def _injector():
    return DriftInjector(random_state=42, auto_report=False)


# ---------------------------------------------------------------------------
# 1. inject_functional_drift — exponential: high driver → larger drift
# ---------------------------------------------------------------------------

def test_functional_drift_exponential_magnitude():
    """Rows with higher driver_col should receive a larger absolute drift."""
    df = _make_df()
    injector = _injector()

    df_result = injector.inject_functional_drift(
        df,
        target_cols=["sensor"],
        driver_col="temperature",
        magnitude_func="exponential",
        magnitude_params={"scale": 0.001, "rate": 0.05},
        drift_type="additive",
    )

    diffs = (df_result["sensor"] - df["sensor"]).values
    temps = df["temperature"].values

    # High temperature rows must have larger drift than low temperature rows
    high_mask = temps > temps.mean()
    low_mask = ~high_mask
    assert np.abs(diffs[high_mask]).mean() > np.abs(diffs[low_mask]).mean()


# ---------------------------------------------------------------------------
# 2. inject_functional_drift — multiplicative mode
# ---------------------------------------------------------------------------

def test_functional_drift_multiplicative():
    """With multiplicative mode, x *= magnitude (not +=)."""
    n = 50
    df = pd.DataFrame({
        "driver": np.ones(n) * 2.0,   # constant driver → constant magnitude
        "target": np.ones(n) * 10.0,
    })
    injector = _injector()

    # linear func: slope=1 → magnitude = 1*driver = 2
    # multiplicative: target *= 2 → result = 20
    df_result = injector.inject_functional_drift(
        df,
        target_cols=["target"],
        driver_col="driver",
        magnitude_func="linear",
        magnitude_params={"slope": 1.0},
        drift_type="multiplicative",
    )
    np.testing.assert_allclose(df_result["target"].values, 20.0, rtol=1e-10)


# ---------------------------------------------------------------------------
# 3. inject_functional_drift — callable
# ---------------------------------------------------------------------------

def test_functional_drift_callable():
    """A custom callable is accepted as magnitude_func."""
    n = 30
    df = pd.DataFrame({
        "driver": np.arange(n, dtype=float),
        "target": np.zeros(n),
    })
    injector = _injector()

    triple = lambda x: 3.0 * x  # noqa: E731

    df_result = injector.inject_functional_drift(
        df,
        target_cols=["target"],
        driver_col="driver",
        magnitude_func=triple,
        drift_type="additive",
    )
    expected = 3.0 * np.arange(n, dtype=float)
    np.testing.assert_allclose(df_result["target"].values, expected, rtol=1e-10)


# ---------------------------------------------------------------------------
# 4. inject_functional_drift — conditions row selection
# ---------------------------------------------------------------------------

def test_functional_drift_conditions():
    """Only rows matching conditions should be modified."""
    n = 100
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "temperature": rng.uniform(20, 60, n),
        "sensor":      np.ones(n),
    })
    injector = _injector()

    threshold = 40.0
    df_result = injector.inject_functional_drift(
        df,
        target_cols=["sensor"],
        driver_col="temperature",
        magnitude_func="linear",
        magnitude_params={"slope": 1.0},
        drift_type="additive",
        conditions=[{"column": "temperature", "operator": ">", "value": threshold}],
    )

    hot_rows = df["temperature"] > threshold
    cold_rows = ~hot_rows

    # Hot rows changed
    assert (df_result.loc[hot_rows, "sensor"] != df.loc[hot_rows, "sensor"]).all()
    # Cold rows unchanged
    pd.testing.assert_series_equal(
        df_result.loc[cold_rows, "sensor"], df.loc[cold_rows, "sensor"]
    )


# ---------------------------------------------------------------------------
# 5. inject_causal_cascade — 3-node DAG: all downstream columns change
# ---------------------------------------------------------------------------

def test_causal_cascade_3_nodes():
    """Trigger=temperature, intermediate=pressure, leaf=sensor. All three change."""
    dag = {
        "temperature": [],
        "pressure":    [{"parent": "temperature", "func": "linear", "params": {"slope": 1.0}}],
        "sensor":      [{"parent": "pressure",    "func": "linear", "params": {"slope": 0.5}}],
    }
    df = _make_df()
    df_orig = df.copy()
    injector = _injector()

    df_result = injector.inject_causal_cascade(
        df,
        dag_config=dag,
        trigger_col="temperature",
        trigger_delta=5.0,
    )

    assert (df_result["temperature"] != df_orig["temperature"]).all()
    assert (df_result["pressure"]    != df_orig["pressure"]).all()
    assert (df_result["sensor"]      != df_orig["sensor"]).all()


# ---------------------------------------------------------------------------
# 6. evolve_features with type="driven_by"
# ---------------------------------------------------------------------------

def test_driven_by_linear():
    """pressure delta should equal slope * temperature for each row."""
    slope = 0.8
    n = 50
    rng = np.random.default_rng(7)
    df = pd.DataFrame({
        "temperature": rng.uniform(10, 40, n),
        "pressure":    rng.uniform(1, 5, n),
    })
    df_orig = df.copy()

    si = ScenarioInjector(seed=42)
    df_evolved = si.evolve_features(df.copy(), evolution_config={
        "pressure": {
            "type":        "driven_by",
            "driver_col":  "temperature",
            "func":        "linear",
            "func_params": {"slope": slope},
        }
    })

    expected_delta = slope * df_orig["temperature"].values
    actual_delta   = df_evolved["pressure"].values - df_orig["pressure"].values

    np.testing.assert_allclose(actual_delta, expected_delta, rtol=1e-8)


def test_driven_by_missing_driver_raises():
    """driven_by with a driver_col not in the DataFrame should raise ValueError."""
    si = ScenarioInjector(seed=42)
    df = pd.DataFrame({"pressure": np.ones(10)})

    with pytest.raises(ValueError, match="driven_by"):
        si.evolve_features(df, evolution_config={
            "pressure": {
                "type":       "driven_by",
                "driver_col": "nonexistent_col",
                "func":       "linear",
            }
        })


# ---------------------------------------------------------------------------
# 7. propagate_numeric_drift free function — same result as old class method
# ---------------------------------------------------------------------------

def test_propagate_numeric_drift_free_function():
    """propagate_numeric_drift must apply delta correctly to a correlated column."""
    rng = np.random.default_rng(99)
    n = 200
    # Create two correlated columns
    base = rng.normal(0, 1, n)
    df = pd.DataFrame({
        "X": base + rng.normal(0, 0.1, n),
        "Y": base * 2.0 + rng.normal(0, 0.1, n),
    })

    # All rows
    rows = df.index
    delta_x = np.full(n, 1.0)
    corr = df[["X", "Y"]].corr()

    df_result = propagate_numeric_drift(
        df.copy(), rows=rows, driver_col="X",
        delta_driver=delta_x, correlations=corr
    )

    # propagate_numeric_drift propagates from driver to OTHER columns, not the driver itself
    # X is unchanged — caller is responsible for applying delta to X
    np.testing.assert_allclose(df_result["X"].values, df["X"].values)

    # Y should have changed (correlated with X, rho≈1, so delta_Y ≈ delta_X * std_Y/std_X)
    assert not np.allclose(df_result["Y"].values, df["Y"].values)
