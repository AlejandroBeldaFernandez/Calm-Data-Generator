"""
Tests for DriftInjector - Advanced Categorical Drift Methods
"""

import pandas as pd
import numpy as np
import pytest
from calm_data_generator.generators.drift import DriftInjector


# ---------------------------------------------------------------------------
# Tests for the new categorical drift injection methods.
# ---------------------------------------------------------------------------

def _make_categorical_df():
    np.random.seed(42)
    n = 100
    return pd.DataFrame(
        {
            "category": np.random.choice(["A", "B", "C"], n),
            "city": np.random.choice(["Madrid", "Barcelona", "Valencia"], n),
            "is_active": np.random.choice([True, False], n),
            "flag": np.random.choice([0, 1], n),
            "value": np.random.randn(n),
        }
    )


def test_categorical_frequency_drift_uniform():
    """Test that frequency drift changes category distribution."""
    df = _make_categorical_df()
    injector = DriftInjector(auto_report=False, random_state=42)
    original_counts = df["category"].value_counts(normalize=True)

    drifted_df = injector.inject_categorical_frequency_drift(
        df=df,
        feature_cols=["category"],
        drift_magnitude=0.5,
        perturbation="uniform",
    )

    drifted_counts = drifted_df["category"].value_counts(normalize=True)

    # Drifted distribution should be different
    assert not original_counts.equals(drifted_counts)
    # Same shape
    assert len(df) == len(drifted_df)


def test_categorical_frequency_drift_invert():
    """Test invert perturbation strategy."""
    df = _make_categorical_df()
    injector = DriftInjector(auto_report=False, random_state=42)
    drifted_df = injector.inject_categorical_frequency_drift(
        df=df,
        feature_cols=["category"],
        drift_magnitude=1.0,
        perturbation="invert",
    )
    assert len(df) == len(drifted_df)


def test_typos_drift():
    """Test typo injection into string columns."""
    df = _make_categorical_df()
    injector = DriftInjector(auto_report=False, random_state=42)
    drifted_df = injector.inject_typos_drift(
        df=df,
        feature_cols=["city"],
        drift_magnitude=0.5,
        typo_density=1,
        typo_type="random",
    )

    # At least some values should have changed
    changed = (df["city"] != drifted_df["city"]).sum()
    assert changed > 0


def test_category_merge_drift():
    """Test merging categories."""
    df = _make_categorical_df()
    injector = DriftInjector(auto_report=False, random_state=42)
    drifted_df = injector.inject_category_merge_drift(
        df=df,
        col="category",
        categories_to_merge=["A", "B"],
        new_category_name="AB",
    )

    # A and B should be gone, AB should exist
    unique_vals = drifted_df["category"].unique()
    assert "AB" in unique_vals
    assert "A" not in unique_vals
    assert "B" not in unique_vals
    assert "C" in unique_vals


def test_boolean_drift_true_false():
    """Test boolean flipping on True/False column."""
    df = _make_categorical_df()
    injector = DriftInjector(auto_report=False, random_state=42)
    original_true_count = df["is_active"].sum()

    drifted_df = injector.inject_boolean_drift(
        df=df,
        feature_cols=["is_active"],
        drift_magnitude=1.0,  # Flip all
    )

    drifted_true_count = drifted_df["is_active"].sum()

    # With magnitude 1.0, all values should be flipped
    # So original True count + drifted True count should equal total rows
    assert original_true_count + drifted_true_count == len(df)


def test_boolean_drift_zero_one():
    """Test boolean flipping on 0/1 integer column."""
    df = _make_categorical_df()
    injector = DriftInjector(auto_report=False, random_state=42)
    original_ones = df["flag"].sum()

    drifted_df = injector.inject_boolean_drift(
        df=df,
        feature_cols=["flag"],
        drift_magnitude=1.0,
    )

    drifted_ones = drifted_df["flag"].sum()
    assert original_ones + drifted_ones == len(df)


def test_conditional_drift_gradual():
    """Test that conditional drift works with gradual method."""
    df = _make_categorical_df()
    injector = DriftInjector(auto_report=False, random_state=42)
    drifted_df = injector.inject_conditional_drift(
        df=df,
        feature_cols=["value"],
        conditions=[{"column": "category", "operator": "==", "value": "A"}],
        drift_type="shift",
        drift_magnitude=0.5,
        drift_method="gradual",
    )

    # Only rows where category == 'A' should be affected
    a_rows_original = df[df["category"] == "A"]["value"]
    a_rows_drifted = drifted_df[df["category"] == "A"]["value"]

    # Values should be different
    assert not a_rows_original.equals(a_rows_drifted)

    # Non-A rows should be unchanged
    non_a_original = df[df["category"] != "A"]["value"]
    non_a_drifted = drifted_df[df["category"] != "A"]["value"]
    pd.testing.assert_series_equal(non_a_original, non_a_drifted)


# ---------------------------------------------------------------------------
# Tests for the unified inject_drift() method.
# ---------------------------------------------------------------------------

def _make_unified_df():
    np.random.seed(42)
    n = 200
    return pd.DataFrame(
        {
            # Numeric columns
            "age": np.random.randint(18, 80, n),
            "income": np.random.normal(50000, 15000, n),
            # Categorical columns
            "gender": np.random.choice(["M", "F", "Other"], n),
            "city": np.random.choice(["Madrid", "Barcelona", "Valencia"], n),
            # Boolean columns
            "is_active": np.random.choice([True, False], n),
            "flag": np.random.choice([0, 1], n),
        }
    )


def test_unified_drift_abrupt():
    """Test unified drift with abrupt mode."""
    df = _make_unified_df()
    injector = DriftInjector(auto_report=False, random_state=42)
    drifted_df = injector.inject_drift(
        df=df,
        columns=["age", "income"],
        drift_mode="abrupt",
        drift_magnitude=0.3,
        start_index=100,
    )

    # Check that drift was applied to second half
    first_half_age = drifted_df.iloc[:100]["age"]
    second_half_age = drifted_df.iloc[100:]["age"]

    # First half should be unchanged
    pd.testing.assert_series_equal(
        first_half_age, df.iloc[:100]["age"], check_names=False
    )

    # Second half should be different (on average)
    assert second_half_age.mean() != df.iloc[100:]["age"].mean()


def test_unified_drift_gradual():
    """Test unified drift with gradual mode."""
    df = _make_unified_df()
    injector = DriftInjector(auto_report=False, random_state=42)
    drifted_df = injector.inject_drift(
        df=df,
        columns=["income"],
        drift_mode="gradual",
        drift_magnitude=0.5,
        center=100,
        width=50,
        profile="sigmoid",
    )

    # Values should change gradually around the center
    assert len(drifted_df) == len(df)
    # Income values should be modified
    assert not df["income"].equals(drifted_df["income"])


def test_unified_drift_incremental():
    """Test unified drift with incremental mode."""
    df = _make_unified_df()
    injector = DriftInjector(auto_report=False, random_state=42)
    drifted_df = injector.inject_drift(
        df=df,
        columns=["age"],
        drift_mode="incremental",
        drift_magnitude=0.2,
    )

    assert len(drifted_df) == len(df)
    assert not df["age"].equals(drifted_df["age"])


def test_unified_drift_recurrent():
    """Test unified drift with recurrent mode."""
    df = _make_unified_df()
    injector = DriftInjector(auto_report=False, random_state=42)
    drifted_df = injector.inject_drift(
        df=df,
        columns=["income"],
        drift_mode="recurrent",
        drift_magnitude=0.4,
        repeats=3,
    )

    assert len(drifted_df) == len(df)
    assert not df["income"].equals(drifted_df["income"])


def test_unified_drift_mixed_column_types():
    """Test that unified drift handles all column types correctly."""
    df = _make_unified_df()
    injector = DriftInjector(auto_report=False, random_state=42)
    drifted_df = injector.inject_drift(
        df=df,
        columns=["age", "gender", "is_active"],  # Numeric, categorical, boolean
        drift_mode="abrupt",
        drift_magnitude=0.3,
        start_index=100,
    )

    assert len(drifted_df) == len(df)
    # All three columns should potentially be affected
    assert not df["age"].iloc[100:].equals(drifted_df["age"].iloc[100:])


def test_unified_drift_auto_detects_types():
    """Test that column types are correctly auto-detected."""
    df = _make_unified_df()
    injector = DriftInjector(auto_report=False, random_state=42)
    # This should work without specifying operations
    drifted_df = injector.inject_drift(
        df=df,
        columns=["age", "income", "gender", "city", "is_active", "flag"],
        drift_mode="abrupt",
        drift_magnitude=0.2,
        start_index=150,
    )

    assert len(drifted_df) == len(df)


def test_unified_drift_custom_operations():
    """Test that custom operations per column type work."""
    df = _make_unified_df()
    injector = DriftInjector(auto_report=False, random_state=42)
    drifted_df = injector.inject_drift(
        df=df,
        columns=["age", "gender"],
        drift_mode="abrupt",
        drift_magnitude=0.3,
        numeric_operation="scale",
        categorical_operation="frequency",
        start_index=100,
    )

    assert len(drifted_df) == len(df)


def test_unified_drift_invalid_mode_raises_error():
    """Test that invalid drift_mode raises ValueError."""
    df = _make_unified_df()
    injector = DriftInjector(auto_report=False, random_state=42)
    with pytest.raises(ValueError):
        injector.inject_drift(
            df=df,
            columns=["age"],
            drift_mode="invalid_mode",
            drift_magnitude=0.3,
        )


def test_unified_drift_missing_column_warning():
    """Test that missing columns generate warnings but don't fail."""
    import warnings

    df = _make_unified_df()
    injector = DriftInjector(auto_report=False, random_state=42)

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        drifted_df = injector.inject_drift(
            df=df,
            columns=["age", "nonexistent_column"],
            drift_mode="abrupt",
            drift_magnitude=0.3,
        )

        # Should have generated a warning
        assert any("nonexistent_column" in str(warning.message) for warning in w)
        # But should still return valid data
        assert len(drifted_df) == len(df)


def test_inject_multiple_types_of_drift_with_config_objects():
    """Test inject_multiple_types_of_drift with DriftConfig objects."""
    from calm_data_generator.generators.configs import DriftConfig

    df = _make_unified_df()
    injector = DriftInjector(auto_report=False, random_state=42)

    drift_configs = [
        DriftConfig(
            method="inject_drift",
            params={
                "columns": ["age"],
                "drift_mode": "abrupt",
                "drift_magnitude": 0.5,
                "start_index": 50,
            },
        ),
        DriftConfig(
            method="inject_drift",
            params={
                "columns": ["income"],
                "drift_mode": "gradual",
                "drift_magnitude": 0.3,
                "center": 100,
                "width": 50,
            },
        ),
    ]

    # Use inject_multiple_types_of_drift which accepts a schedule list
    drifted_df = injector.inject_multiple_types_of_drift(
        df=df, schedule=drift_configs
    )

    assert len(drifted_df) == len(df)
    # Age should be drifted abruptly (check second half vs first half of affected region)
    # Note: simplistic check
    assert df["age"].iloc[50:].mean() != drifted_df["age"].iloc[50:].mean()
    # Income should be drifted gradually
    assert not df["income"].equals(drifted_df["income"])
