"""
Tests for CausalEngine — DAG-based causal cascade propagation.
"""

import math

import numpy as np
import pandas as pd
import pytest

from calm_data_generator.generators.dynamics.CausalEngine import CausalEngine


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _simple_df(n=50, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "A": rng.uniform(1, 10, n),
        "B": rng.uniform(1, 10, n),
        "C": rng.uniform(1, 10, n),
    })


# ---------------------------------------------------------------------------
# Test 1: linear chain A → B → C, verify delta cascades correctly
# ---------------------------------------------------------------------------

def test_linear_chain_delta():
    """A→B (slope=2), B→C (slope=3). delta_A=5 → delta_B=10, delta_C=30."""
    dag = {
        "A": [],
        "B": [{"parent": "A", "func": "linear", "params": {"slope": 2.0}}],
        "C": [{"parent": "B", "func": "linear", "params": {"slope": 3.0}}],
    }
    engine = CausalEngine(dag)
    df = _simple_df()
    df_orig = df.copy()

    delta = np.full(len(df), 5.0)
    df_result = engine.apply_cascade(df.copy(), trigger_col="A", delta=delta)

    np.testing.assert_allclose(df_result["A"] - df_orig["A"], 5.0)
    np.testing.assert_allclose(df_result["B"] - df_orig["B"], 10.0)
    np.testing.assert_allclose(df_result["C"] - df_orig["C"], 30.0)


# ---------------------------------------------------------------------------
# Test 2: exponential transfer — non-linear delta
# ---------------------------------------------------------------------------

def test_exponential_delta():
    """A→B with exponential func. Verify delta_B = scale*(exp(rate*(v+d)) - exp(rate*v))."""
    scale, rate, d = 1.0, 0.5, 2.0
    dag = {
        "A": [],
        "B": [{"parent": "A", "func": "exponential", "params": {"scale": scale, "rate": rate}}],
    }
    engine = CausalEngine(dag)

    n = 10
    rng = np.random.default_rng(1)
    df = pd.DataFrame({"A": rng.uniform(1, 5, n), "B": rng.zeros(n) if False else np.zeros(n)})
    df_orig = df.copy()

    delta = np.full(n, d)
    df_result = engine.apply_cascade(df.copy(), trigger_col="A", delta=delta)

    v_a = df_orig["A"].values
    expected_delta_b = scale * (np.exp(rate * (v_a + d)) - np.exp(rate * v_a))

    np.testing.assert_allclose(df_result["A"] - df_orig["A"], d, rtol=1e-10)
    np.testing.assert_allclose(df_result["B"] - df_orig["B"], expected_delta_b, rtol=1e-8)


# ---------------------------------------------------------------------------
# Test 3: callable transfer function
# ---------------------------------------------------------------------------

def test_callable_transfer():
    """A→B with a lambda as transfer function."""
    double = lambda x: 2 * x  # noqa: E731

    dag = {
        "A": [],
        "B": [{"parent": "A", "func": double, "params": {}}],
    }
    engine = CausalEngine(dag)
    df = _simple_df()
    df_orig = df.copy()

    d = 3.0
    delta = np.full(len(df), d)
    df_result = engine.apply_cascade(df.copy(), trigger_col="A", delta=delta)

    # f(v+d) - f(v) = 2*(v+d) - 2*v = 2*d
    np.testing.assert_allclose(df_result["B"] - df_orig["B"], 2 * d, rtol=1e-10)


# ---------------------------------------------------------------------------
# Test 4: cycle detection raises ValueError at construction time
# ---------------------------------------------------------------------------

def test_cycle_raises():
    """A→B→A cycle must raise ValueError."""
    dag = {
        "A": [{"parent": "B", "func": "linear", "params": {"slope": 1.0}}],
        "B": [{"parent": "A", "func": "linear", "params": {"slope": 1.0}}],
    }
    with pytest.raises(ValueError, match="[Cc]ycle"):
        CausalEngine(dag)


# ---------------------------------------------------------------------------
# Test 5: unknown parent raises ValueError
# ---------------------------------------------------------------------------

def test_unknown_parent_raises():
    """Referencing a parent not declared as a key must raise ValueError."""
    dag = {
        "A": [],
        "B": [{"parent": "UNKNOWN", "func": "linear", "params": {}}],
    }
    with pytest.raises(ValueError, match="unknown"):
        CausalEngine(dag)


# ---------------------------------------------------------------------------
# Test 6: rows=None affects all rows; partial rows affects only those
# ---------------------------------------------------------------------------

def test_rows_partial():
    """Only specified rows should change when rows param is provided."""
    dag = {
        "A": [],
        "B": [{"parent": "A", "func": "linear", "params": {"slope": 1.0}}],
    }
    engine = CausalEngine(dag)
    df = _simple_df(n=20)
    df_orig = df.copy()

    partial_rows = df.index[:10]
    delta = np.full(len(partial_rows), 5.0)
    df_result = engine.apply_cascade(df.copy(), trigger_col="A", delta=delta, rows=partial_rows)

    # First 10 rows changed
    assert (df_result.loc[partial_rows, "A"] != df_orig.loc[partial_rows, "A"]).all()
    assert (df_result.loc[partial_rows, "B"] != df_orig.loc[partial_rows, "B"]).all()
    # Last 10 rows unchanged
    rest = df.index[10:]
    pd.testing.assert_frame_equal(df_result.loc[rest], df_orig.loc[rest])


def test_rows_none_affects_all():
    """rows=None must affect every row."""
    dag = {
        "A": [],
        "B": [{"parent": "A", "func": "linear", "params": {"slope": 2.0}}],
    }
    engine = CausalEngine(dag)
    df = _simple_df(n=30)
    df_orig = df.copy()

    delta = np.full(len(df), 1.0)
    df_result = engine.apply_cascade(df.copy(), trigger_col="A", delta=delta)

    assert (df_result["A"] != df_orig["A"]).all()
    assert (df_result["B"] != df_orig["B"]).all()


# ---------------------------------------------------------------------------
# Test 7: topological order — chain and diamond DAG
# ---------------------------------------------------------------------------

def test_topo_order_chain():
    """A→B→C: order must respect A before B before C."""
    dag = {
        "A": [],
        "B": [{"parent": "A", "func": "linear", "params": {}}],
        "C": [{"parent": "B", "func": "linear", "params": {}}],
    }
    order = CausalEngine(dag).get_topological_order()
    assert order.index("A") < order.index("B") < order.index("C")


def test_topo_order_diamond():
    """Diamond A→B, A→C, B→D, C→D: A first, D last."""
    dag = {
        "A": [],
        "B": [{"parent": "A", "func": "linear", "params": {}}],
        "C": [{"parent": "A", "func": "linear", "params": {}}],
        "D": [
            {"parent": "B", "func": "linear", "params": {}},
            {"parent": "C", "func": "linear", "params": {}},
        ],
    }
    order = CausalEngine(dag).get_topological_order()
    assert order[0] == "A"
    assert order[-1] == "D"
    assert order.index("B") < order.index("D")
    assert order.index("C") < order.index("D")


# ---------------------------------------------------------------------------
# Test 8: trigger_col not in DAG raises ValueError in apply_cascade
# ---------------------------------------------------------------------------

def test_trigger_not_in_dag_raises():
    dag = {
        "A": [],
        "B": [{"parent": "A", "func": "linear", "params": {}}],
    }
    engine = CausalEngine(dag)
    df = _simple_df()
    with pytest.raises(ValueError, match="trigger_col"):
        engine.apply_cascade(df, trigger_col="NOT_A_NODE", delta=np.ones(len(df)))
