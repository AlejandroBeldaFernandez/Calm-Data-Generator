# CausalEngine - Reference

**Location:** `calm_data_generator.generators.dynamics.CausalEngine`

`CausalEngine` implements DAG-based causal cascade propagation. It allows defining directed acyclic graphs (DAGs) of variable dependencies and propagating a perturbation from a trigger variable to all its descendants using user-defined transfer functions (linear, exponential, power, polynomial, or any callable).

---

## Quick Start

```python
from calm_data_generator.generators.dynamics.CausalEngine import CausalEngine
import numpy as np
import pandas as pd

# IoT sensor chain: temperature drives pressure, which drives sensor failure rate
dag = {
    "temperature": [],
    "pressure": [
        {"parent": "temperature", "func": "linear", "params": {"slope": 1.2}}
    ],
    "sensor_fail_rate": [
        {"parent": "pressure", "func": "exponential", "params": {"scale": 0.001, "rate": 0.3}}
    ],
}

engine = CausalEngine(dag)

df = pd.DataFrame({
    "temperature":    np.random.uniform(20, 40, 500),
    "pressure":       np.random.uniform(1, 5, 500),
    "sensor_fail_rate": np.zeros(500),
})

# Apply a +5 degree shock to all rows
df_shocked = engine.apply_cascade(df.copy(), trigger_col="temperature", delta=np.full(500, 5.0))
```

---

## DAG Format

```python
dag_config = {
    "node_name": [                      # list of parent edges (empty = root node)
        {
            "parent": "parent_col",     # required: name of the parent node
            "func":   "linear",         # required: transfer function (see table below)
            "params": {"slope": 1.2},   # optional: parameters for the function
        }
    ],
}
```

A node with an empty list `[]` is a **root** (no parents). Nodes can have multiple parents — their delta contributions are summed.

---

## Transfer Functions

The propagation uses a **differential approach**:
```
delta_child = f(v_parent + delta_parent) - f(v_parent)
```
This preserves the current absolute values and only propagates the incremental change.

| `func` | Formula | Parameters |
|--------|---------|------------|
| `"linear"` | `slope * delta_parent` | `slope` (default 1.0), `intercept` ignored for deltas |
| `"exponential"` | `scale * (exp(rate*(v+d)) - exp(rate*v))` | `scale` (default 1.0), `rate` (default 1.0) |
| `"power"` | `scale * ((v+d)^exp - v^exp)` | `scale` (default 1.0), `exponent` (default 2.0) |
| `"polynomial"` | `P(v+d) - P(v)` | `coeffs`: list of polynomial coefficients for `np.poly1d` |
| callable | `func(v+d) - func(v)` | — |

---

## API Reference

### `CausalEngine(dag_config)`

```python
engine = CausalEngine(dag_config: dict)
```

Validates the DAG on construction (detects cycles via Kahn's algorithm). Raises `ValueError` if:
- A cycle is detected
- An edge references a node not declared as a key in `dag_config`

### `apply_cascade(df, trigger_col, delta, rows=None)`

```python
df_result = engine.apply_cascade(
    df: pd.DataFrame,
    trigger_col: str,          # must be a DAG node
    delta: np.ndarray,         # perturbation vector (length = len(rows) or len(df))
    rows: pd.Index = None,     # None = all rows
) -> pd.DataFrame              # df modified in-place, also returned
```

Applies `delta` to `trigger_col` and propagates to all descendants in topological order.

### `get_topological_order()`

Returns all nodes as a list ordered from roots to leaves. Useful for introspection and testing.

### `validate()`

Re-validates the DAG. Raises `ValueError` on cycles or unknown node references.

---

## Using with DriftInjector

`DriftInjector.inject_causal_cascade` wraps `CausalEngine` and integrates it with the full row-selection and reporting system:

```python
from calm_data_generator.generators.drift.DriftInjector import DriftInjector

injector = DriftInjector(original_df=df)

dag = {
    "temperature": [],
    "pressure":    [{"parent": "temperature", "func": "linear",      "params": {"slope": 0.5}}],
    "sensor_fail": [{"parent": "pressure",    "func": "exponential", "params": {"scale": 0.01, "rate": 0.3}}],
}

# Apply cascade only to rows where temperature > 35
df_result = injector.inject_causal_cascade(
    df,
    dag_config=dag,
    trigger_col="temperature",
    trigger_delta=10.0,
    conditions=[{"column": "temperature", "operator": ">", "value": 35}],
)
```

---

## Using with ScenarioInjector

`evolve_type: "driven_by"` makes a feature evolve proportionally to another column's current value at each row:

```python
from calm_data_generator.generators.dynamics.ScenarioInjector import ScenarioInjector

si = ScenarioInjector()
df_evolved = si.evolve_features(df, evolution_config={
    "pressure": {
        "type":        "driven_by",
        "driver_col":  "temperature",
        "func":        "linear",
        "func_params": {"slope": 0.8},
    }
})
# Each row: delta_pressure = 0.8 * temperature_value
```

---

## Domain Examples

### Finance: equity price shock propagation

```python
dag = {
    "market_index": [],
    "sector_etf": [
        {"parent": "market_index", "func": "linear", "params": {"slope": 1.1}}
    ],
    "single_stock": [
        {"parent": "market_index", "func": "linear",      "params": {"slope": 0.6}},
        {"parent": "sector_etf",   "func": "exponential", "params": {"scale": 0.3, "rate": 0.05}},
    ],
}
engine = CausalEngine(dag)
df_shocked = engine.apply_cascade(df, "market_index", delta=np.full(len(df), -0.05))
```

### Clinical: treatment effect cascade

```python
dag = {
    "drug_dose": [],
    "biomarker_A": [
        {"parent": "drug_dose", "func": "linear", "params": {"slope": -0.3}}
    ],
    "biomarker_B": [
        {"parent": "biomarker_A", "func": "power", "params": {"scale": 1.0, "exponent": 1.5}}
    ],
}
```

---

## Error Handling

| Situation | Behaviour |
|-----------|-----------|
| Cycle in DAG | `ValueError` at construction time |
| Edge to unknown node | `ValueError` at construction time |
| `trigger_col` not in DAG | `ValueError` in `apply_cascade` |
| Node in DAG but not in df | Silently skipped (delta computed but not applied) |
| Node in df but not in DAG | Not affected by the cascade |
