# DriftInjector - Complete Reference

**Location:** `calm_data_generator.generators.drift.DriftInjector`

The `DriftInjector` is a powerful tool to simulate **data drift** (changes in data distribution over time) in synthetic datasets. It is essential for testing model monitoring systems, drift detection algorithms, and adaptive ML pipelines.

---

## ⚡ Quick Start: Drift from `generate()`

The easiest way to specificy drift is passing a `drift_injection_config` to `RealGenerator.generate()`. We recommend using the `DriftConfig` object for type safety and validation.

### Using `DriftConfig` (Recommended)

```python
from calm_data_generator.generators.configs import DriftConfig

# 1. Define Drift Configuration
drift_conf = DriftConfig(
    method="inject_feature_drift_gradual",
    feature_cols=["age", "income"],  # Columns to drift
    drift_type="shift",              # Operation type (shift, scale, noise, etc.)
    magnitude=0.3,                   # Intensity (0.0 - 1.0)
    center=500,                      # Row where drift peaks
    width=200,                       # Width of transition window
    profile="sigmoid"                # Shape of transition
)

# 2. Generate Data with Drift
synthetic = gen.generate(
    data=real_data,
    n_samples=1000,
    method='ctgan',
    drift_injection_config=[drift_conf]
)
```

### Supported `DriftConfig` Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `method` | str | `"inject_feature_drift"` | DriftInjector method to call |
| `feature_cols` | List[str] | `None` | Columns to apply drift to |
| `drift_type` | str | `"gaussian_noise"` | Type of drift operation (e.g., `shift`, `scale`) |
| `magnitude` | float | `0.2` | Drift intensity (0.0-1.0 typical) |
| `start_index` | int | `None` | Row index where drift starts |
| `end_index` | int | `None` | Row index where drift ends |
| `center` | int | `None` | Center point of drift window (for gradual) |
| `width` | int | `None` | Width of drift transition (for gradual) |
| `profile` | str | `"sigmoid"` | Transition shape (`sigmoid`, `linear`, `cosine`) |

---

## 🌲 Decision Tree: Which Drift Type?

Use this guide to choose the right drift method:

```text
Do you want to change...
├─ Feature values?
│  ├─ Gradually over time? → inject_feature_drift_gradual()
│  └─ Suddenly at a point? → inject_feature_drift() (with start_index)
├─ Target/label distribution?
│  ├─ Flip labels? → inject_label_drift()
│  └─ Force specific distribution? → inject_label_shift()
├─ Feature distributions (not values)?
│  └─ → inject_categorical_frequency_drift() or inject_covariate_shift()
└─ Feature→Target relationship?
   └─ → inject_conditional_drift() (Concept Drift)
```

---

## 📚 Drift Types Explained

| Drift Type | What It Does | Example Scenario |
|------------|--------------|------------------|
| **Feature Drift (Gradual)** | Slowly shifts feature values | Aging population, inflation |
| **Feature Drift (Sudden)** | Abrupt feature change | Sensor replacement, system update |
| **Label Drift** | Changes target distribution | Fraud wave, market shift |
| **Covariate Shift** | Changes feature distribution | New user segment joining |
| **Concept Drift** | Changes Feature→Target logic | "Good" customer definition changes |

---

## 🛠️ DriftInjector Class Reference

If you need more control than `generate()` allows, you can use `DriftInjector` directly on any DataFrame.

**Import:** `from calm_data_generator.generators.drift import DriftInjector`

### Initialization

```python
injector = DriftInjector(
    output_dir="./drift_output",      # Directory for reports/plots
    generator_name="my_drift",        # Prefix for output files
    random_state=42,                  # Reproducibility seed
    auto_report=True,                 # Generate PDF report automatically
)
```

> [!TIP]
> **Dashboard Compatibility**: The generated `report_results.json` now includes `compared_data_files` metadata. This ensures full compatibility with the monitoring dashboard for visualizing drift comparisons.

### Feature Drift Methods

#### `inject_feature_drift()` - Abrupt Shift
Directly changes values starting from `start_index`.

```python
drifted_df = injector.inject_feature_drift(
    df=df,
    feature_cols=["price", "quantity"],
    drift_type="shift",        # Options: shift, scale, gaussian_noise ...
    drift_magnitude=0.3,       # +30% shift
    start_index=500,           # Start at row 500
)
```

#### `inject_feature_drift_gradual()` - Soft Transition
Transition follows a curve (sigmoid, linear) centered at `center`.

```python
drifted_df = injector.inject_feature_drift_gradual(
    df=df,
    feature_cols=["price"],
    drift_type="scale",
    drift_magnitude=0.5,     # Scaling factor increases by 0.5
    center=500,              # Transition center
    width=200,               # Transition duration (rows)
    profile="sigmoid"        # Curve shape
)
```

#### `inject_feature_drift_incremental()` - Continuous Growth
Linear drift that keeps growing or declining over the range.

```python
drifted_df = injector.inject_feature_drift_incremental(
    df=df,
    feature_cols=["usage"],
    drift_type="shift",
    drift_magnitude=0.5,
    start_index=0,
    end_index=1000,
)
```

### Label & Categorical Drift

#### `inject_label_drift()`
Randomly flips labels (good for simulating noise/errors).

```python
drifted_df = injector.inject_label_drift(
    df=df,
    target_cols=["is_fraud"],
    drift_magnitude=0.1,     # Flip 10% of labels
    start_index=500
)
```

#### `inject_categorical_frequency_drift()`
Changes the frequency of categories (e.g., make rare items common).

```python
drifted_df = injector.inject_categorical_frequency_drift(
    df=df,
    feature_cols=["category"],
    drift_magnitude=0.5,
    perturbation="invert"    # Invert frequency distribution
)
```

---

## 🧪 Operation Types (`drift_type`)

### For Numeric Columns

| Type | Formula/Logic | Use Case |
|------|---------------|----------|
| `shift` | `x + (mean * magnitude)` | Moving average, bias |
| `scale` | `mean + (x - mean) * (1 + magnitude)` | Increased variance/amplitude |
| `gaussian_noise` | `x + N(0, magnitude * std)` | Sensor noise, measurement error |
| `add_value` | `x + magnitude` | Fixed offset |
| `multiply_value` | `x * magnitude` | Multiplicative gain |

### For Categorical/Boolean

| Type | Method | Logic |
|------|--------|-------|
| `frequency` | `inject_categorical...` | Resamples to change counts |
| `new_category` | `inject_new_category...` | Injects unknown values |
| `flip` | `inject_boolean_drift` | Flips True/False |
| `typos` | `inject_typos_drift` | Adds character noise |

---

## Pilar 5: Functional Drift & Causal Cascades

### `inject_functional_drift()` — Magnitude driven by another column

Injects drift whose magnitude varies per row as a function of a driver column's current value.

Use case: *sensor noise that scales exponentially with temperature.*

```python
df_result = injector.inject_functional_drift(
    df,
    target_cols=["sensor_reading"],
    driver_col="temperature",
    magnitude_func="exponential",    # "linear"|"exponential"|"power"|"polynomial"|callable
    magnitude_params={"scale": 0.001, "rate": 0.1},
    drift_type="additive",           # "additive" or "multiplicative"
    # Optional row selection (same system as all other methods):
    conditions=[{"column": "temperature", "operator": ">", "value": 30}],
)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `target_cols` | list[str] | Columns to perturb |
| `driver_col` | str | Column whose values determine the per-row magnitude |
| `magnitude_func` | str or callable | Transfer function: `"linear"`, `"exponential"`, `"power"`, `"polynomial"`, or a callable |
| `magnitude_params` | dict | Parameters for `magnitude_func` (e.g. `{"scale": 0.01, "rate": 0.2}`) |
| `drift_type` | str | `"additive"` (x += magnitude) or `"multiplicative"` (x *= magnitude) |

### `inject_causal_cascade()` — DAG-based propagation

Propagates a perturbation from a trigger variable through a user-defined causal DAG.

Use case: *a temperature spike increases pressure, which exponentially degrades sensor reliability.*

```python
dag = {
    "temperature": [],
    "pressure":    [{"parent": "temperature", "func": "linear",      "params": {"slope": 1.2}}],
    "sensor_fail": [{"parent": "pressure",    "func": "exponential", "params": {"scale": 0.001, "rate": 0.3}}],
}

df_result = injector.inject_causal_cascade(
    df,
    dag_config=dag,
    trigger_col="temperature",
    trigger_delta=5.0,           # scalar or array
    # Optional: time_start, conditions, etc.
)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `dag_config` | dict | Causal DAG (see [CAUSAL_ENGINE_REFERENCE.md](CAUSAL_ENGINE_REFERENCE.md)) |
| `trigger_col` | str | Root variable that receives the perturbation |
| `trigger_delta` | float or np.ndarray | Magnitude of the initial perturbation |

Supported transfer functions: `"linear"`, `"exponential"`, `"power"`, `"polynomial"`, callable.
Propagation is differential: `delta_child = f(v_parent + delta) - f(v_parent)`.

---

## Real-World Scenarios

### Case 1: Sensor Degradation (Incremental + Noise)
Simulating an IoT sensor that loses calibration and becomes noisier.

```python
# 1. Calibration loss (Linear Shift)
df = injector.inject_feature_drift_incremental(
    df=sensor_df,
    feature_cols=["reading"],
    drift_type="shift",
    drift_magnitude=0.5
)

# 2. Increasing noise (Gaussian)
df = injector.inject_feature_drift(
    df=df,
    feature_cols=["reading"],
    drift_type="gaussian_noise",
    drift_magnitude=0.3,
    start_index=500
)
```

### Case 2: Seasonal Pattern (Recurrent)
Adding a holiday season effect where sales spike.

```python
df = injector.inject_feature_drift_recurrent(
    df=sales_df,
    feature_cols=["sales"],
    drift_type="multiply_value",
    drift_magnitude=1.5,  # 50% increase
    repeats=3             # 3 seasons
)
```

### Case 3: Concept Drift (Rule-based)
Logic change: High income users suddenly start defaulting.

```python
df = injector.inject_conditional_drift(
    df=loan_df,
    feature_cols=["default"],
    conditions=[
        {"column": "income", "operator": ">", "value": 80000}
    ],
    drift_type="add_value", # Flip 0 -> 1
    drift_magnitude=1.0,
    center=1000
)
```
