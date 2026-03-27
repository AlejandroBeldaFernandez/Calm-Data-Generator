# Presets Reference

Presets are ready-to-use generator configurations that encapsulate method selection, hyperparameters, and reporting for the most common synthetic data scenarios.

## Base Class

```python
from calm_data_generator.presets import GeneratorPreset  # abstract
```

All presets inherit from `GeneratorPreset` and share three constructor parameters:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `random_state` | `int` | `42` | Random seed for reproducibility |
| `verbose` | `bool` | `True` | Print progress messages |
| `fast_dev_run` | `bool` | `False` | Minimal iterations/epochs — for testing pipelines |

All presets expose a single `.generate()` method. Parameters vary by preset (see sections below).

---

## Speed & Prototyping

### `FastPreset`

```python
from calm_data_generator.presets import FastPreset
```

Fastest general-purpose generation. Uses LightGBM with 10 iterations and passes through additional kwargs to the underlying generator.

**`generate(data, n_samples, **kwargs)`**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `data` | required | Real DataFrame to learn from |
| `n_samples` | required | Number of synthetic rows |
| `iterations` | `10` | Number of LightGBM fitting iterations |
| `auto_report` | `False` | Enable quality reporting |

```python
preset = FastPreset(random_state=42)
synthetic_df = preset.generate(data=real_df, n_samples=1000)

# Override iterations
synthetic_df = preset.generate(data=real_df, n_samples=1000, iterations=20)
```

---

### `FastPrototypePreset`

```python
from calm_data_generator.presets import FastPrototypePreset
```

Optimized for CI/CD pipelines and integration tests. Uses LightGBM with a fixed 10 iterations (1 if `fast_dev_run=True`). Does **not** forward kwargs to the underlying generator.

**`generate(data, n_samples, **kwargs)`**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `data` | required | Real DataFrame to learn from |
| `n_samples` | required | Number of synthetic rows |

```python
# Integration test — use fast_dev_run=True for 1 iteration
preset = FastPrototypePreset(fast_dev_run=True)
synthetic_df = preset.generate(data=real_df, n_samples=100)
```

---

## Quality & Fidelity

### `HighFidelityPreset`

```python
from calm_data_generator.presets import HighFidelityPreset
```

Maximum quality for production data. Uses CTGAN with 1000 epochs, batch size 250, and adversarial validation.

**`generate(data, n_samples, auto_report=True)`**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `data` | required | Real DataFrame |
| `n_samples` | required | Number of synthetic rows |
| `auto_report` | `True` | Enable quality reporting |

```python
preset = HighFidelityPreset(random_state=42)
synthetic_df = preset.generate(data=real_df, n_samples=5000)
```

---

### `DiffusionPreset`

```python
from calm_data_generator.presets import DiffusionPreset
```

Uses Tabular DDPM (denoising diffusion probabilistic model) from SynthCity. Captures complex multi-modal distributions better than GANs. Slower but higher structural fidelity.

**`generate(data, n_samples, **kwargs)`**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `data` | required | Real DataFrame |
| `n_samples` | required | Number of synthetic rows |
| `auto_report` | `True` | Enable quality reporting |

Key internal config: `method="ddpm"`, `n_steps=1000` (2 with `fast_dev_run`), `batch_size=256`.

```python
preset = DiffusionPreset(random_state=42)
synthetic_df = preset.generate(data=real_df, n_samples=2000)
```

---

### `CopulaPreset`

```python
from calm_data_generator.presets import CopulaPreset
```

Fast and statistically robust baseline using a Gaussian Copula to model dependencies. Ideal as a benchmark before using heavier models.

**`generate(data, n_samples, **kwargs)`**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `data` | required | Real DataFrame |
| `n_samples` | required | Number of synthetic rows |
| `auto_report` | `True` | Enable quality reporting |

```python
preset = CopulaPreset(random_state=42)
baseline_df = preset.generate(data=real_df, n_samples=1000)
```

---

### `DataQualityAuditPreset`

```python
from calm_data_generator.presets import DataQualityAuditPreset
```

Combines TVAE synthesis with a forced comprehensive quality report (`auto_report=True`, `minimal_report=False`). Use this when you need a full fidelity audit alongside the synthetic data.

**`generate(data, n_samples, **kwargs)`**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `data` | required | Real DataFrame |
| `n_samples` | required | Number of synthetic rows |

Key internal config: `method="tvae"`, `epochs=300` (1 with `fast_dev_run`), full reporting always on.

```python
preset = DataQualityAuditPreset(random_state=42)
synthetic_df = preset.generate(data=real_df, n_samples=3000)
# A full HTML quality report is automatically saved to disk
```

---

## Class Distribution

### `ImbalancedGeneratorPreset`

```python
from calm_data_generator.presets import ImbalancedGeneratorPreset
```

Generates synthetic data with a controlled minority/majority class ratio. Useful for creating imbalanced benchmarks for drift detection and bias analysis.

**`generate(data, n_samples, target_col, imbalance_ratio=0.1, **kwargs)`**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `data` | required | Real DataFrame (binary target required) |
| `n_samples` | required | Number of synthetic rows |
| `target_col` | required | Column to imbalance |
| `imbalance_ratio` | `0.1` | Fraction of the minority class (0–1) |

Key internal config: `method="ctgan"`, `epochs=300`.

```python
preset = ImbalancedGeneratorPreset(random_state=42)

# 5% minority class
synthetic_df = preset.generate(
    data=real_df, n_samples=2000,
    target_col="fraud", imbalance_ratio=0.05
)
```

> **Note**: Currently supports binary targets only.

---

### `BalancedDataGeneratorPreset`

```python
from calm_data_generator.presets import BalancedDataGeneratorPreset
```

Balances an imbalanced dataset using SMOTE oversampling. Generates synthetic minority samples to reach a 50/50 distribution.

**`generate(data, n_samples, target_col, **kwargs)`**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `data` | required | Real (imbalanced) DataFrame |
| `n_samples` | required | Total synthetic rows to generate |
| `target_col` | required | Column to balance |

```python
preset = BalancedDataGeneratorPreset(random_state=42)
balanced_df = preset.generate(data=imbalanced_df, n_samples=2000, target_col="label")
```

---

## Time Series

### `TimeSeriesPreset`

```python
from calm_data_generator.presets import TimeSeriesPreset
```

Generates sequential/temporal data using dedicated time-series models. Supports three backends:

- `timegan` — best for complex irregular temporal patterns
- `timevae` — faster, good for regular time series
- `fflows` — most stable, best for periodic/seasonal data

**`generate(data, n_samples, sequence_key, time_key=None, method="timegan", **kwargs)`**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `data` | required | Time-series DataFrame |
| `n_samples` | required | Number of synthetic sequences |
| `sequence_key` | required | Column identifying each entity/sequence |
| `time_key` | `None` | Column with timestamps (optional) |
| `method` | `"timegan"` | Backend: `"timegan"`, `"timevae"`, `"fflows"` |

Key internal config: `n_iter=500` epochs (1 with `fast_dev_run`), `batch_size=100`.

```python
preset = TimeSeriesPreset(random_state=42)

# TimeGAN (default)
synthetic_df = preset.generate(
    data=ts_df, n_samples=200,
    sequence_key="patient_id", time_key="visit_date"
)

# FourierFlows for periodic data
synthetic_df = preset.generate(
    data=ts_df, n_samples=200,
    sequence_key="sensor_id", method="fflows"
)
```

---

### `SeasonalTimeSeriesPreset`

```python
from calm_data_generator.presets import SeasonalTimeSeriesPreset
```

Two-stage preset: (1) generates base sequences with TimeGAN, then (2) superimposes a sinusoidal seasonal pattern via `ScenarioInjector`.

**`generate(data, n_samples, time_col, seasonal_cols, period=12, amplitude=1.0, **kwargs)`**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `data` | required | Time-series DataFrame |
| `n_samples` | required | Number of synthetic rows |
| `time_col` | required | Timestamp column |
| `seasonal_cols` | required | Columns to inject seasonality into |
| `period` | `12` | Seasonality period (e.g., 12 = monthly in a yearly cycle) |
| `amplitude` | `1.0` | Strength of the seasonal oscillation |

```python
preset = SeasonalTimeSeriesPreset(random_state=42)
synthetic_df = preset.generate(
    data=sales_df, n_samples=500,
    time_col="date", seasonal_cols=["sales", "web_traffic"],
    period=12, amplitude=2.5
)
```

---

## Drift & Scenarios

### `DriftScenarioPreset`

```python
from calm_data_generator.presets import DriftScenarioPreset
```

Generates data with injected drift characteristics. Useful for stress-testing drift detection systems and evaluating model robustness.

**`generate(data, n_samples, drift_scenarios=None, **kwargs)`**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `data` | required | Real DataFrame |
| `n_samples` | required | Number of synthetic rows |
| `drift_scenarios` | `None` | List of drift config dicts (wrapped in `DriftConfig`) |

```python
from calm_data_generator.presets import DriftScenarioPreset

preset = DriftScenarioPreset(random_state=42)

# With explicit drift injection
scenarios = [{"column": "temperature", "type": "shift_mean", "magnitude": 3.0}]
synthetic_df = preset.generate(data=real_df, n_samples=1000, drift_scenarios=scenarios)
```

---

### `GradualDriftPreset`

```python
from calm_data_generator.presets import GradualDriftPreset
```

Simulates gradual linear drift over time/index in specified columns.

**`generate(data, n_samples, drift_cols, slope=0.01, **kwargs)`**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `data` | required | Real DataFrame |
| `n_samples` | required | Number of synthetic rows |
| `drift_cols` | required | List of columns to apply drift to |
| `slope` | `0.01` | Rate of linear drift per step |

```python
preset = GradualDriftPreset(random_state=42)
synthetic_df = preset.generate(
    data=real_df, n_samples=1000,
    drift_cols=["temperature", "humidity"], slope=0.05
)
```

---

### `ConceptDriftPreset`

```python
from calm_data_generator.presets import ConceptDriftPreset
```

Simulates sudden concept drift by altering the P(y|x) relationship between features and target. Use this to test model robustness to distribution shifts in the label boundary.

**`generate(data, n_samples, target_col, drift_magnitude=0.5, **kwargs)`**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `data` | required | Real DataFrame |
| `n_samples` | required | Number of synthetic rows |
| `target_col` | required | Target column to apply concept drift to |
| `drift_magnitude` | `0.5` | Degree of P(y\|x) alteration (0–1) |

```python
preset = ConceptDriftPreset(random_state=42)
synthetic_df = preset.generate(
    data=real_df, n_samples=1000,
    target_col="churn", drift_magnitude=0.7
)
```

---

### `ScenarioInjectorPreset`

```python
from calm_data_generator.presets import ScenarioInjectorPreset
```

Applies complex evolution scenarios directly to an existing dataset using `ScenarioInjector.evolve_features()`. Unlike other presets, this modifies input data rather than generating new samples from scratch.

**`generate(data, scenario_config, **kwargs)`**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `data` | required | Existing DataFrame to transform |
| `scenario_config` | required | Dict with `"evolve_features"` key mapping column names to `EvolutionFeatureConfig` objects |

```python
from calm_data_generator.presets import ScenarioInjectorPreset
from calm_data_generator.generators.configs import EvolutionFeatureConfig

scenario = {
    "evolve_features": {
        "temperature": EvolutionFeatureConfig(
            column="temperature", type="linear", slope=0.1
        ),
        "pressure": EvolutionFeatureConfig(
            column="pressure", type="driven_by",
            driver_col="temperature", func="linear",
            func_params={"slope": 0.5}
        ),
    }
}

preset = ScenarioInjectorPreset(random_state=42)
transformed_df = preset.generate(data=real_df, scenario_config=scenario)
```

---

## Clinical & Omics

### `LongitudinalHealthPreset`

```python
from calm_data_generator.presets import LongitudinalHealthPreset
```

Generates longitudinal clinical data with multi-visit patient records using `ClinicalDataGenerator`.

**`generate(n_samples, n_visits=5, **kwargs)`**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_samples` | required | Number of patients |
| `n_visits` | `5` | Average number of visits per patient |

```python
preset = LongitudinalHealthPreset(random_state=42)
longitudinal_df = preset.generate(n_samples=200, n_visits=8)
```

---

### `RareDiseasePreset`

```python
from calm_data_generator.presets import RareDiseasePreset
```

Simulates a clinical cohort with a rare disease condition. Enforces very low disease prevalence.

**`generate(n_samples, disease_ratio=0.01, **kwargs)`**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_samples` | required | Total number of subjects |
| `disease_ratio` | `0.01` | Disease prevalence (1% default) |

Returns a `Dict[str, pd.DataFrame]` with multiple layers (clinical, omics).

```python
preset = RareDiseasePreset(random_state=42, verbose=True)

# 2% disease prevalence
result = preset.generate(n_samples=500, disease_ratio=0.02)
clinical_df = result["clinical"]
```

---

### `OmicsIntegrationPreset`

```python
from calm_data_generator.presets import OmicsIntegrationPreset
```

Generates correlated multi-omics data across three layers: clinical, gene expression, and proteomics.

**`generate(n_samples, n_genes=100, n_proteins=50, **kwargs)`**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_samples` | required | Number of subjects |
| `n_genes` | `100` | Number of gene expression features |
| `n_proteins` | `50` | Number of protein features |

```python
preset = OmicsIntegrationPreset(random_state=42)
result = preset.generate(n_samples=300, n_genes=200, n_proteins=80)
```

---

### `SingleCellQualityPreset`

```python
from calm_data_generator.presets import SingleCellQualityPreset
```

Generates high-quality single-cell RNA-seq data using scVI (Single-Cell Variational Inference). State-of-the-art for high-dimensional count data.

**`generate(data, n_samples, **kwargs)`**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `data` | required | Real single-cell DataFrame or AnnData |
| `n_samples` | required | Number of synthetic cells |
| `auto_report` | `True` | Enable quality reporting |

Key internal config: `method="scvi"`, `epochs=400`, `n_latent=10`. Additional kwargs are forwarded.

```python
preset = SingleCellQualityPreset(random_state=42)
synthetic_cells = preset.generate(data=adata_df, n_samples=500)
```

---

## Import Summary

```python
from calm_data_generator.presets import (
    # Speed
    FastPreset,
    FastPrototypePreset,
    # Quality
    HighFidelityPreset,
    DiffusionPreset,
    CopulaPreset,
    DataQualityAuditPreset,
    # Class distribution
    ImbalancedGeneratorPreset,
    BalancedDataGeneratorPreset,
    # Time series
    TimeSeriesPreset,
    SeasonalTimeSeriesPreset,
    # Drift & scenarios
    DriftScenarioPreset,
    GradualDriftPreset,
    ConceptDriftPreset,
    ScenarioInjectorPreset,
    # Clinical / omics
    LongitudinalHealthPreset,
    RareDiseasePreset,
    OmicsIntegrationPreset,
    SingleCellQualityPreset,
)
```
