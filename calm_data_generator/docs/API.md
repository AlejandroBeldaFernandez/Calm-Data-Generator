# CalmGenerator API Documentation

## Modules Overview

### generators.tabular - Real Data Synthesis

```python
from calm_data_generator.generators.tabular import RealGenerator, QualityReporter
```

**RealGenerator** - Generate synthetic data from real datasets

| Method | Description |
|--------|-------------|
| `cart` | CART-based iterative synthesis |
| `rf` | Random Forest synthesis |
| `lgbm` | LightGBM synthesis |
| `ctgan` | CTGAN (deep learning) |
| `tvae` | TVAE (variational autoencoder) |
| `bn` | Bayesian Network (causal structure) |
| `smote` | SMOTE oversampling |
| `adasyn` | ADASYN adaptive sampling |
| `timegan` | TimeGAN (time series) |
| `timevae` | TimeVAE (time series VAE) |
| `fflows` | FourierFlows (periodic time series) |
| `scvi` | scVI (Single-Cell VI) |
| `ddpm` | Tabular Diffusion (DDPM) |

**New Parameters (v1.2.0):**
- `differentiation_factor` (float): Enhances class separation in latent space (TVAE/scVI only).
- `clipping_mode` (str): `'strict'`, `'permissive'`, or `'none'` for handling output ranges.
- `use_latent_sampling` (bool): For scVI, sample from real data latent space.


---

### generators.clinical - Clinical Data

```python
from calm_data_generator.generators.clinical import ClinicalDataGenerator
```

**Methods:**
- `generate()` - Generate demographics + omics
- `generate_longitudinal_data()` - Multi-visit patient data

---

### generators.stream - Stream-Based

```python
from calm_data_generator.generators.stream import StreamGenerator
```

**Features:**
- River library compatible
- Balanced generation
- SMOTE post-hoc
- Sequence generation

---

### generators.drift - Drift Injection

```python
from calm_data_generator.generators.drift import DriftInjector
```

**Drift Types:**
- `inject_drift()` **(Unified)**
- `inject_feature_drift_gradual()`
- `inject_feature_drift_abrupt()`
- `inject_feature_drift_recurrent()`
- `inject_label_drift_gradual()`
- `inject_label_drift_abrupt()`
- `inject_label_drift_incremental()`
- `inject_concept_drift()`
- `inject_conditional_drift()`
- `inject_outliers_global()`
- `inject_new_category_drift()`
- `inject_correlation_matrix_drift()`
- `inject_binary_probabilistic_drift()`
- `inject_multiple_types_of_drift()`

---

### generators.dynamics - Scenario Evolution

```python
from calm_data_generator.generators.dynamics import ScenarioInjector
```

**Methods:**
- `evolve_features()` - Apply trends/cycles
- `construct_target()` - Create target variables
- `project_to_future_period()` - Future data

---

### privacy - Privacy Transformations (Integrated)

Privacy features are integrated into the `QualityReporter`. You can assess quality and privacy using:

```python
# Comprehensive Quality Report (including ARI metrics for class separability)
reporter.generate_comprehensive_report(..., privacy_check=True)

# Standalone ARI calculation
ari_scores = reporter.calculate_ari(real_df, synthetic_df, target_col="label")
```

Or use standalone transformations for manual protection:
- `pseudonymize_columns`
- `add_laplace_noise`
- `shuffle_columns`

---

## Installation

```bash
# Basic
pip install calm-data-generator

# Stream (River)
pip install calm-data-generator[stream]

# Full
pip install calm-data-generator[full]
```

> [!NOTE]
> **Privacy Features**: Privacy assessment (DCR metrics) is now integrated into `QualityReporter`. Use `privacy_check=True` when generating reports.
