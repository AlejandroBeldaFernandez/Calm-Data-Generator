# ClinicalDataGenerator - Complete Reference

**Location:** `calm_data_generator.generators.clinical.ClinicalDataGenerator`

The `ClinicalDataGenerator` is a high-fidelity simulator for multi-modal healthcare datasets.

---

## Quick Start Guide

### What is ClinicalDataGenerator?

A specialized generator for **clinical/medical research data** that creates realistic multi-modal datasets including:
- 👥 **Patient Demographics** (age, gender, BMI, etc.)
- 🧬 **Omics Data** (gene expression, proteins)
- 📊 **Longitudinal Records** (multi-visit trajectories)
- �� **Disease Effects** (biomarkers, treatment responses)

### When to Use ClinicalDataGenerator

✅ **Use ClinicalDataGenerator when:**
- You need **clinical trial** or **medical research** data
- Working with **omics data** (RNA-Seq, Microarray, Proteomics)
- Simulating **disease vs control** studies
- Creating **longitudinal patient** trajectories
- Testing **biomarker discovery** algorithms
- Need **correlated demographics** (age, BMI, blood pressure)

❌ **Don't use ClinicalDataGenerator when:**
- You have **simple tabular data** → Use `RealGenerator` instead
- You need **general-purpose** synthetic data → Use `RealGenerator`
- You don't need **multi-modal** structure → Use `RealGenerator`

### Basic Usage (3 Lines)

```python
from calm_data_generator.generators.clinical import ClinicalDataGenerator

gen = ClinicalDataGenerator()
data = gen.generate(n_samples=100, n_genes=500, n_proteins=200)
# Returns: {"demographics": DataFrame, "genes": DataFrame, "proteins": DataFrame}
```

### Common Use Cases

| Scenario | Method | Key Parameters |
|----------|--------|----------------|
| **Static cohort** (single timepoint) | `generate()` | `n_samples`, `n_genes`, `n_proteins` |
| **Longitudinal study** (multiple visits) | `generate_longitudinal_data()` | `longitudinal_config` |
| **Biomarker simulation** (disease effects) | `generate()` | `disease_effects_config` |
| **Population diversity** (correlated demographics) | `generate()` | `demographic_correlations` |

---

The `ClinicalDataGenerator` is a high-fidelity simulator for multi-modal healthcare datasets. It orchestrates the generation of:
1.  **Patient Demographics**: Age, gender, BMI, etc., with inter-dependencies.
2.  **Omics Data**: Gene expression (RNA-Seq/Microarray) and proteins, correlated with demographics.
3.  **Longitudinal Records**: Multi-visit trajectories.

---

## Initialization

```python
from calm_data_generator.generators.clinical import ClinicalDataGenerator

gen = ClinicalDataGenerator(
    seed=42,                # Seed for reproducibility
    auto_report=True,       # Generate reports automatically
    minimal_report=False    # Full detailed reports
)
```

## Main Method: `generate()`

Generates a static cohort (single timepoint) with demographics and omics data.

```python
from calm_data_generator.generators.configs import DateConfig, DriftConfig

data = gen.generate(
    n_samples=100,
    n_genes=500,
    n_proteins=200,
    control_disease_ratio=0.5,
    date_config=DateConfig(start_date="2024-01-01"),
    
    # Drift Configuration (using DriftConfig objects)
    demographics_drift_config=[
        DriftConfig(method="inject_feature_drift", feature_cols=["Age"], magnitude=0.5)
    ],
    
    # Detailed configurations
    demographic_correlations=None,
    gene_correlations=None,
    disease_effects_config=[...],
    custom_demographic_columns={...}
)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `n_samples` | int | 100 | Number of patients (samples) |
| `n_genes` | int | 200 | Number of gene variables |
| `n_proteins` | int | 50 | Number of protein variables |
| `control_disease_ratio` | float | 0.5 | Proportion of "Control" group (0-1) |
| `gene_type` | str | "RNA-Seq" | "RNA-Seq" (integers) or "Microarray" (floats) |
| `demographic_correlations` | array | None | Custom NxN correlation matrix for demographics |
| `custom_demographic_columns`| dict | None | Definitions for custom features (see Use Cases) |
| `disease_effects_config` | list | None | List of effect definitions (see below) |
| `date_config` | DateConfig | None | Settings for the `timestamp` column |

### Return Structure

Returns a dictionary `Dict[str, pd.DataFrame]` with keys:
*   `"demographics"`: Patient metadata (ID, Group, Age, Gender, etc.)
*   `"genes"`: Expression matrix (Rows=Patients, Cols=Genes)
*   `"proteins"`: Expression matrix (Rows=Patients, Cols=Proteins)

---

## Disease Effects Configuration

The `disease_effects_config` allows precise control over biological signals. You can modify specific genes/proteins for the "Disease" group using various mathematical transformations.

### Configuration Format

```python
{
    "target_type": "gene",          # "gene" or "protein"
    "index": [0, 5, 12],            # Integer or List of indices to affect
    "effect_type": "fold_change",   # Type of transformation (see table)
    "effect_value": 2.0,            # Magnitude of the effect
    "group": "Disease"              # Target group (usually "Disease")
}
```

### Supported Effect Types

| Effect Type | Formula | Description |
|-------------|---------|-------------|
| `fold_change` | $x_{new} = x \cdot value$ | Multiplicative scaling (e.g., overexpression) |
| `additive_shift` | $x_{new} = x + value$ | Adds constant background signal |
| `power_transform` | $x_{new} = x^{value}$ | Non-linear distortion |
| `log_transform` | $x_{new} = \ln(x + \epsilon)$ | Logarithmic normalization |
| `variance_scale` | $x_{new} = \mu + (x-\mu)\cdot value$ | Increases/decreases spread |
| `polynomial_transform`| $x_{new} = P(x)$ | Polynomial mapping (coeffs in value) |
| `sigmoid_transform` | $x_{new} = \frac{1}{1 + e^{-k(x-x_0)}}$ | S-curve saturation |

---

## Longitudinal Data: `generate_longitudinal_data()`

Generates multi-visit data (trajectories).

```python
longitudinal_data = gen.generate_longitudinal_data(
    n_samples=50,
    longitudinal_config={
        "n_visits": 5,          # Total number of visits per patient
        "time_step_days": 30,   # Average days between visits
    },
    # Standard generate() args
    n_genes=100
)
```

---

## Advanced Configuration

### Drift & Dynamics Injection

You can pass configuration dictionaries directly to the internal injectors:

*   `demographics_drift_config`: List of `DriftConfig` objects for demographics.
*   `genes_drift_config`: List of `DriftConfig` objects for genes.
*   `proteins_drift_config`: List of `DriftConfig` objects for proteins.
*   `genes_dynamics_config`: Scenarios for gene evolution.

Example:
```python
drift_conf = DriftConfig(
    method="inject_feature_drift_gradual", 
    params={"feature_cols": ["Age"], "drift_magnitude": 0.5}
)

gen.generate(..., demographics_drift_config=[drift_conf])
```

---

## Comprehensive Use Cases

### Case 1: Biomarker Discovery Simulation

**Scenario:** You want to simulate a clinical trial where 5 specific genes are highly upregulated in disease patients.

**Solution:** Use `fold_change` effect.

```python
# Upregulate first 5 genes by 4x in Disease group
biomarker_config = [{
    "target_type": "gene",
    "index": [0, 1, 2, 3, 4],
    "effect_type": "fold_change",
    "effect_value": 4.0,
    "group": "Disease"
}]

data = gen.generate(
    n_samples=200,
    n_genes=1000,
    control_disease_ratio=0.5,
    disease_effects_config=biomarker_config
)
```

### Case 2: Longitudinal Disease Progression

**Scenario:** Simulating Alzheimer's progression where a protein level decays over time.

```python
cohort = gen.generate_longitudinal_data(
    n_samples=100,
    longitudinal_config={
        "n_visits": 12,        # 1 year of monthly data
        "time_step_days": 30
    },
    n_proteins=50
)
# Returns a dictionary containing longitudinal data structures
```

### Case 3: Diverse Population Modeling

**Scenario:** Generating a study with complex demographic correlations (e.g., Age highly correlated with BMI).

**Solution:** Inject a custom `demographic_correlations` matrix.

```python
import numpy as np

# 3x3 matrix: [Age, BMI, BloodPressure]
# High correlation (0.8) between Age and BMI
corr_matrix = np.array([
    [1.0, 0.8, 0.5],
    [0.8, 1.0, 0.4],
    [0.5, 0.4, 1.0]
])

data = gen.generate(
    n_samples=500,
    custom_demographic_columns={
        "Age": {"dist": "normal", "loc": 60, "scale": 10},
        "BMI": {"dist": "normal", "loc": 25, "scale": 5},
        "BP":  {"dist": "normal", "loc": 120, "scale": 15}
    },
    demographic_correlations=corr_matrix
)
```
