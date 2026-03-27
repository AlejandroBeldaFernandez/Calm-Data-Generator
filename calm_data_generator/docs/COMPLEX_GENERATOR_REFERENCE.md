# ComplexGenerator - Reference

**Location:** `calm_data_generator.generators.complex.ComplexGenerator`

`ComplexGenerator` is an abstract intermediate layer between `BaseGenerator` and domain-specific generators (Clinical, Finance, IoT, Insurance). It exposes three reusable mathematical engines based on Gaussian Copula theory and stochastic effect modelling, so that new domain generators can inherit them without duplicating code.

---

## Inheritance Hierarchy

```
BaseGenerator (ABC)
└── ComplexGenerator          ← mathematical engines (this file)
    └── ClinicalDataGenerator ← clinical domain
    └── FinanceGenerator      ← (example: user-defined)
    └── IoTGenerator          ← (example: user-defined)
```

---

## Quick Start: Creating a Domain Generator

```python
from calm_data_generator.generators.complex.ComplexGenerator import ComplexGenerator
import scipy.stats as stats
import numpy as np
import pandas as pd

class FinanceGenerator(ComplexGenerator):
    def generate(self, n_samples, n_assets, correlation_matrix=None, stress_events=None):
        if correlation_matrix is None:
            correlation_matrix = np.identity(n_assets)

        marginals = [stats.norm(loc=0.001, scale=0.02) for _ in range(n_assets)]
        X = self._generate_correlated_module(n_samples, marginals, correlation_matrix)
        df = pd.DataFrame(X, columns=[f"ASSET_{i}" for i in range(n_assets)])

        if stress_events:
            for event in stress_events:
                self.apply_stochastic_effects(df, df.index[event["rows"]], event)

        return df
```

---

## Engine 1: `_generate_correlated_module`

Generates correlated data using an unconditional Gaussian Copula. Handles non-PSD correlation matrices automatically via eigenvalue repair.

### Signature

```python
def _generate_correlated_module(
    self,
    n_samples: int,
    marginals_list: list,      # list of scipy frozen rv objects
    sigma_module: np.ndarray,  # correlation matrix (n_vars x n_vars)
) -> np.ndarray:               # shape: (n_samples, n_vars)
```

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `n_samples` | int | Number of samples to generate |
| `marginals_list` | list | Scipy frozen distributions — one per variable |
| `sigma_module` | np.ndarray | Correlation matrix. Non-PSD matrices are repaired via `scipy.linalg.eigh` |

### How it works

1. Samples a multivariate normal `Z ~ N(0, sigma)` in latent space
2. Applies `U = Phi(Z)` (standard normal CDF) to obtain uniform marginals on [0,1]
3. Applies `X_i = F_i^{-1}(U_i)` (PPF of each marginal) to get target-distributed output

### Example

```python
from scipy import stats

gen = MyGenerator()
marginals = [
    stats.norm(loc=5, scale=1),       # variable 1: normal
    stats.expon(scale=2),             # variable 2: exponential
    stats.lognorm(s=0.5, scale=10),   # variable 3: lognormal
]
correlation = np.array([[1.0, 0.6, 0.3],
                         [0.6, 1.0, 0.4],
                         [0.3, 0.4, 1.0]])

X = gen._generate_correlated_module(500, marginals, correlation)
# X.shape == (500, 3)
```

---

## Engine 2: `_generate_conditional_data`

Generates target variables conditioned on already-observed data using a conditional Gaussian Copula. Supports both continuous and discrete conditioning marginals (via Randomized Quantile Residuals).

### Signature

```python
def _generate_conditional_data(
    self,
    n_samples: int,
    conditioning_data: np.ndarray,      # (n_samples, n_cond)
    conditioning_marginals: list,        # marginals for conditioning vars
    target_marginals: list,              # marginals for target vars
    full_covariance: np.ndarray,         # (n_cond + n_target, n_cond + n_target)
) -> np.ndarray:                         # shape: (n_samples, n_target)
```

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `n_samples` | int | Number of samples |
| `conditioning_data` | np.ndarray | Observed data to condition on, shape `(n_samples, n_cond)` |
| `conditioning_marginals` | list | Marginals for conditioning variables. Discrete distributions (binom, poisson, etc.) are handled via RQR jittering |
| `target_marginals` | list | Marginals for variables to generate |
| `full_covariance` | np.ndarray | Joint covariance over all variables (conditioning + target) |

### How it works

1. Transforms `conditioning_data` to latent space using marginal CDFs (with RQR for discrete)
2. Partitions the covariance matrix into `S_cc`, `S_ct`, `S_tc`, `S_tt`
3. Computes conditional mean and covariance: `mu_t|c = S_tc S_cc^{-1} Z_c`, `Sigma_t|c = S_tt - S_tc S_cc^{-1} S_ct`
4. Samples `Z_target ~ N(mu_t|c, Sigma_t|c)` per sample
5. Applies target marginal PPFs to get final values

### Example: Generate gene expression conditioned on demographics

```python
demographic_data = df[["age", "bmi"]].values  # (100, 2)
demographic_marginals = [stats.norm(50, 10), stats.norm(25, 4)]
gene_marginals = [stats.lognorm(s=0.5) for _ in range(20)]
joint_cov = np.eye(22)  # 2 demo + 20 genes

X_genes = gen._generate_conditional_data(
    n_samples=100,
    conditioning_data=demographic_data,
    conditioning_marginals=demographic_marginals,
    target_marginals=gene_marginals,
    full_covariance=joint_cov,
)
# X_genes.shape == (100, 20)
```

---

## Engine 3: `apply_stochastic_effects`

Applies a single stochastic effect to a subset of entities in-place. Supports 7 effect types for modelling disease signals, market shocks, sensor anomalies, etc.

### Signature

```python
def apply_stochastic_effects(
    self,
    df: pd.DataFrame,     # modified in-place
    entity_ids,           # index labels of entities to affect
    effect_config: dict,  # see below
) -> None:
```

### Effect Configuration

```python
effect_config = {
    "index": [0, 1, 5],          # column indices to affect (int, list, or slice)
    "effect_type": "fold_change", # one of the 7 types below
    "effect_value": 2.0,          # scalar, [low, high] list, or dict (sigmoid only)
}
```

When `effect_value` is a list `[low, high]`, each entity receives an independent sample from `Uniform(low, high)`. When it is a scalar, entities receive independent samples from `Normal(value, |value| * 0.1)`.

### Supported Effect Types

| Effect Type | Formula | Use Case |
|-------------|---------|----------|
| `additive_shift` | `x += offset` | Background signal, sensor bias |
| `fold_change` | `x *= factor` | Gene overexpression, price multiplier |
| `power_transform` | `x **= exponent` | Non-linear distortion |
| `variance_scale` | Rescales around mean | Heteroscedasticity, volatility regimes |
| `log_transform` | `x = log(x + eps)` | Log-normalisation |
| `polynomial_transform` | `x = P(x)` | Polynomial mapping (coefficients in value) |
| `sigmoid_transform` | `x = 1/(1+e^{-k(x-x0)})` | Saturation / clipping |

`simple_additive_shift` is accepted as an alias for `additive_shift`.

### Example: Stress shock on asset returns

```python
stress_event = {
    "index": slice(0, 5),         # affect first 5 assets
    "effect_type": "additive_shift",
    "effect_value": [-0.1, -0.02], # random loss in [-10%, -2%]
}
affected_rows = df.index[:20]     # first 20 time steps
gen.apply_stochastic_effects(df, affected_rows, stress_event)
```

### Notes

- Modifies `df` **in-place** (no return value)
- Passing an empty `entity_ids` is a safe no-op
- Unknown `effect_type` raises `ValueError`

---

## Error Handling

| Situation | Behaviour |
|-----------|-----------|
| Non-PSD `sigma_module` | Repaired via eigenvalue clipping (no exception) |
| Singular `S_cc` in conditional | Regularised with `+1e-6 * I` |
| Non-PSD conditional covariance | Repaired via eigenvalue clipping |
| Shape mismatch in `_generate_conditional_data` | Raises `ValueError` |
| Unknown `effect_type` | Raises `ValueError` |
| Empty `entity_ids` | No-op, returns immediately |
