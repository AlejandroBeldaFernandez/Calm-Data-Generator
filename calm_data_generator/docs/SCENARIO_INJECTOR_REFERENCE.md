# ScenarioInjector - Complete Reference

**Location:** `calm_data_generator.generators.dynamics.ScenarioInjector`

The `ScenarioInjector` simulates **temporal dynamics** and **evolving patterns** in synthetic datasets. Unlike `DriftInjector` which modifies distributions, `ScenarioInjector` creates deterministic or stochastic patterns of evolution (how features change over time) and constructs target variables based on logic.

---

## ⚡ Quick Start: Using Config Objects

We recommend using `ScenarioConfig` and `EvolutionFeatureConfig` for type safety.

```python
from calm_data_generator.generators.configs import ScenarioConfig, EvolutionFeatureConfig

# 1. Define feature evolution (e.g., revenue grows, interest decays)
scenario_conf = ScenarioConfig(
    evolve_features={
        "revenue": EvolutionFeatureConfig(type="trend", slope=100.0),
        "interest": EvolutionFeatureConfig(type="decay", rate=0.01)
    },
    # 2. Construct target based on evolved features
    construct_target={
        "target_col": "high_value_customer",
        "formula": "0.4 * revenue - 100 * interest",
        "threshold": 0.8
    }
)

# 3. Apply to DataFrame (or use inside RealGenerator with dynamics_config)
# Via RealGenerator:
# gen.generate(..., dynamics_config=scenario_conf)

# Via Direct Injection:
from calm_data_generator.generators.dynamics import ScenarioInjector
injector = ScenarioInjector()
df_evolved = injector.apply_config(df, scenario_conf)
```

---

## 🌲 Decision Tree: Usage Guide

```text
What do you want to do?
├─ Make values change over time (Growth, Seasonality)?
│  └─ → inject_feature_evolution() (or ScenarioConfig.evolve_features)
├─ Create a Target variable from Features?
│  └─ → construct_target_from_features() (or ScenarioConfig.construct_target)
├─ Project historical data into the future?
│  └─ → project_to_future_period()
└─ Change distribution properties (Mean shift, Noise)?
   └─ → Use DriftInjector instead.
```

---

## 📚 Evolution Types (`type`)

| Type | Pattern | Use Case | Formula |
|------|---------|----------|---------|
| `trend` / `linear` | Steady change | Sales growth, inflation | `y = x + slope * t` |
| `exponential_growth` | Accelerating increase | Viral growth | `y = x * (1 + rate)^t` |
| `decay` | Decreasing values | Retention loss, radioactivity | `y = x * (1 - rate)^t` |
| `seasonal` | Cyclical pattern | Holidays, weather, daily cycles | `y = x + A * sin(2πt/P)` |
| `step` | Sudden jump | Policy change, price hike | `y = x + value if t > step` |
| `noise` | Random fluctuation | Sensor error, market noise | `y = x + N(0, scale)` |

---

## 🛠️ ScenarioInjector Class Reference

**Import:** `from calm_data_generator.generators.dynamics import ScenarioInjector`

### Method: `evolve_features()`

Evolves numeric columns based on configured patterns.

```python
evolved_df = injector.evolve_features(
    df=df,
    evolution_config={
        "price": {"type": "trend", "slope": 0.01},          # Linear growth
        "demand": {"type": "seasonal", "amplitude": 10, "period": 30} # Monthly cycle
    },
    time_col="date"  # Optional: use date column for time steps
)
```

### Method: `construct_target()`

Creates a target variable based on feature logic. Useful for creating ground truth for synthetic scenarios.

```python
# String Formula
df = injector.construct_target(
    df=df,
    target_col="risk_score",
    formula="0.3 * age + 0.5 * bmi - 0.2 * exercise",
    noise_std=0.1  # Add noise to make it realistic
)

# Python Function (Callable)
def complex_logic(row):
    return 1 if (row["age"] > 50 and row["income"] > 100000) else 0

df = injector.construct_target(
    df=df,
    target_col="is_vip",
    formula=complex_logic
)
```

### Method: `project_to_future_period()`

Extends a dataset into the future by generating new samples and applying evolution.

```python
future_df = injector.project_to_future_period(
    df=historical_df,
    periods=12,                   # Generate 12 future steps (e.g., months)
    time_col="month",
    evolution_config={...},       # Apply trends to future data
    n_samples_per_period=100
)
```

---

## 🌟 Real-World Scenarios

### Case 1: SaaS Growth (Viral + Churn)
Simulate a startup with viral user growth but increasing churn.

```python
scenario_conf = ScenarioConfig(
    evolve_features={
        "users": EvolutionFeatureConfig(type="exponential_growth", rate=0.1), # 10% daily growth
        "churn": EvolutionFeatureConfig(type="trend", slope=0.001)           # Churn slowly creeps up
    }
)
```

### Case 2: Retail Seasonality
Simulate holiday sales spikes.

```python
# Yearly cycle with peak at end of year
seasonal_conf = EvolutionFeatureConfig(
    type="seasonal",
    amplitude=5000,
    period=365,
    phase=300 # Shift peak to ~Day 300 (Nov/Dec)
)
```

### Case 3: Credit Scoring (Ground Truth Generation)
Create a dataset where you KNOW the exact relationship between inputs and target.

```python
# We define the ground truth mechanism:
# Risk = 2 * Debt - 0.5 * Income + Noise
injector.construct_target(
    df=data,
    target_col="default_probability",
    formula="2 * debt_ratio - 0.5 * normalized_income",
    noise_std=0.05
)
```
