# Documentación API de CalmGenerador

## Descripción General de Módulos

### generators.tabular - Síntesis de Datos Reales

```python
from calm_data_generator.generators.tabular import RealGenerator, QualityReporter
```

**RealGenerator** - Genera datos sintéticos a partir de datasets reales

| Método | Descripción |
|--------|-------------|
| `cart` | Síntesis iterativa basada en CART |
| `rf` | Síntesis con Random Forest |
| `lgbm` | Síntesis con LightGBM |
| `ctgan` | CTGAN (deep learning) |
| `tvae` | TVAE (autoencoder variacional) |
| `bn` | Red Bayesiana (estructura causal) |
| `smote` | Sobremuestreo SMOTE |
| `adasyn` | Muestreo adaptativo ADASYN |
| `diffusion` | Difusión Tabular (DDPM) |
| `ddpm` | Synthcity TabDDPM (avanzado) |
| `timegan` | TimeGAN (series temporales) |
| `timevae` | TimeVAE (series temporales) |
| `fflows` | FourierFlows (series periódicas) |
| `scvi` | scVI (Single-Cell VI) |

**Nuevos Parámetros (v1.2.0):**
- `differentiation_factor` (float): Aumenta la separación de clases en el espacio latente (solo TVAE/scVI).
- `clipping_mode` (str): `'strict'`, `'permissive'`, o `'none'` para manejar los rangos de salida.
- `use_latent_sampling` (bool): Para scVI, muestrea desde el espacio latente de datos reales.




---

### generators.clinical - Datos Clínicos

```python
from calm_data_generator.generators.clinical import ClinicalDataGenerator
```

**Métodos:**
- `generate()` - Genera demografía + ómicas
- `generate_longitudinal_data()` - Datos de paciente multi-visita

---

### generators.stream - Basado en Stream

```python
from calm_data_generator.generators.stream import StreamGenerator
```

**Características:**
- Compatible con librería River
- Generación balanceada
- SMOTE post-hoc
- Generación de secuencias

---

### generators.drift - Inyección de Drift

```python
from calm_data_generator.generators.drift import DriftInjector
```

**Tipos de Drift:**
- `inject_drift()` **(Unificado)**
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

### generators.dynamics - Evolución de Escenarios

```python
from calm_data_generator.generators.dynamics import ScenarioInjector
```

**Métodos:**
- `evolve_features()` - Aplica tendencias/ciclos
- `construct_target()` - Crea variables objetivo
- `project_to_future_period()` - Datos futuros

---

### privacy - Transformaciones de Privacidad (Integrado)

Las funciones de privacidad están integradas en el `QualityReporter`. Puedes evaluar calidad y privacidad usando:

```python
# Reporte Completo de Calidad (incluyendo métricas ARI para separabilidad)
reporter.generate_comprehensive_report(..., privacy_check=True)

# Cálculo de ARI independiente
ari_scores = reporter.calculate_ari(real_df, synthetic_df, target_col="label")
```

O usar transformaciones manuales para protección:
- `pseudonymize_columns`
- `add_laplace_noise`
- `shuffle_columns`

## Instalación

```bash
# Básica
pip install calm-data-generator

# Stream (River)
pip install "calm-data-generator[stream]"

# Completa
pip install "calm-data-generator[full]"
```
