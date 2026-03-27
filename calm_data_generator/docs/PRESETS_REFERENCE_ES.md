# Referencia de Presets

Los presets son configuraciones de generador listas para usar que encapsulan selección de método, hiperparámetros y reportes para los escenarios de datos sintéticos más comunes.

## Clase Base

```python
from calm_data_generator.presets import GeneratorPreset  # abstracta
```

Todos los presets heredan de `GeneratorPreset` y comparten tres parámetros de constructor:

| Parámetro | Tipo | Por defecto | Descripción |
|-----------|------|-------------|-------------|
| `random_state` | `int` | `42` | Semilla aleatoria para reproducibilidad |
| `verbose` | `bool` | `True` | Mostrar mensajes de progreso |
| `fast_dev_run` | `bool` | `False` | Iteraciones/épocas mínimas — para probar pipelines |

Todos los presets exponen un único método `.generate()`. Los parámetros varían por preset (ver secciones a continuación).

---

## Velocidad y Prototipado

### `FastPreset`

```python
from calm_data_generator.presets import FastPreset
```

Generación general más rápida. Usa LightGBM con 10 iteraciones y reenvía kwargs adicionales al generador subyacente.

**`generate(data, n_samples, **kwargs)`**

| Parámetro | Por defecto | Descripción |
|-----------|-------------|-------------|
| `data` | requerido | DataFrame real del que aprender |
| `n_samples` | requerido | Número de filas sintéticas |
| `iterations` | `10` | Iteraciones de ajuste de LightGBM |
| `auto_report` | `False` | Activar reporte de calidad |

```python
preset = FastPreset(random_state=42)
synthetic_df = preset.generate(data=real_df, n_samples=1000)

# Sobreescribir iteraciones
synthetic_df = preset.generate(data=real_df, n_samples=1000, iterations=20)
```

---

### `FastPrototypePreset`

```python
from calm_data_generator.presets import FastPrototypePreset
```

Optimizado para pipelines CI/CD y tests de integración. Usa LightGBM con 10 iteraciones fijas (1 si `fast_dev_run=True`). **No** reenvía kwargs al generador subyacente.

**`generate(data, n_samples, **kwargs)`**

| Parámetro | Por defecto | Descripción |
|-----------|-------------|-------------|
| `data` | requerido | DataFrame real del que aprender |
| `n_samples` | requerido | Número de filas sintéticas |

```python
# Test de integración — usa fast_dev_run=True para 1 iteración
preset = FastPrototypePreset(fast_dev_run=True)
synthetic_df = preset.generate(data=real_df, n_samples=100)
```

---

## Calidad y Fidelidad

### `HighFidelityPreset`

```python
from calm_data_generator.presets import HighFidelityPreset
```

Máxima calidad para datos de producción. Usa CTGAN con 1000 épocas, batch size 250 y validación adversarial.

**`generate(data, n_samples, auto_report=True)`**

| Parámetro | Por defecto | Descripción |
|-----------|-------------|-------------|
| `data` | requerido | DataFrame real |
| `n_samples` | requerido | Número de filas sintéticas |
| `auto_report` | `True` | Activar reporte de calidad |

```python
preset = HighFidelityPreset(random_state=42)
synthetic_df = preset.generate(data=real_df, n_samples=5000)
```

---

### `DiffusionPreset`

```python
from calm_data_generator.presets import DiffusionPreset
```

Usa Tabular DDPM (modelo de difusión probabilístico de eliminación de ruido) de SynthCity. Captura distribuciones multimodales complejas mejor que los GANs. Más lento pero con mayor fidelidad estructural.

**`generate(data, n_samples, **kwargs)`**

| Parámetro | Por defecto | Descripción |
|-----------|-------------|-------------|
| `data` | requerido | DataFrame real |
| `n_samples` | requerido | Número de filas sintéticas |
| `auto_report` | `True` | Activar reporte de calidad |

Configuración interna clave: `method="ddpm"`, `n_steps=1000` (2 con `fast_dev_run`), `batch_size=256`.

```python
preset = DiffusionPreset(random_state=42)
synthetic_df = preset.generate(data=real_df, n_samples=2000)
```

---

### `CopulaPreset`

```python
from calm_data_generator.presets import CopulaPreset
```

Línea base rápida y estadísticamente robusta usando una Cópula Gaussiana para modelar dependencias. Ideal como benchmark antes de usar modelos más pesados.

**`generate(data, n_samples, **kwargs)`**

| Parámetro | Por defecto | Descripción |
|-----------|-------------|-------------|
| `data` | requerido | DataFrame real |
| `n_samples` | requerido | Número de filas sintéticas |
| `auto_report` | `True` | Activar reporte de calidad |

```python
preset = CopulaPreset(random_state=42)
baseline_df = preset.generate(data=real_df, n_samples=1000)
```

---

### `DataQualityAuditPreset`

```python
from calm_data_generator.presets import DataQualityAuditPreset
```

Combina síntesis TVAE con un reporte de calidad completo forzado (`auto_report=True`, `minimal_report=False`). Usar cuando se necesita una auditoría completa de fidelidad junto con los datos sintéticos.

**`generate(data, n_samples, **kwargs)`**

| Parámetro | Por defecto | Descripción |
|-----------|-------------|-------------|
| `data` | requerido | DataFrame real |
| `n_samples` | requerido | Número de filas sintéticas |

Configuración interna clave: `method="tvae"`, `epochs=300` (1 con `fast_dev_run`), reporte completo siempre activado.

```python
preset = DataQualityAuditPreset(random_state=42)
synthetic_df = preset.generate(data=real_df, n_samples=3000)
# Un reporte HTML completo de calidad se guarda automáticamente en disco
```

---

## Distribución de Clases

### `ImbalancedGeneratorPreset`

```python
from calm_data_generator.presets import ImbalancedGeneratorPreset
```

Genera datos sintéticos con una proporción controlada de clases minoritaria/mayoritaria. Útil para crear benchmarks desbalanceados para detección de drift y análisis de sesgos.

**`generate(data, n_samples, target_col, imbalance_ratio=0.1, **kwargs)`**

| Parámetro | Por defecto | Descripción |
|-----------|-------------|-------------|
| `data` | requerido | DataFrame real (requiere target binario) |
| `n_samples` | requerido | Número de filas sintéticas |
| `target_col` | requerido | Columna a desbalancear |
| `imbalance_ratio` | `0.1` | Fracción de la clase minoritaria (0–1) |

Configuración interna clave: `method="ctgan"`, `epochs=300`.

```python
preset = ImbalancedGeneratorPreset(random_state=42)

# 5% clase minoritaria
synthetic_df = preset.generate(
    data=real_df, n_samples=2000,
    target_col="fraude", imbalance_ratio=0.05
)
```

> **Nota**: Actualmente solo soporta targets binarios.

---

### `BalancedDataGeneratorPreset`

```python
from calm_data_generator.presets import BalancedDataGeneratorPreset
```

Balancea un dataset desbalanceado usando sobremuestreo SMOTE. Genera muestras sintéticas de la clase minoritaria hasta alcanzar una distribución 50/50.

**`generate(data, n_samples, target_col, **kwargs)`**

| Parámetro | Por defecto | Descripción |
|-----------|-------------|-------------|
| `data` | requerido | DataFrame real (desbalanceado) |
| `n_samples` | requerido | Total de filas sintéticas |
| `target_col` | requerido | Columna a balancear |

```python
preset = BalancedDataGeneratorPreset(random_state=42)
balanced_df = preset.generate(data=imbalanced_df, n_samples=2000, target_col="etiqueta")
```

---

## Series Temporales

### `TimeSeriesPreset`

```python
from calm_data_generator.presets import TimeSeriesPreset
```

Genera datos secuenciales/temporales usando modelos dedicados de series temporales. Soporta tres backends:

- `timegan` — mejor para patrones temporales complejos e irregulares
- `timevae` — más rápido, bueno para series regulares
- `fflows` — más estable, mejor para datos periódicos/estacionales

**`generate(data, n_samples, sequence_key, time_key=None, method="timegan", **kwargs)`**

| Parámetro | Por defecto | Descripción |
|-----------|-------------|-------------|
| `data` | requerido | DataFrame de series temporales |
| `n_samples` | requerido | Número de secuencias sintéticas |
| `sequence_key` | requerido | Columna que identifica cada entidad/secuencia |
| `time_key` | `None` | Columna con timestamps (opcional) |
| `method` | `"timegan"` | Backend: `"timegan"`, `"timevae"`, `"fflows"` |

Configuración interna clave: `n_iter=500` épocas (1 con `fast_dev_run`), `batch_size=100`.

```python
preset = TimeSeriesPreset(random_state=42)

# TimeGAN (por defecto)
synthetic_df = preset.generate(
    data=ts_df, n_samples=200,
    sequence_key="paciente_id", time_key="fecha_visita"
)

# FourierFlows para datos periódicos
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

Preset de dos etapas: (1) genera secuencias base con TimeGAN, luego (2) superpone un patrón estacional sinusoidal mediante `ScenarioInjector`.

**`generate(data, n_samples, time_col, seasonal_cols, period=12, amplitude=1.0, **kwargs)`**

| Parámetro | Por defecto | Descripción |
|-----------|-------------|-------------|
| `data` | requerido | DataFrame de series temporales |
| `n_samples` | requerido | Número de filas sintéticas |
| `time_col` | requerido | Columna de timestamp |
| `seasonal_cols` | requerido | Columnas donde inyectar estacionalidad |
| `period` | `12` | Periodo de la estacionalidad (ej. 12 = mensual en ciclo anual) |
| `amplitude` | `1.0` | Intensidad de la oscilación estacional |

```python
preset = SeasonalTimeSeriesPreset(random_state=42)
synthetic_df = preset.generate(
    data=ventas_df, n_samples=500,
    time_col="fecha", seasonal_cols=["ventas", "trafico_web"],
    period=12, amplitude=2.5
)
```

---

## Drift y Escenarios

### `DriftScenarioPreset`

```python
from calm_data_generator.presets import DriftScenarioPreset
```

Genera datos con características de drift inyectadas. Útil para pruebas de estrés de sistemas de detección de drift y evaluación de robustez de modelos.

**`generate(data, n_samples, drift_scenarios=None, **kwargs)`**

| Parámetro | Por defecto | Descripción |
|-----------|-------------|-------------|
| `data` | requerido | DataFrame real |
| `n_samples` | requerido | Número de filas sintéticas |
| `drift_scenarios` | `None` | Lista de dicts de configuración de drift (envueltos en `DriftConfig`) |

```python
from calm_data_generator.presets import DriftScenarioPreset

preset = DriftScenarioPreset(random_state=42)

# Con inyección de drift explícita
scenarios = [{"column": "temperatura", "type": "shift_mean", "magnitude": 3.0}]
synthetic_df = preset.generate(data=real_df, n_samples=1000, drift_scenarios=scenarios)
```

---

### `GradualDriftPreset`

```python
from calm_data_generator.presets import GradualDriftPreset
```

Simula drift lineal gradual a lo largo del tiempo/índice en columnas especificadas.

**`generate(data, n_samples, drift_cols, slope=0.01, **kwargs)`**

| Parámetro | Por defecto | Descripción |
|-----------|-------------|-------------|
| `data` | requerido | DataFrame real |
| `n_samples` | requerido | Número de filas sintéticas |
| `drift_cols` | requerido | Lista de columnas a las que aplicar drift |
| `slope` | `0.01` | Tasa de drift lineal por paso |

```python
preset = GradualDriftPreset(random_state=42)
synthetic_df = preset.generate(
    data=real_df, n_samples=1000,
    drift_cols=["temperatura", "humedad"], slope=0.05
)
```

---

### `ConceptDriftPreset`

```python
from calm_data_generator.presets import ConceptDriftPreset
```

Simula drift de concepto repentino alterando la relación P(y|x) entre features y target. Usar para probar la robustez del modelo ante cambios de distribución en el límite de decisión.

**`generate(data, n_samples, target_col, drift_magnitude=0.5, **kwargs)`**

| Parámetro | Por defecto | Descripción |
|-----------|-------------|-------------|
| `data` | requerido | DataFrame real |
| `n_samples` | requerido | Número de filas sintéticas |
| `target_col` | requerido | Columna objetivo a la que aplicar drift de concepto |
| `drift_magnitude` | `0.5` | Grado de alteración de P(y\|x) (0–1) |

```python
preset = ConceptDriftPreset(random_state=42)
synthetic_df = preset.generate(
    data=real_df, n_samples=1000,
    target_col="abandono", drift_magnitude=0.7
)
```

---

### `ScenarioInjectorPreset`

```python
from calm_data_generator.presets import ScenarioInjectorPreset
```

Aplica escenarios de evolución complejos directamente a un dataset existente usando `ScenarioInjector.evolve_features()`. A diferencia de otros presets, este modifica los datos de entrada en lugar de generar muestras nuevas desde cero.

**`generate(data, scenario_config, **kwargs)`**

| Parámetro | Por defecto | Descripción |
|-----------|-------------|-------------|
| `data` | requerido | DataFrame existente a transformar |
| `scenario_config` | requerido | Dict con clave `"evolve_features"` que mapea nombres de columna a objetos `EvolutionFeatureConfig` |

```python
from calm_data_generator.presets import ScenarioInjectorPreset
from calm_data_generator.generators.configs import EvolutionFeatureConfig

scenario = {
    "evolve_features": {
        "temperatura": EvolutionFeatureConfig(
            column="temperatura", type="linear", slope=0.1
        ),
        "presion": EvolutionFeatureConfig(
            column="presion", type="driven_by",
            driver_col="temperatura", func="linear",
            func_params={"slope": 0.5}
        ),
    }
}

preset = ScenarioInjectorPreset(random_state=42)
transformed_df = preset.generate(data=real_df, scenario_config=scenario)
```

---

## Clínico y Omics

### `LongitudinalHealthPreset`

```python
from calm_data_generator.presets import LongitudinalHealthPreset
```

Genera datos clínicos longitudinales con registros de pacientes multi-visita usando `ClinicalDataGenerator`.

**`generate(n_samples, n_visits=5, **kwargs)`**

| Parámetro | Por defecto | Descripción |
|-----------|-------------|-------------|
| `n_samples` | requerido | Número de pacientes |
| `n_visits` | `5` | Número promedio de visitas por paciente |

```python
preset = LongitudinalHealthPreset(random_state=42)
longitudinal_df = preset.generate(n_samples=200, n_visits=8)
```

---

### `RareDiseasePreset`

```python
from calm_data_generator.presets import RareDiseasePreset
```

Simula una cohorte clínica con una enfermedad rara. Impone una prevalencia muy baja de la enfermedad.

**`generate(n_samples, disease_ratio=0.01, **kwargs)`**

| Parámetro | Por defecto | Descripción |
|-----------|-------------|-------------|
| `n_samples` | requerido | Número total de sujetos |
| `disease_ratio` | `0.01` | Prevalencia de la enfermedad (1% por defecto) |

Devuelve un `Dict[str, pd.DataFrame]` con múltiples capas (clínica, omics).

```python
preset = RareDiseasePreset(random_state=42, verbose=True)

# 2% de prevalencia
result = preset.generate(n_samples=500, disease_ratio=0.02)
clinical_df = result["clinical"]
```

---

### `OmicsIntegrationPreset`

```python
from calm_data_generator.presets import OmicsIntegrationPreset
```

Genera datos multi-omics correlacionados en tres capas: clínica, expresión génica y proteómica.

**`generate(n_samples, n_genes=100, n_proteins=50, **kwargs)`**

| Parámetro | Por defecto | Descripción |
|-----------|-------------|-------------|
| `n_samples` | requerido | Número de sujetos |
| `n_genes` | `100` | Número de features de expresión génica |
| `n_proteins` | `50` | Número de features de proteínas |

```python
preset = OmicsIntegrationPreset(random_state=42)
result = preset.generate(n_samples=300, n_genes=200, n_proteins=80)
```

---

### `SingleCellQualityPreset`

```python
from calm_data_generator.presets import SingleCellQualityPreset
```

Genera datos de RNA-seq de célula única de alta calidad usando scVI (Single-Cell Variational Inference). Estado del arte para datos de conteo de alta dimensión.

**`generate(data, n_samples, **kwargs)`**

| Parámetro | Por defecto | Descripción |
|-----------|-------------|-------------|
| `data` | requerido | DataFrame real de célula única o AnnData |
| `n_samples` | requerido | Número de células sintéticas |
| `auto_report` | `True` | Activar reporte de calidad |

Configuración interna clave: `method="scvi"`, `epochs=400`, `n_latent=10`. Los kwargs adicionales se reenvían.

```python
preset = SingleCellQualityPreset(random_state=42)
synthetic_cells = preset.generate(data=adata_df, n_samples=500)
```

---

## Resumen de Imports

```python
from calm_data_generator.presets import (
    # Velocidad
    FastPreset,
    FastPrototypePreset,
    # Calidad
    HighFidelityPreset,
    DiffusionPreset,
    CopulaPreset,
    DataQualityAuditPreset,
    # Distribución de clases
    ImbalancedGeneratorPreset,
    BalancedDataGeneratorPreset,
    # Series temporales
    TimeSeriesPreset,
    SeasonalTimeSeriesPreset,
    # Drift y escenarios
    DriftScenarioPreset,
    GradualDriftPreset,
    ConceptDriftPreset,
    ScenarioInjectorPreset,
    # Clínico / omics
    LongitudinalHealthPreset,
    RareDiseasePreset,
    OmicsIntegrationPreset,
    SingleCellQualityPreset,
)
```
