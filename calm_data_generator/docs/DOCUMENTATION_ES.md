# Documentación de Calm Data Generator

## Documentación de Motores Externos
Para ajustes de hiperparámetros avanzados y detalles técnicos de los modelos subyacentes, por favor consulta:
- **Synthcity**: [Manual de Referencia](https://github.com/vanderschaarlab/synthcity)
- **scvi-tools**: [Guía de Usuario](https://docs.scvi-tools.org/)
- **GEARS**: [Detalles de Implementación](https://github.com/snap-stanford/GEARS)

Bienvenido a la documentación completa de **Calm Data Generator**. Esta guía cubre la instalación, configuración y uso avanzado de todos los módulos.

> **Nota:** Para documentos de referencia de API específicos, ver:
> - [RealGenerator API](./REAL_GENERATOR_REFERENCE_ES.md)
> - [DriftInjector API](./DRIFT_INJECTOR_REFERENCE_ES.md)
> - [StreamGenerator API](./STREAM_GENERATOR_REFERENCE_ES.md)
> - [ClinicalGenerator API](./CLINICAL_GENERATOR_REFERENCE_ES.md)
> - [Índice API](./API_ES.md)

---

## Tabla de Contenidos

1. [Instalación](#instalación)
2. [Inicio Rápido](#inicio-rápido)
3. [Generador Real (Tabular)](#realgenerator)
4. [Generador Clínico](#clinicalgenerator)
5. [Generador de Stream](#streamgenerator)
6. [Inyector de Drift](#driftinjector)
7. [Privacidad y Anonimización](#privacidad-y-anonimización)
8. [Generadores de Bloques](#generadores-de-bloques)
9. [Informes de Calidad](#informes-de-calidad)

---

## Instalación

### Instalación Estándar
La librería está disponible en PyPI. Para una instalación estable y rápida, recomendamos usar un entorno virtual:

```bash
# 1. Crear y activar el entorno virtual
python3 -m venv venv
source venv/bin/activate

# 2. Actualizar pip, setuptools y wheel (Crucial para una instalación exitosa)
pip install --upgrade pip setuptools wheel

# 3. Instalar la librería (optimizada para velocidad)
pip install calm-data-generator
```

### Dependencias Opcionales

| Extra | Comando | Incluye |
|-------|---------|---------|
| stream | `pip install "calm-data-generator[stream]"` | River (streaming ML) |
| full | `pip install "calm-data-generator[full]"` | Todas las dependencias anteriores |

> [!NOTE]
> **Velocidad de Instalación**: En la versión 1.0.0, hemos bloqueado dependencias clave (`pydantic`, `xgboost`, `cloudpickle`) para evitar el bucle de resolución de ~40 minutos causado por los requisitos complejos de `synthcity`. La instalación ahora es mucho más rápida.

---

## Inicio Rápido

Ver [README_ES.md](../../README_ES.md) para ejemplos básicos de código.

---

## RealGenerator

**Clase:** `calm_data_generator.generators.tabular.RealGenerator`

El motor principal para generar datos sintéticos que imitan datasets tabulares reales.

### Uso Básico

```python
from calm_data_generator.generators.tabular import RealGenerator

gen = RealGenerator()
synthetic_data = gen.generate(real_data, n_samples=1000, method='lgbm')
```

### Métodos Soportados

| Método | Descripción | Caso de Uso |
|--------|-------------|-------------|
| `cart` | Árboles de Clasificación y Regresión | Iteración rápida, captura estructura básica. |
| `rf` | Random Forest | Mejor calidad que CART, más lento. |
| `copula` | Copula | Copula-based synthesis | Base installation |
| `lgbm` | LightGBM | Alta eficiencia y rendimiento para tablas grandes. |
| `ctgan` | Conditional GAN (Synthcity) | Deep learning para distribuciones complejas multi-modales. |
| `tvae` | Variational Autoencoder (Synthcity) | A menudo más rápido y robusto que GANs para datos tabulares. |
| `copula` | Gaussian Copula | Modela correlaciones multivariadas usando la librería `copulae`. |
| `diffusion` | Difusión Tabular (DDPM) | Estado del arte experimental. Lento pero alta fidelidad. |
| `scvi` | Single-Cell (Genómica) | Modelado biológico especializado para RNA-Seq (scVI/scANVI). |


### Configuración Avanzada (`**kwargs`)

Puedes pasar parámetros específicos al modelo subyacente a través de `**kwargs`.

**Para métodos de Deep Learning (CTGAN, TVAE) vía Synthcity:**
- `epochs`: Número de épocas de entrenamiento (defecto: 300).
- `batch_size`: Tamaño del lote (defecto: 500).
- `n_units_conditional`: Parámetros específicos de Synthcity.
- `cuda`: `True`/`False` para forzar uso de GPU.

**Para métodos basados en ML (LGBM):**
- `n_estimators`: Número de árboles.
- `max_depth`: Profundidad máxima.
- `balance_target`: `True` para reequilibrar clases antes de entrenar.
- `differentiation_factor`: Factor de separación latente (v1.2.0).
- `clipping_mode`: Estrategia de recorte (`'strict'`, `'permissive'`, `'none'`).
- `use_latent_sampling`: `True` para mayor fidelidad biológica.

---

## ClinicalGenerator

**Clase:** `calm_data_generator.generators.clinical.ClinicalDataGenerator`

Diseñado para simular datos sanitarios complejos incluyendo datos demográficos, genómicos (genes) y proteómicos (proteínas).

### Características Clave
- **Correlaciones Biológicas:** Simula dependencias realistas entre edad, género y expresión de biomarcadores.
- **Efectos de Enfermedad:** Permite inyectar señales específicas de enfermedad (ej. sobreexpresión de un gen).
- **Longitudinal:** Genera trayectorias de pacientes a lo largo del tiempo.

Ver [CLINICAL_GENERATOR_REFERENCE_ES.md](./CLINICAL_GENERATOR_REFERENCE_ES.md) para detalles completos de configuración.

---

## StreamGenerator

**Clase:** `calm_data_generator.generators.stream.StreamGenerator`

Un wrapper alrededor de la biblioteca `River` para generar flujos de datos infinitos con concept drift evolutivo.

### Flujo de Trabajo
1. Instanciar un generador de River (ej. `SEA`, `Agrawal`).
2. Pasarlo a `StreamGenerator.generate()`.
3. Aplicar drift, balanceo o inyección de fechas.

```python
from river import synth
from calm_data_generator.generators.stream import StreamGenerator

river_gen = synth.SEA()
gen = StreamGenerator()
df = gen.generate(river_gen, n_samples=5000)
```

Ver [STREAM_GENERATOR_REFERENCE_ES.md](./STREAM_GENERATOR_REFERENCE_ES.md).

---

## DriftInjector

**Clase:** `calm_data_generator.generators.drift.DriftInjector`

Permite modificar datasets existentes para introducir cambios estadísticos controlados (drift), útiles para probar sistemas de monitorización de ML.

### Tipos de Drift
- **Feature Drift:** Cambios en la distribución de las variables de entrada $P(X)$.
- **Label Drift:** Cambios en la distribución de la variable objetivo $P(y)$.
- **Concept Drift:** Cambios en la relación entre entrada y objetivo $P(y|X)$.

### Inyección Unificada

Usa `inject_drift()` para aplicar drift fácilmente a múltiples columnas sin preocuparte por sus tipos de datos.

```python
injector.inject_drift(df, columns=['salary'], drift_mode='gradual', magnitude=0.5)
```

Ver [DRIFT_INJECTOR_REFERENCE_ES.md](./DRIFT_INJECTOR_REFERENCE_ES.md).

---

## Privacidad y Anonimización

> [!NOTE]
> **Módulo de Privacidad Eliminado**: El módulo `anonymizer` independiente ha sido eliminado en favor de características de privacidad integradas.

Las características de privacidad ahora están disponibles a través de:

1. **QualityReporter con métricas DCR**: Usa `privacy_check=True` para calcular métricas de Distance to Closest Record (DCR), que miden el riesgo de re-identificación.

```python
from calm_data_generator.generators.tabular import QualityReporter

reporter = QualityReporter()
reporter.generate_report(real_df, synthetic_df, privacy_check=True)
```

2. **Modelos de Privacidad Diferencial de Synthcity**: Algunos plugins de Synthcity soportan privacidad diferencial de forma nativa. Consulta la documentación de Synthcity para más detalles.

---


## Generadores de Bloques

Permiten crear datasets compuestos de múltiples partes ("bloques"), donde cada bloque puede representar un periodo de tiempo, ubicación o concepto diferente.

### Cómo Funciona

1.  **Partición**: Los datos de entrada se dividen en trozos basados en `block_column` (ej. Año, Región).
2.  **Modelado Independiente**: Se entrena un modelo generativo separado para **cada bloque**. Esto captura las propiedades estadísticas locales.
3.  **Generación**: Se generan datos sintéticos para cada bloque independientemente.
4.  **Ensamblaje**: Los bloques sintéticos se concatenan.

### Clases Soportadas

| Generador | Descripción |
|-----------|-------------|
| `RealBlockGenerator` | Divide un dataset real en bloques y aprende de cada uno. |
| `StreamBlockGenerator` | Concatena generadores de stream para simular drift sintético puro. |
| `ClinicalDataGeneratorBlock` | Genera datos clínicos multi-centro (ej. varios hospitales). |

### Ejemplo: RealBlockGenerator

```python
from calm_data_generator.generators.tabular import RealBlockGenerator

gen = RealBlockGenerator()

# Generar datos divididos por "Año"
synthetic_blocks = gen.generate(
    data=data,
    output_dir="./output",
    block_column="Year",
    target_col="Churn"
)
```

---

## Informes de Calidad

**Clase:** `calm_data_generator.generators.tabular.QualityReporter`

Genera informes HTML interactivos comparando los datos reales y sintéticos.

```python
from calm_data_generator.generators.tabular import QualityReporter

reporter = QualityReporter()
# Genera un reporte HTML completo incluyendo métricas ARI para separabilidad de clases
reporter.generate_report(real_df, synthetic_df, target_col='target')

# Cálculo de ARI independiente para cuantificar la mejora en separación de clases
ari_metrics = reporter.calculate_ari(real_df, synthetic_df, target_col='target')
# Devuelve: {'ari_original': 0.95, 'ari_synthetic': 0.98, 'ari_improvement': 0.03}
```

**Métricas Incluidas:**
- **Estadísticas Descriptivas:** Comparación de media, std, min, max.
- **Distribuciones:** Histogramas superpuestos.
- **Correlaciones:** Mapas de calor de Pearson/Spearman.
- **PCA/TSNE:** Visualización de la variedad de datos en 2D.
- **Privacidad:** (Opcional) Tests de riesgo de reidentificación.

## Síntesis de Series Temporales

CALM-Data-Generator ahora soporta métodos avanzados de síntesis de series temporales mediante integración con Synthcity.

### Métodos Disponibles para Series Temporales

| Método | Tipo | Mejor Para |
|--------|------|-----------|
| `timegan` | GAN | Patrones temporales complejos, secuencias multi-entidad |
| `timevae` | VAE | Series temporales regulares, entrenamiento más rápido |
| `fflows` | Normalizing Flows | Series periódicas/estacionales, más estable que TimeGAN |
| `bn` | Red Bayesiana | Datos tabulares clínicos con dependencias causales |

### Uso Básico

```python
from calm_data_generator import RealGenerator

gen = RealGenerator()

# TimeGAN para patrones complejos
synth = gen.generate(
    datos_series_temporales,
    method='timegan',
    n_samples=100,
    n_iter=1000
)

# FourierFlows - estable para series periódicas
synth = gen.generate(
    datos_series_temporales,
    method='fflows',
    n_samples=100,
    sequence_key='seq_id',
    time_key='timestamp',
    n_iter=500
)

# Red Bayesiana - para datos tabulares con dependencias causales
synth = gen.generate(
    datos_clinicos,
    method='bn',
    n_samples=500,
    target_col='diagnostico'
)
```

Para parámetros detallados y escenarios de uso, ver [REAL_GENERATOR_REFERENCE_ES.md](REAL_GENERATOR_REFERENCE_ES.md).

## Modelos de Difusión Avanzados

### DDPM vs Difusión Custom

| Característica | `diffusion` (custom) | `ddpm` (Synthcity) |
|----------------|---------------------|-------------------|
| Velocidad | ⚡ Rápido | 🐢 Más lento |
| Calidad | ⭐⭐⭐ Buena | ⭐⭐⭐⭐ Excelente |
| Arquitecturas | MLP | MLP/ResNet/TabNet |
| Caso de Uso | Prototipado | Producción |

```python
# Prototipado rápido
synth = gen.generate(data, method='diffusion', n_samples=1000)

# Calidad de producción
synth = gen.generate(
    data,
    method='ddpm',
    n_samples=1000,
    model_type='resnet',
    scheduler='cosine'
)
```

