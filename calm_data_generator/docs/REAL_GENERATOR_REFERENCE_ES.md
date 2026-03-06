# RealGenerator - Referencia Completa

**Ubicación:** `calm_data_generator.generators.tabular.RealGenerator`

El generador principal para la síntesis de datos tabulares a partir de datasets reales.

---

## Inicialización

```python
from calm_data_generator import RealGenerator

gen = RealGenerator(
    auto_report=True,       # Generar informe automáticamente tras síntesis
    minimal_report=False,   # Si es True, informe más rápido sin correlaciones/PCA
    random_state=42,        # Semilla para reproducibilidad
    logger=None,            # Logger de Python personalizado opcional
)
```

### Parámetros del Constructor

| Parámetro | Tipo | Defecto | Descripción |
|-----------|------|---------|-------------|
| `auto_report` | bool | `True` | Generar informe de calidad automáticamente |
| `minimal_report` | bool | `False` | Informe simplificado (más rápido) |
| `random_state` | int | `None` | Semilla para reproducibilidad |
| `logger` | Logger | `None` | Instancia de Logger de Python personalizada |
| `verbose_training` | bool | `False` | Muestra la pérdida por época de Synthcity en consola durante el entrenamiento |


---

## Método Principal: `generate()`

```python
# Nuevos Imports de Configuración
from calm_data_generator.generators.configs import DriftConfig, ReportConfig, DateConfig

synthetic_df = gen.generate(
    data=df,                          # DataFrame original (requerido)
    n_samples=1000,                   # Número de muestras a generar (requerido)
    method="ctgan",                   # Método de síntesis
    
    # Objetos de Configuración
    report_config=ReportConfig(       # Configuración de informes
        output_dir="./output",
        target_column="target"
    ),
    
    # Inyección de Drift
    drift_injection_config=[
        DriftConfig(
            method="inject_feature_drift",
            feature_cols=["age"],
            drift_type="shift", 
            magnitude=0.5
        )
    ],
    
    # Los argumentos legacy aún son soportados pero se recomiendan los objetos Config
    # target_col="target", 
    # output_dir="./output" 
)
```

### Parámetros de `generate()`

| Parámetro | Tipo | Defecto | Descripción |
|-----------|------|---------|-------------|
| `data` | DataFrame | - | Dataset original (requerido) |
| `n_samples` | int | - | Número de muestras a generar (requerido) |
| `method` | str | `"cart"` | Método de síntesis |
| `target_col` | str | `None` | Columna objetivo para balanceo |
| `output_dir` | str | `None` | Directorio para archivos de salida |
| `generator_name` | str | `"RealGenerator"` | Nombre base para archivos de salida |
| `save_dataset` | bool | `False` | Guardar dataset generado como CSV |
| `custom_distributions` | Dict | `None` | Distribución forzada por columna |
| `date_col` | str | `None` | Nombre de columna de fecha a inyectar |
| `date_start` | str | `None` | Fecha de inicio ("YYYY-MM-DD") |
| `date_step` | Dict | `None` | Incremento temporal (ej., `{"days": 1}`) |
| `date_every` | int | `1` | Incrementar fecha cada N filas |
| `drift_injection_config` | List[Union[Dict, DriftConfig]] | `None` | Configuración de drift post-generación |
| `dynamics_config` | Dict | `None` | Configuración de evolución dinámica |
| `**kwargs` | Dict | `None` | Hiperparámetros específicos  |
| `constraints` | List[Dict] | `None` | Restricciones de integridad |
| `adversarial_validation` | bool | `False` | Activar reporte de discriminador (Real vs Sintético) |

---

## Referencia Completa de `**kwargs`

El diccionario `**kwargs` permite el ajuste fino de parámetros internos para cada método de síntesis.

### Deep Learning (Synthcity)

| Parámetro | Métodos | Descripción |
|-----------|---------|-------------|
| `epochs` | `ctgan`, `tvae` | Número de épocas de entrenamiento (defecto: 300) |
| `batch_size` | `ctgan`, `tvae` | Tamaño del batch de entrenamiento (defecto: 500) |
| `n_units_conditional` | `ctgan`, `tvae` | Unidades en capas condicionales |
| `lr` | `ctgan`, `tvae` | Tasa de aprendizaje (Learning rate) |
| `differentiation_factor` | `ctgan`, `tvae`, `scvi` | *(v1.2.0)* Desplaza los centroides de clase en el espacio latente/de características. `0.0` = sin desplazamiento, `1.0` = moderado, `2.0+` = separación fuerte |


**Ejemplo:**
```python
gen.generate(
    df, 1000,
    method="ctgan",
    epochs=500,
    batch_size=256
)
```


### Machine Learning Clásico (CART, RF, LGBM)

| Parámetro | Métodos | Descripción |
|-----------|---------|-------------|
| `balance_target` | Todos ML | Si es True y `target_col` existe, balancea clases antes de entrenar |
| `n_estimators` | RF, LGBM | Número de árboles |
| `max_depth` | CART, RF | Profundidad máxima |

**Ejemplo:**
```python
method="rf",
target_col="churn",
balance_target=True,
n_estimators=100,

```

### Single-Cell (scVI)

Estos métodos están diseñados específicamente para **datos transcriptómicos (RNA-seq)**. Utilizan modelos generativos profundos para manejar la dispersión (sparsity) y el ruido técnico característico de los datos biológicos. Son ideales para corregir "efectos de lote" (batch effects) y generar perfiles de expresión genética sintéticos coherentes.

**Formato de Entrada:** Acepta tanto `pd.DataFrame` como objetos `AnnData` directamente.

#### scVI (Single-cell Variational Inference)

**Entrada DataFrame:**
```python
synthetic = gen.generate(
    data=expression_df,      # Filas=células, Columnas=genes
    n_samples=1000,
    method="scvi",
    target_col="cell_type",  # Columna de metadatos opcional
    epochs=100,
    n_latent=10,
    n_layers=1,
    
)

# GEARS - Predicción de Perturbaciones basada en Grafos
synthetic = gen.generate(
    expression_df, 500,
    method='gears',
    perturbations=['GENE1', 'GENE2'],  # Requerido: genes a perturbar
    epochs=20,
    batch_size=32,
    device='cpu'
)
> **IMPORTANTE:** GEARS requiere instalación desde el código fuente (`pip install "git+https://github.com/snap-stanford/GEARS.git@f374e43"`) y PyTorch >= 2.4.0.

**Formato de Entrada:** Acepta objetos `pd.DataFrame`, `AnnData` o rutas de archivo (`.h5`, `.h5ad` o `.csv`) directamente.

**Formato de Entrada:** Acepta objetos `pd.DataFrame`, `AnnData` o rutas de archivo (`.h5`, `.h5ad` o `.csv`) directamente.

**Uso de Rutas de Archivo (H5/H5AD/CSV):**
```python
# ¡El generador carga el archivo automáticamente por ti!
synthetic = gen.generate(
    data="datos_single_cell.csv",  # O .h5ad, .h5
    n_samples=1000,
    method="scvi",
    target_col="cell_type"
)
```


**Entrada AnnData (Recomendado para datos single-cell):**
```python
import anndata

# Crear o cargar objeto AnnData
adata = anndata.read_h5ad("single_cell_data.h5ad")

synthetic = gen.generate(
    data=adata,              # Pasar AnnData directamente
    n_samples=1000,
    method="scvi",
    target_col="cell_type",  # Debe estar en adata.obs
    epochs=100,
    n_latent=10,
    n_layers=1,
    
)
# Retorna pd.DataFrame con columnas de genes + metadatos
```

| Parámetro | Descripción |
|-----------|-------------|
| `epochs` | Épocas de entrenamiento (default: 100) |
| `n_latent` | Dimensionalidad del espacio latente (default: 10) |
| `n_layers` | Número de capas ocultas (default: 1) |



> **Soporte AnnData:** Al pasar un objeto `AnnData`, este se utiliza directamente sin conversión, preservando la estructura original. El resultado es siempre un `pd.DataFrame` que contiene tanto la expresión génica como los metadatos de las observaciones (`obs`).


```

| Parámetro | Descripción |
|-----------|-------------|
| `epochs` | Épocas de entrenamiento (default: 100) |
| `n_latent` | Dimensionalidad del espacio latente (default: 10) |
| `condition_col` | Columna con etiquetas de condición/lote (requerido) |



---


---

## Características Avanzadas

### Inyección de Fechas (DateConfig)

Puedes inyectar una columna de fecha/hora en los datos generados usando `DateConfig`.

```python
from calm_data_generator.generators.configs import DateConfig

synthetic = gen.generate(
    data=df,
    n_samples=1000,
    method="cart",
    date_config=DateConfig(
        date_col="timestamp",
        start_date="2024-06-01",
        step={"hours": 1},  # Incremento temporal
        frequency=1         # Incrementar cada N filas
    )
)
```

## Manejo de Datos Desbalanceados

`RealGenerator` ofrece varias estrategias para trabajar con datasets fuertemente desbalanceados (ej. detección de fraude, diagnósticos raros):

### 1. Re-balanceo Automático (`balance_target=True`)
Utiliza técnicas de re-muestreo antes o durante el entrenamiento para generar un dataset sintético equilibrado.
*   **Ideal para:** Entrenar modelos de clasificación robustos que requieren clases balanceadas.
*   **Comportamiento:** Si el original es 99% clase A y 1% clase B, el resultado será aprox. 50% A y 50% B.
*   **Métodos compatibles:** `cart`, `rf`, `lgbm`.

### 2. Control Manual de Distribución
Puede forzar la distribución de la clase objetivo usando `DriftInjector`.
*   **Ideal para:** Escenarios "What-If" (ej. "¿Qué pasa si el fraude aumenta al 10%?").
*   **Método:** `DriftInjector.inject_label_shift` post-generación.

### 3. Técnicas de Oversampling
Métodos clásicos para aumentar la clase minoritaria mediante interpolación.
*   **Métodos:** `smote` (Synthetic Minority Over-sampling Technique), `adasyn` (Adaptive Synthetic Sampling).
*   **Ideal para:** Datasets numéricos pequeños donde se necesita aumentar la representación de casos raros.

### 4. Fidelidad Estadística (Por defecto)
Si no se especifica ninguna opción, los modelos generativos avanzados (`ctgan`, `tvae`) aprenderán y replicarán la distribución original, preservando el desbalance real.
*   **Ideal para:** Análisis exploratorio fiel a la realidad o validación de sistemas en condiciones reales.

---

## Métodos Soportados

| Método | Tipo | Descripción |
|--------|------|-------------|
| `cart` | ML | Árboles de Clasificación y Regresión (Rápido, bueno para estructura) |
| `rf` | ML | Random Forest (Robusto, más lento que CART) |
| `ctgan` | DL | Conditional GAN para tablas (Vía Synthcity) |
| `tvae` | DL | Variational Autoencoder para tablas (Vía Synthcity) |
| `copula` | Estadístico | Síntesis basada en Copulas Gaussianas |
| `diffusion` | DL | Difusión Tabular (DDPM) | **Experimental**. Requiere `calm-data-generator[deeplearning]` |
| `smote` | Aug. | Sobremuestreo SMOTE | Instalación base |
| `adasyn` | Aug. | Muestreo adaptativo ADASYN | Instalación base |

| `gmm` | Estadístico | Modelos de Mezcla Gaussiana | Instalación base |
| `scvi` | Single-Cell | scVI (Variational Inference) para RNA-seq | Requiere `scvi-tools` |
| `gears` | Single-Cell | GEARS (Predicción de Perturbaciones) | Requiere `gears` |

---

## Escenarios de Uso Comunes (Guía Rápida)

### 1. Series Temporales (Time Series)
Para datos de series temporales, usa métodos tabulares estándar (CTGAN, TVAE, etc.) en datos temporales estructurados adecuadamente.
*   **Proyección de Futuro (Forecasting):** No es el caso de uso principal. Usa `StreamGenerator` para flujos infinitos o inyección de fechas manual.


### 2. Clasificación y Regresión (Supervisado)
Si tienes una columna `target` (ej. precio, churn) y la relación $X \rightarrow Y$ es crítica:
*   Usa `method="lgbm"` (LightGBM) o `method="rf"` (Random Forest).
*   Especifica siempre `target_col="nombre_columna"`.
    ```python
    # El generador detecta automáticamente si es Regresión o Clasificación
    gen.generate(data, target_col="precio", method="lgbm") 
    ```

### 3. Clustering (No Supervisado)
Si no hay un target claro y quieres preservar grupos naturales de datos:
*   Usa `method="gmm"` (Gaussian Mixture Models, vía librería externa si disponible) o `method="tvae"` (Variational Autoencoder).
    ```python
    gen.generate(data, method="tvae")
    ```

### 4. Multi-Label (Etiquetas Múltiples)
Si una celda contiene múltiples valores (ej: `["A", "B", "C"]`) o formato string `"A,B,C"`:
*   **Limitación:** Los modelos estándar no manejan bien listas dentro de celdas.
*   **Solución:** Transforma la columna a **One-Hot Encoding** (múltiples columnas binarias `is_A`, `is_B`) antes de pasarla al generador. Los modelos basados en árboles (`lgbm`, `cart`) aprenderán las correlaciones entre etiquetas (ej: si `is_A=1` suele implicar `is_B=1`).

### 5. Datos por Bloques (Blocks)
Si tus datos están fragmentados lógicamente (ej: por Tiendas, Países, Pacientes) y quieres modelos independientes para cada uno:
*   Usa **`RealBlockGenerator`** en lugar de `RealGenerator`.
    ```python
    block_gen = RealBlockGenerator()
    block_gen.generate(data, block_column="TiendaID", method="cart") 
    ```
    *Esto entrena un modelo diferente para cada TiendaID.*

### 6. Manejo de Datos Desbalanceados (Imbalance)
Si tu columna objetivo (`target`) tiene clases muy minoritarias que quieres potenciar:
*   **Balanceo Automático:** Usa `balance_target=True`. El generador aplicará técnicas de sobremuestreo (SMOTE/RandomOverSampler) internamente para que el modelo aprenda por igual de todas las clases.
    ```python
    gen.generate(data, target_col="fraude", balance_target=True, method="cart")
    ```
*   **Distribución Personalizada:** Si quieres una proporción exacta (ej: 70% Clase A, 30% Clase B):
    ```python
    gen.generate(data, target_col="nivel", custom_distributions={"nivel": {"Bajo": 0.7, "Alto": 0.3}})
    ```
    *Nota: `balance_target` es un atajo para `custom_distributions={"col": "balanced"}`. Para desbalanceos extremos, los métodos de Deep Learning como `method="ctgan"` suelen ofrecer mayor estabilidad que los métodos basados en árboles.*
---
---

### `ddpm` - Synthcity TabDDPM (Difusión Tabular Avanzada)

**Tipo:** Deep Learning (Modelo de Difusión)
**Mejor para:** Síntesis tabular de alta calidad, entornos de producción, grandes datasets
**Requisitos:** `synthcity` (incluido en instalación base de deep learning)

**Type:** Deep Learning (Diffusion Model)  
**Best For:** High-quality tabular synthesis, production environments, large datasets  
**Requirements:** `synthcity` (included in base installation)

#### Descripción

TabDDPM (Tabular Denoising Diffusion Probabilistic Model) es la implementación avanzada de modelos de difusión para datos tabulares de Synthcity. Ofrece múltiples arquitecturas, schedulers avanzados y calidad superior comparada con el método `diffusion` personalizado.

#### Cuándo usarlo

✅ **Usa `ddpm` cuando:**
- Necesitas **calidad máxima** en datos sintéticos
- Trabajas con **grandes datasets** (>100k filas)
- En **entornos de producción** que requieren código robusto y mantenido
- Necesitas **arquitecturas avanzadas** (ResNet, TabNet)
- Quieres **cosine scheduling** para una mejor difusión
- Tienes **tiempo para entrenamientos largos** (1000 épocas por defecto)

❌ **No uses `ddpm` cuando:**
- Necesitas **prototipado rápido** (usa `diffusion` en su lugar)
- Trabajas con **datasets muy pequeños** (<1k filas)
- Tienes **recursos computacionales limitados**
- Necesitas **modificaciones personalizadas** al algoritmo

#### Parameters

```python
synth = gen.generate(
    data,
    method='ddpm',
    n_samples=1000,
    
    # Training parameters
    n_iter=1000,                    # Training epochs (default: 1000)
    lr=0.002,                       # Learning rate (default: 0.002)
    batch_size=1024,                # Batch size (default: 1024)
    
    # Diffusion parameters
    num_timesteps=1000,             # Diffusion timesteps (default: 1000)
    scheduler='cosine',             # 'cosine' or 'linear' (default: 'cosine')
    gaussian_loss_type='mse',       # 'mse' or 'kl' (default: 'mse')
    
    # Model architecture
    model_type='mlp',               # 'mlp', 'resnet', or 'tabnet' (default: 'mlp')
    model_params={                  # Architecture-specific parameters
        'n_layers_hidden': 3,
        'n_units_hidden': 256,
        'dropout': 0.0
    },
    
    # Task type
    is_classification=False,        # True for classification tasks
)
```

#### Parameter Details

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `n_iter` | int | 1000 | Number of training epochs |
| `lr` | float | 0.002 | Learning rate for optimizer |
| `batch_size` | int | 1024 | Training batch size |
| `num_timesteps` | int | 1000 | Number of diffusion timesteps |
| `scheduler` | str | `'cosine'` | Beta scheduler: `'cosine'` (recommended) or `'linear'` |
| `gaussian_loss_type` | str | `'mse'` | Loss function: `'mse'` or `'kl'` |
| `model_type` | str | `'mlp'` | Architecture: `'mlp'`, `'resnet'`, or `'tabnet'` |
| `model_params` | dict | See above | Architecture-specific parameters |
| `is_classification` | bool | False | Set to True for classification tasks |

#### Model Types

**MLP (Multi-Layer Perceptron)**
- Best for: General tabular data
- Speed: Fast
- Parameters: `n_layers_hidden`, `n_units_hidden`, `dropout`

**ResNet (Residual Network)**
- Best for: Complex feature relationships
- Speed: Medium
- Parameters: `n_layers_hidden`, `n_units_hidden`, `dropout`

**TabNet**
- Best for: Tabular data with feature importance
- Speed: Slower
- Parameters: Specific to TabNet architecture

#### Comparison: `diffusion` vs `ddpm`

| Aspect | `diffusion` (custom) | `ddpm` (Synthcity) |
|--------|---------------------|-------------------|
| **Speed** | ⚡ Fast (100 epochs) | 🐢 Slower (1000 epochs) |
| **Quality** | ⭐⭐⭐ Good | ⭐⭐⭐⭐ Excellent |
| **Architectures** | MLP only | MLP/ResNet/TabNet |
| **Scheduler** | Linear | Cosine/Linear |
| **Batch Size** | 64 | 1024 |
| **Use Case** | Quick prototyping | Production quality |
| **Customization** | Easy to modify | Black box |
| **Maintenance** | Your responsibility | Synthcity team |

#### Usage Examples

**Basic Usage:**
```python
from calm_data_generator import RealGenerator
import pandas as pd

gen = RealGenerator()
synth = gen.generate(
    data,
    method='ddpm',
    n_samples=1000,
    n_iter=500  # Reduce for faster training
)
```

**Classification Task:**
```python
synth = gen.generate(
    data,
    method='ddpm',
    n_samples=1000,
    is_classification=True,
    target_col='label'
)
```

**Advanced Architecture:**
```python
synth = gen.generate(
    data,
    method='ddpm',
    n_samples=1000,
    model_type='resnet',
    model_params={
        'n_layers_hidden': 5,
        'n_units_hidden': 512,
        'dropout': 0.1
    },
    scheduler='cosine',
    n_iter=2000
)
```

---

### `timegan` - TimeGAN (Time Series GAN)

**Tipo:** Deep Learning (GAN para Series Temporales)
**Mejor para:** Patrones temporales complejos, series temporales multi-entidad
**Requisitos:** `synthcity` (incluido en instalación base)

#### Descripción

TimeGAN (Time-series Generative Adversarial Network) está diseñado específicamente para datos secuenciales/temporales. Aprende tanto la dinámica temporal como la distribución de características, haciéndolo ideal para series temporales con patrones complejos.

#### Cuándo usarlo

✅ **Usa `timegan` cuando:**
- Tienes **datos de series temporales** con dependencias temporales
- Trabajas con **secuencias multi-entidad** (ej. múltiples usuarios/sensores)
- Necesitas preservar **dinámicas temporales**
- Tienes **patrones temporales complejos** para aprender
- Necesitas síntesis de series temporales de **alta calidad**

❌ **No uses `timegan` cuando:**
- Tienes **datos tabulares simples** (usa `ctgan` o `ddpm` en su lugar)
- Trabajas con **secuencias muy cortas** (<10 pasos de tiempo)
- Necesitas **generación rápida** (usa `timevae` en su lugar)
- Tienes **recursos computacionales limitados**

#### Requisitos de Datos

TimeGAN espera datos en un formato temporal específico:
- **Orden temporal**: Los datos deben estar ordenados por tiempo
- **Agrupación por entidad**: Si es multi-entidad, agrupa por ID de entidad
- **Pasos consistentes**: Preferible intervalos de tiempo regulares

#### Parameters

```python
synth = gen.generate(
    data,
    method='timegan',
    n_samples=100,  # Número de secuencias a generar
    
    # Parámetros de entrenamiento
    n_iter=1000,                    # Épocas de entrenamiento (defecto: 1000)
    n_units_hidden=100,             # Unidades ocultas en RNN (defecto: 100)
    batch_size=128,                 # Tamaño de batch (defecto: 128)
    lr=0.001,                       # Tasa de aprendizaje (defecto: 0.001)
)
```

#### Parameter Details

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `n_iter` | int | 1000 | Number of training epochs |
| `n_units_hidden` | int | 100 | Number of hidden units in RNN layers |
| `batch_size` | int | 128 | Training batch size |
| `lr` | float | 0.001 | Learning rate for optimizer |

#### Usage Examples

**Basic Time Series:**
```python
from calm_data_generator import RealGenerator
import pandas as pd

# Data must have temporal structure
# Example: sensor readings over time
gen = RealGenerator()
synth = gen.generate(
    time_series_data,
    method='timegan',
    n_samples=100,  # Generate 100 sequences
    n_iter=1000,
    n_units_hidden=100
)
```

**Multi-Entity Time Series:**
```python
# Data with multiple entities (e.g., users, sensors)
# Ensure data is sorted by entity_id and timestamp
synth = gen.generate(
    multi_entity_data,
    method='timegan',
    n_samples=50,  # Generate 50 entity sequences
    n_iter=2000,
    n_units_hidden=150,
    batch_size=64
)
```

---

### `timevae` - TimeVAE (Time Series VAE)

**Type:** Deep Learning (VAE for Time Series)  
**Best For:** Regular time series, faster training than TimeGAN  
**Requirements:** `synthcity` (included in base installation)

#### Description

TimeVAE is a variational autoencoder designed for temporal data. It's generally faster than TimeGAN and works well for regular time series with consistent patterns.

#### When to Use

✅ **Use `timevae` when:**
- You have **regular time series** data
- You need **faster training** than TimeGAN
- Working with **consistent temporal patterns**
- You want **good quality** with **less computation**
- You have **moderate-length sequences**

❌ **No uses `timevae` when:**
- You have **highly irregular** time series
- You need **maximum quality** (use `timegan` instead)
- Working with **very complex** temporal dynamics
- You have **simple tabular data** (use `ctgan` or `ddpm`)

#### Data Requirements

Similar to TimeGAN:
- **Temporal ordering**: Data sorted by time
- **Regular intervals**: Works best with consistent timesteps
- **Entity grouping**: If multi-entity, group by entity ID

#### Parameters

```python
synth = gen.generate(
    data,
    method='timevae',
    n_samples=100,  # Number of sequences to generate
    
    # Training parameters
    n_iter=1000,                    # Training epochs (default: 1000)
    decoder_n_layers_hidden=2,      # Decoder layers (default: 2)
    decoder_n_units_hidden=100,     # Decoder units (default: 100)
    batch_size=128,                 # Batch size (default: 128)
    lr=0.001,                       # Learning rate (default: 0.001)
)
```

#### Parameter Details

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `n_iter` | int | 1000 | Number of training epochs |
| `decoder_n_layers_hidden` | int | 2 | Number of hidden layers in decoder |
| `decoder_n_units_hidden` | int | 100 | Number of hidden units in decoder |
| `batch_size` | int | 128 | Training batch size |
| `lr` | float | 0.001 | Learning rate for optimizer |

---

## Guardado y Carga de Modelos

`RealGenerator` permite guardar modelos generadores entrenados y cargarlos posteriormente para inferencia sin re-entrenar. Esto es útil para pipelines de producción donde el entrenamiento es costoso.

### Guardar un Modelo

Después de generar datos (lo cual entrena el modelo subyacente), puedes guardar el generador:

```python
# 1. Entrenar y Generar
gen.generate(data, n_samples=1000, method="ctgan", batch_size=500)

# 2. Guardar el generador entrenado
gen.save("models/mi_modelo_ctgan.pkl")
```
> **Nota:** El archivo guardado es un archivo zip que contiene la configuración del `RealGenerator` y el modelo subyacente (ej. estado del plugin de Synthcity).

### Cargar un Modelo

Puedes cargar un modelo guardado usando el método de clase `load()`. Una vez cargado, puedes generar más muestras sin proporcionar los datos de entrenamiento originales.

```python
from calm_data_generator.generators.tabular import RealGenerator

# 1. Cargar el generador
loaded_gen = RealGenerator.load("models/mi_modelo_ctgan.pkl")

# 2. Generar nuevas muestras (¡No se necesita argumento 'data'!)
new_samples = loaded_gen.generate(n_samples=500)
```

> **Advertencia:** Al generar desde un modelo cargado, **no debes** pasar `data` a `generate()`, pero **debes** pasar `n_samples`.

---

## Mejores Prácticas

6. **Desbalance severo:** Usa `smote` o `adasyn` con `target_col`.

#### Comparison: `timegan` vs `timevae`

| Aspect | `timegan` | `timevae` |
|--------|-----------|-----------|
| **Speed** | 🐢 Slower | ⚡ Faster |
| **Quality** | ⭐⭐⭐⭐ Excellent | ⭐⭐⭐ Good |
| **Complexity** | Handles complex patterns | Best for regular patterns |
| **Training Time** | Longer | Shorter |
| **Use Case** | Complex temporal dynamics | Regular time series |

#### Usage Examples

**Basic Time Series:**
```python
from calm_data_generator import RealGenerator
import pandas as pd

gen = RealGenerator()
synth = gen.generate(
    time_series_data,
    method='timevae',
    n_samples=100,
    n_iter=500,  # Faster than TimeGAN
    decoder_n_units_hidden=100
)
```

**Faster Training:**
```python
# Reduce parameters for quick prototyping
synth = gen.generate(
    time_series_data,
    method='timevae',
    n_samples=50,
    n_iter=300,
    decoder_n_layers_hidden=1,
    decoder_n_units_hidden=50,
    batch_size=64
)
```

---

### `fflows` - FourierFlows (Flujos Normalizantes en Dominio de Frecuencia)

**Tipo:** Deep Learning (Normalizing Flows para Series Temporales)  
**Mejor para:** Series temporales periódicas/quasi-periódicas, alternativa estable a TimeGAN  
**Requisitos:** `synthcity`

#### Descripción

`fflows` aplica flujos normalizantes en el dominio de la frecuencia para generar secuencias temporales. Es generalmente más estable que TimeGAN y destaca en series con patrones periódicos (sinusoidales, estacionales).

```python
synth = gen.generate(
    data,
    method='fflows',
    n_samples=100,
    sequence_key='seq_id',   # Columna que identifica cada secuencia
    time_key='timestamp',    # Columna con marcas de tiempo
    n_iter=1000,
    batch_size=128,
    lr=0.001,
)
```

#### Comparación: `timegan` vs `timevae` vs `fflows`

| Aspecto | `timegan` | `timevae` | `fflows` |
|---------|-----------|-----------|----------|
| **Velocidad** | 🐢 Lento | ⚡ Rápido | ⚡ Rápido |
| **Calidad** | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ |
| **Estabilidad** | Baja | Media | Alta |
| **Mejor para** | Patrones complejos | Series regulares | Series periódicas |

---

### `bn` - Red Bayesiana (Bayesian Network)

**Tipo:** Modelo Gráfico Probabilístico  
**Mejor para:** Datos tabulares clínicos/estructurados con dependencias causales entre variables  
**Requisitos:** `synthcity`

#### Descripción

Una Red Bayesiana modela las dependencias condicionales entre variables usando un grafo acíclico dirigido. El aprendizaje de estructura descubre qué variables influyen causalmente en otras. Especialmente útil para datos sanitarios y clínicos.

```python
synth = gen.generate(
    data,
    method='bn',
    n_samples=1000,
    target_col='diagnostico',
)
```

✅ **Usa `bn` cuando:**
- Los datos tienen **relaciones causales** entre variables (ej. diagnóstico ← síntomas ← analíticas)
- Trabajas con datos **clínicos o epidemiológicos**
- Quieres un **modelo interpretable** (la red es inspeccionable)

❌ **No uses `bn` cuando:**
- Los datos son **alta dimensionalidad** (100+ variables) — el aprendizaje de estructura se vuelve lento
- Necesitas datos de **series temporales** (usa `timegan`/`timevae`/`fflows`)

---

## Method Selection Guide

### For Tabular Data

| Scenario | Recommended Method | Alternative |
|----------|-------------------|-------------|
| **Quick prototyping** | `diffusion` | `cart`, `rf` |
| **Production quality** | `ddpm` | `ctgan` |
| **Large datasets (>100k)** | `ddpm`, `lgbm` | `ctgan` |
| **Small datasets (<1k)** | `cart`, `rf` | `diffusion` |
| **Class imbalance** | `smote`, `adasyn` | `ctgan` |
| **Preserve correlations** | `ctgan`, `ddpm` | `copula` |
| **Fast generation** | `cart`, `diffusion` | `rf` |
| **Maximum quality** | `ddpm` (ResNet) | `ctgan` |

### Para Series Temporales

| Escenario | Método Recomendado | Alternativa |
|----------|-------------------|-------------|
| **Patrones temporales complejos** | `timegan` | `fflows` |
| **Series temporales regulares** | `timevae` | `timegan` |
| **Series periódicas/estacionales** | `fflows` | `timevae` |
| **Entrenamiento rápido** | `timevae` | `fflows` |
| **Secuencias multi-entidad** | `timegan` | `fflows` |
| **Calidad máxima** | `timegan` | `fflows` |

### For Special Cases

| Tipo de Dato | Método Recomendado |
|-----------|-------------------|
| **RNA-seq single-cell** | `scvi` |
| **Datos clínicos tabulares** | `bn` o `ClinicalDataGenerator` |
| **Datos clínicos** | Usa `ClinicalDataGenerator` |
| **Datos en streaming** | Usa `StreamGenerator` |
| **Datos por bloques** | Usa `RealBlockGenerator` |

---

## Novedades v1.2.0

### Diferenciación en el Espacio Latente (`differentiation_factor`)

Disponible para `tvae`, `ctgan` y `scvi`. Controla cuánto se separan los centroides de clase en el espacio latente (o de características) durante la síntesis.

```python
synth = gen.generate(
    data=df,
    n_samples=500,
    method="tvae",
    target_col="grupo",
    differentiation_factor=1.5  # Empujar clases más separadas
)
```

| Valor | Efecto |
|-------|--------|
| `0.0` | Sin desplazamiento (comportamiento por defecto) |
| `0.5–1.0` | Separación suave |
| `1.5–2.0` | Separación moderada/fuerte |
| `> 2.0` | Riesgo de muestras fuera de distribución |

> **TVAE:** El desplazamiento se aplica directamente en el espacio latente neuronal (vectores mu). 
> **CTGAN:** Usa desplazamiento en el espacio de características (GAN no tiene encoder explícito). 
> **scVI:** El desplazamiento se aplica en el espacio latente `z` antes de decodificar.

---

### Visibilidad del Entrenamiento (`verbose_training`)

Pasa `verbose_training=True` al instanciar para dejar que Synthcity imprima la pérdida por época:

```python
gen = RealGenerator(verbose_training=True)
gen.generate(data=df, n_samples=500, method="tvae", epochs=200)
# → 2024-03-06 14:01:12 | INFO | tvae | epoch 1/200 | loss: 1.2341
# → 2024-03-06 14:01:15 | INFO | tvae | epoch 2/200 | loss: 1.1872
# → ...
```

Para **scVI**, la barra de progreso de PyTorch Lightning siempre se muestra. Al terminar, la pérdida final también se registra en el logger de Python.

---

### Métodos de Introspección (Accessors)

Tras llamar a `generate()`, estos métodos exponen el estado interno del modelo:

#### `get_encoder()`

Devuelve la red encodificadora del último modelo entrenado.

```python
synth = gen.generate(df, 500, method="tvae", target_col="etiqueta")
encoder = gen.get_encoder()
# TVAE: nn.Module (encoder del VAE interno)
# scVI: module.z_encoder
# Otros: None
```

#### `get_decoder()`

Devuelve la red descodificadora.

```python
decoder = gen.get_decoder()
# Devuelve nn.Module para tvae y scvi
```

#### `get_latest_embeddings()`

Devuelve los embeddings del espacio latente calculados durante la última síntesis que aplicó `differentiation_factor`.

```python
embeddings = gen.get_latest_embeddings()  # np.ndarray o None
if embeddings is not None:
    print(f"Forma del embedding: {embeddings.shape}")  # (n_muestras, n_latente)
    # Ej. para UMAP:
    import umap
    reducer = umap.UMAP()
    proyeccion = reducer.fit_transform(embeddings)
```

> Devuelve `None` si no se aplicó diferenciación o si el modelo usó el fallback en espacio de características.

#### `get_training_history()`

Devuelve el diccionario de historial de entrenamiento (**solo scVI/scANVI**).

```python
synth = gen.generate(df, 500, method="scvi", epochs=100)
history = gen.get_training_history()

if history:
    import matplotlib.pyplot as plt
    elbo = history["elbo_train"]
    plt.plot(elbo.values)
    plt.xlabel("Época")
    plt.ylabel("ELBO")
    plt.title("Evolución del entrenamiento scVI")
    plt.show()
```

| Clave | Descripción |
|-------|-------------|
| `train_loss_epoch` | Pérdida total de entrenamiento por época |
| `elbo_train` | Cota inferior de la evidencia (ELBO) |
| `reconstruction_loss_train` | Pérdida de reconstrucción (expresión) |
| `kl_local_train` | Divergencia KL – término local por célula |
| `kl_global_train` | Divergencia KL – término global |

> Devuelve `None` para modelos basados en Synthcity (TVAE, CTGAN). Usa `verbose_training=True` para esos.

#### `get_synthesizer_model()`

Devuelve el objeto del modelo subyacente (plugin de Synthcity, modelo scVI, modelo sklearn, etc.).

```python
raw = gen.get_synthesizer_model()
# tvae/ctgan: objeto plugin de Synthcity
# scvi:       instancia scvi.model.SCVI
# cart/rf:    instancia FCSModel
```
