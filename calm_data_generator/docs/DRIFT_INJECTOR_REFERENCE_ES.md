# DriftInjector - Referencia Completa

**Ubicación:** `calm_data_generator.injectors.DriftInjector`

El `DriftInjector` es una herramienta potente para simular **drift de datos** (cambios en la distribución de datos a lo largo del tiempo) en datasets sintéticos. Es esencial para probar sistemas de monitoreo de modelos, algoritmos de detección de drift y pipelines de ML adaptativos.

---

## ⚡ Inicio Rápido: Drift desde `generate()`

La forma más sencilla de especificar drift es pasando una `drift_injection_config` a `RealGenerator.generate()`. Exhortamos el uso del objeto `DriftConfig` para validación y seguridad de tipos.

### Usando `DriftConfig` (Recomendado)

```python
from calm_data_generator.generators.configs import DriftConfig

# 1. Definir Configuración de Drift
drift_conf = DriftConfig(
    method="inject_feature_drift_gradual",
    feature_cols=["age", "income"],  # Columnas a afectar
    drift_type="shift",              # Tipo de operación (shift, scale, noise, etc.)
    magnitude=0.3,                   # Intensidad (0.0 - 1.0)
    center=500,                      # Fila donde el drift alcanza su pico
    width=200,                       # Ancho de la ventana de transición
    profile="sigmoid"                # Forma de la transición
)

# 2. Generar Datos con Drift
synthetic = gen.generate(
    data=real_data,
    n_samples=1000,
    method='ctgan',
    drift_injection_config=[drift_conf]
)
```

### Parámetros Soportados por `DriftConfig`

| Parámetro | Tipo | Defecto | Descripción |
|-----------|------|---------|-------------|
| `method` | str | `"inject_feature_drift"` | Método de DriftInjector a llamar |
| `feature_cols` | List[str] | `None` | Columnas a las que aplicar drift |
| `drift_type` | str | `"gaussian_noise"` | Tipo de operación de drift (ej. `shift`, `scale`) |
| `magnitude` | float | `0.2` | Intensidad del drift (típicamente 0.0-1.0) |
| `start_index` | int | `None` | Fila donde empieza el drift |
| `end_index` | int | `None` | Fila donde termina el drift |
| `center` | int | `None` | Punto central de la ventana (para gradual) |
| `width` | int | `None` | Ancho de la transición (para gradual) |
| `profile` | str | `"sigmoid"` | Forma de transición (`sigmoid`, `linear`, `cosine`) |

---

## 🌲 Árbol de Decisión: ¿Qué Tipo de Drift Usar?

Usa esta guía para elegir el método correcto:

```text
¿Qué quieres cambiar?
├─ ¿Valores de una característica (Feature)?
│  ├─ ¿Gradualmente en el tiempo? → inject_feature_drift_gradual()
│  └─ ¿Repentinamente en un punto? → inject_feature_drift() (con start_index)
├─ ¿Distribución del objetivo/etiqueta (Label)?
│  ├─ ¿Invertir etiquetas? → inject_label_drift()
│  └─ ¿Forzar una distribución específica? → inject_label_shift()
├─ ¿Distribución de características (no valores)?
│  └─ → inject_categorical_frequency_drift() o inject_covariate_shift()
└─ ¿Relación Feature→Target?
   └─ → inject_conditional_drift() (Concept Drift)
```

---

## 📚 Tipos de Drift Explicados

| Tipo de Drift | Qué Hace | Escenario de Ejemplo |
|---------------|----------|----------------------|
| **Feature Drift (Gradual)** | Cambia valores lentamente | Población envejeciendo, inflación |
| **Feature Drift (Repentino)**| Cambio abrupto | Reemplazo de sensor, actualización de sistema |
| **Label Drift** | Cambia distribución del target | Ola de fraudes, cambio de mercado |
| **Covariate Shift** | Cambia distribución de inputs | Nuevo segmento de usuarios |
| **Concept Drift** | Cambia lógica Feature→Target | Definición de "buen cliente" cambia |

---

## 🛠️ Referencia de Clase `DriftInjector`

Si necesitas más control del que permite `generate()`, puedes usar `DriftInjector` directamente sobre cualquier DataFrame.

**Importar:** `from calm_data_generator.injectors import DriftInjector`

### Inicialización

```python
injector = DriftInjector(
    output_dir="./drift_output",      # Directorio para reportes/gráficos
    generator_name="my_drift",        # Prefijo para archivos
    random_state=42,                  # Semilla de reproducibilidad
    auto_report=True,                 # Generar reporte PDF automáticamente
)
```

> [!TIP]
> **Compatibilidad con Dashboard**: El archivo `report_results.json` generado ahora incluye metadatos de `compared_data_files`. Esto asegura la compatibilidad total con el dashboard de monitorización para visualizar las comparaciones de drift.

### Métodos de Feature Drift

#### `inject_feature_drift()` - Cambio Abrupto
Cambia valores directamente a partir de `start_index`.

```python
drifted_df = injector.inject_feature_drift(
    df=df,
    feature_cols=["price", "quantity"],
    drift_type="shift",        # Opciones: shift, scale, gaussian_noise ...
    drift_magnitude=0.3,       # +30% desplazamiento
    start_index=500,           # Empieza en fila 500
)
```

#### `inject_feature_drift_gradual()` - Transición Suave
La transición sigue una curva (sigmoide, lineal) centrada en `center`.

```python
drifted_df = injector.inject_feature_drift_gradual(
    df=df,
    feature_cols=["price"],
    drift_type="scale",
    drift_magnitude=0.5,     # Factor de escala aumenta en 0.5
    center=500,              # Centro de transición
    width=200,               # Duración de transición (filas)
    profile="sigmoid"        # Forma de curva
)
```

#### `inject_feature_drift_incremental()` - Crecimiento Continuo
Drift lineal que sigue creciendo o decreciendo sobre el rango.

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

### Drift de Etiquetas (Label) y Categórico

#### `inject_label_drift()`
Invierte etiquetas aleatoriamente (bueno para simular ruido/errores).

```python
drifted_df = injector.inject_label_drift(
    df=df,
    target_cols=["is_fraud"],
    drift_magnitude=0.1,     # Invierte 10% de etiquetas
    start_index=500
)
```

#### `inject_categorical_frequency_drift()`
Cambia la frecuencia de categorías (ej. hacer comunes los ítems raros).

```python
drifted_df = injector.inject_categorical_frequency_drift(
    df=df,
    feature_cols=["category"],
    drift_magnitude=0.5,
    perturbation="invert"    # Invierte distribución de frecuencia
)
```

---

## 🧪 Tipos de Operación (`drift_type`)

### Para Columnas Numéricas

| Tipo | Fórmula/Lógica | Caso de Uso |
|------|----------------|-------------|
| `shift` | `x + (mean * magnitude)` | Promedio móvil, sesgo |
| `scale` | `mean + (x - mean) * (1 + magnitude)` | Aumento de varianza/amplitud |
| `gaussian_noise` | `x + N(0, magnitude * std)` | Ruido de sensor, error de medición |
| `add_value` | `x + magnitude` | Offset fijo |
| `multiply_value` | `x * magnitude` | Ganancia multiplicativa |

### Para Categóricas/Booleanas

| Tipo | Método | Lógica |
|------|--------|--------|
| `frequency` | `inject_categorical...` | Remuestrea para cambiar conteos |
| `new_category` | `inject_new_category...` | Inyecta valores desconocidos |
| `flip` | `inject_boolean_drift` | Invierte True/False |
| `typos` | `inject_typos_drift` | Añade ruido de caracteres |

---

## 🌟 Escenarios del Mundo Real

### Caso 1: Degradación de Sensor (Incremental + Ruido)
Simular un sensor IoT que pierde calibración y se vuelve más ruidoso.

```python
# 1. Pérdida de calibración (Shift Lineal)
df = injector.inject_feature_drift_incremental(
    df=sensor_df,
    feature_cols=["reading"],
    drift_type="shift",
    drift_magnitude=0.5
)

# 2. Ruido creciente (Gaussiano)
df = injector.inject_feature_drift(
    df=df,
    feature_cols=["reading"],
    drift_type="gaussian_noise",
    drift_magnitude=0.3,
    start_index=500
)
```

### Caso 2: Patrón Estacional (Recurrente)
Añadir efecto de temporada vacacional donde las ventas se disparan.

```python
df = injector.inject_feature_drift_recurrent(
    df=sales_df,
    feature_cols=["sales"],
    drift_type="multiply_value",
    drift_magnitude=1.5,  # 50% aumento
    repeats=3             # 3 temporadas
)
```

### Caso 3: Concept Drift (Basado en Reglas)
Cambio de lógica: Usuarios de altos ingresos empiezan a impagar repentinamente.

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
