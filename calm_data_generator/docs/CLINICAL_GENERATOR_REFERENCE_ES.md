# ClinicalDataGenerator - Referencia Completa

**Ubicacion:** `calm_data_generator.generators.clinical.ClinicalDataGenerator`
**Hereda de:** `ComplexGenerator` → `BaseGenerator`

El `ClinicalDataGenerator` es un simulador de alta fidelidad para datasets sanitarios multimodales. Utiliza los tres motores matematicos de `ComplexGenerator` (Copula Gaussiana incondicional, Copula Gaussiana condicional, y efectos estocasticos) para generar datos correlacionados y biologicamente realistas. Consulta [COMPLEX_GENERATOR_REFERENCE_ES.md](COMPLEX_GENERATOR_REFERENCE_ES.md) para detalles de los motores.

Orquestra la generacion de:
1.  **Demografía de Pacientes**: Edad, género, IMC, etc., con interdependencias.
2.  **Datos Ómicos**: Expresión génica (RNA-Seq/Microarray) y proteínas, correlacionados con la demografía.
3.  **Registros Longitudinales**: Trayectorias de visitas múltiples.


---

## Guía de Inicio Rápido

### ¿Qué es ClinicalDataGenerator?

Un generador especializado para **datos de investigación clínica/médica** que crea datasets multimodales realistas que incluyen:
- 👥 **Demografía de Pacientes** (edad, género, IMC, etc.)
- 🧬 **Datos Ómicos** (expresión génica, proteínas)
- 📊 **Registros Longitudinales** (trayectorias de múltiples visitas)
- 🌡️ **Efectos de Enfermedad** (biomarcadores, respuestas al tratamiento)

### Cuándo usar ClinicalDataGenerator

✅ **Usa ClinicalDataGenerator cuando:**
- Necesites datos de **ensayos clínicos** o **investigación médica**
- Trabajes con **datos ómicos** (RNA-Seq, Microarray, Proteómica)
- Simules estudios de **enfermedad vs control**
- Crees trayectorias de **pacientes longitudinales**
- Pruebes algoritmos de **descubrimiento de biomarcadores**
- Necesites **demografía correlacionada** (edad, IMC, presión arterial)

❌ **No uses ClinicalDataGenerator cuando:**
- Tengas **datos tabulares simples** → Usa `RealGenerator` en su lugar
- Necesites datos sintéticos de **propósito general** → Usa `RealGenerator`
- No necesites una estructura **multimodal** → Usa `RealGenerator`

### Uso Básico (3 Líneas)

```python
from calm_data_generator.generators.clinical import ClinicalDataGenerator

gen = ClinicalDataGenerator()
data = gen.generate(n_samples=100, n_genes=500, n_proteins=200)
# Retorna: {"demographics": DataFrame, "genes": DataFrame, "proteins": DataFrame}
```

### Casos de Uso Comunes

| Escenario | Método | Parámetros Clave |
|----------|--------|------------------|
| **Cohorte estática** (punto temporal único) | `generate()` | `n_samples`, `n_genes`, `n_proteins` |
| **Estudio longitudinal** (múltiples visitas) | `generate_longitudinal_data()` | `longitudinal_config` |
| **Simulación de biomarcadores** (efectos de enfermedad) | `generate()` | `disease_effects_config` |
| **Diversidad poblacional** (demografía correlacionada) | `generate()` | `demographic_correlations` |

---

## Inicialización

```python
from calm_data_generator.generators.clinical import ClinicalDataGenerator

gen = ClinicalDataGenerator(
    seed=42,                # Semilla para reproducibilidad
    auto_report=True,       # Generar informes automáticamente
    minimal_report=False    # Informes detallados completos
)
```

## Método Principal: `generate()`

Genera una cohorte estática (un solo punto temporal) con demografía y datos ómicos.

```python
from calm_data_generator.generators.configs import DateConfig, DriftConfig

data = gen.generate(
    n_samples=100,
    n_genes=500,
    n_proteins=200,
    control_disease_ratio=0.5,
    date_config=DateConfig(start_date="2024-01-01"),
    
    # Configuración de Drift (usando objetos DriftConfig)
    demographics_drift_config=[
        DriftConfig(method="inject_feature_drift", feature_cols=["Age"], magnitude=0.5)
    ],
    
    # Configuraciones detalladas
    demographic_correlations=None,
    gene_correlations=None,
    disease_effects_config=[...],
    custom_demographic_columns={...}
)
```

### Parámetros

| Parámetro | Tipo | Defecto | Descripción |
|-----------|------|---------|-------------|
| `n_samples` | int | 100 | Número de pacientes (muestras) |
| `n_genes` | int | 200 | Número de variables génicas |
| `n_proteins` | int | 50 | Número de variables de proteínas |
| `control_disease_ratio` | float | 0.5 | Proporción del grupo "Control" (0-1) |
| `gene_type` | str | "RNA-Seq" | "RNA-Seq" (enteros) o "Microarray" (flotantes) |
| `demographic_correlations` | array | None | Matriz de correlación personalizada NxN para demografía |
| `custom_demographic_columns`| dict | None | Definiciones para features personalizadas (ver Casos de Uso) |
| `disease_effects_config` | list | None | Lista de definiciones de efectos (ver abajo) |
| `date_config` | DateConfig | None | Configuración para la columna `timestamp` |

### Estructura de Retorno

Retorna un diccionario `Dict[str, pd.DataFrame]` con las claves:
*   `"demographics"`: Metadatos del paciente (ID, Grupo, Edad, Género, etc.)
*   `"genes"`: Matriz de expresión (Filas=Pacientes, Cols=Genes)
*   `"proteins"`: Matriz de expresión (Filas=Pacientes, Cols=Proteínas)

---

## Configuración de Efectos de Enfermedad

El `disease_effects_config` permite un control preciso sobre señales biológicas. Puedes modificar genes/proteínas específicos para el grupo "Disease" usando varias transformaciones matemáticas.

### Formato de Configuración

```python
{
    "target_type": "gene",          # "gene" o "protein"
    "index": [0, 5, 12],            # Entero o lista de índices a afectar
    "effect_type": "fold_change",   # Tipo de transformación (ver tabla)
    "effect_value": 2.0,            # Magnitud del efecto
    "group": "Disease"              # Grupo objetivo (usualmente "Disease")
}
```

### Tipos de Efecto Soportados

Todos los efectos se aplican via `ComplexGenerator.apply_stochastic_effects`. Cada entidad (paciente) recibe un offset aleatorio independiente muestreado de la distribucion de `effect_value`.

| Tipo de Efecto | Formula | Descripcion |
|----------------|---------|-------------|
| `fold_change` | $x_{new} = x \cdot value$ | Escalado multiplicativo (ej. sobreexpresion) |
| `additive_shift` | $x_{new} = x + value$ | Desplazamiento aditivo directo |
| `power_transform` | $x_{new} = x^{value}$ | Distorsion no lineal |
| `log_transform` | $x_{new} = \ln(x + \epsilon)$ | Normalizacion logaritmica |
| `variance_scale` | $x_{new} = \mu + (x-\mu)\cdot value$ | Aumenta/disminuye dispersion |
| `polynomial_transform`| $x_{new} = P(x)$ | Mapeo polinomial (coeficientes en value) |
| `sigmoid_transform` | $x_{new} = \frac{1}{1 + e^{-k(x-x_0)}}$ | Saturacion en curva S |

> **Nota para datos de proteinas:** Usa `fold_change` para distribuciones lognormales de proteinas. El uso de `additive_shift` en proteinas emite un aviso de logger porque ahora aplica un desplazamiento aditivo directo (no el desplazamiento historico en espacio logaritmico). Usa `simple_additive_shift` como alias explicito si quieres un desplazamiento aditivo directo sin el aviso.

---

## Datos Longitudinales: `generate_longitudinal_data()`

Genera datos multi-visita (trayectorias).

```python
longitudinal_data = gen.generate_longitudinal_data(
    n_samples=50,
    longitudinal_config={
        "n_visits": 5,          # Número total de visitas por paciente
        "time_step_days": 30,   # Días promedio entre visitas
    },
    # Argumentos estándar de generate()
    n_genes=100
)
```

---

## Configuración Avanzada

### Inyección de Drift y Dinámicas

Puedes pasar diccionarios de configuración directamente a los inyectores internos:

*   `demographics_drift_config`: Lista de objetos `DriftConfig` para demografía.
*   `genes_drift_config`: Lista de objetos `DriftConfig` para genes.
*   `proteins_drift_config`: Lista de objetos `DriftConfig` para proteínas.
*   `genes_dynamics_config`: Escenarios para evolución de genes.

Ejemplo:
```python
drift_conf = DriftConfig(
    method="inject_feature_drift_gradual", 
    params={"feature_cols": ["Age"], "drift_magnitude": 0.5}
)

gen.generate(..., demographics_drift_config=[drift_conf])
```

---

## Casos de Uso Exhaustivos

### Caso 1: Simulación para Descubrimiento de Biomarcadores

**Escenario:** Quieres simular un ensayo clínico donde 5 genes específicos están altamente sobreexpresados en pacientes enfermos.

**Solución:** Usa el efecto `fold_change`.

```python
# Sobreexpresar primeros 5 genes por 4x en grupo Disease
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

### Caso 2: Progresión de Enfermedad Longitudinal

**Escenario:** Simular progresión de Alzheimer donde un nivel de proteína decae con el tiempo.

```python
cohort = gen.generate_longitudinal_data(
    n_samples=100,
    longitudinal_config={
        "n_visits": 12,        # 1 año de datos mensuales
        "time_step_days": 30
    },
    n_proteins=50
)
# Retorna un diccionario con estructuras de datos longitudinales
```

### Caso 3: Modelado de Población Diversa

**Escenario:** Generar un estudio con correlaciones demográficas complejas (ej. Edad altamente correlacionada con IMC).

**Solución:** Inyectar una matriz personalizada `demographic_correlations`.

```python
import numpy as np

# Matriz 3x3: [Age, BMI, BloodPressure]
# Alta correlación (0.8) entre Edad e IMC
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
