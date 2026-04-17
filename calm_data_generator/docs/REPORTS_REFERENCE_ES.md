# Referencia de Reportes

La biblioteca `calm_data_generator` incluye un conjunto de herramientas de reporte diseñadas para evaluar la calidad, privacidad y características de los datos generados.

---

## Referencia de la Clase ReportConfig

**Importar:** `from calm_data_generator.generators.configs import ReportConfig`

`ReportConfig` es un modelo Pydantic que proporciona configuración con tipos seguros para la generación de reportes en todas las clases de reportes.

### Parámetros

| Parámetro | Tipo | Por Defecto | Descripción |
|-----------|------|-------------|-------------|
| `output_dir` | str | `"output"` | Directorio para guardar reportes generados |
| `auto_report` | bool | `True` | Generar reportes automáticamente después de la generación de datos |
| `minimal` | bool | `False` | Generar reportes mínimos (más rápido, menos detalle) |
| `target_column` | str | `None` | Columna objetivo/etiqueta para análisis de clasificación/regresión |
| `time_col` | str | `None` | Columna de tiempo para análisis de series temporales |
| `block_column` | str | `None` | Columna identificadora de bloques para datos basados en bloques |
| `resample_rule` | str/int | `None` | Regla de remuestreo para series temporales (ej., `"1D"`, `"1H"`) |
| `privacy_check` | bool | `False` | Habilitar evaluación de privacidad (métricas DCR) |
| `adversarial_validation` | bool | `False` | Habilitar validación basada en discriminador |
| `focus_columns` | List[str] | `None` | Columnas específicas en las que enfocar el análisis |
| `constraints_stats` | Dict[str, int] | `None` | Estadísticas de violación de restricciones |
| `sequence_config` | Dict | `None` | Configuración para análisis basado en secuencias |
| `per_block_external_reports` | bool | `False` | Generar reportes separados por bloque |
| `use_scgft` | bool | `False` | Habilitar evaluación especializada scGFT para single-cell |

### Ejemplos de Uso

**Configuración Básica de Reporte:**
```python
from calm_data_generator.generators.configs import ReportConfig

report_config = ReportConfig(
    output_dir="./mis_reportes",
    target_column="target",
    privacy_check=True,
    adversarial_validation=True
)
```

**Reporte de Series Temporales:**
```python
report_config = ReportConfig(
    output_dir="./reporte_series_temporales",
    time_col="timestamp",
    resample_rule="1D",  # Agregación diaria
    target_column="ventas"
)
```

**Reporte Basado en Bloques:**
```python
report_config = ReportConfig(
    output_dir="./reporte_bloques",
    block_column="paciente_id",
    per_block_external_reports=True,
    target_column="diagnostico"
)
```

**Reporte Mínimo (Rápido):**
```python
report_config = ReportConfig(
    output_dir="./reporte_rapido",
    minimal=True,
    focus_columns=["edad", "ingresos", "target"]
)
```

---

## Reporter de Calidad (`Tabular`)
**Módulo:** `calm_data_generator.generators.tabular.QualityReporter`

Genera informes completos comparando datos tabulares reales y sintéticos.

### `generate_comprehensive_report`
Genera un informe estático incluyendo:
- **Puntuaciones de Calidad Global**: Métricas de similitud generales y por columna.
- **Evaluación de Privacidad**: Métricas de Distancia al Registro Más Cercano (DCR).
- **Visualizaciones**: Histogramas, gráficos de densidad, proyecciones PCA/UMAP.
- **Métricas ARI (Separabilidad de Clases)**: Índice de Rand Ajustado (ARI) mediante K-Means (k=2) para cuantificar qué tan bien se separan las clases (Casos vs Controles) tanto en datos reales como sintéticos.
- **Análisis de Drift**: Comparación visual de distribuciones de features.

```python
from calm_data_generator.generators.configs import ReportConfig

reporter = QualityReporter(verbose=True)
reporter.generate_comprehensive_report(
    real_df=original_df,
    synthetic_df=synthetic_df,
    generator_name="MyGenerator",
    report_config=ReportConfig(
        output_dir="./report_output",
        target_column="target_col"
    )
)
```

### `calculate_quality_metrics`
Calcula métricas de calidad (SDMetrics) para dos conjuntos de datos sin generar un informe completo.

```python
reporter = QualityReporter(verbose=False)
metrics = reporter.calculate_quality_metrics(
    real_df=df1, 
    synthetic_df=df2
)
# Devuelve: {'overall_quality_score': 0.85, 'weighted_quality_score': 0.82}
```

### `calculate_ari`
Cálculo independiente del Índice de Rand Ajustado (ARI) para cuantificar la separabilidad de clases.

```python
ari_metrics = reporter.calculate_ari(
    real_df=df1, 
    synthetic_df=df2,
    target_col="label"
)
# Devuelve: {'ari_original': 0.95, 'ari_synthetic': 0.98, 'ari_improvement': 0.03}
```

## Reporter Discriminador (Adversarial Validation)
**Módulo:** `calm_data_generator.reports.DiscriminatorReporter`

Este reporter entrena un modelo clasificador (Random Forest) para intentar distinguir entre datos reales y sintéticos. Se utiliza para detectar drift o evaluar la fidelidad general.

### Métricas Clave
- **Similarity Score (Indistinguishability)**: (0.0 - 1.0).
    - **Fórmula**: `1 - 2 * |AUC - 0.5|`
    - `1.0`: Datos indistinguibles (AUC = 0.5). Excelente Calidad.
    - `0.0`: Datos fácilmente distinguibles (AUC = 1.0 o 0.0). Drift detectado o baja calidad.
- **Confusion Score**: Capacidad de los datos para "confundir" al discriminador (basado en Accuracy inversamente).
- **Explicabilidad**:
    - **Feature Importance**: Qué variables permitieron al modelo distinguir los datos.
    - **SHAP Values**: Explicación detallada del impacto de cada feature.

### Uso
Este reporter se integra automáticamente en `QualityReporter` si se activa el parámetro opcional:
```python
reporter.generate_comprehensive_report(
    ...,
    report_config=ReportConfig(
        output_dir="./report_output",
        adversarial_validation=True  # Activar Discriminator
    )
)
```

## Reporter de Stream (`Stream`)
**Módulo:** `calm_data_generator.generators.stream.StreamReporter`

Diseñado para analizar flujos de datos sintéticos sin un dataset de referencia "real" directo (aunque puede comparar contra expectativas).

### `generate_report`
Genera un informe para un dataset sintético:
- **Perfilado de Datos**: Integración con YData Profiling.
- **Visualizaciones**: Gráficos de densidad y reducción de dimensionalidad.
- **Análisis por Bloques**: Puede generar informes separados para cada bloque de datos.

```python
reporter = StreamReporter()
reporter.generate_report(
    synthetic_df=stream_df,
    generator_name="StreamGen",
    report_config=ReportConfig(output_dir="./stream_report")
)
```


## Evaluación Single-Cell (scGFT)
**Módulo:** `calm_data_generator.reports.QualityReporter`

La biblioteca integra `scGFT_evaluador` para proporcionar validación especializada para datos de secuenciación de ARN de célula única (scRNA-seq). Este método utiliza Transformadas de Fourier en Grafos (GFT) para evaluar si los datos sintéticos preservan el manifold subyacente y la estructura biológica de las células originales.

### Características Clave
- **Preservación del Manifold**: Evalúa si se mantienen las relaciones célula a célula.
- **Integridad de Clusters/Poblaciones**: Proporciona métricas sobre qué tan bien las células sintéticas representan las poblaciones reales.
- **Integración en el Dashboard**: Genera una pestaña dedicada `scgft_report.html` en el panel HTML.

### Uso
Para activar esta evaluación, asegúrese de tener instalados `scGFT_evaluador`, `scanpy` y `anndata`, luego establezca `use_scgft=True` en su `ReportConfig`:

```python
from calm_data_generator.generators.configs import ReportConfig

reporter.generate_comprehensive_report(
    ...,
    report_config=ReportConfig(
        output_dir="./sc_report",
        use_scgft=True,
        target_column="cell_type"  # Opcional: usar para evaluar conservación de poblaciones
    )
)
```

> [!IMPORTANT]
> **Formato de Datos**: Este método está diseñado específicamente para datos de célula única donde las columnas representan genes y las filas representan células. **No se recomienda** para datos estándar de bulk o tabulares.

## Reporter Clínico (`Clinical`)
**Módulo:** `calm_data_generator.generators.clinical.ClinicReporter`

Una versión especializada de `StreamReporter` para datos clínicos. Hereda capacidades de reporte estándar pero está adaptado para manejar conjuntos de características clínicas y puede incluir verificaciones específicas de dominio en el futuro.

```python
reporter = ClinicReporter()
reporter.generate_report(...)
```

### JSON de Resultados del Reporte (`report_results.json`)
Cada reporte genera un archivo `report_results.json` que contiene las métricas en bruto:

```json
{
  "generator_name": "MyGenerator",
  "generation_timestamp": "2024-01-01T12:00:00",
  "real_rows": 1000,
  "synthetic_rows": 1000,
  "quality_scores": {
    "overall_quality_score": 0.85,
    "weighted_quality_score": 0.82
  },
  "compared_data_files": {
    "original": "real_data",
    "generated": "synthetic_data"
  }
}
```

> [!NOTE]
> **Informes de Privacidad**: Las características de privacidad (métricas DCR) ahora están integradas en `QualityReporter`. Usa `privacy_check=True` al generar informes.

