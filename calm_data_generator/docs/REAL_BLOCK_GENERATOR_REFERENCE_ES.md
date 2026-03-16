# Referencia del Real Block Generator

`calm_data_generator.generators.tabular.RealBlockGenerator` extiende `RealGenerator` para soportar generación de datos basada en bloques e inyección de drift programada. Es ideal para escenarios donde los datos llegan naturalmente en lotes o bloques (ej. ventas mensuales, logs diarios).

## Clase: `RealBlockGenerator`

### Uso
```python
from calm_data_generator.generators.tabular.RealBlockGenerator import RealBlockGenerator
from calm_data_generator.generators.configs import DriftConfig
import pandas as pd

# Cargar datos
data = pd.read_csv("my_data.csv")

# Inicializar
generator = RealBlockGenerator(auto_report=True)

# Generar con bloques definidos por una columna 'Mes'
synthetic_data = generator.generate(
    data=data,
    output_dir="./output_blocks",
    method="lgbm",
    block_column="Mes",
    drift_config=[
        DriftConfig(
            method="inject_feature_drift",
            params={
                "drift_type": "shift",
                "drift_magnitude": 0.5,
                "feature_cols": ["FeatureA"]
            }
        )
    ]
)
```

### `__init__`
**Firma:** `__init__(auto_report: bool = True, random_state: int = 42, verbose: bool = True)`

- **Argumentos:**
    - `auto_report`: Si es True, genera un informe completo tras procesar todos los bloques.
    - `random_state`: Semilla para reproducibilidad.
    - `verbose`: Habilita logging detallado.

### `generate`
**Firma:** `generate(...)`

Genera un dataset sintético completo procesando cada bloque y aplicando un programa de drift.

- **Argumentos:**
    - `data` (pd.DataFrame): El dataset original completo.
    - `output_dir` (str): Directorio donde guardar.
    - `method` (str): Método de síntesis (ej., 'cart', 'lgbm', 'ctgan').
    - `target_col` (Optional[str]): Nombre variable objetivo.
    - `block_column` (Optional[str]): Nombre de la columna que define los bloques.
    - `chunk_size` (Optional[int]): Crea bloques de tamaño fijo (alternativa a `block_column`).
    - `chunk_by_timestamp` (Optional[str]): Crea bloques basados en cambios de timestamp (alternativa).
    - `n_samples_block` (Union[int, Dict]): Número de muestras por bloque (uniforme o dict por bloque).
    - `drift_config` (List[Dict]): Programa de drift a aplicar al dataset generado.
    - `custom_distributions` (Dict): Distribuciones marginales personalizadas.
    - `date_start`, `date_step`, `date_col`: Configuración para inyectar timestamps alineados con bloques.

- **Retorna:** `pd.DataFrame`: El dataset sintético completo.

### `save_block_dataset`
**Firma:** `save_block_dataset(...)`

Guarda el dataset, opcionalmente dividiendo cada bloque en un archivo separado.

- **Argumentos:**
    - `synthetic_dataset`: El DataFrame a guardar.
    - `output_path`: Ruta donde guardar.
    - `block_column`: Nombre columna de bloque.
    - `separate_blocks` (bool): Si es True, guarda archivos individuales por bloque.
