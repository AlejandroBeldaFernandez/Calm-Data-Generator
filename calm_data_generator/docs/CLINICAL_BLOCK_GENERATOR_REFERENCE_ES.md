# Referencia del Clinical Block Generator

El módulo `calm_data_generator.generators.clinical.ClinicGeneratorBlock` proporciona `ClinicalDataGeneratorBlock`, un generador especializado para crear datos clínicos estructurados en bloques. Utiliza internamente `ClinicalDataGenerator` para el mapeo de características específicas del dominio y asegura consistencia a través de visitas de pacientes o lotes de datos.

## Clase: `ClinicalDataGeneratorBlock`

### Uso
```python
from calm_data_generator.generators.clinical.ClinicGeneratorBlock import ClinicalDataGeneratorBlock
from river import synth

# Inicializar
block_gen = ClinicalDataGeneratorBlock()

# Crear generador River para lógica subyacente
base_gen = synth.Agrawal(seed=42)

# Generar bloques de datos clínicos
path = block_gen.generate(
    output_dir="./output_clinical_blocks",
    filename="clinical_blocks.csv",
    n_blocks=4,
    total_samples=4000,
    generators=base_gen,       # Puede ser una instancia única reusada o una lista
    n_samples_block=[1000, 1000, 1000, 1000],
    target_col="Diagnosis",
    drift_config=[ ... ] # Drift opcional
)
```

### `generate`
**Firma:** `generate(...)`

Genera un dataset clínico estructurado en bloques. Orquesta la generación de cada bloque usando `ClinicalDataGenerator`, concatenándolos en un único dataset final (y archivos de bloque individuales si se configura).

- **Argumentos:**
    - `output_dir` (str): Directorio de salida.
    - `filename` (str): Nombre archivo CSV final.
    - `n_blocks` (int): Número de bloques.
    - `total_samples` (int): Muestras totales.
    - `n_samples_block`: Lista de muestras por bloque (o int único).
    - `generators`: Instancia(s) de generador River.
    - `target_col`: Nombre columna objetivo.
    - `balance`: Si balancear clases.
    - `date_start`, `date_step`, `date_col`: Config para inyectar fechas.
    - `generate_report`: Si generar informe clínico.
    - `drift_config`: Lista config de inyección de drift.
    - `dynamics_config`: Config para dinámicas (ej. evolución).

- **Retorna:** `str`: Ruta completa al archivo CSV generado.

### Características Clave
- **Basado en Bloques**: Genera bloques distintos de datos, simulando periodos de tiempo o diferentes fuentes de datos.
- **Mapeo de Features Clínicas**: Mapea automáticamente nombres de features genéricos (x0, x1...) a términos clínicos (Systolic_BP, BMI...) usando `ClinicalDataGenerator`.
- **Drift y Dinámicas**: Soporta inyectar drift y evolucionar features a lo largo del tiempo/bloques.
- **Reporting Especializado**: Usa `ClinicReporter` para generar informes adaptados a distribuciones de datos clínicos.
