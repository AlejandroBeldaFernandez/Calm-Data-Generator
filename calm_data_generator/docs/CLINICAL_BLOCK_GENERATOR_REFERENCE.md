# Clinical Block Generator Reference

The `calm_data_generator.generators.clinical.ClinicGeneratorBlock` module provides `ClinicalDataGeneratorBlock`, a specialized generator for creating block-structured clinical data. It utilizes `ClinicalDataGenerator` for domain-specific feature mapping and ensures consistency across patient visits or data batches.

## Class: `ClinicalDataGeneratorBlock`

### Usage
```python
from calm_data_generator.generators.clinical.ClinicGeneratorBlock import ClinicalDataGeneratorBlock
from river import synth

# Initialize
block_gen = ClinicalDataGeneratorBlock()

# Create River generator for underlying logic
base_gen = synth.Agrawal(seed=42)

# Generate clinical data blocks
path = block_gen.generate(
    output_dir="./output_clinical_blocks",
    filename="clinical_blocks.csv",
    n_blocks=4,
    total_samples=4000,
    generators=base_gen,       # Can be a single instance reused or a list
    n_samples_block=[1000, 1000, 1000, 1000],
    target_col="Diagnosis",
    drift_config=[ ... ] # Optional drift
)
```

### `generate`
**Signature:** `generate(...)`

Generates a block-structured clinical dataset. It orchestrates the generation of each block using `ClinicalDataGenerator`, concatenating them into a single final dataset (and individual block files if configured).

- **Args:**
    - `output_dir` (str): Output directory.
    - `filename` (str): Final CSV filename.
    - `n_blocks` (int): Number of blocks.
    - `total_samples` (int): Total samples.
    - `n_samples_block`: List of samples per block (or single int).
    - `generators`: River generator instance(s).
    - `target_col`: Target column name.
    - `balance`: Whether to balance classes.
    - `date_start`, `date_step`, `date_col`: Configuration for injecting dates.
    - `generate_report`: Whether to generate a clinical report.
    - `drift_config`: List of drift injection configs.
    - `dynamics_config`: Configuration for dynamics (e.g. evolution).

- **Returns:** `str`: Full path to the generated CSV file.

### Key Features
- **Block-Based**: Generates distinct blocks of data, simulating time periods or different data sources.
- **Clinical Feature Mapping**: Automatically maps generic feature names (x0, x1...) to clinical terms (Systolic_BP, BMI...) using `ClinicalDataGenerator`.
- **Drift & Dynamics**: Supports injecting drift and evolving features over time/blocks.
- **Specialized Reporting**: Uses `ClinicReporter` to generate reports tailored to clinical data distributions.
