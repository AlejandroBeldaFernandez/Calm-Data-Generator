# Real Block Generator Reference

The `calm_data_generator.generators.tabular.RealBlockGenerator` extends `RealGenerator` to support block-based data generation and scheduled drift injection. It is ideal for scenarios where data naturally arrives in batches or blocks (e.g., monthly sales, daily logs).

## Class: `RealBlockGenerator`

### Usage
```python
from calm_data_generator.generators.tabular.RealBlockGenerator import RealBlockGenerator
from calm_data_generator.generators.configs import DriftConfig
import pandas as pd

# Load data
data = pd.read_csv("my_data.csv")

# Initialize
generator = RealBlockGenerator(auto_report=True)

# Generate with blocks defined by a column 'Month'
synthetic_data = generator.generate(
    data=data,
    output_dir="./output_blocks",
    method="lgbm",
    block_column="Month",
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
**Signature:** `__init__(auto_report: bool = True, random_state: int = 42, verbose: bool = True)`

- **Args:**
    - `auto_report`: If True, generates a comprehensive report after processing all blocks.
    - `random_state`: Seed for reproducibility.
    - `verbose`: Enables verbose logging.

### `generate`
**Signature:** `generate(...)`

Generates a complete synthetic dataset by processing each block and applying a drift schedule.

- **Args:**
    - `data` (pd.DataFrame): The full, original dataset.
    - `output_dir` (str): Directory to save output.
    - `method` (str): Synthesis method (e.g., 'cart', 'lgbm', 'ctgan').
    - `target_col` (Optional[str]): Target variable name.
    - `block_column` (Optional[str]): Name of column defining blocks.
    - `chunk_size` (Optional[int]): Create fixed-size blocks (alternative to `block_column`).
    - `chunk_by_timestamp` (Optional[str]): Create blocks based on timestamp changes (alternative).
    - `n_samples_block` (Union[int, Dict]): Number of samples per block (uniform or per-block dict).
    - `drift_config` (List[Dict]): Drift schedule to apply to the generated dataset.
    - `custom_distributions` (Dict): Custom marginal distributions.
    - `date_start`, `date_step`, `date_col`: Configuration for injecting block-aligned timestamps.

- **Returns:** `pd.DataFrame`: The complete synthetic dataset.

### `save_block_dataset`
**Signature:** `save_block_dataset(...)`

Saves the dataset, optionally splitting each block into a separate file.

- **Args:**
    - `synthetic_dataset`: The DataFrame to save.
    - `output_path`: Path to save to.
    - `block_column`: Block column name.
    - `separate_blocks` (bool): If True, saves individual files per block.
