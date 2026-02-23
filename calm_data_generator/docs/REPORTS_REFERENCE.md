# Reports Reference

The `calm_data_generator` library includes a suite of reporting tools designed to assess the quality, privacy, and characteristics of generated data.

---

## ReportConfig Class Reference

**Import:** `from calm_data_generator.generators.configs import ReportConfig`

`ReportConfig` is a Pydantic model that provides type-safe configuration for report generation across all reporter classes.

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `output_dir` | str | `"output"` | Directory to save generated reports |
| `auto_report` | bool | `True` | Automatically generate reports after data generation |
| `minimal` | bool | `False` | Generate minimal reports (faster, less detail) |
| `target_column` | str | `None` | Target/label column for classification/regression analysis |
| `time_col` | str | `None` | Time column for time-series analysis |
| `block_column` | str | `None` | Block identifier column for block-based data |
| `resample_rule` | str/int | `None` | Resampling rule for time-series (e.g., `"1D"`, `"1H"`) |
| `privacy_check` | bool | `False` | Enable privacy assessment (DCR metrics) |
| `adversarial_validation` | bool | `False` | Enable discriminator-based validation |
| `focus_columns` | List[str] | `None` | Specific columns to focus analysis on |
| `constraints_stats` | Dict[str, int] | `None` | Constraint violation statistics |
| `sequence_config` | Dict | `None` | Configuration for sequence-based analysis |
| `per_block_external_reports` | bool | `False` | Generate separate reports per block |

### Usage Examples

**Basic Report Configuration:**
```python
from calm_data_generator.generators.configs import ReportConfig

report_config = ReportConfig(
    output_dir="./my_reports",
    target_column="target",
    privacy_check=True,
    adversarial_validation=True
)
```

**Time-Series Report:**
```python
report_config = ReportConfig(
    output_dir="./timeseries_report",
    time_col="timestamp",
    resample_rule="1D",  # Daily aggregation
    target_column="sales"
)
```

**Block-Based Report:**
```python
report_config = ReportConfig(
    output_dir="./block_report",
    block_column="patient_id",
    per_block_external_reports=True,
    target_column="diagnosis"
)
```

**Minimal Report (Fast):**
```python
report_config = ReportConfig(
    output_dir="./quick_report",
    minimal=True,
    focus_columns=["age", "income", "target"]
)
```

---

## Quality Reporter (`Tabular`)
**Module:** `calm_data_generator.generators.tabular.QualityReporter`

Generates comprehensive reports comparing real and synthetic tabular data.

### `generate_comprehensive_report`
Generates a static report including:
- **Overall Quality Scores**: Overall and column-wise similarity metrics.
- **Privacy Assessment**: Distance to Closest Record (DCR) metrics.
- **Visualizations**: Histograms, density plots, PCA/UMAP projections.
- **Drift Analysis**: Visual comparison of feature distributions.

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
Calculates quality metrics (SDMetrics) for two datasets without generating a full report.

```python
reporter = QualityReporter(verbose=False)
metrics = reporter.calculate_quality_metrics(
    real_df=df1, 
    synthetic_df=df2
)
# Returns: {'overall_quality_score': 0.85, 'weighted_quality_score': 0.82}
```

## Discriminator Reporter (Adversarial Validation)
**Module:** `calm_data_generator.reports.DiscriminatorReporter`

This reporter trains a classifier model (Random Forest) to attempt to distinguish between real and synthetic data. It is used to detect drift or assess general fidelity.

### Key Metrics
- **Similarity Score (Indistinguishability)**: (0.0 - 1.0).
    - **Formula**: `1 - 2 * |AUC - 0.5|`
    - `1.0`: Indistinguishable data (AUC = 0.5). Excellent Quality.
    - `0.0`: Easily distinguishable data (AUC = 1.0 or 0.0). Drift detected or poor quality.
- **Confusion Score**: Ability of the data to "confuse" the discriminator (based on inverted Accuracy).
- **Explainability**:
    - **Feature Importance**: Which variables allowed the model to distinguish the data.
    - **SHAP Values**: Detailed explanation of the impact of each feature.

### Usage
This reporter is automatically integrated into `QualityReporter` if the optional parameter is activated:
```python
reporter.generate_comprehensive_report(
    ...,
    report_config=ReportConfig(
        output_dir="./report_output",
        adversarial_validation=True  # Activate Discriminator
    )
)
```

## Stream Reporter (`Stream`)
**Module:** `calm_data_generator.generators.stream.StreamReporter`

Designed for analyzing synthetic data streams without a direct "real" reference dataset (though it can compare against expectations).

### `generate_report`
Generates a report for a synthetic dataset:
- **Data Profiling**: YData Profiling integration.
- **Visualizations**: Density plots and dimensionality reduction.
- **Block-wise Analysis**: Can generate separate reports for each data block.

```python
reporter = StreamReporter()
reporter.generate_report(
    synthetic_df=stream_df,
    generator_name="StreamGen",
    report_config=ReportConfig(output_dir="./stream_report")
)
```


## Clinic Reporter (`Clinical`)
**Module:** `calm_data_generator.generators.clinical.ClinicReporter`

A specialized version of `StreamReporter` for clinical data. It inherits standard reporting capabilities but is tailored to handle clinical feature sets and may include domain-specific checks in the future.

```python
reporter = ClinicReporter()
reporter.generate_report(...)
```

### Report Results JSON (`report_results.json`)
Every report generates a `report_results.json` file containing the raw metrics:

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
> **Privacy Reporting**: Privacy features (DCR metrics) are now integrated into `QualityReporter`. Use `privacy_check=True` when generating reports.

