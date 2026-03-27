import pandas as pd
import shutil
import os

from calm_data_generator.generators.clinical.ClinicGeneratorBlock import (
    ClinicalDataGeneratorBlock,
)
from calm_data_generator.generators.configs import DriftConfig, ReportConfig


def test_initialization():
    gen = ClinicalDataGeneratorBlock()
    assert isinstance(gen, ClinicalDataGeneratorBlock)


def test_generate_clinical_blocks():
    output_dir = "test_output_clinic_block"
    os.makedirs(output_dir, exist_ok=True)
    filename = "test_clinic_blocks.csv"
    try:
        gen = ClinicalDataGeneratorBlock()

        full_path = gen.generate(
            output_dir=output_dir,
            filename=filename,
            n_blocks=2,
            total_samples=20,
            n_samples_block=[10, 10],
            n_genes=5,
            n_proteins=5,
            target_col="diagnosis",
            generate_report=False,
        )

        assert os.path.exists(full_path)
        df = pd.read_csv(full_path)
        assert len(df) == 20
        assert "block" in df.columns
        assert set(df["block"].unique()) == {1, 2}
        assert "Age" in df.columns
    finally:
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)


def test_generate_with_drift_and_report_config():
    """Test with DriftConfig and ReportConfig."""
    output_dir = "test_output_clinic_block"
    os.makedirs(output_dir, exist_ok=True)
    try:
        gen = ClinicalDataGeneratorBlock()

        drift_conf = DriftConfig(
            method="inject_feature_drift",
            params={"missing_fraction": 0.1, "columns": ["Age"]},
        )
        report_conf = ReportConfig(
            output_dir=output_dir, target_column="diagnosis"
        )

        full_path = gen.generate(
            output_dir=output_dir,
            filename="test_clinic_config.csv",
            n_blocks=1,
            total_samples=10,
            n_samples_block=[10],
            n_genes=5,
            n_proteins=5,
            target_col="diagnosis",
            drift_config=[drift_conf],
            report_config=report_conf,
            generate_report=False,
        )
        assert os.path.exists(full_path)
    finally:
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
