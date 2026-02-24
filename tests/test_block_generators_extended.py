import pytest
import pandas as pd
import numpy as np
import tempfile
import os
from calm_data_generator.generators.tabular.RealBlockGenerator import RealBlockGenerator
from calm_data_generator.generators.clinical.ClinicGeneratorBlock import (
    ClinicalDataGeneratorBlock,
)
from calm_data_generator.generators.configs import DriftConfig


@pytest.fixture
def block_data():
    """Create data with block structure."""
    np.random.seed(42)
    df = pd.DataFrame(
        {
            "block_id": ["A"] * 50 + ["B"] * 50,
            "feature": np.concatenate(
                [np.random.normal(0, 1, 50), np.random.normal(10, 2, 50)]
            ),
            "target": np.random.choice([0, 1], 100),
        }
    )
    return df


def test_real_block_generator_samples_per_block(block_data):
    """Test block generation with variable samples per block."""
    with tempfile.TemporaryDirectory() as tmpdir:
        gen = RealBlockGenerator(auto_report=False)

        n_samples_block = {"A": 10, "B": 20}
        synth = gen.generate(
            data=block_data,
            output_dir=tmpdir,
            method="cart",
            block_column="block_id",
            target_col="target",
            n_samples_block=n_samples_block,
        )

        assert len(synth) == 30  # 10 + 20
        assert synth["block_id"].value_counts()["A"] == 10
        assert synth["block_id"].value_counts()["B"] == 20
        assert os.path.exists(os.path.join(tmpdir, "complete_block_dataset_cart.csv"))


def test_clinical_block_generator_multiple_blocks():
    """Test generating clinical data for multiple blocks using ClinicalDataGeneratorBlock."""
    with tempfile.TemporaryDirectory() as tmpdir:
        gen = ClinicalDataGeneratorBlock()

        full_path = gen.generate(
            output_dir=tmpdir,
            filename="multi_block_clinical.csv",
            n_blocks=3,
            total_samples=30,
            n_samples_block=[10, 10, 10],
            n_genes=5,
            n_proteins=5,
            target_col="diagnosis",
            generate_report=False,
        )

        assert os.path.exists(full_path)
        df = pd.read_csv(full_path)
        assert len(df) == 30
        assert "block" in df.columns
        assert set(df["block"].unique()) == {1, 2, 3}
        assert "Age" in df.columns


def test_clinical_block_generator_with_custom_labels():
    """Test generating clinical blocks with custom block labels."""
    with tempfile.TemporaryDirectory() as tmpdir:
        gen = ClinicalDataGeneratorBlock()

        full_path = gen.generate(
            output_dir=tmpdir,
            filename="labeled_blocks.csv",
            n_blocks=2,
            total_samples=20,
            n_samples_block=[10, 10],
            n_genes=5,
            n_proteins=5,
            block_labels=["pre_treatment", "post_treatment"],
            generate_report=False,
        )

        assert os.path.exists(full_path)
        df = pd.read_csv(full_path)
        assert set(df["block"].unique()) == {"pre_treatment", "post_treatment"}


def test_clinical_block_generator_control_disease_ratio():
    """Test that control_disease_ratio is respected across blocks."""
    with tempfile.TemporaryDirectory() as tmpdir:
        gen = ClinicalDataGeneratorBlock()

        full_path = gen.generate(
            output_dir=tmpdir,
            filename="ratio_blocks.csv",
            n_blocks=2,
            total_samples=100,
            n_samples_block=[50, 50],
            n_genes=5,
            n_proteins=5,
            control_disease_ratio=0.8,  # Mostly controls
            generate_report=False,
        )

        assert os.path.exists(full_path)
        df = pd.read_csv(full_path)
        assert len(df) == 100


def test_clinical_block_generator_with_drift():
    """Test that drift is applied correctly to clinical blocks."""
    with tempfile.TemporaryDirectory() as tmpdir:
        gen = ClinicalDataGeneratorBlock()

        drift = DriftConfig(
            method="inject_feature_drift",
            params={"missing_fraction": 0.1, "columns": ["Age"]},
        )

        full_path = gen.generate(
            output_dir=tmpdir,
            filename="drift_blocks.csv",
            n_blocks=2,
            total_samples=20,
            n_samples_block=[10, 10],
            n_genes=5,
            n_proteins=5,
            drift_config=[drift],
            generate_report=False,
        )

        assert os.path.exists(full_path)
        df = pd.read_csv(full_path)
        # After drift with missing_fraction=0.1, some Age values should be NaN
        assert "Age" in df.columns
