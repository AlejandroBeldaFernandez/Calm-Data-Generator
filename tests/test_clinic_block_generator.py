import unittest
import pandas as pd
import shutil
import os

from calm_data_generator.generators.clinical.ClinicGeneratorBlock import (
    ClinicalDataGeneratorBlock,
)
from calm_data_generator.generators.configs import DriftConfig, ReportConfig


class TestClinicalBlockGenerator(unittest.TestCase):
    def setUp(self):
        self.output_dir = "test_output_clinic_block"
        os.makedirs(self.output_dir, exist_ok=True)
        self.filename = "test_clinic_blocks.csv"

    def tearDown(self):
        if os.path.exists(self.output_dir):
            shutil.rmtree(self.output_dir)

    def test_initialization(self):
        gen = ClinicalDataGeneratorBlock()
        self.assertIsInstance(gen, ClinicalDataGeneratorBlock)

    def test_generate_clinical_blocks(self):
        gen = ClinicalDataGeneratorBlock()

        full_path = gen.generate(
            output_dir=self.output_dir,
            filename=self.filename,
            n_blocks=2,
            total_samples=20,
            n_samples_block=[10, 10],
            n_genes=5,
            n_proteins=5,
            target_col="diagnosis",
            generate_report=False,
        )

        self.assertTrue(os.path.exists(full_path))
        df = pd.read_csv(full_path)
        self.assertEqual(len(df), 20)
        self.assertTrue("block" in df.columns)
        self.assertEqual(set(df["block"].unique()), {1, 2})
        self.assertTrue("Age" in df.columns)

    def test_generate_with_drift_and_report_config(self):
        """Test with DriftConfig and ReportConfig."""
        gen = ClinicalDataGeneratorBlock()

        drift_conf = DriftConfig(
            method="inject_feature_drift",
            params={"missing_fraction": 0.1, "columns": ["Age"]},
        )
        report_conf = ReportConfig(
            output_dir=self.output_dir, target_column="diagnosis"
        )

        full_path = gen.generate(
            output_dir=self.output_dir,
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
        self.assertTrue(os.path.exists(full_path))


if __name__ == "__main__":
    unittest.main()
