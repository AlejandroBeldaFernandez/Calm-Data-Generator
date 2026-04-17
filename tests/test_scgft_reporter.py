import os
import shutil
import unittest
import pandas as pd
import numpy as np
import importlib
from unittest.mock import MagicMock, patch

import calm_data_generator.reports.QualityReporter as QRModule
if not hasattr(QRModule, '__file__'):
    QRModule = importlib.import_module('calm_data_generator.reports.QualityReporter')

from calm_data_generator.reports.QualityReporter import QualityReporter
from calm_data_generator.generators.configs import ReportConfig


class TestScGFTReporter(unittest.TestCase):

    def setUp(self):
        self.output_dir = "test_scgft_output"
        os.makedirs(self.output_dir, exist_ok=True)

        genes = [f"Gene_{i}" for i in range(50)]
        self.real_df = pd.DataFrame(
            np.random.poisson(lam=1.0, size=(100, 50)), columns=genes
        )
        self.real_df["cell_type"] = np.random.choice(["TypeA", "TypeB"], size=100)

        self.synth_df = pd.DataFrame(
            np.random.poisson(lam=0.9, size=(100, 50)), columns=genes
        )
        self.synth_df["cell_type"] = np.random.choice(["TypeA", "TypeB"], size=100)

    def tearDown(self):
        if os.path.exists(self.output_dir):
            shutil.rmtree(self.output_dir)

    def test_scgft_integration(self):
        mock_adata = MagicMock()
        mock_adata.obs.__getitem__.return_value.unique.return_value.tolist.return_value = [
            "TypeA", "TypeB"
        ]

        def mock_run_all(a1, a2, genes_top, col_grupo, grupo_a, grupo_b, **kwargs):
            print("scGFT ARI: 0.85")
            return pd.DataFrame([{"ARI_TypeA": 0.85}])

        with patch.object(QRModule, 'SCGFT_AVAILABLE', True, create=True), \
             patch.object(QRModule, 'ScGFT_Evaluator', create=True) as mock_cls, \
             patch.object(QRModule, 'sc', create=True) as mock_sc, \
             patch.object(QRModule, 'ad', create=True) as mock_ad:

            mock_ad.AnnData.return_value = mock_adata
            mock_cls.run_all.side_effect = mock_run_all

            reporter = QualityReporter(verbose=True)
            reporter._run_scgft_evaluation(
                real_df=self.real_df,
                synthetic_df=self.synth_df,
                output_dir=self.output_dir,
                target_col="cell_type",
            )

            mock_cls.run_all.assert_called_once()

            report_file = os.path.join(self.output_dir, "scgft_report.html")
            self.assertTrue(os.path.exists(report_file), "scGFT HTML report was not created")

            with open(report_file) as f:
                content = f.read()
                self.assertIn("scGFT Single-Cell Evaluation Report", content)
                self.assertIn("scGFT ARI: 0.85", content)

    def test_scgft_not_available_graceful_exit(self):
        with patch.object(QRModule, 'SCGFT_AVAILABLE', False, create=True):
            reporter = QualityReporter(verbose=True)
            reporter._run_scgft_evaluation(
                real_df=self.real_df,
                synthetic_df=self.synth_df,
                output_dir=self.output_dir,
                target_col="cell_type",
            )
            report_file = os.path.join(self.output_dir, "scgft_report.html")
            self.assertFalse(os.path.exists(report_file))

    def test_scgft_run_all_signature(self):
        """run_all must be called with genes_top, col_grupo, grupo_a, grupo_b."""
        mock_adata = MagicMock()
        mock_adata.obs.__getitem__.return_value.unique.return_value.tolist.return_value = [
            "TypeA", "TypeB"
        ]

        call_kwargs = {}

        def capture_run_all(a1, a2, genes_top, col_grupo, grupo_a, grupo_b, **kwargs):
            call_kwargs.update({
                "genes_top": genes_top,
                "col_grupo": col_grupo,
                "grupo_a": grupo_a,
                "grupo_b": grupo_b,
            })
            return pd.DataFrame([{"ARI_TypeA": 0.5}])

        with patch.object(QRModule, 'SCGFT_AVAILABLE', True, create=True), \
             patch.object(QRModule, 'ScGFT_Evaluator', create=True) as mock_cls, \
             patch.object(QRModule, 'sc', create=True), \
             patch.object(QRModule, 'ad', create=True) as mock_ad:

            mock_ad.AnnData.return_value = mock_adata
            mock_cls.run_all.side_effect = capture_run_all

            reporter = QualityReporter(verbose=False)
            reporter._run_scgft_evaluation(
                real_df=self.real_df,
                synthetic_df=self.synth_df,
                output_dir=self.output_dir,
                target_col="cell_type",
            )

        self.assertEqual(call_kwargs["col_grupo"], "cell_type")
        self.assertIn(call_kwargs["grupo_a"], ["TypeA", "TypeB"])
        self.assertIn(call_kwargs["grupo_b"], ["TypeA", "TypeB"])
        self.assertIsInstance(call_kwargs["genes_top"], list)


if __name__ == "__main__":
    unittest.main()
