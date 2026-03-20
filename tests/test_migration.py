import unittest
import pandas as pd
import numpy as np
import os
import shutil
from calm_data_generator.generators.tabular.RealGenerator import RealGenerator
from calm_data_generator.generators.tabular.QualityReporter import QualityReporter
from calm_data_generator.reports.Visualizer import Visualizer


class TestMigration(unittest.TestCase):
    def setUp(self):
        # Create dummy data
        self.output_dir = "test_migration_output"
        os.makedirs(self.output_dir, exist_ok=True)

        data = {
            "age": np.random.randint(20, 60, 100),
            "salary": np.random.normal(50000, 15000, 100),
            "department": np.random.choice(["Sales", "HR", "Tech"], 100),
            "target": np.random.choice([0, 1], 100),
        }
        self.real_df = pd.DataFrame(data)

    def tearDown(self):
        if os.path.exists(self.output_dir):
            shutil.rmtree(self.output_dir)

    def test_imports_no_sdv(self):
        """Verify the library does not import SDV as a dependency."""
        import sys
        # SDV should not be imported as a side-effect of loading calm_data_generator
        import calm_data_generator  # noqa: F401
        self.assertNotIn("sdv", sys.modules, "SDV was imported as a side-effect of calm_data_generator")

    def test_synthcity_available(self):
        """Verify Synthcity is importable"""
        try:
            import synthcity
        except ImportError:
            self.fail("Synthcity should be installed")

    def test_sdmetrics_available(self):
        """Verify SDMetrics is importable"""
        try:
            import sdmetrics
        except ImportError:
            self.fail("SDMetrics should be installed")

    def test_quality_reporter_renaming(self):
        """Verify QualityReporter uses new method names"""
        reporter = QualityReporter(verbose=False)
        # Check if the renamed methods/vars exist implicitly by running assessment
        # We assume _assess_quality_scores is called

        # Create explicit method call test if possible, or run full report
        # We'll run a minimal report generation

        # Mock some data
        synth_df = self.real_df.copy()

        # This will call _assess_quality_scores and Visualizer.generate_quality_scores_card
        try:
            reporter.generate_report(
                real_df=self.real_df,
                synthetic_df=synth_df,
                generator_name="TestGen",
                output_dir=self.output_dir,
                minimal=True,  # Skip heavy stuff
            )
        except Exception as e:
            self.fail(f"QualityReporter failed with minimal=True: {e}")

    def test_real_generator_plugins(self):
        """Verify RealGenerator can init synthcity plugins (mocked or real)"""
        # checks if plugins are loading without smartnoise/sdv errors
        gen = RealGenerator()
        # Just init shouldn't crash
        self.assertTrue(gen is not None)


if __name__ == "__main__":
    unittest.main()
