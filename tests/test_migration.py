import pandas as pd
import numpy as np
import os
import shutil
import pytest
from calm_data_generator.generators.tabular.RealGenerator import RealGenerator
from calm_data_generator.generators.tabular.QualityReporter import QualityReporter
from calm_data_generator.reports.Visualizer import Visualizer


def _make_real_df():
    data = {
        "age": np.random.randint(20, 60, 100),
        "salary": np.random.normal(50000, 15000, 100),
        "department": np.random.choice(["Sales", "HR", "Tech"], 100),
        "target": np.random.choice([0, 1], 100),
    }
    return pd.DataFrame(data)


def test_imports_no_sdv():
    """Verify the library does not import SDV as a dependency."""
    import sys
    # SDV should not be imported as a side-effect of loading calm_data_generator
    import calm_data_generator  # noqa: F401
    assert "sdv" not in sys.modules, "SDV was imported as a side-effect of calm_data_generator"


def test_synthcity_available():
    """Verify Synthcity is importable"""
    try:
        import synthcity
    except ImportError:
        pytest.fail("Synthcity should be installed")


def test_sdmetrics_available():
    """Verify SDMetrics is importable"""
    try:
        import sdmetrics
    except ImportError:
        pytest.fail("SDMetrics should be installed")


def test_quality_reporter_renaming():
    """Verify QualityReporter uses new method names"""
    output_dir = "test_migration_output"
    os.makedirs(output_dir, exist_ok=True)
    real_df = _make_real_df()
    try:
        reporter = QualityReporter(verbose=False)
        # Check if the renamed methods/vars exist implicitly by running assessment
        # We assume _assess_quality_scores is called

        # Create explicit method call test if possible, or run full report
        # We'll run a minimal report generation

        # Mock some data
        synth_df = real_df.copy()

        # This will call _assess_quality_scores and Visualizer.generate_quality_scores_card
        try:
            reporter.generate_report(
                real_df=real_df,
                synthetic_df=synth_df,
                generator_name="TestGen",
                output_dir=output_dir,
                minimal=True,  # Skip heavy stuff
            )
        except Exception as e:
            pytest.fail(f"QualityReporter failed with minimal=True: {e}")
    finally:
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)


def test_real_generator_plugins():
    """Verify RealGenerator can init synthcity plugins (mocked or real)"""
    # checks if plugins are loading without smartnoise/sdv errors
    gen = RealGenerator()
    # Just init shouldn't crash
    assert gen is not None
