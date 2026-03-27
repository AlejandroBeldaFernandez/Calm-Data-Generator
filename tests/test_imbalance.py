import pandas as pd
import numpy as np
import shutil
import tempfile
import os
import pytest
from calm_data_generator.generators.tabular.RealGenerator import RealGenerator


def test_balance_imbalanced_data():
    """Test A: Feed Imbalanced Data (90/10) -> Generate Balanced (50/50)."""
    output_dir = tempfile.mkdtemp()
    np.random.seed(42)
    try:
        # Create imbalanced data (90% class 0, 10% class 1)
        n = 1000
        df = pd.DataFrame(
            {
                "feature1": np.random.normal(0, 1, n),
                "feature2": np.random.normal(5, 2, n),
                "target": np.concatenate([np.zeros(900), np.ones(100)]),
            }
        )

        try:
            # Try to use a faster method if possible, but CART/Bayesian is default
            # We use Bayesian if available or just default 'cart'
            generator = RealGenerator()

            # Request balancing
            synth_df = generator.generate(
                data=df,
                n_samples=200,
                target_col="target",
                balance=True,
                output_dir=output_dir,
                save_dataset=False,
            )

            if synth_df is None:
                pytest.fail("Generator returned None")

            # Check counts
            counts = synth_df["target"].value_counts(normalize=True)
            print(f"\nBalanced Output Counts:\n{counts}")

            # Should be roughly 0.5 each (allow some variance for iterative convergence)
            assert 0.3 <= counts[0] <= 0.7, f"Class 0 proportion {counts[0]} not balanced"
            assert 0.3 <= counts[1] <= 0.7, f"Class 1 proportion {counts[1]} not balanced"

        except ImportError as e:
            print(
                f"Skipping test_balance_imbalanced_data due to missing dependency: {e}"
            )
            # User asked to report errors, raising it as failure or printing is fine.
            # I will let it fail so it shows in the report.
            raise e
        except Exception as e:
            pytest.fail(f"Test failed with error: {e}")
    finally:
        shutil.rmtree(output_dir)


def test_create_imbalance_from_balanced():
    """Test B: Feed Balanced Data (50/50) -> Generate Imbalanced (90/10)."""
    output_dir = tempfile.mkdtemp()
    np.random.seed(42)
    try:
        # Create balanced data
        n = 1000
        df = pd.DataFrame(
            {
                "feature1": np.random.normal(0, 1, n),
                "target": np.concatenate([np.zeros(500), np.ones(500)]),
            }
        )

        # Force imbalance: 90% Class 0
        custom_dist = {"target": {0: 0.9, 1: 0.1}}

        try:
            generator = RealGenerator()

            synth_df = generator.generate(
                data=df,
                n_samples=500,
                target_col="target",
                custom_distributions=custom_dist,
                output_dir=output_dir,
                save_dataset=False,
            )

            # Check counts
            counts = synth_df["target"].value_counts(normalize=True)
            print(f"\nImposed Imbalance Output Counts:\n{counts}")

            # Should be roughly 0.9 for class 0
            assert (
                0.8 <= counts[0] <= 1.0
            ), f"Class 0 proportion {counts[0]} did not match custom dist 0.9"

        except ImportError as e:
            print(
                f"Skipping test_create_imbalance_from_balanced due to missing dependency: {e}"
            )
            raise e
        except Exception as e:
            pytest.fail(f"Test failed with error: {e}")
    finally:
        shutil.rmtree(output_dir)
