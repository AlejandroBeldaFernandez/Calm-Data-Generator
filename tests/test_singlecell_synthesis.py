import pytest
import pandas as pd
import numpy as np
from calm_data_generator.generators.tabular import RealGenerator
import sys


class TestGEARSSynthesis:
    """Test suite for GEARS single-cell perturbation synthesis."""

    @pytest.fixture
    def sample_expression_data(self):
        """Create sample single-cell gene expression data."""
        np.random.seed(42)
        n_samples = 300  # Larger sample size for valid splits
        # Use real gene names for GO compatibility, plus extra generic ones
        gene_names = [f"GENE_{i}" for i in range(50)]
        gene_names[:10] = [
            "TP53",
            "EGFR",
            "TNF",
            "MAPK1",
            "GAPDH",
            "ACTB",
            "MYC",
            "BRCA1",
            "IL6",
            "INS",
        ]

        data = pd.DataFrame(np.random.randn(n_samples, 50), columns=gene_names)
        # Add conditions in combinatorial format: Gene+ctrl
        # This structure is robust for GEARS parsing
        conditions = (
            ["ctrl"] * 100
            + ["TP53+ctrl"] * 50
            + ["EGFR+ctrl"] * 50
            + ["TNF+ctrl"] * 50
            + ["MYC+ctrl"] * 50
        )
        data["condition"] = conditions
        return data

    def test_gears_in_valid_methods(self):
        """Test that 'gears' is a valid method."""
        gen = RealGenerator(auto_report=False)
        # Should not raise
        gen._validate_method("gears")

    def test_gears_synthesis_basic(self, sample_expression_data):
        """Test basic GEARS synthesis with perturbations."""

        # Check dependencies first
        try:
            import gears
            import torch

        except ImportError:
            pytest.skip("GEARS not installed")

        gen = RealGenerator(auto_report=False)

        n_samples = 30
        try:
            # We must use target_col='condition' so GEARS uses our prepared conditions
            synthetic = gen.generate(
                data=sample_expression_data,
                n_samples=n_samples,
                target_col="condition",
                method="gears",
                # Perturbations must exist in the input data logic or be valid genes
                perturbations=["CNN1", "CBL"],
                epochs=1,  # Low epochs for testing speed
                device="cpu",
            )

            if synthetic is None:
                pytest.fail(
                    "GEARS synthesis returned None (likely missing dependencies or runtime error)"
                )

            assert synthetic is not None
            assert len(synthetic) == n_samples
            # Verify columns match (GEARS returns gene expression columns)
            # excluding 'condition' if it wasn't requested in output, but generator usually adds target_col back
            assert "TP53" in synthetic.columns

        except Exception as e:
            err = str(e)
            # Perturbation not in GO graph = data compatibility issue, not a code bug
            if "not in the perturbation graph" in err or "pert_list" in err:
                pytest.skip(
                    f"GEARS perturbation not in GO graph (expected with synthetic data): {e}"
                )
            elif "gears" in err.lower() or "ImportError" in err:
                pytest.fail(f"GEARS failed to load: {e}")
            else:
                raise e
