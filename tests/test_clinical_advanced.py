import os
import tempfile
import numpy as np
import shutil
import pytest
from calm_data_generator.generators.clinical.Clinic import ClinicalDataGenerator


def test_clinical_generation_flow():
    """Test full Clinical Generator flow."""
    output_dir = tempfile.mkdtemp()
    np.random.seed(42)
    try:
        try:
            generator = ClinicalDataGenerator(seed=42)

            # ClinicalDataGenerator generates a dictionary of components
            # using its own synthetic logic (not necessarily from a template df)
            results = generator.generate(
                n_samples=50, n_genes=100, n_proteins=50, save_dataset=False
            )

            assert results is not None
            assert "demographics" in results
            assert "genes" in results
            assert "proteins" in results

            demo_df = results["demographics"]
            assert len(demo_df) == 50
            assert "Group" in demo_df.columns

        except ImportError as e:
            print(
                f"Skipping test_clinical_generation_flow due to missing dependency: {e}"
            )
        except Exception as e:
            pytest.fail(f"Clinical generator failed: {e}")
    finally:
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
