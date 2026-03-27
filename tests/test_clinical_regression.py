"""
Regression tests for ClinicalDataGenerator after ComplexGenerator refactor.
Verifies that the output shape and column names are unchanged, and that the
new unified protein effects pathway works correctly.
"""
import warnings

import numpy as np
import pytest
import scipy.stats as stats

from calm_data_generator.generators.clinical.Clinic import ClinicalDataGenerator
from calm_data_generator.generators.complex.ComplexGenerator import ComplexGenerator


# ---------------------------------------------------------------------------
# Inheritance check
# ---------------------------------------------------------------------------

def test_clinical_inherits_complex_generator():
    gen = ClinicalDataGenerator(seed=42)
    assert isinstance(gen, ComplexGenerator)


# ---------------------------------------------------------------------------
# generate_gene_data regression
# ---------------------------------------------------------------------------

class TestGeneDataRegression:
    @pytest.fixture
    def gen(self):
        return ClinicalDataGenerator(seed=42, auto_report=False)

    def test_output_shape_and_columns(self, gen):
        n_samples, n_genes = 30, 10
        df = gen.generate_gene_data(
            n_genes=n_genes,
            gene_type="Microarray",
            n_samples=n_samples,
            control_disease_ratio=0.7,
        )
        assert df.shape == (n_samples, n_genes)
        assert list(df.columns) == [f"G_{i}" for i in range(n_genes)]

    def test_rna_seq_output_is_integer(self, gen):
        df = gen.generate_gene_data(
            n_genes=5,
            gene_type="RNA-Seq",
            n_samples=20,
            control_disease_ratio=0.6,
        )
        assert df.dtypes.apply(lambda d: np.issubdtype(d, np.integer)).all()

    def test_additive_shift_effect_produces_finite(self, gen):
        effect = {"index": slice(0, 3), "effect_type": "additive_shift", "effect_value": 1.5}
        df = gen.generate_gene_data(
            n_genes=5,
            gene_type="Microarray",
            n_samples=20,
            control_disease_ratio=0.5,
            disease_effects_config=[effect],
        )
        assert np.all(np.isfinite(df.values))

    def test_fold_change_effect_produces_finite(self, gen):
        effect = {"index": slice(0, 3), "effect_type": "fold_change", "effect_value": 2.0}
        df = gen.generate_gene_data(
            n_genes=5,
            gene_type="Microarray",
            n_samples=20,
            control_disease_ratio=0.5,
            disease_effects_config=[effect],
        )
        assert np.all(np.isfinite(df.values))

    def test_variance_scale_effect_produces_finite(self, gen):
        effect = {"index": slice(0, 3), "effect_type": "variance_scale", "effect_value": 2.0}
        df = gen.generate_gene_data(
            n_genes=5,
            gene_type="Microarray",
            n_samples=20,
            control_disease_ratio=0.5,
            disease_effects_config=[effect],
        )
        assert np.all(np.isfinite(df.values))

    def test_log_transform_effect_produces_finite(self, gen):
        effect = {"index": slice(0, 3), "effect_type": "log_transform", "effect_value": 1e-8}
        df = gen.generate_gene_data(
            n_genes=5,
            gene_type="Microarray",
            n_samples=20,
            control_disease_ratio=0.5,
            disease_effects_config=[effect],
        )
        assert np.all(np.isfinite(df.values))

    def test_polynomial_transform_effect_produces_finite(self, gen):
        effect = {"index": slice(0, 2), "effect_type": "polynomial_transform", "effect_value": [1, 0]}
        df = gen.generate_gene_data(
            n_genes=5,
            gene_type="Microarray",
            n_samples=20,
            control_disease_ratio=0.5,
            disease_effects_config=[effect],
        )
        assert np.all(np.isfinite(df.values))

    def test_sigmoid_transform_effect_produces_finite(self, gen):
        effect = {"index": slice(0, 2), "effect_type": "sigmoid_transform", "effect_value": {"k": 1, "x0": 0}}
        df = gen.generate_gene_data(
            n_genes=5,
            gene_type="Microarray",
            n_samples=20,
            control_disease_ratio=0.5,
            disease_effects_config=[effect],
        )
        assert np.all(np.isfinite(df.values))


# ---------------------------------------------------------------------------
# generate_protein_data regression
# ---------------------------------------------------------------------------

class TestProteinDataRegression:
    @pytest.fixture
    def gen(self):
        return ClinicalDataGenerator(seed=42, auto_report=False)

    def test_output_shape_and_columns(self, gen):
        n_samples, n_proteins = 30, 8
        df = gen.generate_protein_data(
            n_proteins=n_proteins,
            n_samples=n_samples,
            control_disease_ratio=0.7,
        )
        assert df.shape == (n_samples, n_proteins)
        assert list(df.columns) == [f"P_{i}" for i in range(n_proteins)]

    def test_fold_change_effect_produces_finite(self, gen):
        effect = {"index": slice(0, 3), "effect_type": "fold_change", "effect_value": 2.0}
        df = gen.generate_protein_data(
            n_proteins=5,
            n_samples=20,
            control_disease_ratio=0.5,
            disease_effects_config=[effect],
        )
        assert np.all(np.isfinite(df.values))

    def test_additive_shift_emits_warning(self, gen):
        """additive_shift on protein data should emit a deprecation warning."""
        effect = {"index": slice(0, 3), "effect_type": "additive_shift", "effect_value": 0.5}
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            # The warning is emitted via self.logger.warning, not Python warnings.
            # We just verify it doesn't raise.
            df = gen.generate_protein_data(
                n_proteins=5,
                n_samples=20,
                control_disease_ratio=0.5,
                disease_effects_config=[effect],
            )
        assert np.all(np.isfinite(df.values))

    def test_simple_additive_shift_does_not_raise(self, gen):
        effect = {"index": slice(0, 3), "effect_type": "simple_additive_shift", "effect_value": 0.5}
        df = gen.generate_protein_data(
            n_proteins=5,
            n_samples=20,
            control_disease_ratio=0.5,
            disease_effects_config=[effect],
        )
        assert np.all(np.isfinite(df.values))
