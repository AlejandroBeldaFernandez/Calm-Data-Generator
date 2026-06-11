"""
Tests for build_correlation_matrix and get_conditioning_columns.

Covers:
- get_conditioning_columns returns correct columns in correct order
- build_correlation_matrix dimensions match actual conditioning columns
- demo_idx as string column name works correctly
- Mismatch between n_demo and actual columns raises ValueError
- Correlation structure is preserved in generated data (2 groups + noise)
- Full pipeline: demographics + gene_correlations + target variable
"""
import numpy as np
import pytest

from calm_data_generator.generators.clinical.Clinic import ClinicalDataGenerator
from calm_data_generator.tutorials.clinical_generator import build_correlation_matrix

CUSTOM_DEMO_2 = {
    "Age": {"distribution": "truncnorm", "a": -2.0, "b": 2.5, "loc": 60, "scale": 10},
    "Sex": {"distribution": "binom", "n": 1, "p": 0.5},
}

CUSTOM_DEMO_3 = {
    "Age": {"distribution": "truncnorm", "a": -2.0, "b": 2.5, "loc": 60, "scale": 10},
    "Sex": {"distribution": "binom", "n": 1, "p": 0.5},
    "BMI": {"distribution": "norm", "loc": 27, "scale": 4},
}


@pytest.fixture
def gen():
    return ClinicalDataGenerator(seed=42, auto_report=False)


# ---------------------------------------------------------------------------
# get_conditioning_columns
# ---------------------------------------------------------------------------

class TestGetConditioningColumns:
    def test_default_columns(self, gen):
        _, raw = gen.generate_demographic_data(n_samples=20)
        cols = ClinicalDataGenerator.get_conditioning_columns(raw)
        assert isinstance(cols, list)
        assert len(cols) >= 1
        assert all(isinstance(c, str) for c in cols)

    def test_two_custom_columns(self, gen):
        _, raw = gen.generate_demographic_data(n_samples=20, custom_demographic_columns=CUSTOM_DEMO_2)
        cols = ClinicalDataGenerator.get_conditioning_columns(raw)
        assert "Age" in cols
        assert "Sex_Binario" in cols
        assert "Binary_Group" not in cols
        assert "Disease_Subgroup" not in cols

    def test_three_custom_columns(self, gen):
        _, raw = gen.generate_demographic_data(n_samples=20, custom_demographic_columns=CUSTOM_DEMO_3)
        cols = ClinicalDataGenerator.get_conditioning_columns(raw)
        assert "Age" in cols
        assert "BMI" in cols
        assert len(cols) == 3

    def test_excludes_internal_columns(self, gen):
        _, raw = gen.generate_demographic_data(n_samples=20, custom_demographic_columns=CUSTOM_DEMO_2)
        cols = ClinicalDataGenerator.get_conditioning_columns(raw)
        excluded = {"Group", "Binary_Group", "Disease_Subgroup", "Patient_ID"}
        assert excluded.isdisjoint(set(cols))

    def test_len_equals_n_demo_for_matrix(self, gen):
        _, raw = gen.generate_demographic_data(n_samples=20, custom_demographic_columns=CUSTOM_DEMO_3)
        cols = ClinicalDataGenerator.get_conditioning_columns(raw)
        n_demo = len(cols)
        # Matrix built with this n_demo should not raise
        matrix = build_correlation_matrix(
            n_demo, [5, 5, 5],
            [{"internal": 0.5}, {"internal": 0.3}, {"internal": 0.0}],
        )
        assert matrix.shape == (n_demo + 15, n_demo + 15)


# ---------------------------------------------------------------------------
# build_correlation_matrix
# ---------------------------------------------------------------------------

class TestBuildCorrelationMatrix:
    def test_shape_two_groups(self, gen):
        _, raw = gen.generate_demographic_data(n_samples=10, custom_demographic_columns=CUSTOM_DEMO_2)
        cols = ClinicalDataGenerator.get_conditioning_columns(raw)
        matrix = build_correlation_matrix(len(cols), [10, 10, 10], [
            {"internal": 0.6}, {"internal": 0.4}, {"internal": 0.0},
        ])
        assert matrix.shape == (len(cols) + 30, len(cols) + 30)

    def test_diagonal_is_one(self, gen):
        _, raw = gen.generate_demographic_data(n_samples=10, custom_demographic_columns=CUSTOM_DEMO_2)
        cols = ClinicalDataGenerator.get_conditioning_columns(raw)
        matrix = build_correlation_matrix(len(cols), [5, 5], [
            {"internal": 0.6}, {"internal": 0.4},
        ])
        np.testing.assert_array_almost_equal(np.diag(matrix), 1.0)

    def test_intra_group_correlation_set(self, gen):
        _, raw = gen.generate_demographic_data(n_samples=10, custom_demographic_columns=CUSTOM_DEMO_2)
        cols = ClinicalDataGenerator.get_conditioning_columns(raw)
        n_demo = len(cols)
        n_A = 5
        matrix = build_correlation_matrix(n_demo, [n_A, 5], [
            {"internal": 0.7}, {"internal": 0.3},
        ])
        # Group A block starts at n_demo
        block = matrix[n_demo:n_demo + n_A, n_demo:n_demo + n_A]
        off_diag = block[np.triu_indices(n_A, k=1)]
        np.testing.assert_array_almost_equal(off_diag, 0.7)

    def test_noise_group_is_identity(self, gen):
        _, raw = gen.generate_demographic_data(n_samples=10, custom_demographic_columns=CUSTOM_DEMO_2)
        cols = ClinicalDataGenerator.get_conditioning_columns(raw)
        n_demo = len(cols)
        n_noise = 6
        matrix = build_correlation_matrix(n_demo, [5, n_noise], [
            {"internal": 0.5}, {"internal": 0.0},
        ])
        noise_start = n_demo + 5
        block = matrix[noise_start:noise_start + n_noise, noise_start:noise_start + n_noise]
        np.testing.assert_array_almost_equal(block, np.eye(n_noise))

    def test_demo_corr_by_string_name(self, gen):
        _, raw = gen.generate_demographic_data(n_samples=10, custom_demographic_columns=CUSTOM_DEMO_2)
        cols = ClinicalDataGenerator.get_conditioning_columns(raw)
        n_demo = len(cols)
        matrix = build_correlation_matrix(
            n_demo, [5, 5],
            [{"internal": 0.6, "demo_idx": "Age", "demo_corr": 0.4},
             {"internal": 0.3}],
            demo_col_names=cols,
        )
        age_idx = cols.index("Age")
        # Age row should have 0.4 for all genes in group A
        np.testing.assert_array_almost_equal(
            matrix[age_idx, n_demo:n_demo + 5], 0.4
        )

    def test_demo_corr_by_integer_index(self, gen):
        _, raw = gen.generate_demographic_data(n_samples=10, custom_demographic_columns=CUSTOM_DEMO_2)
        cols = ClinicalDataGenerator.get_conditioning_columns(raw)
        n_demo = len(cols)
        matrix = build_correlation_matrix(
            n_demo, [5, 5],
            [{"internal": 0.6, "demo_idx": 0, "demo_corr": 0.3},
             {"internal": 0.3}],
        )
        np.testing.assert_array_almost_equal(
            matrix[0, n_demo:n_demo + 5], 0.3
        )

    def test_string_demo_idx_without_col_names_raises(self, gen):
        _, raw = gen.generate_demographic_data(n_samples=10, custom_demographic_columns=CUSTOM_DEMO_2)
        cols = ClinicalDataGenerator.get_conditioning_columns(raw)
        with pytest.raises(ValueError, match="demo_col_names must be provided"):
            build_correlation_matrix(
                len(cols), [5],
                [{"internal": 0.5, "demo_idx": "Age", "demo_corr": 0.3}],
                demo_col_names=None,
            )

    def test_unknown_column_name_raises(self, gen):
        _, raw = gen.generate_demographic_data(n_samples=10, custom_demographic_columns=CUSTOM_DEMO_2)
        cols = ClinicalDataGenerator.get_conditioning_columns(raw)
        with pytest.raises(ValueError, match="not found in demo_col_names"):
            build_correlation_matrix(
                len(cols), [5],
                [{"internal": 0.5, "demo_idx": "NonExistentCol", "demo_corr": 0.3}],
                demo_col_names=cols,
            )

    def test_is_symmetric(self, gen):
        _, raw = gen.generate_demographic_data(n_samples=10, custom_demographic_columns=CUSTOM_DEMO_2)
        cols = ClinicalDataGenerator.get_conditioning_columns(raw)
        matrix = build_correlation_matrix(
            len(cols), [8, 8, 4],
            [{"internal": 0.6, "demo_idx": "Age", "demo_corr": 0.3},
             {"internal": 0.5},
             {"internal": 0.0}],
            demo_col_names=cols,
        )
        np.testing.assert_array_almost_equal(matrix, matrix.T)


# ---------------------------------------------------------------------------
# Dimension mismatch raises clear error
# ---------------------------------------------------------------------------

class TestDimensionMismatch:
    def test_wrong_n_demo_raises_valueerror(self, gen):
        _, raw = gen.generate_demographic_data(n_samples=20, custom_demographic_columns=CUSTOM_DEMO_3)
        # n_demo=2 but actual is 3 → should raise ValueError inside generate_gene_data
        wrong_matrix = build_correlation_matrix(2, [5, 5, 5], [
            {"internal": 0.6}, {"internal": 0.5}, {"internal": 0.0},
        ])
        demo_df, raw_demo = gen.generate_demographic_data(
            n_samples=50, custom_demographic_columns=CUSTOM_DEMO_3
        )
        with pytest.raises(ValueError, match="mismatch"):
            gen.generate_gene_data(
                n_genes=15,
                gene_type="Microarray",
                demographic_df=demo_df,
                demographic_id_col="Patient_ID",
                raw_demographic_data=raw_demo,
                demographic_gene_correlations=wrong_matrix,
                n_samples=50,
            )


# ---------------------------------------------------------------------------
# Correlation structure preserved in generated data
# ---------------------------------------------------------------------------

class TestCorrelationStructure:
    """Generated data should respect the specified intra-group correlations."""

    def test_two_groups_plus_noise(self, gen):
        n_samples = 600
        n_A, n_B, n_noise = 15, 15, 10
        target_rA, target_rB = 0.65, 0.50

        corr = np.eye(n_A + n_B + n_noise)
        corr[:n_A, :n_A] = target_rA
        np.fill_diagonal(corr[:n_A, :n_A], 1.0)
        corr[n_A:n_A + n_B, n_A:n_A + n_B] = target_rB
        np.fill_diagonal(corr[n_A:n_A + n_B, n_A:n_A + n_B], 1.0)

        demo_df, raw_demo = gen.generate_demographic_data(n_samples=n_samples)
        genes_df = gen.generate_gene_data(
            n_genes=n_A + n_B + n_noise,
            gene_type="Microarray",
            demographic_df=demo_df,
            demographic_id_col="Patient_ID",
            raw_demographic_data=raw_demo,
            gene_correlations=corr,
            n_samples=n_samples,
        )
        arr = genes_df.values.astype(float)
        real_corr = np.corrcoef(arr.T)

        rA    = real_corr[:n_A, :n_A][np.triu_indices(n_A, 1)].mean()
        rB    = real_corr[n_A:n_A + n_B, n_A:n_A + n_B][np.triu_indices(n_B, 1)].mean()
        cross = real_corr[:n_A, n_A:n_A + n_B].mean()
        noise = real_corr[n_A + n_B:, n_A + n_B:][np.triu_indices(n_noise, 1)].mean()

        assert abs(rA - target_rA) < 0.10, f"Intra-A {rA:.3f} far from {target_rA}"
        assert abs(rB - target_rB) < 0.10, f"Intra-B {rB:.3f} far from {target_rB}"
        assert abs(cross) < 0.10, f"Cross {cross:.3f} should be ~0"
        assert abs(noise) < 0.10, f"Noise {noise:.3f} should be ~0"

    def test_demographic_gene_correlation(self, gen):
        n_samples = 600
        n_A, n_B, n_noise = 10, 10, 5

        _, raw_tmp = gen.generate_demographic_data(
            n_samples=10, custom_demographic_columns=CUSTOM_DEMO_2
        )
        cols = ClinicalDataGenerator.get_conditioning_columns(raw_tmp)
        n_demo = len(cols)

        matrix = build_correlation_matrix(
            n_demo, [n_A, n_B, n_noise],
            [{"internal": 0.6, "demo_idx": "Age", "demo_corr": 0.4},
             {"internal": 0.5},
             {"internal": 0.0}],
            demo_col_names=cols,
        )

        demo_df, raw_demo = gen.generate_demographic_data(
            n_samples=n_samples, custom_demographic_columns=CUSTOM_DEMO_2
        )
        genes_df = gen.generate_gene_data(
            n_genes=n_A + n_B + n_noise,
            gene_type="Microarray",
            demographic_df=demo_df,
            demographic_id_col="Patient_ID",
            raw_demographic_data=raw_demo,
            demographic_gene_correlations=matrix,
            n_samples=n_samples,
        )

        from scipy.stats import pearsonr
        age = raw_demo["Age"].values.astype(float)
        gene_A_mean = genes_df.iloc[:, :n_A].mean(axis=1).values
        r, _ = pearsonr(age, gene_A_mean)
        assert abs(r - 0.4) < 0.15, f"Age-GrupoA corr {r:.3f} far from 0.4"


# ---------------------------------------------------------------------------
# Full pipeline: demographics + groups + target variable
# ---------------------------------------------------------------------------

class TestFullPipeline:
    def test_two_groups_noise_demographics_target(self, gen):
        n_samples = 300
        n_A, n_B, n_noise = 10, 10, 5

        _, raw_tmp = gen.generate_demographic_data(
            n_samples=10, custom_demographic_columns=CUSTOM_DEMO_2
        )
        cols = ClinicalDataGenerator.get_conditioning_columns(raw_tmp)
        n_demo = len(cols)

        matrix = build_correlation_matrix(
            n_demo, [n_A, n_B, n_noise],
            [{"internal": 0.6, "demo_idx": "Age", "demo_corr": 0.3},
             {"internal": 0.5},
             {"internal": 0.0}],
            demo_col_names=cols,
        )

        datasets = gen.generate(
            n_samples=n_samples,
            n_genes=n_A + n_B + n_noise,
            n_proteins=0,
            custom_demographic_columns=CUSTOM_DEMO_2,
            gene_type="Microarray",
            demographic_gene_correlations=matrix,
            target_variable_config={
                "weights": {"Age": 0.3, "G_0": 0.1, "G_1": 0.1},
                "binary_threshold": "median",
                "name": "diagnosis",
            },
            save_dataset=False,
        )

        demo_df  = datasets["demographics"]
        genes_df = datasets["genes"]

        assert demo_df.shape[0] == n_samples
        assert genes_df.shape == (n_samples, n_A + n_B + n_noise)
        assert "diagnosis" in demo_df.columns
        # median split → ~50/50
        balance = demo_df["diagnosis"].mean()
        assert 0.4 < balance < 0.6, f"Diagnosis balance {balance:.2f} far from 0.5"

    def test_three_custom_demo_columns(self, gen):
        """Adding BMI must not cause dimension mismatch."""
        n_samples = 200
        n_A, n_B, n_noise = 8, 8, 4

        _, raw_tmp = gen.generate_demographic_data(
            n_samples=10, custom_demographic_columns=CUSTOM_DEMO_3
        )
        cols = ClinicalDataGenerator.get_conditioning_columns(raw_tmp)
        n_demo = len(cols)
        assert n_demo == 3

        matrix = build_correlation_matrix(
            n_demo, [n_A, n_B, n_noise],
            [{"internal": 0.6, "demo_idx": "Age", "demo_corr": 0.3},
             {"internal": 0.4},
             {"internal": 0.0}],
            demo_col_names=cols,
        )
        assert matrix.shape == (n_demo + n_A + n_B + n_noise, n_demo + n_A + n_B + n_noise)

        datasets = gen.generate(
            n_samples=n_samples,
            n_genes=n_A + n_B + n_noise,
            n_proteins=0,
            custom_demographic_columns=CUSTOM_DEMO_3,
            gene_type="Microarray",
            demographic_gene_correlations=matrix,
            save_dataset=False,
        )
        assert datasets["genes"].shape == (n_samples, n_A + n_B + n_noise)
