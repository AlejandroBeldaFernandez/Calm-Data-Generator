"""
Tests for ComplexGenerator intermediate abstract layer.
Covers all 3 mathematical engines independently.
"""
import numpy as np
import pandas as pd
import pytest
import scipy.stats as stats

from calm_data_generator.generators.complex.ComplexGenerator import ComplexGenerator


class ConcreteGenerator(ComplexGenerator):
    """Minimal concrete subclass for testing."""

    def generate(self, *args, **kwargs):
        pass


@pytest.fixture
def gen():
    return ConcreteGenerator(random_state=42)


# ---------------------------------------------------------------------------
# _generate_correlated_module
# ---------------------------------------------------------------------------

class TestGenerateCorrelatedModule:
    def test_output_shape(self, gen):
        marginals = [stats.norm(loc=0, scale=1) for _ in range(3)]
        sigma = np.identity(3)
        X = gen._generate_correlated_module(50, marginals, sigma)
        assert X.shape == (50, 3)

    def test_finite_values(self, gen):
        marginals = [stats.expon(scale=1) for _ in range(4)]
        sigma = np.identity(4)
        X = gen._generate_correlated_module(100, marginals, sigma)
        assert np.all(np.isfinite(X))

    def test_non_psd_matrix_repaired(self, gen):
        """A non-PSD matrix should not raise; the method repairs it via eigh."""
        marginals = [stats.norm() for _ in range(3)]
        # Build a clearly non-PSD matrix
        bad_sigma = np.array([
            [1.0,  0.9,  0.9],
            [0.9,  1.0,  0.9],
            [0.9,  0.9,  1.0],
        ])
        # Make it non-PSD by introducing a negative eigenvalue
        bad_sigma[0, 1] = bad_sigma[1, 0] = 1.5
        # Should not raise
        X = gen._generate_correlated_module(30, marginals, bad_sigma)
        assert X.shape == (30, 3)
        assert np.all(np.isfinite(X))

    def test_single_variable(self, gen):
        marginals = [stats.norm(loc=5, scale=2)]
        sigma = np.array([[1.0]])
        X = gen._generate_correlated_module(20, marginals, sigma)
        assert X.shape == (20, 1)


# ---------------------------------------------------------------------------
# _generate_conditional_data
# ---------------------------------------------------------------------------

class TestGenerateConditionalData:
    def _make_full_cov(self, n_cond, n_target, rho=0.5):
        n = n_cond + n_target
        cov = np.full((n, n), rho)
        np.fill_diagonal(cov, 1.0)
        return cov

    def test_output_shape(self, gen):
        n_samples, n_cond, n_target = 40, 2, 3
        cond_data = np.random.randn(n_samples, n_cond)
        cond_marginals = [stats.norm() for _ in range(n_cond)]
        tgt_marginals = [stats.norm() for _ in range(n_target)]
        full_cov = self._make_full_cov(n_cond, n_target)
        X = gen._generate_conditional_data(
            n_samples, cond_data, cond_marginals, tgt_marginals, full_cov
        )
        assert X.shape == (n_samples, n_target)

    def test_finite_values(self, gen):
        n_samples, n_cond, n_target = 60, 3, 3
        cond_data = np.random.randn(n_samples, n_cond)
        cond_marginals = [stats.norm() for _ in range(n_cond)]
        tgt_marginals = [stats.lognorm(s=0.5) for _ in range(n_target)]
        full_cov = self._make_full_cov(n_cond, n_target, rho=0.3)
        X = gen._generate_conditional_data(
            n_samples, cond_data, cond_marginals, tgt_marginals, full_cov
        )
        assert np.all(np.isfinite(X))

    def test_discrete_marginal_rqr_path(self, gen):
        """Discrete conditioning marginal activates the RQR (jittering) path."""
        n_samples, n_cond, n_target = 30, 1, 2
        cond_data = np.random.randint(0, 10, size=(n_samples, n_cond)).astype(float)
        cond_marginals = [stats.binom(n=10, p=0.5)]  # discrete
        tgt_marginals = [stats.norm() for _ in range(n_target)]
        full_cov = self._make_full_cov(n_cond, n_target, rho=0.4)
        X = gen._generate_conditional_data(
            n_samples, cond_data, cond_marginals, tgt_marginals, full_cov
        )
        assert X.shape == (n_samples, n_target)
        assert np.all(np.isfinite(X))

    def test_shape_mismatch_raises(self, gen):
        n_samples, n_cond, n_target = 20, 2, 2
        # Pass wrong shape for conditioning_data
        cond_data = np.random.randn(n_samples, n_cond + 1)
        cond_marginals = [stats.norm() for _ in range(n_cond)]
        tgt_marginals = [stats.norm() for _ in range(n_target)]
        full_cov = self._make_full_cov(n_cond, n_target)
        with pytest.raises(ValueError, match="mismatch"):
            gen._generate_conditional_data(
                n_samples, cond_data, cond_marginals, tgt_marginals, full_cov
            )


# ---------------------------------------------------------------------------
# apply_stochastic_effects
# ---------------------------------------------------------------------------

class TestApplyStochasticEffects:
    def _make_df(self, n=50, n_cols=5):
        np.random.seed(0)
        return pd.DataFrame(
            np.abs(np.random.randn(n, n_cols)) + 1,
            columns=[f"F{i}" for i in range(n_cols)],
        )

    def test_additive_shift(self, gen):
        df = self._make_df()
        before = df["F0"].copy()
        gen.apply_stochastic_effects(
            df, df.index, {"index": [0], "effect_type": "additive_shift", "effect_value": 10.0}
        )
        assert (df["F0"] != before).any()

    def test_fold_change(self, gen):
        df = self._make_df()
        before = df["F0"].copy()
        gen.apply_stochastic_effects(
            df, df.index, {"index": [0], "effect_type": "fold_change", "effect_value": 2.0}
        )
        assert (df["F0"] != before).any()
        assert np.all(df["F0"] > 0)

    def test_power_transform(self, gen):
        df = self._make_df()
        before = df["F0"].copy()
        gen.apply_stochastic_effects(
            df, df.index, {"index": [0], "effect_type": "power_transform", "effect_value": 2.0}
        )
        assert (df["F0"] != before).any()

    def test_variance_scale(self, gen):
        df = self._make_df()
        before_std = df["F0"].std()
        gen.apply_stochastic_effects(
            df, df.index, {"index": [0], "effect_type": "variance_scale", "effect_value": 2.0}
        )
        # Variance should have changed
        assert abs(df["F0"].std() - before_std) > 1e-6

    def test_log_transform(self, gen):
        df = self._make_df()
        gen.apply_stochastic_effects(
            df, df.index, {"index": [0], "effect_type": "log_transform", "effect_value": 1.0}
        )
        assert np.all(np.isfinite(df["F0"]))

    def test_polynomial_transform(self, gen):
        df = self._make_df()
        before = df["F0"].copy()
        gen.apply_stochastic_effects(
            df, df.index, {"index": [0], "effect_type": "polynomial_transform", "effect_value": [1, 0, 0]}
        )
        # poly([1,0,0]) = x^2, so all positive values squared
        np.testing.assert_allclose(df["F0"].values, before.values ** 2, rtol=1e-5)

    def test_sigmoid_transform(self, gen):
        df = self._make_df()
        gen.apply_stochastic_effects(
            df, df.index, {"index": [0], "effect_type": "sigmoid_transform", "effect_value": {"k": 1, "x0": 0}}
        )
        # Sigmoid output must be in (0, 1)
        assert np.all(df["F0"] > 0)
        assert np.all(df["F0"] < 1)

    def test_simple_additive_shift_alias(self, gen):
        """simple_additive_shift should be treated as additive_shift (alias)."""
        df = self._make_df()
        before = df["F0"].copy()
        gen.apply_stochastic_effects(
            df, df.index, {"index": [0], "effect_type": "simple_additive_shift", "effect_value": 5.0}
        )
        assert (df["F0"] != before).any()

    def test_empty_entity_ids_is_noop(self, gen):
        df = self._make_df()
        before = df.copy()
        gen.apply_stochastic_effects(
            df, [], {"index": [0], "effect_type": "additive_shift", "effect_value": 100.0}
        )
        pd.testing.assert_frame_equal(df, before)

    def test_unknown_effect_type_raises(self, gen):
        df = self._make_df()
        with pytest.raises(ValueError, match="Unsupported effect_type"):
            gen.apply_stochastic_effects(
                df, df.index, {"index": [0], "effect_type": "unknown_effect", "effect_value": 1.0}
            )

    def test_range_effect_value(self, gen):
        """effect_value as [low, high] list should sample from uniform."""
        df = self._make_df()
        before = df["F0"].copy()
        gen.apply_stochastic_effects(
            df, df.index, {"index": [0], "effect_type": "additive_shift", "effect_value": [1.0, 5.0]}
        )
        assert (df["F0"] != before).any()
