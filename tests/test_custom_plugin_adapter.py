"""
Tests for CustomPluginAdapter.

Covers auto-detection of fit/generate interfaces and explicit override via lambdas.
"""
import pytest
import pandas as pd
import numpy as np
from calm_data_generator.generators.tabular.CustomPluginAdapter import CustomPluginAdapter


@pytest.fixture
def simple_df():
    np.random.seed(1)
    return pd.DataFrame({
        "x": np.random.randn(80),
        "y": np.random.randn(80),
        "z": np.random.randn(80),
    })


# ---------------------------------------------------------------------------
# Minimal model stubs
# ---------------------------------------------------------------------------

class FitGenerateModel:
    """Synthcity-style model: .fit() + .generate(count=n).dataframe()."""
    def fit(self, data):
        self._cols = data.columns.tolist()
        self._n = len(data)

    class _GenResult:
        def __init__(self, df):
            self._df = df
        def dataframe(self):
            return self._df

    def generate(self, count):
        df = pd.DataFrame(np.random.randn(count, 3), columns=["x", "y", "z"])
        return self._GenResult(df)


class FitSampleModel:
    """sklearn-style model: .fit() + .sample(n)."""
    def fit(self, X):
        self._shape = X.shape[1]

    def sample(self, n):
        return np.random.randn(n, self._shape), None  # returns tuple (samples, log_prob)


class TrainRandomModel:
    """Alternative interface: .train() + .random(n)."""
    def train(self, data):
        self._cols = list(data.columns)

    def random(self, n):
        return np.random.randn(n, len(self._cols))


class NoInterfaceModel:
    """Model with no recognized fit or generate methods."""
    pass


# ---------------------------------------------------------------------------
# Tests: Auto-detection
# ---------------------------------------------------------------------------

def test_auto_detect_fit_generate(simple_df):
    adapter = CustomPluginAdapter(model=FitGenerateModel(), method_name="fg")
    adapter.fit(simple_df)
    result = adapter.generate(10)
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 10


def test_auto_detect_fit_sample(simple_df):
    cols = list(simple_df.columns)
    adapter = CustomPluginAdapter(
        model=FitSampleModel(),
        columns=cols,
        method_name="fs",
    )
    # FitSampleModel.fit expects raw array
    adapter._fit_fn = lambda m, data: m.fit(data.values)
    adapter.fit(simple_df)
    result = adapter.generate(15)
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 15


def test_auto_detect_train_random(simple_df):
    cols = list(simple_df.columns)
    adapter = CustomPluginAdapter(
        model=TrainRandomModel(),
        columns=cols,
        method_name="tr",
    )
    adapter.fit(simple_df)
    result = adapter.generate(20)
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 20


# ---------------------------------------------------------------------------
# Tests: Explicit overrides
# ---------------------------------------------------------------------------

def test_explicit_fit_fn_and_generate_fn(simple_df):
    from sklearn.neighbors import KernelDensity
    kde = KernelDensity(bandwidth=0.5)
    cols = list(simple_df.columns)

    adapter = CustomPluginAdapter(
        model=kde,
        fit_fn=lambda m, data: m.fit(data.values),
        generate_fn=lambda m, n: pd.DataFrame(m.sample(n), columns=cols),
        columns=cols,
        method_name="explicit_kde",
    )
    adapter.fit(simple_df)
    result = adapter.generate(12)
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 12
    assert list(result.columns) == cols


def test_postprocess_fn(simple_df):
    def postprocess(df):
        df = df.copy()
        df["x"] = df["x"].abs()  # Force non-negative
        return df

    adapter = CustomPluginAdapter(
        model=FitGenerateModel(),
        postprocess_fn=postprocess,
        method_name="with_post",
    )
    adapter.fit(simple_df)
    result = adapter.generate(10)
    assert (result["x"] >= 0).all(), "postprocess_fn should have made x non-negative"


# ---------------------------------------------------------------------------
# Tests: Error cases
# ---------------------------------------------------------------------------

def test_generate_before_fit_raises(simple_df):
    adapter = CustomPluginAdapter(model=FitGenerateModel(), method_name="early")
    with pytest.raises(ValueError, match="not trained"):
        adapter.generate(10)


def test_no_fit_interface_raises():
    with pytest.raises(ValueError, match="No fit method"):
        CustomPluginAdapter(model=NoInterfaceModel(), method_name="bad")


def test_no_generate_interface_raises():
    class FitOnly:
        def fit(self, data): pass
    with pytest.raises(ValueError, match="No generate method"):
        CustomPluginAdapter(model=FitOnly(), method_name="fit_only")


def test_generate_priority_over_sample(simple_df):
    """Models with both .generate() and .sample() should use .generate() first."""
    class BothMethods:
        def fit(self, data):
            self._cols = list(data.columns)
        def generate(self, count):
            df = pd.DataFrame(np.ones((count, 3)), columns=["x", "y", "z"])
            class R:
                def __init__(self, df): self._df = df
                def dataframe(self): return self._df
            return R(df)
        def sample(self, n):
            # Returns zeros — should NOT be called
            return np.zeros((n, 3)), None

    adapter = CustomPluginAdapter(model=BothMethods(), method_name="both")
    adapter.fit(simple_df)
    result = adapter.generate(5)
    # If generate() was used, values are 1; if sample() was used, they'd be 0
    assert (result.values == 1.0).all(), "Should have used .generate(), not .sample()"