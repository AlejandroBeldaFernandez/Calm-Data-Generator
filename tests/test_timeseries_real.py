"""
Tests for time series synthesis methods using Synthcity.

This file tests TimeGAN and TimeVAE methods for temporal data synthesis.
Both models require multi-sequence data with a sequence_key.
"""

import pandas as pd
import numpy as np
import pytest


@pytest.fixture
def time_series_data():
    """Create multi-sequence time series data for testing.

    synthcity's TimeGAN and TimeVAE require multiple sequences, so we create
    10 sequences of 20 steps each, identifiable by a sequence_key column.
    """
    np.random.seed(42)
    n_sequences = 10
    seq_len = 20

    data = []
    for i in range(n_sequences):
        for t in range(seq_len):
            data.append(
                {
                    "seq_id": i,
                    "time": t,
                    "feature1": np.sin(t / 5.0) + np.random.normal(0, 0.1),
                    "feature2": np.cos(t / 5.0) + np.random.normal(0, 0.1),
                    "feature3": t / 10.0 + np.random.normal(0, 0.5),
                }
            )
    return pd.DataFrame(data)


def test_timegan_synthesis(time_series_data):
    """Test TimeGAN for time series synthesis."""
    from calm_data_generator.generators.tabular import RealGenerator

    gen = RealGenerator(auto_report=False)

    try:
        synth = gen.generate(
            time_series_data,
            n_samples=5,
            method="timegan",
            sequence_key="seq_id",
            time_key="time",
            n_iter=10,
            n_units_hidden=50,
            batch_size=16,
        )
        assert synth is not None
        assert len(synth) > 0
        print("✅ TimeGAN test passed")
    except ImportError:
        pytest.skip("Synthcity not available for TimeGAN")
    except Exception as e:
        pytest.fail(f"TimeGAN test failed: {e}")


def test_timevae_synthesis(time_series_data):
    """Test TimeVAE for time series synthesis."""
    from calm_data_generator.generators.tabular import RealGenerator

    gen = RealGenerator(auto_report=False)

    try:
        synth = gen.generate(
            time_series_data,
            n_samples=5,
            method="timevae",
            sequence_key="seq_id",
            time_key="time",
            n_iter=10,
            decoder_n_units_hidden=50,
            batch_size=16,
        )
        assert synth is not None
        assert len(synth) > 0
        print("✅ TimeVAE test passed")
    except ImportError:
        pytest.skip("Synthcity not available for TimeVAE")
    except Exception as e:
        pytest.fail(f"TimeVAE test failed: {e}")


def test_timevae_parameters(time_series_data):
    """Test TimeVAE with different decoder parameters."""
    from calm_data_generator.generators.tabular import RealGenerator

    gen = RealGenerator(auto_report=False)

    try:
        synth = gen.generate(
            time_series_data,
            n_samples=5,
            method="timevae",
            sequence_key="seq_id",
            time_key="time",
            n_iter=5,
            decoder_n_layers_hidden=1,
            decoder_n_units_hidden=32,
            batch_size=8,
        )
        assert synth is not None
        assert len(synth) > 0
        print("✅ TimeVAE parameters test passed")
    except ImportError:
        pytest.skip("Synthcity not available for TimeVAE")
    except Exception as e:
        pytest.fail(f"TimeVAE parameters test failed: {e}")


@pytest.fixture
def fflows_data():
    """Create large multi-sequence time series data for FourierFlows.

    fflows uses internal cross-validation with n_splits=3, so it needs
    significantly more sequences than 10 to avoid errors. We use 30 sequences
    of 10 steps each.
    """
    np.random.seed(42)
    n_sequences = 30
    seq_len = 10

    data = []
    for i in range(n_sequences):
        for t in range(seq_len):
            data.append(
                {
                    "seq_id": i,
                    "time": t,
                    "feature1": np.sin(t / 3.0) + np.random.normal(0, 0.1),
                    "feature2": np.cos(t / 3.0) + np.random.normal(0, 0.1),
                }
            )
    return pd.DataFrame(data)


def test_fflows_synthesis(fflows_data):
    """Test FourierFlows (fflows) for time series synthesis.

    fflows uses normalizing flows in the frequency domain and is typically
    more stable than TimeGAN, especially for periodic/quasi-periodic series.
    Requires at least ~20+ sequences to satisfy internal cross-validation.
    """
    from calm_data_generator.generators.tabular import RealGenerator

    gen = RealGenerator(auto_report=False)

    try:
        synth = gen.generate(
            fflows_data,
            n_samples=10,
            method="fflows",
            sequence_key="seq_id",
            time_key="time",
            n_iter=10,
            batch_size=16,
        )
        assert synth is not None
        assert len(synth) > 0
        print("✅ FourierFlows test passed")
    except ImportError:
        pytest.skip("Synthcity not available for fflows")
    except Exception as e:
        pytest.fail(f"FourierFlows test failed: {e}")
