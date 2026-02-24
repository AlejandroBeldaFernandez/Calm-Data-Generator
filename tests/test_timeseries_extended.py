import pytest
import pandas as pd
import numpy as np
from calm_data_generator.generators.tabular import RealGenerator


@pytest.fixture
def sequential_data():
    """Create a sequential dataset."""
    np.random.seed(42)
    n_sequences = 10
    seq_len = 20

    data = []
    for i in range(n_sequences):
        seq_id = i
        start_date = pd.Timestamp("2024-01-01")
        for t in range(seq_len):
            data.append(
                {
                    "seq_id": seq_id,
                    "timestamp": start_date + pd.Timedelta(days=t),
                    "value": np.sin(t / 5.0) + np.random.normal(0, 0.1),
                    "target": 1 if i % 2 == 0 else 0,
                }
            )
    return pd.DataFrame(data)


def test_timegan_availability(sequential_data):
    """Test TimeGAN generation (skips if dependencies missing or validation fails)."""
    gen = RealGenerator(auto_report=False)

    try:
        synth = gen.generate(
            data=sequential_data,
            n_samples=5,  # 5 sequences
            method="timegan",
            sequence_key="seq_id",
            time_key="timestamp",
            epochs=1,
        )
        assert len(synth) > 0
    except ImportError:
        pytest.skip("TimeGAN dependencies (synthcity) missing")
    except Exception as e:
        # Catch pydantic validaton errors or implementation placeholders
        # TimeGAN in synthcity is fragile to input format
        pytest.fail(f"TimeGAN execution failed: {e}")


def test_real_generator_with_time_col(sequential_data):
    """Test standard tabular generation preserving time column structure/type."""
    gen = RealGenerator(auto_report=False)

    # Pre-process: convert time to int for stability in basic models
    df_processed = sequential_data.copy()
    df_processed["timestamp"] = df_processed["timestamp"].astype(int)

    synth = gen.generate(
        data=df_processed, n_samples=20, method="cart", target_col="target"
    )

    assert len(synth) == 20
    assert "timestamp" in synth.columns
