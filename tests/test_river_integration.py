import pandas as pd
import numpy as np
import shutil
import tempfile
import os
import pytest

# Try to import river to skip if not available
try:
    from river import synth

    RIVER_AVAILABLE = True
except ImportError:
    try:
        from river.datasets import synth

        RIVER_AVAILABLE = True
    except ImportError:
        RIVER_AVAILABLE = False
        synth = None

from calm_data_generator.generators.stream.StreamGenerator import StreamGenerator


def test_river_drift_detection():
    """Test StreamGenerator with River synth generator."""
    output_dir = tempfile.mkdtemp()
    np.random.seed(42)
    try:
        if not RIVER_AVAILABLE:
            pytest.skip("River not available")

        # Create a River generator
        agrawal_gen = synth.Agrawal(seed=42)

        generator = StreamGenerator(auto_report=False)

        try:
            # Generate from River stream
            n_samples = 100
            synthetic_stream = generator.generate(
                generator_instance=agrawal_gen,
                n_samples=n_samples,
                output_dir=output_dir,
            )

            assert synthetic_stream is not None
            assert len(synthetic_stream) == n_samples

            # Check if columns are present (Agrawal features: salary, commission, age, etc.)
            assert "target" in synthetic_stream.columns
            assert len(synthetic_stream.columns) > 1

            print("River integration test passed (River Generator)")

        except Exception as e:
            import traceback

            traceback.print_exc()
            pytest.fail(f"StreamGenerator failed with River available: {e}")
    finally:
        shutil.rmtree(output_dir)
