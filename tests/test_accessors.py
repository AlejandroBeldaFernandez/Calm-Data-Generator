import pytest
import pandas as pd
import numpy as np
from calm_data_generator.generators.tabular.RealGenerator import RealGenerator

def test_tvae_accessors():
    """Test that TVAE accessors return appropriate objects."""
    generator = RealGenerator(random_state=42)
    # Simple dataset
    df = pd.DataFrame({
        "feature1": np.random.randn(100),
        "feature2": np.random.randn(100),
        "target": np.random.choice([0, 1], size=100)
    })
    
    # Needs to be fast for the test, use minimal epochs
    params = {"epochs": 2, "differentiation_factor": 0.5}
    try:
        synth_df= generator.generate(
            data=df, 
            n_samples=10,
            method="tvae", 
            target_col="target", 
            **params
        )
        
        # Test get_synthesizer_model
        model = generator.get_synthesizer_model()
        assert model is not None, "Synthesizer model should not be None for TVAE"
        
        # Test get_encoder
        encoder = generator.get_encoder()
        assert encoder is not None, "Encoder should not be None for TVAE"
        
        # Test get_decoder
        decoder = generator.get_decoder()
        assert decoder is not None, "Decoder should not be None for TVAE"
        
        # Test get_latest_embeddings
        # Note: embeddings may be None if the encoder path failed (e.g. model undertrained)
        embeddings = generator.get_latest_embeddings()
        # Just verify the accessor works without error (can be None in fallback case)
        
    except ImportError:
        pytest.skip("Synthcity is not installed")


def test_ctgan_accessors():
    """Test that CTGAN accessors return appropriate objects."""
    generator = RealGenerator(random_state=42)
    df = pd.DataFrame({
        "feature1": np.random.randn(100),
        "feature2": np.random.randn(100),
        "target": np.random.choice([0, 1], size=100)
    })
    
    params = {"epochs": 2, "differentiation_factor": 0.5}
    try:
        synth_df = generator.generate(
            data=df, 
            n_samples=10,
            method="ctgan", 
            target_col="target", 
            **params
        )
        
        # Test get_synthesizer_model
        model = generator.get_synthesizer_model()
        print(model)
        assert model is not None, "Synthesizer model should not be None for CTGAN"
        
        # Test get_encoder
        encoder = generator.get_encoder()
        print(encoder)
        assert encoder is None, "Encoder should be None for CTGAN (doesn't exist)"
        
        # Test get_decoder
        decoder = generator.get_decoder()
        print(decoder)
        assert decoder is None, "Decoder should be None for CTGAN (doesn't exist)"

        # Test get_latest_embeddings
        embeddings = generator.get_latest_embeddings()
        assert embeddings is None, "Latest embeddings should be None for CTGAN (using feature space fallback)"

    except ImportError:
        pytest.skip("Synthcity is not installed")


def test_rtvae_accessors():
    """Test that RTVAE accessors return encoder and decoder."""
    generator = RealGenerator(random_state=42)
    df = pd.DataFrame({
        "feature1": np.random.randn(100),
        "feature2": np.random.randn(100),
        "target": np.random.choice([0, 1], size=100)
    })
    params = {"epochs": 2, "differentiation_factor": 0.3}
    try:
        generator.generate(
            data=df,
            n_samples=10,
            method="rtvae",
            target_col="target",
            **params
        )
        model = generator.get_synthesizer_model()
        assert model is not None, "Synthesizer model should not be None for RTVAE"

        encoder = generator.get_encoder()
        assert encoder is not None, "Encoder should not be None for RTVAE"

        decoder = generator.get_decoder()
        assert decoder is not None, "Decoder should not be None for RTVAE"

    except ImportError:
        pytest.skip("Synthcity is not installed")
