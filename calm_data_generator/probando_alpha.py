import pandas as pd
import numpy as np
from calm_data_generator.generators.tabular.RealGenerator import RealGenerator

generator = RealGenerator(random_state=42)
# Simple dataset
df = pd.DataFrame({
    "feature1": np.random.randn(100),
    "feature2": np.random.randn(100),
    "target": np.random.choice([0, 1], size=100)
})


# Needs to be fast for the test, use minimal epochs
params = {"epochs": 2, "differentiation_factor": 2}

synth_df= generator.generate(
    data=df, 
    n_samples=10,
    method="tvae", 
    target_col="target", 
    **params
)

# Test get_synthesizer_model
model = generator.get_synthesizer_model()


# Test get_encoder
encoder = generator.get_encoder()


# Test get_decoder
decoder = generator.get_decoder()

