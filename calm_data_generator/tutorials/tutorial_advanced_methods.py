"""
Tutorial: Advanced Synthesis Methods (DDPM, TimeGAN, TimeVAE)

This tutorial demonstrates the new synthesis methods added to RealGenerator:
- ddpm: Advanced tabular diffusion from Synthcity
- timegan: Time series GAN
- timevae: Time series VAE

Author: CALM Data Generator Team
Date: 2026-02-06
"""

import pandas as pd
import numpy as np
from calm_data_generator.generators.tabular import RealGenerator

# ============================================================================
# Part 1: DDPM - Advanced Tabular Diffusion
# ============================================================================

print("=" * 80)
print("PART 1: DDPM (Synthcity TabDDPM)")
print("=" * 80)

# Create sample tabular data
np.random.seed(42)
tabular_data = pd.DataFrame(
    {
        "age": np.random.randint(20, 70, 200),
        "income": np.random.normal(50000, 15000, 200),
        "credit_score": np.random.randint(300, 850, 200),
        "debt_ratio": np.random.uniform(0, 1, 200),
        "approved": np.random.choice([0, 1], 200, p=[0.6, 0.4]),
    }
)

print("\nOriginal data shape:", tabular_data.shape)
print(tabular_data.head())

# Initialize generator
gen = RealGenerator(auto_report=False)

# Example 1: Basic DDPM usage
print("\n--- Example 1: Basic DDPM ---")
try:
    synth_basic = gen.generate(
        tabular_data,
        method="ddpm",
        n_samples=50,
        n_iter=100,  # Reduced for tutorial speed
        batch_size=64,
    )
    print(f"✅ Generated {len(synth_basic)} samples with DDPM")
    print(synth_basic.head())
except Exception as e:
    print(f"⚠️ DDPM requires synthcity: {e}")

# Example 2: DDPM with advanced architecture
print("\n--- Example 2: DDPM with ResNet architecture ---")
try:
    synth_resnet = gen.generate(
        tabular_data,
        method="ddpm",
        n_samples=50,
        n_iter=100,
        model_type="resnet",  # Advanced architecture
        scheduler="cosine",  # Better scheduler
        batch_size=128,
    )
    print(f"✅ Generated {len(synth_resnet)} samples with ResNet DDPM")
except Exception as e:
    print(f"⚠️ DDPM ResNet: {e}")

# Example 3: DDPM vs Custom Diffusion comparison
print("\n--- Example 3: DDPM vs Custom Diffusion ---")
print("DDPM: Production quality, slower (1000 epochs default)")
print("Diffusion: Quick prototyping, faster (100 epochs default)")

try:
    # Quick diffusion
    synth_diff = gen.generate(tabular_data, method="diffusion", n_samples=50, epochs=50)
    print(f"✅ Custom diffusion: {len(synth_diff)} samples")
except Exception as e:
    print(f"⚠️ Custom diffusion: {e}")

# ============================================================================
# Part 2: TimeGAN - Time Series Synthesis
# ============================================================================

print("\n" + "=" * 80)
print("PART 2: TimeGAN (Time Series GAN)")
print("=" * 80)

# Create sample time series data
np.random.seed(42)
n_timesteps = 100
time_series_data = pd.DataFrame(
    {
        "time": np.arange(n_timesteps),
        "sensor1": np.sin(np.arange(n_timesteps) / 10)
        + np.random.normal(0, 0.1, n_timesteps),
        "sensor2": np.cos(np.arange(n_timesteps) / 10)
        + np.random.normal(0, 0.1, n_timesteps),
        "sensor3": np.arange(n_timesteps) / 20 + np.random.normal(0, 0.2, n_timesteps),
    }
)

print("\nTime series data shape:", time_series_data.shape)
print(time_series_data.head())

# Example 4: TimeGAN for complex temporal patterns
print("\n--- Example 4: TimeGAN for complex patterns ---")
try:
    synth_timegan = gen.generate(
        time_series_data,
        method="timegan",
        n_samples=10,  # Generate 10 sequences
        n_iter=100,  # Reduced for tutorial
        n_units_hidden=50,
        batch_size=16,
    )
    print(f"✅ Generated time series with TimeGAN")
    print(f"Shape: {synth_timegan.shape}")
except Exception as e:
    print(f"⚠️ TimeGAN: {e}")

# ============================================================================
# Part 3: TimeVAE - Faster Time Series Synthesis
# ============================================================================

print("\n" + "=" * 80)
print("PART 3: TimeVAE (Time Series VAE)")
print("=" * 80)

# Example 5: TimeVAE for regular time series
print("\n--- Example 5: TimeVAE for faster generation ---")
try:
    synth_timevae = gen.generate(
        time_series_data,
        method="timevae",
        n_samples=10,
        n_iter=50,  # Faster than TimeGAN
        decoder_n_units_hidden=50,
        batch_size=16,
    )
    print(f"✅ Generated time series with TimeVAE")
    print(f"Shape: {synth_timevae.shape}")
except Exception as e:
    print(f"⚠️ TimeVAE: {e}")

# Example 6: TimeGAN vs TimeVAE comparison
print("\n--- Example 6: TimeGAN vs TimeVAE ---")
print("TimeGAN: Complex temporal patterns, slower, higher quality")
print("TimeVAE: Regular time series, faster, good quality")

# ============================================================================
# Part 4: Method Selection Guide
# ============================================================================

print("\n" + "=" * 80)
print("PART 4: Method Selection Guide")
print("=" * 80)

print("""
TABULAR DATA:
- Quick prototyping → diffusion (custom, fast)
- Production quality → ddpm (Synthcity, advanced)
- Large datasets → ddpm or lgbm
- Small datasets → cart or rf

TIME SERIES DATA:
- Complex temporal patterns → timegan
- Regular time series → timevae
- Fast training → timevae

SPECIAL CASES:
- Single-cell RNA-seq → scvi
- Clinical/Medical → ClinicalDataGenerator
- Streaming data → StreamGenerator
""")

# ============================================================================
# Part 5: Complete Example Workflow
# ============================================================================

print("\n" + "=" * 80)
print("PART 5: Complete Workflow Example")
print("=" * 80)

print("\n--- Scenario: Credit Risk Modeling ---")

# Create realistic credit data
np.random.seed(42)
credit_data = pd.DataFrame(
    {
        "age": np.random.randint(18, 75, 500),
        "income": np.random.lognormal(10.5, 0.5, 500),
        "credit_history_years": np.random.randint(0, 30, 500),
        "num_accounts": np.random.randint(1, 15, 500),
        "debt_to_income": np.random.uniform(0, 2, 500),
        "default": np.random.choice([0, 1], 500, p=[0.85, 0.15]),
    }
)

print("Original credit data:", credit_data.shape)

# Generate high-quality synthetic data with DDPM
try:
    synthetic_credit = gen.generate(
        credit_data,
        method="ddpm",
        n_samples=1000,
        n_iter=200,
        model_type="mlp",
        scheduler="cosine",
        is_classification=True,
        target_col="default",
    )
    print(f"✅ Generated {len(synthetic_credit)} synthetic credit records")
    print("\nSynthetic data preview:")
    print(synthetic_credit.head())
    print(f"\nDefault rate - Original: {credit_data['default'].mean():.2%}")
    print(f"Default rate - Synthetic: {synthetic_credit['default'].mean():.2%}")
except Exception as e:
    print(f"⚠️ DDPM generation: {e}")

print("\n" + "=" * 80)
print("Tutorial Complete!")
print("=" * 80)
print("\nFor more information, see:")
print("- REAL_GENERATOR_REFERENCE.md for detailed parameters")
print("- DOCUMENTATION.md for usage guides")
print("- API.md for quick reference")
