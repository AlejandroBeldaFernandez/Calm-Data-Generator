# CALM-Data-Generator - Synthetic Data Generation Library

from calm_data_generator import presets
from calm_data_generator.generators.clinical import ClinicalDataGenerator
from calm_data_generator.generators.complex import ComplexGenerator
from calm_data_generator.generators.drift import DriftInjector
from calm_data_generator.generators.dynamics import CausalEngine, ScenarioInjector
from calm_data_generator.generators.tabular import QualityReporter, RealGenerator

# Optional imports that may fail
try:
    from calm_data_generator.generators.stream import StreamGenerator
except ImportError:
    StreamGenerator = None

__version__ = "2.2.1"

__all__ = [
    # Generators
    "RealGenerator",
    "QualityReporter",
    "ClinicalDataGenerator",
    "ComplexGenerator",
    "StreamGenerator",
    # Drift & dynamics
    "DriftInjector",
    "ScenarioInjector",
    "CausalEngine",
    # Presets
    "presets",
]
