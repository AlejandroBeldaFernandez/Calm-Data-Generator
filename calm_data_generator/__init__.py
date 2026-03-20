# CALM-Data-Generator - Synthetic Data Generation Library

from calm_data_generator.generators.tabular import RealGenerator, QualityReporter
from calm_data_generator.generators.clinical import ClinicalDataGenerator
from calm_data_generator.generators.drift import DriftInjector
from calm_data_generator.generators.dynamics import ScenarioInjector

# Optional imports that may fail
try:
    from calm_data_generator.generators.stream import StreamGenerator
except ImportError:
    StreamGenerator = None

__version__ = "1.2.0"

__all__ = [
    "RealGenerator",
    "QualityReporter",
    "ClinicalDataGenerator",
    "StreamGenerator",
    "DriftInjector",
    "ScenarioInjector",
]
