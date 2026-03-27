# Backward-compatibility shim.
# QualityReporter has been moved to calm_data_generator.reports.QualityReporter
from calm_data_generator.reports.QualityReporter import QualityReporter  # noqa: F401

__all__ = ["QualityReporter"]
