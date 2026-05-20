import logging

from calm_data_generator.generators.configs import DriftConfig
from calm_data_generator.generators.tabular import RealGenerator

from .base import GeneratorPreset

logger = logging.getLogger(__name__)


class ConceptDriftPreset(GeneratorPreset):
    """
    Simulates sudden concept drift by altering the relationship between features and target.
    Useful for testing model robustness to P(y|x) changes.
    """

    def generate(self, data, n_samples, target_col, drift_magnitude=0.5, **kwargs):
        # Configuration to invert or shift the target relationship
        drift_conf = [
            {
                "column": target_col,
                "type": "concept_drift",  # Mapped internally or custom logic
                "magnitude": drift_magnitude,
            }
        ]

        # RealGenerator with drift config
        gen = RealGenerator(
            auto_report=kwargs.pop("auto_report", True), random_state=self.random_state
        )

        if self.verbose:
            logger.info(
                f"[ConceptDriftPreset] Injecting concept drift into '{target_col}'..."
            )

        # Enforce CTGAN for concept drift to ensuring learning distribution
        epochs = 1 if self.fast_dev_run else 300

        config = {"method": "ctgan", "epochs": epochs}

        return gen.generate(
            data=data,
            n_samples=n_samples,
            drift_injection_config=DriftConfig(drift_injection_config=drift_conf),
            **config,
        )
