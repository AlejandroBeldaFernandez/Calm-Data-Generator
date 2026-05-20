import logging

from calm_data_generator.generators.dynamics.ScenarioInjector import ScenarioInjector

from .base import GeneratorPreset

logger = logging.getLogger(__name__)


class ScenarioInjectorPreset(GeneratorPreset):
    """
    Directly leverages the ScenarioInjector to apply defined complex scenarios
    to an existing dataset, without necessarily generating new samples from scratch (unless desired).
    """

    def generate(self, data, scenario_config, **kwargs):
        injector = ScenarioInjector(seed=self.random_state)

        if self.verbose:
            logger.info("[ScenarioInjectorPreset] Applying scenario configuration...")

        return injector.evolve_features(
            df=data, evolution_config=scenario_config.get("evolve_features")
        )
