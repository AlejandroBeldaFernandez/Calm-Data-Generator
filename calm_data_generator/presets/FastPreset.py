from .base import GeneratorPreset
from calm_data_generator.generators.tabular import RealGenerator


class FastPreset(GeneratorPreset):
    def generate(self, data, n_samples, **kwargs):
        # Uses LightGBM for speed, minimal reporting
        gen = RealGenerator(
            auto_report=kwargs.pop("auto_report", False),
            minimal_report=kwargs.pop("minimal_report", True),
            random_state=self.random_state,
        )

        # 10 iterations is very fast but provides decent enough structure
        return gen.generate(
            data=data,
            n_samples=n_samples,
            method="lgbm",
            iterations=kwargs.pop("iterations", 10),
            **kwargs,
        )
