from .base import GeneratorPreset
from calm_data_generator.generators.tabular import RealGenerator


class TimeSeriesPreset(GeneratorPreset):
    """
    Preset optimized for generating sequential or time-series data.
    Uses TimeGAN, TimeVAE, or FourierFlows (fflows) to capture temporal dynamics.

    TimeGAN: best for complex temporal patterns.
    TimeVAE: faster, good for regular series.
    fflows: most stable, best for periodic/seasonal series.
    """

    def generate(
        self, data, n_samples, sequence_key, time_key=None, method="timegan", **kwargs
    ):
        gen = RealGenerator(
            auto_report=kwargs.pop("auto_report", True), random_state=self.random_state
        )

        if self.verbose:
            print(
                f"[TimeSeriesPreset] Generating time-series data using {method} "
                f"(seq_key='{sequence_key}')..."
            )

        epochs = 1 if self.fast_dev_run else 500

        return gen.generate(
            data=data,
            n_samples=n_samples,
            method=method,
            sequence_key=sequence_key,
            time_key=time_key,
            n_iter=epochs,
            batch_size=100,
        )
