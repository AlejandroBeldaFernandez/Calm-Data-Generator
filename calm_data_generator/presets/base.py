from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Union

import pandas as pd


class GeneratorPreset(ABC):
    """
    Abstract base class for all generator presets.

    A preset encapsulates a specific generator configuration (method, hyperparameters,
    reporting, etc.) tailored for a particular use case.
    """

    def __init__(
        self,
        random_state: Optional[int] = 42,
        verbose: bool = True,
        fast_dev_run: bool = False,
    ):
        self.random_state = random_state
        self.verbose = verbose
        # fast_dev_run forces minimal iterations/epochs for testing/debugging
        self.fast_dev_run = fast_dev_run

    @abstractmethod
    def generate(
        self, data: Any, n_samples: int, **kwargs
    ) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
        """
        Main method to generate synthetic data using this preset's configuration.

        Args:
            data: The original dataset (DataFrame, AnnData, etc.) to learn from.
            n_samples: Number of synthetic samples to generate.
            **kwargs: Overrides for configuration parameters.

        Returns:
            pd.DataFrame or Dict: The synthetic dataset(s).
        """
        pass
