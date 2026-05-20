#!/usr/bin/env python3
"""
Base Reporter - Abstract Base Class for Data Reporters.

This module provides the BaseReporter class, an abstract base class that defines
the common interface and shared functionality for all data reporters in the library.
"""

import logging
import os
from abc import ABC, abstractmethod
from typing import Any, Optional

from calm_data_generator.logger import get_logger


class BaseReporter(ABC):
    """
    Abstract Base Class for all data reporters.

    This class defines the common initialization logic and interface that all
    reporters (QualityReporter, StreamReporter, PrivacyReporter) should follow.
    """

    def __init__(
        self,
        verbose: bool = True,
        minimal: bool = False,
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initializes the BaseReporter.

        Args:
            verbose: If True, prints progress messages to the console.
            minimal: If True, generates minimal reports (faster, no correlations/PCA).
            logger: An external logger instance. If None, a default one is created.
        """
        self.verbose = verbose
        self.minimal = minimal
        self.logger = logger if logger else get_logger(self.__class__.__name__)

    @staticmethod
    def ensure_output_dir(output_dir: str) -> str:
        """Ensures the output directory exists and returns its path."""
        os.makedirs(output_dir, exist_ok=True)
        return output_dir

    def log(self, message: str, level: str = "info") -> None:
        """Logs a message if verbose mode is enabled."""
        getattr(self.logger, level)(message)

    @abstractmethod
    def generate_report(self, *args, **kwargs) -> Any:
        """
        Main method to generate a report. Must be implemented by subclasses.
        """
        pass
