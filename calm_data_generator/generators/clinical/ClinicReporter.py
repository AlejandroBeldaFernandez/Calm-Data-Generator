"""
Specialized Clinical Data Reporter
"""

import logging

from calm_data_generator.generators.stream.StreamReporter import StreamReporter

logger = logging.getLogger("ClinicReporter")


class ClinicReporter(StreamReporter):
    """
    A specialized version of StreamReporter for clinical data.
    Tailored to handle clinical feature sets and domain-specific checks.
    """

    def __init__(self, verbose: bool = True, minimal_report: bool = False):
        """
        Initializes the ClinicReporter.
        """
        super().__init__(verbose=verbose, minimal_report=minimal_report)

    def generate_report(self, *args, **kwargs):
        """
        Generates a comprehensive clinical data report.
        Currently uses StreamReporter's base logic.
        """
        if self.verbose:
            logger.info("CLINICAL DATA REPORT")

        return super().generate_report(*args, **kwargs)
