import logging

from calm_data_generator.generators.clinical import ClinicalDataGenerator

from .base import GeneratorPreset

logger = logging.getLogger(__name__)


class OmicsIntegrationPreset(GeneratorPreset):
    """
    Generates multi-omics data (Clinical + Gene Expression + Proteomics)
    with high correlation integrity between layers.
    """

    def generate(self, n_samples, n_genes=100, n_proteins=50, **kwargs):
        gen = ClinicalDataGenerator(
            auto_report=kwargs.pop("auto_report", True), seed=self.random_state
        )

        if self.verbose:
            logger.info(
                f"[OmicsIntegrationPreset] Simulating multi-omics data (Clinical + {n_genes} Genes + {n_proteins} Proteins)..."
            )

        # Forces generation of all layers
        return gen.generate(n_samples=n_samples, n_genes=n_genes, n_proteins=n_proteins)
