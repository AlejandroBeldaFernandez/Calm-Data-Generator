from calm_data_generator.generators.clinical import ClinicalDataGenerator
from calm_data_generator.generators.configs import DateConfig


def test_disease_effects_fix():
    """Regression test: 'index' key accepted and 'name' key not required."""
    gen = ClinicalDataGenerator(seed=42, auto_report=False, minimal_report=True)

    biomarker_config = {
        "target_type": "gene",
        "index": [0, 5, 12],
        "effect_type": "fold_change",
        "effect_value": 2.0,
        "group": "Disease",
    }

    data = gen.generate(
        n_samples=100,
        n_genes=500,
        n_proteins=0,
        control_disease_ratio=0.5,
        date_config=DateConfig(start_date="2024-01-01"),
        disease_effects_config=[biomarker_config],
    )

    assert "demographics" in data
    assert "genes" in data
    assert data["demographics"].shape[0] == 100
    assert data["genes"].shape[1] == 500
