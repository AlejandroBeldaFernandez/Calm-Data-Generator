"""
Script de evaluación de calidad de scVI con múltiples datasets reales de scvi.data.
Genera datos sintéticos con parámetros por defecto y evalúa quality score (SDMetrics).
"""

import numpy as np
import pandas as pd
import warnings
import traceback
from calm_data_generator.generators.tabular.RealGenerator import RealGenerator

warnings.filterwarnings("ignore")

MAX_CELLS = 2000  # Subsample datasets grandes para que no tarde horas


def subsample(adata, max_cells=MAX_CELLS):
    if adata.n_obs > max_cells:
        idx = np.random.choice(adata.n_obs, max_cells, replace=False)
        adata = adata[idx].copy()
    return adata


# ---------------------------------------------------------------------------
# Loaders — cada uno devuelve (adata, target_col, name)
# ---------------------------------------------------------------------------

def load_synthetic_iid():
    import scvi
    adata = scvi.data.synthetic_iid()
    return adata, "labels", "synthetic_iid"


def load_cortex():
    import scvi
    adata = scvi.data.cortex()
    adata = subsample(adata)
    return adata, "cell_type", "cortex"


def load_pbmc():
    import scvi
    adata = scvi.data.pbmc_dataset()
    adata = subsample(adata)
    target = "str_labels" if "str_labels" in adata.obs.columns else "labels"
    return adata, target, "pbmc"


def load_heart_cell_atlas():
    import scvi
    adata = scvi.data.heart_cell_atlas_subsampled()
    adata = subsample(adata)
    target = "cell_type" if "cell_type" in adata.obs.columns else adata.obs.columns[0]
    return adata, target, "heart_cell_atlas"


def load_purified_pbmc():
    import scvi
    adata = scvi.data.purified_pbmc_dataset()
    adata = subsample(adata)
    target = "str_labels" if "str_labels" in adata.obs.columns else "labels"
    return adata, target, "purified_pbmc"


def load_brainlarge():
    import scvi
    adata = scvi.data.brainlarge_dataset(save_path="/tmp/scvi_data/")
    adata = subsample(adata, max_cells=1000)
    # brainlarge may not have labels — use a dummy binary split
    if not any(c for c in adata.obs.columns if "label" in c.lower() or "type" in c.lower()):
        adata.obs["group"] = np.random.choice(["A", "B"], size=adata.n_obs)
        target = "group"
    else:
        target = [c for c in adata.obs.columns if "label" in c.lower() or "type" in c.lower()][0]
    return adata, target, "brainlarge"


def load_retina():
    import scvi
    adata = scvi.data.retina()
    adata = subsample(adata)
    target = "labels" if "labels" in adata.obs.columns else adata.obs.columns[0]
    return adata, target, "retina"


def load_breast_cancer():
    import scvi
    adata = scvi.data.breast_cancer_dataset()
    adata = subsample(adata)
    if not any(c for c in adata.obs.columns if "label" in c.lower() or "type" in c.lower()):
        adata.obs["group"] = np.random.choice(["A", "B"], size=adata.n_obs)
        target = "group"
    else:
        target = [c for c in adata.obs.columns if "label" in c.lower() or "type" in c.lower()][0]
    return adata, target, "breast_cancer"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

ALL_LOADERS = [
    load_synthetic_iid,
    load_cortex,
    load_pbmc,
    load_heart_cell_atlas,
    load_purified_pbmc,
    load_brainlarge,
    load_retina,
    load_breast_cancer,
]


def run_all():
    all_results = []

    for loader in ALL_LOADERS:
        name = loader.__name__.replace("load_", "")
        print(f"\n{'='*70}")
        print(f"  Dataset: {name}")
        print(f"{'='*70}")

        # Load dataset
        try:
            data, target_col, name = loader()
        except Exception as e:
            print(f"  SKIP (download/load failed): {e}")
            all_results.append({"dataset": name, "quality_score": "LOAD ERROR"})
            continue

        print(f"  Shape: {data.n_obs} obs x {data.n_vars} vars | target: '{target_col}'")
        if target_col in data.obs.columns:
            print(f"  Classes: {data.obs[target_col].nunique()} unique values")

        gen = RealGenerator(random_state=42, auto_report=False)
        n_samples = min(data.n_obs, 500)

        # Generate
        try:
            synth_df = gen.generate(
                data=data,
                method="scvi",
                n_samples=n_samples,
                target_col=target_col,
            )
        except Exception as e:
            print(f"  GENERATION FAILED: {e}")
            traceback.print_exc()
            all_results.append({"dataset": name, "quality_score": "GEN ERROR"})
            continue

        if synth_df is None:
            print("  GENERATION FAILED: returned None")
            all_results.append({"dataset": name, "quality_score": "GEN ERROR (None)"})
            continue

        print(f"  Generated: {len(synth_df)} samples, {synth_df.shape[1]} columns")

        # Build comparable real DataFrame
        real_df = pd.DataFrame(
            data.X.toarray() if hasattr(data.X, "toarray") else data.X,
            columns=data.var_names,
        )
        real_df[target_col] = data.obs[target_col].values

        # Quality score
        try:
            qm = gen.reporter.calculate_quality_metrics(real_df, synth_df)
            score = qm.get("overall_quality_score", None)
            if score is not None:
                score = round(score, 4)
        except Exception as e:
            score = f"METRIC ERROR: {e}"

        all_results.append({"dataset": name, "quality_score": score})
        print(f"  Quality Score: {score}")

        # Show model/encoder/decoder only for the first dataset
        if len(all_results) == 1:
            model = gen.get_synthesizer_model()
            encoder = gen.get_encoder()
            decoder = gen.get_decoder()
            print(f"\n  --- Model internals (first dataset only) ---")
            print(f"  Model:   {type(model).__name__} -> {model}")
            print(f"  Encoder: {type(encoder).__name__} -> {encoder}")
            print(f"  Decoder: {type(decoder).__name__} -> {decoder}")

    # Summary
    print(f"\n\n{'='*70}")
    print("  SUMMARY — scVI Quality Scores (default params)")
    print(f"{'='*70}")
    for r in all_results:
        s = r["quality_score"]
        if isinstance(s, float):
            print(f"  {r['dataset']:25s}  {s:.2%}")
        else:
            print(f"  {r['dataset']:25s}  {s}")

    scores = [r["quality_score"] for r in all_results if isinstance(r["quality_score"], float)]
    if scores:
        print(f"\n  Average:                    {np.mean(scores):.2%}")
        print(f"  Min:                        {min(scores):.2%}")
        print(f"  Max:                        {max(scores):.2%}")


if __name__ == "__main__":
    run_all()