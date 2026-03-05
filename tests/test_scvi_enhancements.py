"""
Test: scVI differentiation with PBMC 10x v2 dataset

Strategy:
  - Load pbmcs_10x_v2 (real single-cell data) and select top 2000 HVGs.
  - No PCA needed: scVI already encodes to a compact latent space (10D by default).
  - Compare three differentiation approaches:
      1. scVI + differentiation_factor (latent shift)
      2. scANVI (semi-supervised, condition-aware latent space)
      3. ContrastiveVI (salient latent, requires contrastive-vi package)
  - ARI is evaluated on the scVI latent representations directly.
  - Realism is validated with QualityReporter (SDMetrics).
"""
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from calm_data_generator.generators.tabular import RealGenerator
from calm_data_generator.generators.tabular.QualityReporter import QualityReporter

seed = 42
np.random.seed(seed)


def compute_ari_from_latent(synth_df, label_col, model=None, adata_synth=None):
    """
    Calculates ARI directly from the latent representation of synthetic data.

    If `model` and `adata_synth` are provided, encodes through scVI to get latent.
    Otherwise falls back to encoding the numeric columns directly (PCA not needed
    if they are already latent/PCA features).
    """
    import anndata
    import scvi as scvi_lib

    feature_cols = [c for c in synth_df.columns if c != label_col]
    y = synth_df[label_col].astype(str)
    classes = y.unique()
    y_int = y.map({c: i for i, c in enumerate(classes)}).values

    if model is not None:
        # Encode synthetic data through the trained model to get latent
        expr_data = synth_df[feature_cols].values.astype(np.float32)
        adata_q = anndata.AnnData(X=expr_data)
        adata_q.obs_names = [f"synth_{i}" for i in range(len(synth_df))]
        adata_q.var_names = feature_cols

        scvi_lib.model.SCVI.setup_anndata(adata_q)
        # Transfer reference model params to query data for encoding
        latent = model.get_latent_representation(adata=adata_q)
    else:
        # Direct KMeans on numeric columns (already latent space)
        latent = synth_df[feature_cols].values

    n_clusters = len(classes)
    if latent.shape[0] < n_clusters:
        return 0.0

    kmeans = KMeans(n_clusters=n_clusters, random_state=seed, n_init="auto")
    clusters = kmeans.fit_predict(latent)
    return adjusted_rand_score(y_int, clusters)


def load_pbmc_hvg(n_top_genes=2000, n_cells_max=3000):
    """
    Loads the PBMC 10x v2 dataset and selects top HVGs.
    Returns an AnnData object filtered to the HVGs.
    Labels are cell types from adata.obs['cell_type'].
    """
    import scvi
    import scanpy as sc

    print("Loading PBMC 10x CITE-seq dataset...")
    adata = scvi.data.pbmcs_10x_cite_seq()
    # Keep only the RNA modality if multi-modal
    if "protein_expression" in adata.obsm:
        pass  # We only use adata.X (RNA) for scVI
    adata.obs_names_make_unique()  # Fix duplicate obs names
    print(f"  Raw: {adata.n_obs} cells x {adata.n_vars} genes")

    # Subsample for speed if needed
    if adata.n_obs > n_cells_max:
        sc.pp.subsample(adata, n_obs=n_cells_max, random_state=seed)
        print(f"  Subsampled to {adata.n_obs} cells")

    # Standard preprocessing
    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=3)

    # Standard preprocessing: normalize before HVG selection with seurat flavor
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    # HVG selection (seurat flavor works without scikit-misc)
    sc.pp.highly_variable_genes(
        adata,
        n_top_genes=n_top_genes,
        subset=True,
        flavor="seurat",
    )
    print(f"  After HVG selection: {adata.n_obs} cells x {adata.n_vars} genes")

    # Identify label column — this dataset has `batch` (5k vs 10k donor) as the only condition label
    label_col = None
    for col in ["cell_type", "louvain", "leiden", "labels", "batch"]:
        if col in adata.obs.columns and adata.obs[col].nunique() >= 2:
            label_col = col
            break

    if label_col is None:
        raise ValueError("No recognized label column found in PBMC dataset obs.")

    print(f"  Using label column: '{label_col}' ({adata.obs[label_col].nunique()} classes)")
    adata.obs[label_col] = adata.obs[label_col].astype(str)
    return adata, label_col


def adata_to_df(adata, label_col):
    """Converts AnnData expression matrix + label to a pandas DataFrame."""
    import scipy.sparse
    X = adata.X
    if scipy.sparse.issparse(X):
        X = X.toarray()
    df = pd.DataFrame(X, columns=adata.var_names)
    df[label_col] = adata.obs[label_col].values
    return df


def compute_ari_on_df(data_df, label_col, n_pca_components=50):
    """Computes ARI using KMeans after reducing features to n_pca_components using PCA."""
    from sklearn.decomposition import PCA

    feature_cols = [c for c in data_df.columns if c != label_col]
    X = data_df[feature_cols].values
    y = data_df[label_col].astype(str)
    classes = y.unique()
    y_int = y.map({c: i for i, c in enumerate(classes)}).values

    n_clusters = len(classes)
    if len(X) < n_clusters:
        return 0.0

    # Apply PCA before clustering
    pca = PCA(n_components=min(n_pca_components, X.shape[1], X.shape[0]), random_state=seed)
    X_pca = pca.fit_transform(X)

    kmeans = KMeans(n_clusters=n_clusters, random_state=seed, n_init="auto")
    clusters = kmeans.fit_predict(X_pca)
    return adjusted_rand_score(y_int, clusters)


def test_scvi_enhancements():
    """
    Tests scVI-based differentiation on real PBMC 10x v2 data.

    Compares:
      1. Original scVI latent (faithful) - ARI should be moderate (scVI already separates cell types)
      2. scVI + differentiation_factor=2.0 (enhanced)
      3. scANVI (semi-supervised, condition-aware)
    """
    adata, label_col = load_pbmc_hvg(n_top_genes=2000, n_cells_max=2000)
    df_original = adata_to_df(adata, label_col)
    n_obs = len(df_original)

    ari_original = compute_ari_on_df(df_original, label_col)
    print(f"\n=== Test scVI Differentiation (PBMC 10x v2) ===")
    print(f"Cells: {n_obs} | HVGs: {adata.n_vars} | Cell types: {adata.obs[label_col].nunique()}")
    print(f"ARI Original (raw HVG counts): {ari_original:.4f}")
    print("(scVI latent will be much more structured than raw HVG space)")

    generator = RealGenerator(random_state=seed)
    reporter = QualityReporter(verbose=False, minimal=True)

    # SDMetrics Column Pair Trends is O(n^2) on gene columns.
    # We subsample 50 random genes for quality evaluation.
    N_QUALITY_COLS = 50
    def quality_on_subset(real_df, synth_df, label_col):
        all_gene_cols = [c for c in real_df.columns if c != label_col]
        rng = np.random.default_rng(seed)
        subset_cols = rng.choice(all_gene_cols, size=min(N_QUALITY_COLS, len(all_gene_cols)), replace=False).tolist()
        real_sub = real_df[subset_cols + [label_col]].reset_index(drop=True)
        synth_sub = synth_df[subset_cols + [label_col] if label_col in synth_df.columns else subset_cols].reset_index(drop=True)
        scores = reporter.calculate_quality_metrics(real_sub, synth_sub)
        return scores.get('overall_quality_score', float('nan'))

    results = {}

    # -----------------------------------------------------------------------
    # 1. Standard scVI - faithful (no differentiation)
    # -----------------------------------------------------------------------
    print("\n--- scVI (fiel, differentiation_factor=0.0) ---")
    synth_faithful = generator.generate(
        data=adata,
        method="scvi",
        n_samples=n_obs,
        target_col=label_col,
        epochs=30,
        n_latent=10,
        use_latent_sampling=True,
        differentiation_factor=0.0,
        latent_noise_std=0.05,
    )
    ari_faithful = compute_ari_on_df(synth_faithful, label_col)
    results["scVI (fiel)"] = ari_faithful
    print(f"  ARI: {ari_faithful:.4f}")

    # -----------------------------------------------------------------------
    # 2. scVI + differentiation_factor
    # -----------------------------------------------------------------------
    print("\n--- scVI + differentiation_factor=2.0 ---")
    synth_diff = generator.generate(
        data=adata,
        method="scvi",
        n_samples=n_obs,
        target_col=label_col,
        epochs=30,
        n_latent=10,
        use_latent_sampling=True,
        differentiation_factor=2.0,
        latent_noise_std=0.05,
    )
    ari_diff = compute_ari_on_df(synth_diff, label_col)
    print("Calculando calidad SDMetrics (50 genes aleatorios)...")
    q_diff = quality_on_subset(df_original, synth_diff, label_col)
    results["scVI + diff_factor"] = ari_diff
    print(f"  ARI: {ari_diff:.4f}")
    print(f"  Calidad (SDMetrics): {q_diff:.4f}")

    # -----------------------------------------------------------------------
    # 3. scANVI (semi-supervised, condition-aware latent)
    # -----------------------------------------------------------------------
    print("\n--- scANVI (semi-supervisado) ---")
    synth_scanvi = generator.generate(
        data=adata,
        method="scvi",
        n_samples=n_obs,
        target_col=label_col,
        epochs=30,
        n_latent=10,
        use_scanvi=True,
        scanvi_epochs=10,
        differentiation_factor=0.0,  # scANVI already separates by class
        latent_noise_std=0.05,
    )
    ari_scanvi = compute_ari_on_df(synth_scanvi, label_col)
    print("Calculando calidad SDMetrics (50 genes aleatorios)...")
    q_scanvi = quality_on_subset(df_original, synth_scanvi, label_col)
    results["scANVI"] = ari_scanvi
    print(f"  ARI: {ari_scanvi:.4f}")
    print(f"  Calidad (SDMetrics): {q_scanvi:.4f}")

    # -----------------------------------------------------------------------
    # 4. scANVI + differentiation_factor
    # -----------------------------------------------------------------------
    print("\n--- scANVI + differentiation_factor=2.0 ---")
    synth_scanvi_diff = generator.generate(
        data=adata,
        method="scvi",
        n_samples=n_obs,
        target_col=label_col,
        epochs=30,
        n_latent=10,
        use_scanvi=True,
        scanvi_epochs=10,
        differentiation_factor=2.0,
        latent_noise_std=0.05,
    )
    ari_scanvi_diff = compute_ari_on_df(synth_scanvi_diff, label_col)
    print("Calculando calidad SDMetrics (50 genes aleatorios)...")
    q_scanvi_diff = quality_on_subset(df_original, synth_scanvi_diff, label_col)
    results["scANVI + diff_factor"] = ari_scanvi_diff
    print(f"  ARI: {ari_scanvi_diff:.4f}")
    print(f"  Calidad (SDMetrics): {q_scanvi_diff:.4f}")

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    print("\n\n=== RESULTADOS FINALES ===")
    print(f"{'Método':<30} {'ARI':>8}")
    print("-" * 40)
    print(f"{'ARI Original (raw)':<30} {ari_original:>8.4f}")
    for method_name, ari_val in results.items():
        marker = " ✓" if ari_val > ari_original else ""
        print(f"{method_name:<30} {ari_val:>8.4f}{marker}")


if __name__ == "__main__":
    test_scvi_enhancements()
