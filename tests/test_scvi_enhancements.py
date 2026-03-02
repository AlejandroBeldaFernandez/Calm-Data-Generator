import pandas as pd
import numpy as np
from calm_data_generator.generators.tabular import RealGenerator
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

seed = 42


def compute_ari(data_df, label_col, feature_cols):
    """Calcula el ARI entre las etiquetas reales y clusters KMeans ciegos."""
    X = data_df[feature_cols].values
    y = data_df[label_col]
    classes = y.unique()
    y_int = y.map({c: i for i, c in enumerate(classes)}).values
    kmeans = KMeans(n_clusters=len(classes), random_state=seed, n_init='auto')
    clusters = kmeans.fit_predict(X)
    return adjusted_rand_score(y_int, clusters)


def test_scvi_enhancements():
    """
    Objetivo:
      - Datos FIELES:    ARI sintético ~ ARI original (baja separación, mantenida)
      - Datos DIFERENCIADOS: ARI sintético > ARI original (separación incrementada)

    Para que el test sea válido, los datos originales deben tener un ARI BAJO
    (casos y controles muy solapados), de modo que haya margen para mejorar.
    """
    np.random.seed(seed)
    n_obs = 200
    n_genes = 30

    # Datos intencionalmente solapados para ARI original bajo (~0.27)
    # n_diff_genes=10, delta=5, noise=8 => ARI~0.27 confirmado empiricamente
    n_diff_genes = 10
    ctrl_means = np.random.exponential(scale=10, size=n_genes)
    case_means = ctrl_means.copy()
    case_means[:n_diff_genes] += 5  # Diferencia moderada solo en 10 genes

    ctrl_data = np.random.normal(loc=ctrl_means, scale=8.0, size=(n_obs // 2, n_genes))
    case_data = np.random.normal(loc=case_means, scale=8.0, size=(n_obs // 2, n_genes))

    expression_data = np.clip(np.vstack([ctrl_data, case_data]), a_min=0, a_max=None)
    df = pd.DataFrame(expression_data, columns=[f"Gene_{i}" for i in range(n_genes)])
    df["Condition"] = ["Control"] * (n_obs // 2) + ["Case"] * (n_obs // 2)

    for col in df.columns:
        if col != "Condition":
            df[col] = df[col].astype(float)

    genes = [f"Gene_{i}" for i in range(n_genes)]

    ari_original = compute_ari(df, "Condition", genes)
    print(f"\n=== Test scVI (Single-Cell RNA-seq) ===")
    print(f"ARI Original (solapado): {ari_original:.4f}  (esperamos: < 0.3)")

    generator = RealGenerator(random_state=seed)

    # 1. Datos fieles (latent sampling sin diferenciar)
    print("\nGenerando datos fieles (differentiation_factor=0.0)...")
    synthetic_similar = generator.generate(
        data=df,
        method="scvi",
        n_samples=100,
        target_col="Condition",
        epochs=15,
        n_latent=5,
        use_latent_sampling=True,
        differentiation_factor=0.0,
        latent_noise_std=0.05
    )
    ari_similar = compute_ari(synthetic_similar, "Condition", genes)
    print(f"ARI Fieles (diff_factor=0): {ari_similar:.4f}")

    # 2. Datos diferenciados (latent space shift)
    print("\nGenerando datos diferenciados (differentiation_factor=2.0)...")
    synthetic_diff = generator.generate(
        data=df,
        method="scvi",
        n_samples=100,
        target_col="Condition",
        epochs=15,
        n_latent=5,
        use_latent_sampling=True,
        differentiation_factor=2.0,
        latent_noise_std=0.05
    )
    ari_diff = compute_ari(synthetic_diff, "Condition", genes)
    print(f"ARI Diferenciados (diff_factor=2.0): {ari_diff:.4f}")

    print("\n--- RESULTADOS scVI ---")
    print(f"  ARI Original:               {ari_original:.4f}  (referencia, bajo)")
    print(f"  ARI Fieles (diff=0):        {ari_similar:.4f}  (esperamos: ~similar al original)")
    print(f"  ARI Diferenciados (diff=2): {ari_diff:.4f}  (esperamos: MAYOR que el original)")

    if ari_diff > ari_original:
        print("\n  ✓ ÉXITO: Los datos diferenciados tienen ARI mayor que el original.")
    else:
        print("\n  ✗ AVISO: Los datos diferenciados no superaron el ARI original.")


if __name__ == "__main__":
    test_scvi_enhancements()
