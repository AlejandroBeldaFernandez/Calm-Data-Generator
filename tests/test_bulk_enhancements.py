import pandas as pd
import numpy as np
from calm_data_generator.generators.tabular import RealGenerator
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
import warnings

warnings.filterwarnings('ignore')

seed = 42


def compute_ari(data_df, label_col, feature_cols):
    X = data_df[feature_cols].values
    y = data_df[label_col]
    classes = y.unique()
    y_int = y.map({c: i for i, c in enumerate(classes)}).values
    kmeans = KMeans(n_clusters=len(classes), random_state=seed, n_init='auto')
    clusters = kmeans.fit_predict(X)
    return adjusted_rand_score(y_int, clusters)


def test_bulk_enhancements():
    """
    Objetivo:
      - Datos FIELES (split_by_class + match_original_separation):
            ARI sintético ~ ARI original (~0.21)
      - Datos DIFERENCIADOS (split_by_class):
            ARI sintético > 0.7
    """
    np.random.seed(seed)
    n_obs = 500
    n_genes = 20

    # Datos solapados: ARI original ~ 0.21
    n_diff_genes = 10
    ctrl_means = np.random.uniform(10, 50, size=n_genes)
    case_means = ctrl_means.copy()
    case_means[:n_diff_genes] += 4

    ctrl_data = np.random.normal(loc=ctrl_means, scale=8.0, size=(n_obs // 2, n_genes))
    case_data = np.random.normal(loc=case_means, scale=8.0, size=(n_obs // 2, n_genes))
    expression_data = np.clip(np.vstack([ctrl_data, case_data]), a_min=0, a_max=None)
    df = pd.DataFrame(expression_data, columns=[f"Gene_{i}" for i in range(n_genes)])
    df["Condition"] = ["Control"] * (n_obs // 2) + ["Case"] * (n_obs // 2)
    genes = [f"Gene_{i}" for i in range(n_genes)]

    ari_original = compute_ari(df, "Condition", genes)
    print(f"\n=== Test Bulk RNA-seq (TVAE / CTGAN) ===")
    print(f"ARI Original: {ari_original:.4f}  (referencia, esperamos < 0.3)")

    generator = RealGenerator(random_state=seed)

    for method in ["tvae", "ctgan"]:
        print(f"\n--- {method.upper()} ---")

        print(f"Generando datos fieles (split_by_class + match_original_separation)...")
        synthetic_faithful = generator.generate(
            data=df, method=method, n_samples=n_obs,
            target_col="Condition", epochs=20, batch_size=50,
            split_by_class=True, match_original_separation=True
        )
        ari_faithful = compute_ari(synthetic_faithful, "Condition", genes)

        print(f"Generando datos diferenciados (split_by_class)...")
        synthetic_diff = generator.generate(
            data=df, method=method, n_samples=n_obs,
            target_col="Condition", epochs=20, batch_size=50,
            split_by_class=True
        )
        ari_diff = compute_ari(synthetic_diff, "Condition", genes)

        print(f"\n  Resultados {method.upper()}:")
        print(f"  ARI Original:               {ari_original:.4f}")
        print(f"  ARI Fieles:                 {ari_faithful:.4f}  (esperamos: ~similar al original)")
        print(f"  ARI Diferenciados:          {ari_diff:.4f}  (esperamos: > 0.7)")

        if ari_diff > 0.7:
            print(f"  ✓ ÉXITO: Diferenciados superan ARI=0.7.")
        else:
            print(f"  ✗ AVISO: Diferenciados no superaron ARI=0.7.")

        if abs(ari_faithful - ari_original) < 0.15:
            print(f"  ✓ ÉXITO: Fieles tienen ARI similar al original.")
        else:
            print(f"  ✗ AVISO: Fieles difieren demasiado del original (diff={abs(ari_faithful - ari_original):.3f}).")


if __name__ == "__main__":
    test_bulk_enhancements()
