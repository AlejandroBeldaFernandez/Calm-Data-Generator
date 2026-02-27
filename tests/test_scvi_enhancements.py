import pandas as pd
import numpy as np
from calm_data_generator.generators.tabular import RealGenerator

def test_scvi_enhancements():
    # 1. Crear datos ficticios (expresión génica simulada)
    np.random.seed(42)
    n_obs = 200
    n_genes = 30

    # Simular casos y controles con diferencias reales en algunos genes para que el modelo pueda aprender algo
    control_means = np.random.exponential(scale=10, size=n_genes)
    case_means = control_means.copy()
    case_means[:5] += 20 # Los primeros 5 genes están sobrexpresados en Casos

    control_data = np.random.poisson(lam=control_means, size=(n_obs // 2, n_genes))
    case_data = np.random.poisson(lam=case_means, size=(n_obs // 2, n_genes))

    expression_data = np.vstack([control_data, case_data])

    df = pd.DataFrame(expression_data, columns=[f"Gene_{i}" for i in range(n_genes)])
    df["Condition"] = ["Control"] * (n_obs // 2) + ["Case"] * (n_obs // 2)

    # Convertir a float porque scvi suele funcionar con float32
    for col in df.columns:
        if col != "Condition":
            df[col] = df[col].astype(float)

    generator = RealGenerator(random_state=42)

    # 2. Prueba 1: Generación base con latent sampling (differentiation_factor = 0)
    print("Iniciando prueba base (differentiation_factor=0.0)...")
    synthetic_base = generator.generate(
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
    
    # 3. Prueba 2: Generación con diferenciación exagerada
    print("\nIniciando prueba con diferenciación (differentiation_factor=2.0)...")
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

    # 4. Análisis de distancias
    def get_class_centroid(data_df, cls, cols):
        return data_df[data_df["Condition"] == cls][cols].mean().values

    genes = [f"Gene_{i}" for i in range(n_genes)]
    
    orig_case_cent = get_class_centroid(df, "Case", genes)
    orig_ctrl_cent = get_class_centroid(df, "Control", genes)
    orig_dist = np.linalg.norm(orig_case_cent - orig_ctrl_cent)

    base_case_cent = get_class_centroid(synthetic_base, "Case", genes)
    base_ctrl_cent = get_class_centroid(synthetic_base, "Control", genes)
    base_dist = np.linalg.norm(base_case_cent - base_ctrl_cent)

    diff_case_cent = get_class_centroid(synthetic_diff, "Case", genes)
    diff_ctrl_cent = get_class_centroid(synthetic_diff, "Control", genes)
    diff_dist = np.linalg.norm(diff_case_cent - diff_ctrl_cent)

    print("\n--- Resultados de Distancia entre Casos y Controles (Euclidiana) ---")
    print(f"Distancia Original: {orig_dist:.2f}")
    print(f"Distancia Base (diff_factor=0): {base_dist:.2f}")
    print(f"Distancia Exagerada (diff_factor=2.0): {diff_dist:.2f}")
    
    if diff_dist > base_dist:
        print("\n¡ÉXITO! El factor de diferenciación aumentó la separación entre las clases.")
    else:
        print("\nFALLO: El factor de diferenciación no aumentó la separación.")

if __name__ == "__main__":
    test_scvi_enhancements()
