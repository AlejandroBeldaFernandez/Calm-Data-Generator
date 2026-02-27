import pandas as pd
import numpy as np
from calm_data_generator.generators.tabular import RealGenerator
import warnings

warnings.filterwarnings('ignore')

def test_bulk_enhancements():
    # 1. Crear datos ficticios (expresión génica simulada estilo bulk RNA-seq)
    # Bulk suele tener diferencias continuas y no tantos ceros como scRNA-seq
    np.random.seed(42)
    n_obs = 500  # Más muestras para que tvae/ctgan aprendan bien
    n_genes = 20

    # Simular casos y controles con diferencias reales en algunos genes para que el modelo pueda aprender
    control_means = np.random.uniform(10, 50, size=n_genes)
    case_means = control_means.copy()
    case_means[:5] += 30 # Los primeros 5 genes están visiblemente sobrexpresados en Casos

    # Usamos normal para simular datos log-normalizados de bulk
    control_data = np.random.normal(loc=control_means, scale=5.0, size=(n_obs // 2, n_genes))
    case_data = np.random.normal(loc=case_means, scale=5.0, size=(n_obs // 2, n_genes))

    expression_data = np.vstack([control_data, case_data])
    # Evitar negativos
    expression_data = np.clip(expression_data, a_min=0, a_max=None)

    df = pd.DataFrame(expression_data, columns=[f"Gene_{i}" for i in range(n_genes)])
    df["Condition"] = ["Control"] * (n_obs // 2) + ["Case"] * (n_obs // 2)

    generator = RealGenerator(random_state=42)

    # 4. Análisis de distancias
    def get_class_centroid(data_df, cls, cols):
        return data_df[data_df["Condition"] == cls][cols].mean().values

    genes = [f"Gene_{i}" for i in range(n_genes)]
    
    orig_case_cent = get_class_centroid(df, "Case", genes)
    orig_ctrl_cent = get_class_centroid(df, "Control", genes)
    orig_dist = np.linalg.norm(orig_case_cent - orig_ctrl_cent)

    print("\n--- Resultados Originales ---")
    print(f"Distancia entre centroides (Casos vs Controles) Original: {orig_dist:.2f}")

    # ==========================================
    # PRUEBA TVAE
    # ==========================================
    print("\n--- Iniciando prueba Bulk con TVAE ---")
    try:
        synthetic_tvae_base = generator.generate(
            data=df,
            method="tvae",
            n_samples=n_obs,
            target_col="Condition",
            epochs=20, # Pocas epocas para test rápido
            batch_size=50,
            differentiation_factor=0.0
        )
        
        tvae_base_case_cent = get_class_centroid(synthetic_tvae_base, "Case", genes)
        tvae_base_ctrl_cent = get_class_centroid(synthetic_tvae_base, "Control", genes)
        tvae_base_dist = np.linalg.norm(tvae_base_case_cent - tvae_base_ctrl_cent)
        
        print(f"Distancia TVAE Base (con factor 0): {tvae_base_dist:.2f}")

        synthetic_tvae = generator.generate(
            data=df,
            method="tvae",
            n_samples=n_obs,
            target_col="Condition",
            epochs=20, # Pocas epocas para test rápido
            batch_size=50,
            differentiation_factor=2.0
        )
        
        tvae_case_cent = get_class_centroid(synthetic_tvae, "Case", genes)
        tvae_ctrl_cent = get_class_centroid(synthetic_tvae, "Control", genes)
        tvae_dist = np.linalg.norm(tvae_case_cent - tvae_ctrl_cent)
        
        print(f"Distancia TVAE (con factor 2.0): {tvae_dist:.2f}")
        
        if tvae_dist >= orig_dist:
            print("ÉXITO: TVAE logra superar la distancia original (diferenciación aumentada).")
        else:
            print("AVISO: TVAE no superó la distancia original.")
    except Exception as e:
        print(f"Error con TVAE: {e}")

    # ==========================================
    # PRUEBA CTGAN
    # ==========================================
    print("\n--- Iniciando prueba Bulk con CTGAN ---")
    try:
        synthetic_ctgan = generator.generate(
            data=df,
            method="ctgan",
            n_samples=n_obs,
            target_col="Condition",
            epochs=20, # Pocas epocas para test rápido
            batch_size=50,
            differentiation_factor=2.0
        )
        
        ctgan_case_cent = get_class_centroid(synthetic_ctgan, "Case", genes)
        ctgan_ctrl_cent = get_class_centroid(synthetic_ctgan, "Control", genes)
        ctgan_dist = np.linalg.norm(ctgan_case_cent - ctgan_ctrl_cent)
        
        print(f"Distancia CTGAN (con factor 2.0): {ctgan_dist:.2f}")
        
        if ctgan_dist >= orig_dist:
            print("ÉXITO: CTGAN logra superar la distancia original (diferenciación aumentada).")
        else:
            print("AVISO: CTGAN no superó la distancia original.")
    except Exception as e:
        print(f"Error con CTGAN: {e}")

if __name__ == "__main__":
    test_bulk_enhancements()
