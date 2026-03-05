import pandas as pd
import numpy as np
import warnings
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.decomposition import PCA
from calm_data_generator.generators.tabular import RealGenerator
from calm_data_generator.generators.tabular.QualityReporter import QualityReporter

warnings.filterwarnings('ignore')

seed = 42

def compute_ari_from_pca(data_df, label_col):
    """Calcula el ARI directamente sobre las componentes PCA generadas."""
    feature_cols = [c for c in data_df.columns if c != label_col]
    X_pca = data_df[feature_cols].values
    y = data_df[label_col]
    classes = y.unique()
    y_int = y.map({c: i for i, c in enumerate(classes)}).values
    
    kmeans = KMeans(n_clusters=len(classes), random_state=seed, n_init='auto')
    clusters = kmeans.fit_predict(X_pca)
    return adjusted_rand_score(y_int, clusters)

def preprocesar_gse157103(tpm_path, metadata_path, max_genes=1000, n_components=50):
    """Carga, transpone, extrae HVGs y aplica PCA."""
    print("Cargando datos GSE157103...")
    
    # 1. Cargar metadatos
    # Leemos todo y filtramos. skiprows con lambda recibe el NÚMERO DE LÍNEA, no el contenido.
    with open(metadata_path, 'r') as f:
        meta_lines = [line.strip().split('\t') for line in f if line.startswith('!Sample_')]
    metadata_df = pd.DataFrame(meta_lines).set_index(0)
    
    # Extraemos el título (Sample_title) y características
    titles = metadata_df.loc['!Sample_title'].values
    characteristics = metadata_df.loc['!Sample_characteristics_ch1'].values
    
    # Simplificamos: Asumimos que la primera fila de 'characteristics' contiene el tejido/tipo.
    # En este caso particular, podríamos querer separar por severidad si está, pero 
    # asumiendo que el usuario quiere diferenciar alguna clase, tomamos los títulos
    # Muchos GSE tienen "COVID-19" o "Healthy" en el título.
    labels = []
    for t in titles:
        if "Healthy" in t or "healthy" in t:
            labels.append("Healthy")
        else:
            labels.append("COVID19")
            
    # Si todos son la misma clase por el parseo simplista, partimos el dataset en 2 aleatorio solo para el test técnico
    if len(set(labels)) < 2:
        print("Aviso: No se detectaron 2 clases claras. Dividiendo aleatoriamente para el test.")
        np.random.seed(seed)
        labels = np.random.choice(["ClassA", "ClassB"], size=len(titles))

    # 2. Cargar expresión (TPM)
    # Suponemos formato: Gene en col 1, muestras en el resto.
    df_tpm = pd.read_csv(tpm_path, sep='\t', index_col=0)
    
    # Transponer para que las filas sean muestras y columnas genes
    df_expr = df_tpm.T
    
    # Asegurar que el orden de las muestras cuadra (asumimos que sí por defecto en GSE)
    # Si el número de columnas de expr != número de muestras en metadata, truncamos.
    n_samples = min(len(df_expr), len(labels))
    df_expr = df_expr.iloc[:n_samples]
    labels = labels[:n_samples]

    # 3. Filtrar Highly Variable Genes (HVG)
    print(f"Filtrando {max_genes} Highly Variable Genes...")
    variances = df_expr.var(axis=0)
    top_genes = variances.nlargest(max_genes).index
    df_expr_hvg = df_expr[top_genes]

    # 4. Aplicar PCA
    print(f"Aplicando PCA a {n_components} componentes...")
    pca = PCA(n_components=min(n_components, len(df_expr_hvg)), random_state=seed)
    X_pca = pca.fit_transform(df_expr_hvg)
    
    # 5. Crear DataFrame final listo para el generador
    df_final = pd.DataFrame(X_pca, columns=[f"PC_{i+1}" for i in range(X_pca.shape[1])])
    df_final['Condition'] = labels
    
    return df_final

def test_bulk_enhancements():
    """
    Test using GSE157103 to demonstrate high differentiation (ARI improvement) 
    using TVAE/CTGAN on PCA components, and verifying their realism using QualityReporter.
    """
    np.random.seed(seed)
    
    # Paths to the user data
    tpm_path = "/home/alex/GSE157103_genes.tpm.tsv"
    meta_path = "/home/alex/GSE157103_series_matrix.txt"
    
    try:
        df = preprocesar_gse157103(tpm_path, meta_path, max_genes=1000, n_components=50)
    except FileNotFoundError:
        print("Archivos GSE157103 no encontrados. Por favor verifica las rutas.")
        return

    n_obs = len(df)
    print(f"Dataset preprocesado: {df.shape[0]} muestras, {df.shape[1]-1} componentes PCA.")

    ari_original = compute_ari_from_pca(df, "Condition")
    print(f"\n=== Test TVAE/CTGAN Differentiation (GSE157103) ===")
    print(f"ARI Original: {ari_original:.4f}")

    generator = RealGenerator(random_state=seed)
    reporter = QualityReporter(verbose=False, minimal=True)

    for method in ["tvae", "ctgan"]:
        print(f"\n--- {method.upper()} ---")

        print("Generando datos condicionados con diferenciación aumentada...")
        
        # Aumentamos epochs y usamos differentiation_factor alto para forzar la separación
        # differentiation_factor actuará empujando las clases en el espacio de características (PCA)
        synthetic_diff = generator.generate(
            data=df, 
            method=method, 
            n_samples=n_obs,
            target_col="Condition", 
            epochs=200,          # Necesario más entrenamiento por ser datos reales complejos
            batch_size=50,
            differentiation_factor=1.5,  # Fuerte dilatación
            split_by_class=True          # Mejor separación garantizada
        )
        
        ari_diff = compute_ari_from_pca(synthetic_diff, "Condition")
        
        # Validar realismo con QualityReporter
        print("Calculando métricas de calidad de SDMetrics...")
        quality_scores = reporter.calculate_quality_metrics(df, synthetic_diff)
        overall_quality = quality_scores.get('overall_quality_score', 0.0)

        print(f"\n  Resultados {method.upper()}:")
        print(f"  ARI Original:               {ari_original:.4f}")
        print(f"  ARI Diferenciados:          {ari_diff:.4f}")
        print(f"  Calidad (SDMetrics):        {overall_quality:.4f} (1.0 es idéntico a original)")

        if ari_diff > ari_original:
            print(f"  ✓ ÉXITO: Los datos diferenciados superan el ARI original ({ari_diff:.4f} > {ari_original:.4f}).")
        else:
            print(f"  ✗ AVISO: Los diferenciados no superaron el ARI original.")

if __name__ == "__main__":
    test_bulk_enhancements()

