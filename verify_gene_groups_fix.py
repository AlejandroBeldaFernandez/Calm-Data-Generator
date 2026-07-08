"""
Standalone verification script — reproduces the 1_gex_MCY_belda.ipynb scenario
(basic MCY scenario: 4 gene groups A/B/D/NS correlated with age/sex, target Y
built from group weights), using the new `gene_groups` + group-name-referencing
`weights` support in ClinicalDataGenerator, and prints whether group-average
correlations with Y land closer to the configured targets than the old
single-representative-gene approach.

Run:
    python verify_gene_groups_fix.py
"""

import numpy as np
from scipy.stats import pearsonr

from calm_data_generator.generators.clinical.Clinic import ClinicalDataGenerator
from calm_data_generator.tutorials.clinical_generator import build_correlation_matrix

SEED = 45
np.random.seed(SEED)

N_SAMPLES = 2000
group_names = ["A", "B", "D", "NS"]
group_sizes = [100, 200, 300, 400]
gene_groups = dict(zip(group_names, group_sizes))

n_demo = 4
custom_demo = {
    "INR": {"distribution": "truncnorm", "a": 0, "b": 5, "loc": 1.1, "scale": 0.3},
    "batch": {"distribution": "randint", "low": 1, "high": 5},
}

# Same design as the notebook: A <-> age, B <-> sex, D weak intra-group, NS pure noise.
correlations_config = [
    {"internal": (0.3, 0.6), "demo_idx": 0, "demo_corr": 0.4},   # A & A <-> age
    {"internal": (0.2, 0.4), "demo_idx": 1, "demo_corr": 0.4},   # B & B <-> sex
    {"internal": (0.2, 0.3)},                                    # D
    {"internal": 0.0},                                           # NS
]
corr_mx = build_correlation_matrix(n_demo, group_sizes, correlations_config)

gen = ClinicalDataGenerator(seed=SEED)

datasets = gen.generate(
    n_samples=N_SAMPLES,
    n_genes=sum(group_sizes),
    gene_type="Microarray",
    gene_groups=gene_groups,                     # NEW: names groups by size
    custom_demographic_columns=custom_demo,
    demographic_gene_correlations=corr_mx,
    target_variable_config={
        "weights": {
            "Age": 3,
            "Sex_Binario": 1,
            "A": 2,                               # NEW: whole group, not one gene
            "B": 5,
        },
        "binary_threshold": "median",
    },
)

demo_df = datasets["demographics"]
genes_df = datasets["genes"]

print("=" * 70)
print("demographics columns:", list(demo_df.columns))
print("(expect: no 'Group'/'Disease_Subgroup' -- Y replaces them)")
print("=" * 70)

Y = demo_df["Y"].values.astype(float)

print("\n==== Group mean |correlation| of genes with Y ====")
print("Target: A~0.2  B~0.5  D~0  NS~0")
for name in group_names:
    cols = gen.gene_groups[name]
    corrs = [abs(pearsonr(genes_df[c].values, Y)[0]) for c in cols]
    print(
        f"  {name:<4} avg|r|={np.mean(corrs):.4f}  "
        f"min|r|={np.min(corrs):.4f}  max|r|={np.max(corrs):.4f}  (n={len(cols)} genes)"
    )

print("\n==== Covariate correlation with Y ====")
print("Target: age~0.3  sex~0.1")
for col in ["Age", "Sex"]:
    vals = demo_df[col]
    if vals.dtype == object:
        vals = (vals == "Male").astype(int)
    r, _ = pearsonr(vals.astype(float), Y)
    print(f"  {col:<6} r={r:.4f}")

print("\n==== Sanity: intra-group and group<->demographic (unchanged mechanism) ====")
a_cols, b_cols = gen.gene_groups["A"], gen.gene_groups["B"]
age = demo_df["Age"].astype(float).values
sex_bin = (demo_df["Sex"] == "Male").astype(int).values

a_age_r = [abs(pearsonr(genes_df[c].values, age)[0]) for c in a_cols]
b_sex_r = [abs(pearsonr(genes_df[c].values, sex_bin)[0]) for c in b_cols]
print(f"  age vs A(signal)  avg|r|={np.mean(a_age_r):.4f}  (target ~0.40)")
print(f"  sex vs B(signal)  avg|r|={np.mean(b_sex_r):.4f}  (target ~0.40)")
