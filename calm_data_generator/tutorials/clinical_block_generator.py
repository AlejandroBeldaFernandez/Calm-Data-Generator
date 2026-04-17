"""
Tutorial 8: ClinicalBlockGenerator - Multi-Center Clinical Studies
==================================================================

This tutorial demonstrates how to use ClinicalDataGeneratorBlock to simulate data
from multiple clinical sites (blocks) or time periods, potentially with different
underlying distributions (simulating center effects or drift).
"""

import pandas as pd
import shutil
import os

from calm_data_generator.generators.clinical.ClinicGeneratorBlock import (
    ClinicalDataGeneratorBlock,
)

# Setup output directory
OUTPUT_DIR = "tutorial_output/08_clinic_block"
if os.path.exists(OUTPUT_DIR):
    shutil.rmtree(OUTPUT_DIR)
os.makedirs(OUTPUT_DIR, exist_ok=True)

gen = ClinicalDataGeneratorBlock()

print("\n--- Generating Multi-Center Clinical Data ---")

full_path = gen.generate(
    output_dir=OUTPUT_DIR,
    filename="multi_center_study.csv",
    n_blocks=3,
    total_samples=150,  # 50 patients per hospital
    n_samples_block=[50, 50, 50],
    target_col="diagnosis",
    date_start="2024-01-01",
    date_step={
        "months": 1
    },  # Each block is separated by a month (or could be simultaneous)
    generate_report=True,  # Generate aggregated clinical report
)

print(f"Generated multi-center clinical data at: {full_path}")
df = pd.read_csv(full_path)

print("Columns generated:", df.columns.tolist()[:10], "...")
print("\nPatient counts per Hospital (Block):")
print(df["block"].value_counts())

print("\nSample records:")
print(df[["Patient_ID", "Age", "Sex", "diagnosis", "block"]].head())

print(f"\n✅ Tutorial completed! Outputs saved to {OUTPUT_DIR}")
