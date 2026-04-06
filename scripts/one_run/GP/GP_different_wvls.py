from utils.paths import DATABASE_DIR,PREDICTIONS_DIR

from preprocessing.gp_workflow import ExperimentalGPApproach
from preprocessing.data_loading import load_table
from preprocessing.processing import merge_data
from preprocessing.splitting import DataSplitter
from preprocessing.Descriptors import AlloyDescriptorCalculator

import pandas as pd
import numpy as np


df_optical = load_table(DATABASE_DIR / "Ternary_round1.db", "Optical_properties")
df_comp = load_table(DATABASE_DIR / "Ternary_round1.db", "compositions")
df_resis_ANN = load_table(DATABASE_DIR / "Ternary_round1.db", "ANN_resistivity")
# --- Merge ---
df_merged = merge_data(df_optical, df_comp)
df_merged_all = merge_data(df_merged, df_resis_ANN[['ID','resistivity']])


def evaluate_wavelength(df, wavelength, descriptor_spec):
    
    # --- Filter ---
    df_wl = df[df["wavelength_nm"] == wavelength].reset_index(drop=True)

    # --- Define X and y ---
    X = df_wl[["Cu", "Ni", "Al", "resistivity"]]
    y = df_wl["e2"]

    # --- Split ---
    splitter = DataSplitter(
        test_size=0.2,
        random_state=42,
        split_method="cluster",
        n_clusters=5
    )
    split_dict = splitter.split(X, y)

    # --- Descriptors ---
    calc = AlloyDescriptorCalculator()

    x_train = calc.transform(split_dict["X_train"], descriptor_spec=descriptor_spec, include_entropy=True)
    x_test  = calc.transform(split_dict["X_test"],  descriptor_spec=descriptor_spec, include_entropy=True)

    y_train = split_dict["y_train"]
    y_test  = split_dict["y_test"]

    # --- GP ---
    gp_exp = ExperimentalGPApproach(
        random_state=42,
        n_splits=5,
        scoring="r2",
        n_restarts_optimizer=5,
    )

    results = gp_exp.fit_and_evaluate(
        X_train=x_train,
        y_train=y_train,
        X_test=x_test,
        y_test=y_test,
    )

    return {
        "wavelength": wavelength,
        "r2": results["test_metrics"]["r2"]
    }
    
def scan_wavelengths(df, wavelengths, descriptor_spec):

    results = []

    for wl in wavelengths:
        out = evaluate_wavelength(df, wl, descriptor_spec)
        results.append(out)

    return pd.DataFrame(results).sort_values("r2", ascending=False)


descriptor_spec = {
    "atomic_radius": ["avg","mismatch"],
    "metallic_radius": ["avg","mismatch"],
    "electronegativity_pauling": ["avg","mismatch"],
    "electronegativity_allen": ["avg","mismatch"],
    "valence_electrons": ["avg","mismatch"],
    "density": ["avg","mismatch"],
    "electron_affinity": ["avg","mismatch"],
    "ionization_energy_1": ["avg","mismatch"],
    "ionization_energy_2": ["avg","mismatch"],
    "valence_s": ["avg","mismatch"],
    "valence_d": ["avg","mismatch"]
}

wavelengths = sorted(df_merged_all["wavelength_nm"].unique())
results_df = scan_wavelengths(df_merged_all, wavelengths, descriptor_spec)

results_df.to_csv(PREDICTIONS_DIR/"GP_wavelenght_r2.csv")

print(results_df.head())
