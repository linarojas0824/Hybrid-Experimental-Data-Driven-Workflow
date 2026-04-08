from pathlib import Path
import time
import pandas as pd
import numpy as np

from src.utils.paths import DATABASE_DIR, PREDICTIONS_DIR
from src.preprocessing.gp_workflow import ExperimentalGPApproach
from src.preprocessing.data_loading import load_table
from src.preprocessing.processing import merge_data
from src.preprocessing.splitting import DataSplitter
from src.preprocessing.Descriptors import AlloyDescriptorCalculator


# -------------------- LOAD DATA -------------------- #
df_optical = load_table(DATABASE_DIR / "Ternary_round1.db", "Optical_properties")
df_comp = load_table(DATABASE_DIR / "Ternary_round1.db", "compositions")
df_resis_ANN = load_table(DATABASE_DIR / "Ternary_round1.db", "ANN_resistivity")

df_merged = merge_data(df_optical, df_comp)
df_merged_all = merge_data(df_merged, df_resis_ANN[["ID", "resistivity"]])


# -------------------- CONFIG -------------------- #
descriptor_spec = {
    "atomic_radius": ["avg", "mismatch"],
    "metallic_radius": ["avg", "mismatch"],
    "electronegativity_pauling": ["avg", "mismatch"],
    "electronegativity_allen": ["avg", "mismatch"],
    "valence_electrons": ["avg", "mismatch"],
    "density": ["avg", "mismatch"],
    "electron_affinity": ["avg", "mismatch"],
    "ionization_energy_1": ["avg", "mismatch"],
    "ionization_energy_2": ["avg", "mismatch"],
    "valence_s": ["avg", "mismatch"],
    "valence_d": ["avg", "mismatch"]
}

wavelengths = sorted(df_merged_all["wavelength_nm"].unique())

SAVE_PATH = PREDICTIONS_DIR / "GP_wavelength_r2_partial.csv"
PROGRESS_PATH = PREDICTIONS_DIR / "GP_wavelength_progress.txt"


# -------------------- FUNCTIONS -------------------- #
def write_progress(progress_path, wavelength):
    with open(progress_path, "w") as f:
        f.write(f"{wavelength}\n")


def evaluate_wavelength(df, wavelength, descriptor_spec):
    t0 = time.time()
    print(f"Starting {wavelength}", flush=True)

    df_wl = df[df["wavelength_nm"] == wavelength].reset_index(drop=True)

    X = df_wl[["Cu", "Ni", "Al", "resistivity"]]
    y = df_wl["e2"]

    splitter = DataSplitter(
        test_size=0.2,
        random_state=42,
        split_method="cluster",
        n_clusters=5
    )
    split_dict = splitter.split(X, y)

    calc = AlloyDescriptorCalculator()

    x_train = calc.transform(split_dict["X_train"], descriptor_spec=descriptor_spec, include_entropy=True)
    x_test  = calc.transform(split_dict["X_test"],  descriptor_spec=descriptor_spec, include_entropy=True)

    y_train = split_dict["y_train"]
    y_test  = split_dict["y_test"]

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

    print(f"{wavelength}: done in {time.time()-t0:.2f}s", flush=True)

    return {
        "wavelength": wavelength,
        "r2": results["test_metrics"]["r2"]
    }


def scan_wavelengths(df, wavelengths, descriptor_spec, save_path, progress_path):
    save_path = Path(save_path)

    if save_path.exists():
        existing = pd.read_csv(save_path)
        done_wls = set(existing["wavelength"])
    else:
        existing = pd.DataFrame(columns=["wavelength", "r2"])
        done_wls = set()

    for wl in wavelengths:
        if wl in done_wls:
            print(f"Skipping {wl}", flush=True)
            continue

        write_progress(progress_path, wl)

        out = evaluate_wavelength(df, wl, descriptor_spec)

        existing = pd.concat([existing, pd.DataFrame([out])], ignore_index=True)

        # save after each wavelength
        existing.to_csv(save_path, index=False)

    return existing.sort_values("r2", ascending=False)


# -------------------- RUN -------------------- #
results_df = scan_wavelengths(
    df_merged_all,
    wavelengths,
    descriptor_spec,
    SAVE_PATH,
    PROGRESS_PATH
)

results_df.to_csv(PREDICTIONS_DIR / "GP_wavelength_r2.csv", index=False)

print(results_df.head(), flush=True)
