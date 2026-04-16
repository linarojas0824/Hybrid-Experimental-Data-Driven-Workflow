from utils.paths import DATABASE_DIR,RAW_DIR,PREDICTIONS_DIR
from models_scripts.gp_workflow import ExperimentalGPApproach

from data.data_loading import load_table
from data.processing import merge_data, filter_by_wavelength
from data.splitting import DataSplitter
from data.preprocessing import prepare_train_test
from data.Descriptors import AlloyDescriptorCalculator
from plots.TernaryPlot import plot_ternary_heatmap

from evaluation.evaluation import compute_permutation_importance, importance_to_df

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ternary
import mpltern

comp_space = pd.read_csv(RAW_DIR/"CuNiAl_descriptors.csv")

df_optical = load_table(DATABASE_DIR / "Ternary_round1.db", "Optical_properties")
df_comp = load_table(DATABASE_DIR / "Ternary_round1.db", "compositions")
df_resis_ANN = load_table(DATABASE_DIR / "Ternary_round1.db", "ANN_resistivity")

# --- Merge full data ---
df_merged = merge_data(df_optical, df_comp)

# -------------------------------
# Extract pure compositions
# -------------------------------
pure_ids = [29, 30, 31]

df_pure = df_merged[df_merged["ID"].isin(pure_ids)].copy()
df_no_pure = df_merged[~df_merged["ID"].isin(pure_ids)].copy()

# -------------------------------
# Main dataframe for splitting
# -------------------------------
tern_1550 = filter_by_wavelength(df_no_pure, 1550).copy()
tern_1550 = tern_1550.reset_index(drop=True)
tern_1550 = merge_data(tern_1550, df_resis_ANN[["ID", "resistivity"]])

X = tern_1550[["Cu", "Ni", "Al", "resistivity"]].copy()
y = tern_1550["e2"].copy()

splitter = DataSplitter(
    test_size=0.20,
    random_state=42,
    split_method="cluster",
    n_clusters=8
)

split_dict = splitter.split(X, y)

# -------------------------------
# Add pure metals back to training
# -------------------------------
pure_1550 = filter_by_wavelength(df_pure, 1552.0).copy()
pure_1550 = merge_data(pure_1550, df_resis_ANN[["ID", "resistivity"]])

X_pure = pure_1550[["Cu", "Ni", "Al", "resistivity"]].copy()
y_pure = pure_1550["e2"].copy()

split_dict["X_train"] = pd.concat([split_dict["X_train"], X_pure], ignore_index=True)
split_dict["y_train"] = pd.concat([split_dict["y_train"], y_pure], ignore_index=True)

train_comp_cluster = np.asarray(split_dict['X_train'][['Cu','Ni','Al']], dtype=float)*100
test_comp_cluster = np.asarray(split_dict['X_test'][['Cu','Ni','Al']], dtype=float)*100

scale = 100
fig, tax = ternary.figure(scale=scale)
fig.set_size_inches(8, 7)

def to_ternary_points(arr_pct, scale=100):

    pts = (arr_pct / 100.0 * scale)
    return [tuple(p) for p in pts]

pts_A = to_ternary_points(train_comp_cluster, scale)
pts_B = to_ternary_points(test_comp_cluster, scale)

tax.scatter(pts_A, marker="o", s=60, color="blue", label="Train")
tax.scatter(pts_B, marker="o", s=70, color="orange", label="Test")


tax.boundary(linewidth=2)
tax.gridlines(color="gray", multiple=5)
tax.ticks(axis="lbr", multiple=20, linewidth=1, fontsize=16)

tax.left_axis_label("Al",fontsize=16,offset=0.1)
tax.right_axis_label("Ni",fontsize=16,offset=0.1)
tax.bottom_axis_label("Cu",fontsize=16,offset=0.05)

tax.clear_matplotlib_ticks()
tax.get_axes().axis("off")
tax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.05), frameon=False)

calc = AlloyDescriptorCalculator()
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

x_train = calc.transform(split_dict['X_train'], descriptor_spec=descriptor_spec, include_entropy=True)
x_test = calc.transform(split_dict['X_test'], descriptor_spec=descriptor_spec, include_entropy=True)

y_train = split_dict['y_train']
y_test = split_dict['y_test']

# Property calculator
calc = AlloyDescriptorCalculator()
descriptor_spec = {
    "metallic_radius": ["avg","mismatch"],
    "valence_electrons": ["avg"],
    "density": ["avg"],
    "ionization_energy_2": ["avg","mismatch"],
    "valence_s": ["mismatch"],
    "valence_d": ["mismatch"]
}

x_train = calc.transform(split_dict['X_train'], descriptor_spec=descriptor_spec, include_entropy=True)
x_test = calc.transform(split_dict['X_test'], descriptor_spec=descriptor_spec, include_entropy=True)

# Optical properties
y_train = split_dict['y_train']
y_test = split_dict['y_test']

x_train = x_train.drop(columns=['Cu','Ni','Al','resistivity'])
x_test = x_test.drop(columns=['Cu','Ni','Al','resistivity'])

import pickle

gpy_dic = {
    "X_train": x_train,
    "X_test": x_test,
    "y_train": y_train,
    "y_test": y_test
}

with open("gpy_dic.pkl", "wb") as f:
    pickle.dump(gpy_dic, f)
    

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

model = gp_exp.model_

print("Best CV score:", results["best_score_cv"])
print("Best params:")
print(results["best_params"])
print("Train metrics:", results["train_metrics"])
print("Test metrics:", results["test_metrics"])


x_space = calc.transform(comp_space[['Cu','Ni','Al']], descriptor_spec=descriptor_spec, include_entropy=True)
x_space_pred = x_space.drop(columns=['Cu','Ni','Al'])

e2_predict, predict_std= model.predict(x_space_pred, return_std=True)

x_space_pred = x_space[['Cu','Ni','Al']].copy()
x_space_pred['e2'] = e2_predict
x_space_pred['std'] = predict_std

plot_ternary_heatmap(
    df=x_space_pred,
    value_col="e2",
    duplicate_handling="mean"
)

plot_ternary_heatmap(
    df=x_space_pred,
    value_col="std",
    duplicate_handling="mean"
)

feature_names = x_test.columns

imp = compute_permutation_importance(model, x_test, y_test)
df_imp = importance_to_df(imp, feature_names)


plt.figure(figsize=(6,8))
plt.barh(df_imp["feature"], df_imp["importance"])
plt.gca().invert_yaxis()
plt.xlabel("Importance")
plt.tight_layout()
plt.show()


df_wave = pd.read_csv(PREDICTIONS_DIR/"GP-Experiments/GP_wavelength_r2.csv")

import matplotlib.pyplot as plt

df_wave = df_wave.sort_values('wavelength')
plt.plot(df_wave['wavelength'], df_wave['r2'])
plt.xlabel('wavelength')
plt.ylabel('r2')
plt.show()


import pickle

with open("gpy_dic.pkl", "rb") as f:
    data = pickle.load(f)

x_train = data["X_train"]
x_test  = data["X_test"]
y_train = data["y_train"]
y_test  = data["y_test"]


from models_scripts.gp_GPy import GPyMeanModel

model = GPyMeanModel()

model.fit(x_train, y_train)

metrics = model.evaluate(x_test, y_test)
print(metrics)

y_pred, y_std = model.predict(x_test)