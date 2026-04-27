import numpy as np
import pandas as pd

def merge_data(dfs, on="ID", how="inner"):
    result = dfs[0]
    for df in dfs[1:]:
        result = result.merge(df, on=on, how=how)
    return result


def filter_by_wavelength(df, wavelength, tol=1.0):
    w = pd.to_numeric(df["wavelength_nm"], errors="coerce")
    
    closest = w.iloc[(w - wavelength).abs().argmin()]
    
    return df[w == closest]

#========= Space predictions =============================

def predict_composition_space(
    comp_space,
    calc,
    descriptor_spec,
    model,
    include_entropy=True,
    composition_cols=("Cu", "Ni", "Al"),
    include_composition=False,
    include_resistivity=False,
    resistivity_col="resistivity",
    pred_col="e2_pred",
    std_col="e2_std",
):
    comp_cols = list(composition_cols)

    x_space = calc.transform(
        comp_space[comp_cols],
        descriptor_spec=descriptor_spec,
        include_entropy=include_entropy
    )

    # Start from descriptors only
    x_model = x_space.drop(columns=comp_cols, errors="ignore")

    # Optionally add composition columns
    if include_composition:
        x_model = pd.concat([comp_space[comp_cols], x_model], axis=1)

    # Optionally add resistivity column
    if include_resistivity:
        if resistivity_col not in comp_space.columns:
            raise ValueError(
                f"Column '{resistivity_col}' is required, but it is not in comp_space."
            )
        x_model[resistivity_col] = comp_space[resistivity_col]

    # Match model training columns if available
    if hasattr(model, "feature_names_in_"):
        x_model = x_model[list(model.feature_names_in_)]

    y_pred, y_std = model.predict(x_model, return_std=True)

    out = x_space[comp_cols].copy()
    out[pred_col] = y_pred
    out[std_col] = y_std

    return out