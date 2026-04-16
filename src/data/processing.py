import numpy as np
import pandas as pd

def merge_data(df1, df2, on="ID", how="inner"):
    return df1.merge(df2, on=on, how=how)


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
    pred_col="e2_pred",
    std_col="e2_std",
):
    comp_cols = list(composition_cols)

    x_space = calc.transform(
        comp_space[comp_cols],
        descriptor_spec=descriptor_spec,
        include_entropy=include_entropy
    )

    x_model = x_space.drop(columns=comp_cols)

    y_pred, y_std = model.predict(x_model, return_std=True)

    out = x_space[comp_cols].copy()
    out[pred_col] = y_pred
    out[std_col] = y_std

    return out