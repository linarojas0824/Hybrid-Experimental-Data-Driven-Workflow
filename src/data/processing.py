import numpy as np

def merge_data(df1, df2, on="ID", how="inner"):
    return df1.merge(df2, on=on, how=how)


def filter_by_wavelength(df, wavelength, tol=1.0):
    return df[
        np.isclose(df["wavelength_nm"].astype(float), wavelength, atol=tol)
    ]