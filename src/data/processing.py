import numpy as np
import pandas as pd

def merge_data(df1, df2, on="ID", how="inner"):
    return df1.merge(df2, on=on, how=how)


def filter_by_wavelength(df, wavelength, tol=1.0):
    w = pd.to_numeric(df["wavelength_nm"], errors="coerce")
    
    closest = w.iloc[(w - wavelength).abs().argmin()]
    
    return df[w == closest]