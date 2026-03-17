from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

class ElementPropertyLoader:

    def __init__(self, path_root: str):
        self.path_root = path_root

    def load(self):
        atomic_radius = pd.read_pickle(f"{self.path_root}/Atomic_radius.pkl")
        electronegativity = pd.read_pickle(f"{self.path_root}/electronegativity.pkl")
        valence_electrons = pd.read_pickle(f"{self.path_root}/Valence_electrons.pkl")

        atomic_radius = atomic_radius.set_index("symbol")
        electronegativity = electronegativity.set_index("Symbol")
        valence_electrons = valence_electrons.set_index("symbol")

        atomic_radius = atomic_radius.rename(columns={"Metallic": "atomic_radius"})
        valence_electrons = valence_electrons.rename(columns={"valence": "valence_electrons"})

        elem_props = (
            atomic_radius
            .join(electronegativity)
            .join(valence_electrons)
        )

        return elem_props


@dataclass(frozen=True)
class AlloyDescriptorCalculator:
    elem_props: pd.DataFrame
    R: float = 8.314
    
    def elem_cols(self, df_comp: pd.DataFrame) -> pd.Index:
        return df_comp.columns.intersection(self.elem_props.index)
    
    def _C(self, df_comp: pd.DataFrame, elem_cols: Optional[pd.Index] = None) -> pd.DataFrame:
        if elem_cols is None:
            elem_cols = self.elem_cols(df_comp)
        return df_comp.loc[:, elem_cols].apply(pd.to_numeric, errors="coerce")
    
    def _P(self, elem_cols: pd.Index, prop: str) -> pd.Series:
        if prop not in self.elem_props.columns:
            raise KeyError(f"Property '{prop}' not found in elem_props columns.")
        
        P = pd.to_numeric(self.elem_props.loc[elem_cols, prop], errors="coerce")
        
        # Convert atomic radius from pm -A
        if prop == "atomic_radius":
            P=P/100.0
        return P
    
    def average_property(self, df_comp: pd.DataFrame, prop: str, out_col: str) -> pd.DataFrame:
        """Weighted average: sum_i C_i * P_i."""
        df = df_comp.copy()
        elem_cols = self.elem_cols(df)

        C = self._C(df, elem_cols)
        P = self._P(elem_cols, prop)

        df[out_col] = C.mul(P, axis=1).sum(axis=1)
        return df
    def atomic_radius_mismatch(
        self,
        df_comp: pd.DataFrame,
        r_col: str,
        out_col: str = "del_r",
        r_ave_col: str = "r_ave",
    ) -> pd.DataFrame:
        """delta_r = sqrt( sum_i C_i * (1 - r_i / r_ave)^2 )
        """
        df = df_comp.copy()
        elem_cols = self.elem_cols(df)

        C = self._C(df, elem_cols)
        r = self._P(elem_cols, r_col)

        if r_ave_col not in df.columns:
            df[r_ave_col] = C.mul(r, axis=1).sum(axis=1)

        r_ave = pd.to_numeric(df[r_ave_col], errors="coerce").to_numpy()  # (n_rows,)
        r_vec = r.to_numpy()  # (n_elems,)

        ratio = r_vec[None, :] / r_ave[:, None]
        delta2 = (C.to_numpy() * (1 - ratio) ** 2).sum(axis=1)

        df[out_col] = np.sqrt(delta2)
        return df
    def electronegativity_diff(
        self,
        df_comp: pd.DataFrame,
        en_col: str,
        out_col: str = "delta_EN",
    ) -> pd.DataFrame:
        """delta_EN = sqrt( sum_i C_i * (EN_avg - EN_i)^2 )
        """
        df = df_comp.copy()
        elem_cols = self.elem_cols(df)

        C = self._C(df, elem_cols)
        EN = self._P(elem_cols, en_col)

        EN_avg = C.mul(EN, axis=1).sum(axis=1)  # (n_rows,)
        delta2 = C.mul((EN_avg.values[:, None] - EN.values[None, :]) ** 2, axis=1).sum(axis=1)

        df[out_col] = np.sqrt(delta2)
        return df
    def mixing_entropy(
        self,
        df_comp: pd.DataFrame,
        out_col: str = "S_mix",
        R: Optional[float] = None,
    ) -> pd.DataFrame:
        """
        S_mix = -R * sum_i C_i ln(C_i)
        """
        df = df_comp.copy()
        elem_cols = self.elem_cols(df)

        C = self._C(df, elem_cols)

        R_use = self.R if R is None else float(R)
        C_safe = C.replace(0, np.nan)

        df[out_col] = (-R_use * (C_safe * np.log(C_safe)).sum(axis=1)).fillna(0.0)
        return df
    def add_all_descriptors(self, df_comp: pd.DataFrame) -> pd.DataFrame:
        df = df_comp.copy()

        df = self.average_property(df, prop="atomic_radius", out_col="r")  # avg radius
        df = self.atomic_radius_mismatch(df, r_col="atomic_radius", out_col="del_r", r_ave_col="r")
        df = self.electronegativity_diff(df, en_col="electronegativity", out_col="del_EN")
        df = self.mixing_entropy(df, out_col="S", R=self.R)
        df = self.average_property(df, prop="valence_electrons", out_col="VEC")

        return df
    
