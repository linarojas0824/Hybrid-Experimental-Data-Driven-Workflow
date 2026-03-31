from dataclasses import dataclass, field
from typing import Optional
import numpy as np
import pandas as pd
from mendeleev import element


@dataclass
class AlloyDescriptorCalculator:
    R: float = 8.314
    property_cache: dict = field(default_factory=dict)

    def elem_cols(self, df_comp: pd.DataFrame) -> pd.Index:
        elem_cols = []

        for col in df_comp.columns:
            try:
                element(col)
                elem_cols.append(col)
            except Exception:
                pass

        return pd.Index(elem_cols)

    def _C(
        self,
        df_comp: pd.DataFrame,
        elem_cols: Optional[pd.Index] = None
    ) -> pd.DataFrame:
        if elem_cols is None:
            elem_cols = self.elem_cols(df_comp)
        return df_comp.loc[:, elem_cols].apply(pd.to_numeric, errors="coerce")

    def _get_element_property(self, elem_symbol: str, prop: str):
        key = (elem_symbol, prop)

        if key not in self.property_cache:
            el = element(elem_symbol)

            alias_map = {
                "valence_electrons": "nvalence",
            }

        # ---- special cases FIRST ----
            if prop == "ionization_energy_1":
                value = el.ionenergies.get(1)

            elif prop == "ionization_energy_2":
                value = el.ionenergies.get(2)

            else:
                real_prop = alias_map.get(prop, prop)

                try:
                    value = getattr(el, real_prop)
                except AttributeError:
                    raise KeyError(
                        f"Property '{prop}' is not available for element '{elem_symbol}'."
                    )

                if callable(value):
                    value = value()

            self.property_cache[key] = value

        return self.property_cache[key]

    def _P(self, elem_cols: pd.Index, prop: str) -> pd.Series:
        values = {}
        for elem_symbol in elem_cols:
            values[elem_symbol] = self._get_element_property(elem_symbol, prop)

        P = pd.Series(values, index=elem_cols, dtype=float)

        if prop == "atomic_radius":
            P = P / 100.0

        return P

    def average_property(
        self,
        df_comp: pd.DataFrame,
        prop: str,
        out_col: Optional[str] = None
    ) -> pd.DataFrame:
        df = df_comp.copy()
        elem_cols = self.elem_cols(df)
        C = self._C(df, elem_cols)
        P = self._P(elem_cols, prop)

        if out_col is None:
            out_col = f"{''.join([w[0] for w in prop.split('_')])}_avg"

        df[out_col] = C.mul(P, axis=1).sum(axis=1)
        return df

    def mismatch_property(
        self,
        df_comp: pd.DataFrame,
        prop: str,
        out_col: Optional[str] = None
    ) -> pd.DataFrame:
        df = df_comp.copy()
        elem_cols = self.elem_cols(df)
        C = self._C(df, elem_cols)
        P = self._P(elem_cols, prop)

        P_avg = C.mul(P, axis=1).sum(axis=1)
        delta2 = C.mul((P_avg.values[:, None] - P.values[None, :]) ** 2, axis=1).sum(axis=1)

        if out_col is None:
            out_col = f"del_{''.join([w[0] for w in prop.split('_')])}"

        df[out_col] = np.sqrt(delta2)
        return df

    def range_property(
        self,
        df_comp: pd.DataFrame,
        prop: str,
        out_col: Optional[str] = None
    ) -> pd.DataFrame:
        df = df_comp.copy()
        elem_cols = self.elem_cols(df)
        C = self._C(df, elem_cols)
        P = self._P(elem_cols, prop)

        C_np = C.to_numpy()
        P_np = P.to_numpy()

        present_mask = C_np > 0
        p_max = np.where(present_mask, P_np[None, :], -np.inf).max(axis=1)
        p_min = np.where(present_mask, P_np[None, :], np.inf).min(axis=1)

        if out_col is None:
            out_col = f"min_max{''.join([w[0] for w in prop.split('_')])}"

        df[out_col] = p_max - p_min
        return df

    def mixing_entropy(
        self,
        df_comp: pd.DataFrame,
        out_col: str = "S"
    ) -> pd.DataFrame:
        df = df_comp.copy()
        elem_cols = self.elem_cols(df)
        C = self._C(df, elem_cols)

        C_safe = C.replace(0, np.nan)
        df[out_col] = (-self.R * (C_safe * np.log(C_safe)).sum(axis=1)).fillna(0.0)
        return df

    def transform(
        self,
        df_comp: pd.DataFrame,
        descriptor_spec: Optional[dict] = None,
        include_entropy: bool = True
    ) -> pd.DataFrame:
        if descriptor_spec is None:
            descriptor_spec = {
                "atomic_radius": ["avg", "mismatch"],
                "en_pauling": ["mismatch"],
                "valence_electrons": ["avg"]
            }

        df_out = df_comp.copy()

        for prop, modes in descriptor_spec.items():
            for mode in modes:
                if mode == "avg":
                    df_out = self.average_property(df_out, prop)
                elif mode == "mismatch":
                    df_out = self.mismatch_property(df_out, prop)
                elif mode == "range":
                    df_out = self.range_property(df_out, prop)
                else:
                    raise ValueError(f"Unknown mode: {mode}")

        if include_entropy:
            df_out = self.mixing_entropy(df_out)

        return df_out