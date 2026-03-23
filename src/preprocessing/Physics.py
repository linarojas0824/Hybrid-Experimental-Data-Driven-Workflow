from dataclasses import dataclass, field
from typing import Sequence, Union, Optional
import numpy as np
import pandas as pd


ArrayLike = Union[Sequence[float], np.ndarray]

@dataclass
class DrudeAlloyModel:

    elem_cols: Sequence[str]
    ne_pure: ArrayLike
    eps_inf: float = 1.0
    m_eff: float = 9.1093837e-31
    wavelength_unit: str = "nm"

    # Physical constants
    eps0: float = field(default=8.8541878128e-12, init=False)   # F/m
    e: float = field(default=1.602176634e-19, init=False)       # C
    c: float = field(default=299792458.0, init=False)           # m/s

    def __post_init__(self):
        self.elem_cols = list(self.elem_cols)
        self.ne_pure = np.asarray(self.ne_pure, dtype=float)

        if len(self.elem_cols) != len(self.ne_pure):
            raise ValueError(
                "elem_cols and ne_pure must have the same length. "
                f"Got {len(self.elem_cols)} and {len(self.ne_pure)}."
            )

    def _convert_wavelength_to_m(self, wvl: Union[float, ArrayLike], unit: Optional[str] = None):
        
        unit = unit or self.wavelength_unit
        wvl = np.asarray(wvl, dtype=float)

        if unit == "nm":
            return wvl * 1e-9
        elif unit == "um":
            return wvl * 1e-6
        elif unit == "m":
            return wvl
        else:
            raise ValueError(f"Unsupported wavelength unit: {unit}. Use 'nm', 'um', or 'm'.")

    def alloy_electron_density(self, comp: ArrayLike) -> float:
        comp = np.asarray(comp, dtype=float)

        if comp.shape[0] != len(self.ne_pure):
            raise ValueError(
                "Composition vector length must match number of elements. "
                f"Got {comp.shape[0]} and {len(self.ne_pure)}."
            )

        return float(np.dot(self.ne_pure, comp))

    def omega_from_wavelength(self, wvl: Union[float, ArrayLike], unit: Optional[str] = None):
        """Convert wavelength to angular frequency ω."""
        wvl_m = self._convert_wavelength_to_m(wvl, unit=unit)
        return 2 * np.pi * self.c / wvl_m

    def drude_from_rho(
        self,
        comp: ArrayLike,
        rho_alloy: float,
        wvl: Union[float, ArrayLike],
        eps_inf: Optional[float] = None,
        m_eff: Optional[float] = None,
        wvl_unit: Optional[str] = None,
    ):

        eps_inf = self.eps_inf if eps_inf is None else eps_inf
        m_eff = self.m_eff if m_eff is None else m_eff

        n_alloy = self.alloy_electron_density(comp)
        omega = self.omega_from_wavelength(wvl, unit=wvl_unit)

        omega_p2 = n_alloy * self.e**2 / (self.eps0 * m_eff)
        tau = m_eff / (n_alloy * self.e**2 * rho_alloy)

        denom = 1.0 + (omega * tau) ** 2
        e1 = eps_inf - (omega_p2 * tau**2) / denom
        e2 = (omega_p2 * tau) / (omega * denom)

        return e1, e2

    def row_to_result(
        self,
        row: pd.Series,
        rho_col: str,
        wvl: Union[float, ArrayLike],
        eps_inf: Optional[float] = None,
        m_eff: Optional[float] = None,
        wvl_unit: Optional[str] = None,
    ):

        comp = row[self.elem_cols].to_numpy(dtype=float)
        rho_alloy = float(row[rho_col])

        return self.drude_from_rho(
            comp=comp,
            rho_alloy=rho_alloy,
            wvl=wvl,
            eps_inf=eps_inf,
            m_eff=m_eff,
            wvl_unit=wvl_unit,
        )

    def predict_dataframe(
        self,
        df: pd.DataFrame,
        rho_col: str,
        wvl: Union[float, ArrayLike],
        eps_inf: Optional[float] = None,
        m_eff: Optional[float] = None,
        wvl_unit: Optional[str] = None,
        e1_col: str = "e1_drude",
        e2_col: str = "e2_drude",
    ) -> pd.DataFrame:

        df_out = df.copy()

        results = df_out.apply(
            lambda row: self.row_to_result(
                row=row,
                rho_col=rho_col,
                wvl=wvl,
                eps_inf=eps_inf,
                m_eff=m_eff,
                wvl_unit=wvl_unit,
            ),
            axis=1,
        )

        df_out[[e1_col, e2_col]] = pd.DataFrame(results.tolist(), index=df_out.index)
        return df_out