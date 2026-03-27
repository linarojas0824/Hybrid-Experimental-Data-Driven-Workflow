from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal, Optional

import numpy as np
import pandas as pd

from scipy.signal import medfilt, savgol_filter
from scipy.ndimage import gaussian_filter1d


SmoothMethod = Literal["savgol", "gaussian"]
AxisMode = Literal["time", "wavelength"]


@dataclass
class SignalProcessor:
    """
    Signal-processing utilities for 1D signals and 2D pump–probe matrices.
    """

    # Default despike kernel (median filter). Set to None to disable despiking.
    despike_kernel: Optional[int] = 5

    # Defaults for smoothers
    savgol_window: int = 31
    savgol_poly: int = 3
    savgol_mode: str = "interp"

    gaussian_sigma: float = 3.0

    # -----------------------------
    # 1) Despike (median filter)
    # -----------------------------
    def despike_median_1d(self, y: np.ndarray, kernel: Optional[int] = None) -> np.ndarray:
        y = np.asarray(y, dtype=float)

        k = self.despike_kernel if kernel is None else kernel
        if k is None or k < 3:
            return y

        k = int(k)
        if k % 2 == 0:
            k += 1

        # medfilt pads internally; safe for 1D arrays
        return medfilt(y, kernel_size=k)

    # -----------------------------
    # 2) Window for Savitzky-Golay
    # -----------------------------
    @staticmethod
    def _make_valid_sg_window(n: int, window: int, poly: int) -> int:
        if n <= 0:
            raise ValueError("Signal length n must be > 0.")

        win = int(window)

        # Must be odd
        if win % 2 == 0:
            win += 1

        # Window cannot exceed n
        max_odd_n = n if n % 2 == 1 else n - 1
        win = min(win, max_odd_n)

        # Must be at least poly + 2 and odd
        min_win = poly + 2
        if win < min_win:
            win = min_win
        if win % 2 == 0:
            win += 1

        # Re-check upper bound
        if win > max_odd_n:
            win = max_odd_n

        if win <= poly:
            raise ValueError(
                f"Invalid Savitzky–Golay settings: window={win}, poly={poly}, n={n}. "
                "Need window > poly and window <= n."
            )

        return win

    # ---------------------------------
    # 3) Smooth 1D (single entry point)
    # ---------------------------------
    def smooth_1d(
        self,
        y: np.ndarray,
        method: SmoothMethod = "savgol",
        *,
        savgol_window: Optional[int] = None,
        savgol_poly: Optional[int] = None,
        savgol_mode: Optional[str] = None,
        gaussian_sigma: Optional[float] = None,
        despike_kernel: Optional[int] = None,
    ) -> np.ndarray:
        """
        Smooth a 1D signal with optional despiking.

        Parameters
        ----------
        y : array-like
            Input signal.
        method : {"savgol", "gaussian"}
        
        Returns
        -------
        np.ndarray
            Smoothed signal.
        """
        y = np.asarray(y, dtype=float)
        y = self.despike_median_1d(y, kernel=despike_kernel)

        if method == "savgol":
            win = savgol_window if savgol_window is not None else self.savgol_window
            poly = savgol_poly if savgol_poly is not None else self.savgol_poly
            mode = savgol_mode if savgol_mode is not None else self.savgol_mode

            n = len(y)
            win = self._make_valid_sg_window(n, win, poly)
            return savgol_filter(y, window_length=win, polyorder=poly, mode=mode)

        if method == "gaussian":
            sigma = gaussian_sigma if gaussian_sigma is not None else self.gaussian_sigma
            return gaussian_filter1d(y, sigma=float(sigma))

        raise ValueError(f"Unknown method: {method!r}. Use 'savgol' or 'gaussian'.")

    # ----------------------------------
    # 4) Smooth DataFrame along an axis
    # ----------------------------------
    def denoise_df(
        self,
        df: pd.DataFrame,
        *,
        axis_mode: AxisMode,
        method: SmoothMethod = "savgol",
        # Overrides for this call:
        savgol_window: Optional[int] = None,
        savgol_poly: Optional[int] = None,
        savgol_mode: Optional[str] = None,
        gaussian_sigma: Optional[float] = None,
        despike_kernel: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Denoise a 2D pump–probe matrix stored in a DataFrame.
        Returns a DataFrame with the same shape/index/columns.
        """
        df_num = df.apply(pd.to_numeric, errors="coerce")

        if axis_mode == "time":
            # smooth each row; keep columns
            out = df_num.apply(
                lambda r: pd.Series(
                    self.smooth_1d(
                        r.to_numpy(),
                        method=method,
                        savgol_window=savgol_window,
                        savgol_poly=savgol_poly,
                        savgol_mode=savgol_mode,
                        gaussian_sigma=gaussian_sigma,
                        despike_kernel=despike_kernel,
                    ),
                    index=df_num.columns,
                ),
                axis=1,
            )
            out.index = df.index
            return out

        if axis_mode == "wavelength":
            # smooth each column; keep index
            out = df_num.apply(
                lambda c: pd.Series(
                    self.smooth_1d(
                        c.to_numpy(),
                        method=method,
                        savgol_window=savgol_window,
                        savgol_poly=savgol_poly,
                        savgol_mode=savgol_mode,
                        gaussian_sigma=gaussian_sigma,
                        despike_kernel=despike_kernel,
                    ),
                    index=df_num.index,
                ),
                axis=0,
            )
            out.columns = df.columns
            return out

        raise ValueError("axis_mode must be 'time' or 'wavelength'.")

    # -----------------------------
    # Convenience wrappers
    # -----------------------------
    def denoise_df_time(self, df: pd.DataFrame, method: SmoothMethod = "savgol", **kwargs) -> pd.DataFrame:
        return self.denoise_df(df, axis_mode="time", method=method, **kwargs)

    def denoise_df_by_wvl(self, df: pd.DataFrame, method: SmoothMethod = "savgol", **kwargs) -> pd.DataFrame:
        return self.denoise_df(df, axis_mode="wavelength", method=method, **kwargs)