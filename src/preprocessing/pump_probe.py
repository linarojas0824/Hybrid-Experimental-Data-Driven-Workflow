from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Tuple, List, Union,Sequence

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize, LogNorm

from scipy.signal import medfilt, savgol_filter
from scipy.optimize import least_squares

# -------- Data loading + cleaning ------------------------- #

@dataclass
class DataCollector:
    index_col: str = "0.00000E+0"
    
    def load_csv(self,
        path: str,
        round_cols: Optional[int] = 1,
        round_index: Optional[int] = 1,
        header: Union[int, None] = 0,
        low:Optional[float] = 0,
        high:Optional[float] = 39
    ) -> pd.DataFrame:
        df = pd.read_csv(path, header=header)

        # convert everything to numeric when possible
        df = df.apply(pd.to_numeric, errors="coerce")
        
        if self.index_col not in df.columns:
            raise KeyError(
                f"Index column '{self.index_col}' not found. Available: {list(df.columns)[:10]} ..."
        )
            
        df = df.set_index(self.index_col)
        df.index = pd.to_numeric(df.index, errors="coerce").round(round_index)
        
        colnums = pd.to_numeric(df.columns, errors="coerce").round(round_cols)
        mask = (colnums >= low) & (colnums <= high)
        df = df.loc[:, mask]
        
        df.columns = colnums[mask]
        
        df = df.loc[:, ~df.columns.duplicated()]
        
        df = df.dropna()

        return df

# -------- Preprocesing ------------------------- #

@dataclass
class SignalProcessor:
    # Despike params
    k: int = 9
    thr: float = 15

    # Smoothing params
    window: int = 51
    poly: int = 2

    @staticmethod
    def despike_and_smooth(
        y: Union[np.ndarray, pd.Series],
        k: int = 9,
        thr: float = 15,
        window: int = 51,
        poly: int = 2,
    ) -> np.ndarray:
        y = np.asarray(y, dtype=float)

        if y.size == 0:
            return y
        if np.all(~np.isfinite(y)):
            return y

        if np.any(~np.isfinite(y)):
            s = pd.Series(y)
            y = s.interpolate(limit_direction="both").to_numpy(dtype=float)

        n = len(y)
        if n < 3:
  
            return y.copy()

        # --- despike ---
        k = int(k)
        if k % 2 == 0:
            k += 1
        k = max(k, 3)
        k = min(k, n if n % 2 == 1 else n - 1)  
        if k < 3:
            y2 = y.copy()
        else:
            y_med = medfilt(y, kernel_size=k)
            resid = y - y_med
            mad = np.median(np.abs(resid - np.median(resid)))

            if mad is None or mad < 1e-6:
                y2 = y.copy()
            else:
                z = 0.6745 * resid / mad
                y2 = y.copy()
                mask = np.abs(z) > thr
                y2[mask] = y_med[mask]

        # --- smooth ---
        win = int(window)
        if win % 2 == 0:
            win += 1
        win = max(win, 3)
        win = min(win, n if n % 2 == 1 else n - 1) 
        if win < 3:
            return y2

        poly = int(poly)
        poly = min(max(poly, 0), win - 1)

        return savgol_filter(y2, window_length=win, polyorder=poly)

    def denoise_df_time(self, df: pd.DataFrame) -> pd.DataFrame:
        # ensure numeric only
        df_num = df.apply(pd.to_numeric, errors="coerce")

        out = df_num.apply(
            lambda r: pd.Series(
                self.despike_and_smooth(r.to_numpy(), k=5, thr=15, window=11, poly=2),
                index=df_num.columns,
            ),
            axis=1,
        )
        out.index = df.index 
        return out

    def denoise_df_by_wvl(self, df: pd.DataFrame) -> pd.DataFrame:
        df_num = df.apply(pd.to_numeric, errors="coerce")

        out = df_num.apply(
            lambda c: pd.Series(
                self.despike_and_smooth(c.to_numpy(), k=9, thr=15, window=51, poly=2),
                index=df_num.index,
            ),
            axis=0,
        )
        out.columns = df.columns
        return out


# -------- Plotting ------------------------- #


class Plotter:
    @staticmethod
    def plot_columns_in_range(
        df: pd.DataFrame,
        low_time: float = 0.5, 
        high_time: float = 2.0,
        step: int = 1,
        figsize: Tuple[int, int] = (8, 8),
        xlim: Optional[Tuple[float, float]] = None,
        xlabel: str = "Wavelength",
        ylabel: str = "dT/T",
        legend: bool = True,
        ax=None,
        
        # Time range
        times: Optional[Sequence[float]] = None,
        time_ranges: Optional[Sequence[Tuple[float, float]]] = None,
        
        # Colormap controls
        use_time_cmap: bool = True,
        add_colorbar: bool = True,
        cmap_colors: Tuple[Tuple[int,int,int], Tuple[int,int,int], Tuple[int,int,int]] = (
            (128,128,128),  # gray
            (0,0,255),      # blue
            (255,128,0)     # orange
        ),
        linewidth: float = 1.8,
    ) -> None:
        x = df.index.to_numpy()

        cols = df.columns.to_numpy(dtype=float)
        cols_un = np.unique(cols)
        
        #----- Time plot -------
        if times is not None:
            cols_pick = []
            for t in times:
                idx = int(np.argmin(np.abs(cols_un-float(t))))
                cols_pick.append(cols_un[idx])
                col_range = np.array(cols_pick, dtype=float)
        elif time_ranges is not None:
            # Include all the columns fall into the ranges
            mask = np.zeros_like(cols_un, dtype=bool)
            for lo, hi in time_ranges:
                lo,hi = float(lo), float(hi)
                if lo <= hi:
                    mask |= (cols_un >= lo) & (cols_un <= hi)
                else:
                    mask |= (cols_un >= hi) & (cols_un <= lo)
            col_range = cols_un[mask]
            col_range = col_range[::step]
            
        else:
            col_range = cols_un[(cols_un >= low_time) & (cols_un <= high_time)][::step]
        
        if ax is None:
            fig,ax = plt.subplots(figsize=figsize)
        
        if use_time_cmap:
            def rgb01(rgb255):
                return tuple(c/255 for c in rgb255)
            
            gray,blue,orange = map(rgb01,cmap_colors)
            custom_cmap = LinearSegmentedColormap.from_list("time_cmap", [gray, blue, orange], N=256)
            norm = Normalize(vmin=np.min(col_range), vmax=np.max(col_range))
        else:
            custom_cmap = None
            norm = None

    # ------ Plot --------- #
    
        for t in col_range:
            y = df[t]
            if isinstance(y, pd.DataFrame): ## If there are duplicated rows
                y = y.iloc[:,0]
            color = custom_cmap(norm(t)) if use_time_cmap else None
            
            ax.plot(
                x,y.to_numpy(),
                color=color,
                linewidth=linewidth,
                label = f"{t:g}"
            )
        
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        if xlim is not None:
            ax.set_xlim(*xlim)
        
        if legend:
            ax.legend(title='Time', fontsize=8)
        
        if use_time_cmap and add_colorbar:
            sm  = plt.cm.ScalarMappable(cmap=custom_cmap, norm=norm)
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=ax)
            cbar.set_label("Time")

# -------- Lifetime Fitting ------------------------- #

@dataclass
class LifetimeFitter:
    n_samples: int = 5000
    k_best: int = 20
    seed: int = 0
    loss: str = "linear"  # or "soft_l1"
    scale_residuals: bool = True
    PARAM_NAMES = ["dt_t_nt", "tau_th", "tau_p", "dt_t_th", "dt_t_l"]

    @staticmethod
    def model(time, dt_t_nt, tau_th, tau_p, dt_t_th, dt_t_l):
        time = np.asarray(time, dtype=float)
        return (
            dt_t_nt * np.exp(-time * ((1 / tau_th) + (1 / tau_p)))
            + dt_t_th * (np.exp(-time / tau_p) * (1 - np.exp(-time / tau_th)))
            + dt_t_l
        )

    def objective(self, p, time, y_exp):
        r = self.model(time, *p) - y_exp
        return np.sum(r * r)

    def bounds_from_data(self, time: np.ndarray, y_exp: np.ndarray):
        y_scale = np.std(y_exp) if np.std(y_exp) > 0 else 1.0
        tpos = time[time > 0]
        if tpos.size == 0:
            raise ValueError("time must contain positive values to define tau bounds.")
        tmin_pos = np.min(tpos)
        tmax = np.max(time)

        return [
            (-10 * y_scale, 10 * y_scale),        # dt_t_nt
            (tmin_pos * 0.1, tmax * 10),          # tau_th
            (tmin_pos * 0.1, tmax * 10),          # tau_p
            (-10 * y_scale, 10 * y_scale),        # dt_t_th
            (np.min(y_exp), np.max(y_exp)),       # dt_t_l
        ]

    def find_initial_guesses(self, bounds, time, y_exp) -> List[np.ndarray]:
        rng = np.random.default_rng(self.seed)
        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)

        candidates = []
        for _ in range(self.n_samples):
            p = lo + (hi - lo) * rng.random(len(bounds))
            val = self.objective(p, time, y_exp)
            candidates.append((val, p))

        candidates.sort(key=lambda x: x[0])
        return [p for (_, p) in candidates[: self.k_best]]

    def fit(self, time: np.ndarray, y_exp: np.ndarray, bounds=None) -> np.ndarray:
        time = np.asarray(time, dtype=float)
        y_exp = np.asarray(y_exp, dtype=float)

        if bounds is None:
            bounds = self.bounds_from_data(time, y_exp)

        lb = np.array([b[0] for b in bounds], dtype=float)
        ub = np.array([b[1] for b in bounds], dtype=float)

        # initial candidates
        p0_list = self.find_initial_guesses(bounds, time, y_exp)

        # scaling (optional)
        scale = np.std(y_exp) if self.scale_residuals else 1.0
        scale = scale if scale > 0 else 1.0

        best_res = None
        best_cost = np.inf

        for p0 in p0_list:
            res = least_squares(
                fun=lambda p: (self.model(time, *p) - y_exp) / scale,
                x0=p0,
                bounds=(lb, ub),
                loss=self.loss,
            )
            if res.cost < best_cost:
                best_cost = res.cost
                best_res = res

        if best_res is None:
            raise RuntimeError("Fitting failed to produce a solution.")

        p_opt = best_res.x
        y_fit = self.model(time, *p_opt)

        residual = y_fit - y_exp
        sq_error = residual**2
        abs_error = np.abs(residual)

        mse = float(np.mean(sq_error))
        rmse = float(np.sqrt(mse))
        mae = float(np.mean(abs_error))
        sse = float(np.sum(sq_error))

        # optional R^2
        ss_tot = float(np.sum((y_exp - np.mean(y_exp))**2))
        r2 = float(1.0 - sse / ss_tot) if ss_tot > 0 else np.nan

        # named params (easy to print/save)
        params = dict(zip(self.PARAM_NAMES, p_opt))

        # return a bundle you can save + plot later
        return {
            "params": params,                   
            "y_fit": y_fit,
            "residual": residual,
            "sq_error": sq_error,
            "abs_error": abs_error,
            "mse": mse,
            "rmse": rmse,
            "mae": mae,
            "sse": sse,
            "r2": r2,

        }