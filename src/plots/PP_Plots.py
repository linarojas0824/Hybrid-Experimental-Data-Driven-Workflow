from typing import Optional, Tuple, Union,Sequence

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable

import numpy as np
import pandas as pd


class Plotter:
    @staticmethod
    def _cmap_from_rgb255(
        cmap_colors: Tuple[Tuple[int,int,int], Tuple[int,int,int], Tuple[int,int,int]]
    ):
        def rgb01(rgb255):
            return tuple(c / 255 for c in rgb255)

        gray, blue, orange = map(rgb01, cmap_colors)
        return LinearSegmentedColormap.from_list("custom_cmap", [gray, blue, orange], N=256)
    
    @staticmethod
    def plot_spectra_at_times(
        df: pd.DataFrame,
        *,
        times: Optional[Sequence[float]] = None,
        time_range: Optional[Tuple[float, float]] = (0.5, 1.0),
        step: int = 1,
        figsize: Tuple[int, int] = (8, 6),
        xlim: Optional[Tuple[float, float]] = None,
        ylim: Optional[Tuple[float, float]] = None,
        xlabel: str = "Wavelength (nm)",
        ylabel: str = "dT/T",
        legend: bool = True,
        use_cmap: bool = True,
        add_colorbar: bool = True,
        cmap_colors=((128, 128, 128), (0, 0, 255), (255, 128, 0)),
        linewidth: float = 1.6,
        linestyle: str = "-",
        ax=None,
    ):
        x = pd.to_numeric(df.index, errors="coerce").to_numpy(dtype=float)
        cols = pd.to_numeric(df.columns, errors="coerce").to_numpy(dtype=float)
        cols_un = np.unique(cols)

        # --- choose time columns ---
        if times is not None:
            col_range = []
            for t in times:
                idx = int(np.argmin(np.abs(cols_un - float(t))))
                col_range.append(cols_un[idx])
            col_range = np.array(col_range, dtype=float)

        else:
            if time_range is None:
                raise ValueError("Provide a time range")
            lo, hi = map(float, time_range)
            if lo > hi:
                lo, hi = hi, lo
            col_range = cols_un[(cols_un >= lo) & (cols_un <= hi)][::step]

        if len(col_range) == 0:
            raise ValueError("No time columns selected. Check times/time_range/step.")

        if ax is None:
            _, ax = plt.subplots(figsize=figsize)

        # --- colormap ---
        if use_cmap:
            def rgb01(rgb255): return tuple(c/255 for c in rgb255)
            gray, blue, orange = map(rgb01, cmap_colors)
            cmap = LinearSegmentedColormap.from_list("time_cmap", [gray, blue, orange], N=256)
            norm = Normalize(vmin=float(np.min(col_range)), vmax=float(np.max(col_range)))
        else:
            cmap, norm = None, None

        # --- plot ---
        for t in col_range:
            y = df[t]
            if isinstance(y, pd.DataFrame):  # duplicate time columns
                y = y.iloc[:, 0]
            color = cmap(norm(t)) if use_cmap else None
            ax.plot(
                x, y.to_numpy(), 
                color=color, 
                linewidth=linewidth, 
                linestyle=linestyle, 
                label=f"{t:g}")

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        if xlim is not None:
            ax.set_xlim(*xlim)
        
        if ylim is not None:
            ax.set_ylim(*ylim)

        if legend:
            ax.legend(title="Time", fontsize=8)

        if use_cmap and add_colorbar:
            sm = ScalarMappable(norm=norm, cmap=cmap)
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=ax)
            cbar.set_label("Time")

        return ax

    @staticmethod
    def kinetic_plot(
        df: pd.DataFrame,
        wavelengths: Optional[Sequence[float]] = None,
        wavelength_range: Optional[Tuple[float, float]] = None,
        wavelength_step: int = 1,
        figsize: Tuple[int, int] = (8, 6),
        xlim: Optional[Tuple[float, float]] = None,
        ylim: Optional[Tuple[float, float]] = None,
        xlabel: str = "Time (ps)",
        ylabel: str = "dT/T",
        legend: bool = False,
        linewidth: float = 1.6,
        alpha: float = 0.9,
        add_colorbar: bool = True,
        linestyle: str = "-",
        cbar_label: str = "Wavelength (nm)",
        label: Optional[str] = None,
        cmap_colors: Tuple[Tuple[int,int,int], Tuple[int,int,int], Tuple[int,int,int]] = (
            (128, 128, 128),
            (0, 0, 255),
            (255, 128, 0),
        ),
        ax=None,
    ):

        time = pd.to_numeric(df.columns, errors="coerce").to_numpy(dtype=float)
        wl_all = pd.to_numeric(df.index, errors="coerce").to_numpy(dtype=float)

        if np.any(np.isnan(time)):
            raise ValueError("Some time columns could not be converted to float.")
        if np.any(np.isnan(wl_all)):
            raise ValueError("Some wavelength index values could not be converted to float.")

        mask = np.ones_like(wl_all, dtype=bool)

        if wavelength_range is not None:
            lo, hi = float(wavelength_range[0]), float(wavelength_range[1])
            if lo <= hi:
                mask &= (wl_all >= lo) & (wl_all <= hi)
            else:
                mask &= (wl_all >= hi) & (wl_all <= lo)

        if wavelengths is not None:
            wl_req = np.atleast_1d(np.asarray(wavelengths, dtype=float))
            pick = np.zeros_like(wl_all, dtype=bool)
            for w in wl_req:
                idx = int(np.argmin(np.abs(wl_all - w)))
                pick[idx] = True
            mask &= pick

        wl = wl_all[mask][::wavelength_step]
        Y = df.iloc[mask].iloc[::wavelength_step].to_numpy(dtype=float)

        if len(wl) == 0:
            raise ValueError("No wavelengths selected. Check wavelength_range / wavelengths.")

        cmap = Plotter._cmap_from_rgb255(cmap_colors)
        norm = Normalize(vmin=float(np.min(wl)), vmax=float(np.max(wl)))

        if ax is None:
            _, ax = plt.subplots(figsize=figsize)

        for w, y in zip(wl, Y):
            line_label = label if label is not None else f"{w:g} nm"
            ax.plot(
                time, 
                y, 
                color=cmap(norm(w)), 
                linewidth=linewidth, 
                alpha=alpha,
                linestyle=linestyle, 
                label=line_label)

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        if xlim is not None:
            ax.set_xlim(*xlim)
        if ylim is not None:
            ax.set_ylim(*ylim)

        if legend:
            ax.legend(title="Wavelength", fontsize=8)

        if add_colorbar:
            sm = ScalarMappable(norm=norm, cmap=cmap)
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=ax)
            cbar.set_label(cbar_label)

        return ax