from typing import Optional, Tuple, Union,Sequence

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable

import numpy as np
import pandas as pd
import math


class Plotter:
    @staticmethod
    def _cmap_from_rgb255(
        cmap_colors: Tuple[Tuple[int,int,int], Tuple[int,int,int], Tuple[int,int,int]]
    ):
        def rgb01(rgb255):
            return tuple(c / 255 for c in rgb255)

        gray, blue, orange = map(rgb01, cmap_colors)
        return LinearSegmentedColormap.from_list("custom_cmap", [gray, blue, orange], N=256)
    
    
    #============ Spectra plot at time (color bar) ==========================#
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

    #============ Kinetic plot at wavelengths (color bar) ==========================#
    
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
    
    
    #============ Lifetime fit (TTM) ==========================#
    @staticmethod
    def plot_lifetime_fit(
        lifetime: dict,
        time,
        y_exp,
        time_range=None,
        ylim=None,
        figsize=(6, 4),
        ax=None,
        show_r2=True,
        param_name=None,
        legend=True
    ):
        """
        Plot experimental data vs fitted curve.

        Parameters
        ----------
        lifetime : dict
            Fit result dictionary. Must contain 'y_fit'. May contain 'r2' and 'params'.
        time : array-like
            Time axis.
        y_exp : array-like
            Experimental signal.
        time_range : tuple or None
            Time range to display.
        ylim : tuple or None
            Y-axis limits.
        figsize : tuple
            Figure size if ax is None.
        ax : matplotlib axis 

        Returns
        -------
        ax
        """
        fit_out = lifetime
        y_fit = np.asarray(fit_out["y_fit"], dtype=float)
        r2 = fit_out.get("r2", None)
        params = fit_out.get("params", {})

        time = np.asarray(time, dtype=float)
        y_exp = np.asarray(y_exp, dtype=float)

        created_fig = False
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
            created_fig = True

        # Time mask
        if time_range is not None:
            tmin, tmax = time_range
            mask = (time >= tmin) & (time <= tmax)
        else:
            mask = slice(None)

        # Build fit label
        label_parts = ["fit"]

        if show_r2 and r2 is not None:
            label_parts.append(f"R²={r2:.3f}")

        if param_name is not None and param_name in params:
            label_parts.append(f"{param_name}={params[param_name]:.3f}")

        fit_label = " | ".join(label_parts)

        # Plot
        ax.plot(time[mask], y_exp[mask], "o", label="exp")
        ax.plot(time[mask], y_fit[mask], "-", label=fit_label)

        # Limits
        if time_range is not None:
            ax.set_xlim(time_range)
        if ylim is not None:
            ax.set_ylim(ylim)

        ax.set_xlabel("Time (ps)")
        ax.set_ylabel("dT/T")

        if legend:
            ax.legend(fontsize=8)

        if created_fig:
            plt.tight_layout()
            plt.show()

        return ax
    
    
    #============ Panel Spectra Plot ==========================#
    @classmethod
    def plot_spectra_panel(
        cls,
        master_wvl,
        df_compo,
        sample_ids: Optional[Sequence] = None,
        ncols: int = 6,
        figsize_scale: Tuple[float, float] = (3.2, 2.6),
        sharex: bool = True,
        sharey: bool = True,
        drop_columns: Sequence[str] = ("sample", "power_mW"),
        title_col: str = "symbol",
        time_range=(0, 0.5),
        step: int = 3,
        legend: bool = False,
        add_colorbar: bool = True,
        title_fontsize: int = 9
    ):
        """
        Plot a panel of spectra, one subplot per sample.

        Parameters
        ----------
        master_wvl : dict
            Dictionary of sample -> dataframe.
        df_compo : pd.DataFrame
            Dataframe with metadata, including ID and title column.
        sample_ids : sequence or None
            Which sample IDs to plot. If None, uses all keys in master_wvl.
        ncols : int
            Number of columns in the subplot grid.
        figsize_scale : tuple
            Scale factors for figure size per subplot.
        sharex, sharey : bool
            Shared axes options.
        drop_columns : sequence of str
            Columns to drop before passing dataframe to plot method.
        title_col : str
            Column from df_compo used as subplot title.
        time_range, step, legend, add_colorbar
            Passed to plot_spectra_at_times.
        title_fontsize : int
            Font size for subplot titles.

        Returns
        -------
        fig, axes
        """
        if sample_ids is None:
            sample_ids = list(master_wvl.keys())

        sample_ids = np.array(sample_ids)
        n = len(sample_ids)
        nrows = math.ceil(n / ncols)

        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(figsize_scale[0] * ncols, figsize_scale[1] * nrows),
            sharex=sharex,
            sharey=sharey
        )

        axes = np.atleast_1d(axes).ravel()

        for ax, sample in zip(axes, sample_ids):
            df = master_wvl[sample].drop(columns=list(drop_columns), errors="ignore")

            cls.plot_spectra_at_times(
                df,
                time_range=time_range,
                step=step,
                legend=legend,
                ax=ax,
                add_colorbar=add_colorbar,
                figsize=(6, 5)
            )

            match = df_compo.loc[df_compo["ID"] == sample, title_col]
            if not match.empty:
                ax.set_title(str(match.iloc[0]), fontsize=title_fontsize)
            else:
                ax.set_title(str(sample), fontsize=title_fontsize)

        for ax in axes[n:]:
            ax.axis("off")

        fig.tight_layout()
        return fig, axes
    
    #============ Panel Kinetic Plot ==========================#
    
    @classmethod
    def plot_kinetics_panel(
        cls,
        master_time,
        df_compo,
        sample_ids: Optional[Sequence] = None,
        wavelengths: Sequence[float] = (1300, 1500, 1600),
        ncols: int = 6,
        figsize_scale: Tuple[float, float] = (3.2, 2.6),
        sharex: bool = True,
        sharey: bool = True,
        drop_columns: Sequence[str] = ("sample", "power_mW"),
        title_col: str = "symbol",
        xlim=None,
        ylim=None,
        legend: bool = True,
        add_colorbar: bool = False,
        title_fontsize: int = 9
    ):
        """
        Plot kinetics for multiple samples in a panel layout.

        Parameters
        ----------
        master_time : dict
            Dictionary of sample -> dataframe.
        df_compo : pd.DataFrame
            DataFrame containing metadata, including ID and title column.
        sample_ids : sequence or None
            Sample IDs to plot. If None, uses all keys in master_time.
        wavelengths : sequence
            Wavelengths passed to kinetic_plot.
        ncols : int
            Number of subplot columns.
        figsize_scale : tuple
            Figure size scaling per subplot.
        sharex, sharey : bool
            Passed to plt.subplots.
        drop_columns : sequence of str
            Columns to drop before plotting.
        title_col : str
            Column in df_compo used for subplot titles.
        xlim, ylim
            Passed to kinetic_plot.
        legend : bool
            Passed to kinetic_plot.
        add_colorbar : bool
            Passed to kinetic_plot.
        title_fontsize : int
            Font size for subplot titles.

        Returns
        -------
        fig, axes
        """
        if sample_ids is None:
            sample_ids = list(master_time.keys())

        sample_ids = np.array(sample_ids)
        n = len(sample_ids)
        nrows = math.ceil(n / ncols)

        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(figsize_scale[0] * ncols, figsize_scale[1] * nrows),
            sharex=sharex,
            sharey=sharey
        )

        axes = np.atleast_1d(axes).ravel()

        for ax, sample in zip(axes, sample_ids):
            df = master_time[sample].drop(columns=list(drop_columns), errors="ignore")

            cls.kinetic_plot(
                df,
                wavelengths=wavelengths,
                xlim=xlim,
                ylim=ylim,
                ax=ax,
                legend=legend,
                add_colorbar=add_colorbar
            )

            match = df_compo.loc[df_compo["ID"] == sample, title_col]
            if not match.empty:
                ax.set_title(str(match.iloc[0]), fontsize=title_fontsize)
            else:
                ax.set_title(str(sample), fontsize=title_fontsize)

        for ax in axes[n:]:
            ax.axis("off")

        fig.tight_layout()
        return fig, axes
    
    #============ Panel Experimental vs fitted Lifetime ==========================#
    
    @classmethod
    def plot_lifetime_fit_panel(
        cls,
        master_time,
        lifetime_results,
        df_compo,
        wvl,
        sample_ids=None,
        ncols=6,
        figsize_scale=(3.2, 2.6),
        sharex=True,
        sharey=True,
        drop_columns=("sample", "power_mW"),
        title_col="symbol",
        time_range=None,
        ylim=None,
        show_r2=True,
        param_name=None,
        legend=True,
        title_fontsize=9
    ):
        """
        Plot experimental vs fitted lifetime curves in a panel layout.

        Parameters
        ----------
        master_time : dict
            Dictionary: sample_id -> dataframe
        lifetime_results : dict
            Dictionary: sample_id -> fit result
        df_compo : pd.DataFrame
            Dataframe with sample metadata
        wvl : float
            Target wavelength. Closest wavelength in each dataframe is used.
        sample_ids : sequence or None
            Samples to plot. If None, uses all keys in master_time.
        ncols : int
            Number of subplot columns.
        figsize_scale : tuple
            Figure size scaling per subplot.
        sharex, sharey : bool
            Passed to plt.subplots.
        drop_columns : tuple
            Columns to drop before plotting.
        title_col : str
            Column from df_compo used for subplot titles.
        time_range : tuple or None
            Passed to plot_lifetime_fit.
        ylim : tuple or None
            Passed to plot_lifetime_fit.
        show_r2 : bool
            To show R² in legend.
        param_name : str or None
            Parameter from lifetime['params'] to show in legend.
        legend : bool
            To show legend in each subplot.
        title_fontsize : int
            Font size for titles.

        Returns
        -------
        fig, axes
        """
        if sample_ids is None:
            sample_ids = list(master_time.keys())

        sample_ids = np.array(sample_ids)
        n = len(sample_ids)
        nrows = math.ceil(n / ncols)

        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(figsize_scale[0] * ncols, figsize_scale[1] * nrows),
            sharex=sharex,
            sharey=sharey
        )

        axes = np.atleast_1d(axes).ravel()

        for ax, sample in zip(axes, sample_ids):
            df = master_time[sample].drop(columns=list(drop_columns), errors="ignore")

            # Time axis
            x = df.columns.to_numpy(dtype=float)

            # Closest wavelength
            wvl_array = df.index.to_numpy(dtype=float)
            idx = np.abs(wvl_array - wvl).argmin()

            # Experimental signal
            y = df.iloc[idx].to_numpy(dtype=float)

            # Stored fit result
            lifetime = lifetime_results[sample]

            cls.plot_lifetime_fit(
                lifetime=lifetime,
                time=x,
                y_exp=y,
                time_range=time_range,
                ylim=ylim,
                ax=ax,
                show_r2=show_r2,
                param_name=param_name,
                legend=legend
            )

            match = df_compo.loc[df_compo["ID"] == sample, title_col]
            if not match.empty:
                ax.set_title(str(match.iloc[0]), fontsize=title_fontsize)
            else:
                ax.set_title(str(sample), fontsize=title_fontsize)

        for ax in axes[n:]:
            ax.axis("off")

        fig.tight_layout()
        return fig, axes