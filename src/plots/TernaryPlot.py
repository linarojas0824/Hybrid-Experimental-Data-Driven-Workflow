from pathlib import Path
from typing import Sequence, Optional, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import ternary


def plot_ternary_heatmap(
    df: pd.DataFrame,
    comp_cols: Sequence[str] = ("Cu", "Ni", "Al"),
    value_col: str = "e2",
    scale: int = 50,
    cmap: str = "Spectral_r",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    show: bool = True,
    save_path: Optional[str] = None,
    figsize: Tuple[float, float] = (12, 10),
    style: str = "hexagonal",
    duplicate_handling: str = "error",
    grid_multiple: int = 5,
    tick_multiple: int = 10,
    font_family: str = "serif",
    font_name: str = "Times New Roman",
    axis_label_fontsize: int = 35,
    tick_fontsize: int = 35,
    cbar_ticksize: int = 35,
):
    
    if len(comp_cols) != 3:
        raise ValueError("comp_cols must contain exactly 3 column names.")

    required_cols = list(comp_cols) + [value_col]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in DataFrame: {missing}")

    if scale <= 0:
        raise ValueError("scale must be a positive integer.")

    valid_duplicate_options = {"error", "mean", "median", "min", "max", "first"}
    if duplicate_handling not in valid_duplicate_options:
        raise ValueError(
            f"duplicate_handling must be one of {valid_duplicate_options}, "
            f"got '{duplicate_handling}'."
        )

    # Copy and drop rows with missing values in required columns
    data_df = df.copy().dropna(subset=required_cols)

    if data_df.empty:
        raise ValueError("No valid rows remain after dropping NaNs.")

    # Convert compositions from 0-100% to ternary lattice coordinates
    coords = (data_df[list(comp_cols)] / 100 * scale).round().astype(int)

    # Fix rounding drift so each row sums exactly to 'scale'
    drift = scale - coords.sum(axis=1)
    maxcol = data_df[list(comp_cols)].to_numpy().argmax(axis=1)

    for i in range(len(coords)):
        coords.iat[i, maxcol[i]] += drift.iat[i]

    # Replace composition columns with rounded integer lattice coordinates
    data_df = data_df.copy()
    data_df[list(comp_cols)] = coords

    # Detect duplicates after rounding
    dups = data_df.duplicated(subset=list(comp_cols), keep=False)

    if dups.any():
        dup_rows = (
            data_df.loc[dups, list(comp_cols) + [value_col]]
            .sort_values(list(comp_cols))
            .reset_index(drop=True)
        )

        if duplicate_handling == "error":
            raise ValueError(
                "Duplicate ternary coordinates found after scaling/rounding.\n"
                "This means multiple rows collapsed into the same lattice point.\n\n"
                f"{dup_rows.to_string(index=False)}"
            )

        elif duplicate_handling in {"mean", "median", "min", "max"}:
            data = (
                data_df.groupby(list(comp_cols))[value_col]
                .agg(duplicate_handling)
                .to_dict()
            )

        elif duplicate_handling == "first":
            data = (
                data_df.drop_duplicates(subset=list(comp_cols), keep="first")
                .set_index(list(comp_cols))[value_col]
                .to_dict()
            )

    else:
        data = data_df.set_index(list(comp_cols))[value_col].to_dict()

    # Plot settings
    mpl.rcParams["font.family"] = font_family
    mpl.rcParams["font.serif"] = [font_name]

    fig, tax = ternary.figure(scale=scale)
    fig.set_size_inches(*figsize)

    tax.heatmap(
        data,
        style=style,
        cmap=cmap,
        colorbar=True,
        vmin=vmin,
        vmax=vmax,
    )

    # Format colorbar ticks
    cax = fig.axes[-1]
    cax.tick_params(labelsize=cbar_ticksize)

    # Ternary formatting
    tax.boundary(linewidth=2.0)
    tax.gridlines(color="gray", multiple=grid_multiple)

    tax.left_axis_label(comp_cols[2], fontsize=axis_label_fontsize, offset=0.20)
    tax.right_axis_label(comp_cols[1], fontsize=axis_label_fontsize, offset=0.20)
    tax.bottom_axis_label(comp_cols[0], fontsize=axis_label_fontsize, offset=0.15)

    tax.ticks(
        axis="lbr",
        multiple=tick_multiple,
        offset=0.025,
        linewidth=1,
        fontsize=tick_fontsize,
    )

    # Convert tick labels from lattice scale to percent
    pct_per_unit = 100 / scale
    ax = tax.get_axes()

    for txt in ax.texts:
        label = txt.get_text().strip()
        try:
            value = float(label)
            txt.set_text(str(int(value * pct_per_unit)))
        except ValueError:
            pass

    tax.clear_matplotlib_ticks()

    ax.set_frame_on(False)
    ax.axis("off")

    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig, tax