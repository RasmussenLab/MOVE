__all__ = ["find_grid_dimensions", "facet_grid", "generate_grid"]

import math
from typing import Literal, Optional

import matplotlib.axes
import matplotlib.figure
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import LogNorm, Normalize

from move.visualization.style import style_settings

Orientation = Literal["vertical", "horizontal"]
Location = Literal["left", "right", "top", "bottom"]


def find_grid_dimensions(num_subplots: int) -> tuple[int, int]:
    """Compute minimum number of columns and number of rows necessary to
    accommodate a given number of subplots into a nearly square grid."""
    # Adapted from: https://gist.github.com/pganssle/5e921b0dfc93ac54f3c35fea2cbcff2f

    num_sqrt_f = math.sqrt(num_subplots)
    num_sqrt = math.ceil(math.sqrt(num_subplots))

    if num_sqrt == num_sqrt_f:
        # perfect square
        return num_sqrt, num_sqrt
    elif num_subplots <= num_sqrt * (num_sqrt - 1):
        # try horizontal rectangle
        x, y = num_sqrt, num_sqrt - 1
    elif not (num_sqrt % 2) and num_subplots % 2:
        # try horizontal rectangle
        x, y = num_sqrt + 1, num_sqrt - 1
    else:
        # square grid
        x, y = num_sqrt, num_sqrt
    return x, y


def facet_grid(
    data: pd.DataFrame,
    x_name: str,
    y_name: str,
    hue_name: str,
    facet_name: str,
    facet_title_fmt: str,
    hue_label: Optional[str] = None,
    x_label: Optional[str] = None,
    y_label: Optional[str] = None,
    cbar_orientation: Orientation = "vertical",
    cbar_location: Location = "right",
    sharex: bool = False,
    sharey: bool = False,
    use_lognorm: bool = True,
) -> matplotlib.figure.Figure:
    """Form a matrix of panels to visualize four variables. The facet variable
    should have discrete values, whereas the other three be continuous.

    Two variables will be represented by a scatter plot, and the remaining
    variable will be represented by the hue of the scatter dots. An
    accompanying color bar will be generated.

    Args:
        x_name:
            Name of column corresponding to variable represented by x-axis
        y_name:
            Name of column corresponding to variable represented by y-axis
        hue_name:
            Name of column corresponding to variable represented by hue
        facet_name:
            Name of column corresponding to variable represented by facet
        facet_title_fmt:
            Format used to create the label of each subplot
        x_label:
            Label of the x-axis
        y_label:
            Label of the y-axis
        hue_label:
            Label of the colorbar
        cbar_orientation:
            Whether the colorbar is displayed horizontally or vertically
        cbar_location:
            Where the colorbar should be positioned
        sharex:
            Whether all subplots should share the same x-axis
        sharey:
            Whether all subplots should share the same y-axis
        use_lognorm:
            Whether the hue variable should be represented in the log dimension
    """
    levels = data[facet_name].unique()
    if len(levels) == len(data):
        raise ValueError(f"f{facet_name} is not discrete.")

    vmin, vmax = data[hue_name].min(), data[hue_name].max()
    norm_class = LogNorm if use_lognorm else Normalize
    norm = norm_class(vmin, vmax)

    with style_settings("ggplot"):
        fig, axs, cax = generate_grid(
            len(levels),
            x_label,
            y_label,
            cbar_orientation,
            cbar_location,
            sharex,
            sharey,
        )
        markers = None

        for i, level in enumerate(levels):
            subset = data[data[facet_name] == level]

            ax = axs[i]
            ax.plot(
                subset[x_name], subset[y_name], color="k", alpha=0.75, linestyle=":"
            )
            markers = ax.scatter(
                subset[x_name],
                subset[y_name],
                c=subset[hue_name],
                norm=norm,
                zorder=100,
            )
            ax.set(title=facet_title_fmt.format(level))

        assert markers is not None
        fig.colorbar(markers, cax, orientation=cbar_orientation)
        if hue_label:
            if cbar_orientation == "horizontal":
                cax.set(xlabel=hue_label)
            else:
                cax.set(ylabel=hue_label)

        fig.tight_layout()

    return fig


def generate_grid(
    num_subplots: int,
    x_label: Optional[str] = None,
    y_label: Optional[str] = None,
    cbar_orientation: Orientation = "vertical",
    cbar_location: Location = "right",
    sharex: bool = False,
    sharey: bool = False,
) -> tuple[matplotlib.figure.Figure, list[matplotlib.axes.Axes], matplotlib.axes.Axes]:
    """Form a matrix of panels to visualize multiple variables.

    Args:
        num_subplots:
            Number of subplots to accomodate in the grid
        x_label:
            Label of the x-axis
        y_label:
            Label of the y-axis
        cbar_orientation:
            Whether the colorbar is displayed horizontally or vertically
        cbar_location:
            Where the colorbar should be positioned
        sharex:
            Whether all subplots should share the same x-axis
        sharey:
            Whether all subplots should share the same y-axis
    """
    if cbar_orientation == "horizontal":
        if cbar_location not in ("top", "bottom"):
            raise ValueError(
                "Only 'top' or 'bottom' location is valid for 'horizontal' alignment"
            )
    elif cbar_orientation == "vertical":
        if cbar_location not in ("left", "right"):
            raise ValueError(
                "Only 'left' or 'right' location is valid for 'vertical' alignment"
            )
    else:
        raise ValueError("Only 'horizontal' or 'vertical' alignment allowed")

    ncols, nrows = find_grid_dimensions(num_subplots)
    num_unused = ncols * nrows - num_subplots

    fig = plt.figure(figsize=(4 * ncols, 3 * nrows))

    if cbar_orientation == "horizontal":
        if cbar_location == "top":
            cax_idx = 0
            height_ratios = [1] + [3 * ncols] * nrows
        else:
            cax_idx = -1
            height_ratios = [3 * ncols] * nrows + [1]

        gs = fig.add_gridspec(nrows + 1, ncols, height_ratios=height_ratios)
        cax = fig.add_subplot(gs[cax_idx, :])
    else:
        if cbar_location == "left":
            cax_idx = 0
            width_ratios = [1] + [4 * nrows] * ncols
        else:
            cax_idx = -1
            width_ratios = [4 * nrows] * ncols + [1]

        gs = fig.add_gridspec(nrows, ncols + 1, width_ratios=width_ratios)
        cax = fig.add_subplot(gs[:, cax_idx])

    axs = []
    for i in range(num_subplots):
        x_coord = (i // ncols) + 1 * (cbar_location == "top")
        y_coord = (i % ncols) + 1 * (cbar_location == "left")

        kwargs = {}
        if sharex and len(axs) > 0:
            kwargs["sharex"] = axs[0]
        if sharey and len(axs) > 0:
            kwargs["sharey"] = axs[0]

        ax = fig.add_subplot(gs[x_coord, y_coord], **kwargs)
        axs.append(ax)

    if x_label:
        for ax in axs[-(ncols - num_unused) :]:
            ax.set(xlabel=x_label)

    if y_label:
        for i in range(0, len(axs), ncols):
            axs[i].set(ylabel=y_label)

    return fig, axs, cax
