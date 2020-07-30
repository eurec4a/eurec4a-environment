import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import matplotlib.dates as mdates
import seaborn as sns
import typhon

sns.set(
    context="notebook",
    style="white",
    palette="deep",
    font="sans-serif",
    font_scale=2,
    color_codes=True,
    rc=None,
)


def profile_plot_2D(
    ds,
    variable,
    ax=None,
    y="alt",
    x="launch_time",
    cbar_label=None,
    height_labels=None,
    height_labels_color="green",
    **kwargs
):

    """Plot a 2D (default time-height) plot and optionally another height level as scatter points

    **kwargs: Additional keyword arguments passed to
            :func:`matplotlib.pyplot.contourf`.
    """

    # ensure we have x (default launch time) as a coordinate
    time = x
    org_dim = None
    try:
        ds.isel(**{time: 0})
    except ValueError:
        org_dim = ds[time].dims[0]
        ds = ds.swap_dims({org_dim: time})

    # Plot
    fig, ax = plt.subplots(figsize=(20, 10))

    var = ds[variable]

    var.plot.contourf(
        ax=ax,
        x=x,
        y=y,
        add_colorbar=True,
        robust=True,
        cbar_kwargs=dict(label=cbar_label),
        **kwargs
    )

    if height_labels == True:
        varh = ds[height_labels]

        varh.plot(
            ax=ax,
            linestyle="None",
            marker="p",
            color=height_labels_color,
            label=height_labels,
        )
        ax.legend()
    if x == "launch_time":
        ax.set_xlabel("time")
        myFmt = mdates.DateFormatter("%m-%d")
        ax.xaxis.set_major_formatter(myFmt)
    if y == "alt":
        ax.set_ylim(0, 18000)
        ax.set_ylabel("height / m")
    plt.close()
    return fig
    # fig.savefig("ProfilePlot2D_{}.pdf".format(variable), bbox_inches="tight")
