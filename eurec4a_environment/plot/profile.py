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
    interpolate_between_profiles=True,
    cbar_label=None,
    height_labels=None,
    height_labels_color="green",
    **kwargs
):

    """Plot a 2D (default time-height) plot and optionally another height level as scatter points

    **kwargs: Additional keyword arguments passed to
            :func:`matplotlib.pyplot.contourf` if interpolate_between_profiles, else to :func:`matplotlib.pyplot.pcolormesh`
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

    if interpolate_between_profiles:
        var.plot.contourf(
            ax=ax,
            x=x,
            y=y,
            add_colorbar=True,
            robust=True,
            cbar_kwargs=dict(label=cbar_label),
            **kwargs
        )
    else:
        var.plot.pcolormesh(
            ax=ax,
            x=x,
            y=y,
            add_colorbar=True,
            robust=True,
            cbar_kwargs=dict(label=cbar_label),
            **kwargs
        )

    if height_labels:
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


def plot_profile_1D(
    ds_plot,
    variables=["rh", "q"],
    axis_labels=["relative humidity (%)", "specific humidity (g/kg)"],
    height_labels=None,
    **kwargs
):
    """Plot a 1D profile plot of two variables and a height level 
    input a 1D dataset (that is profiles e.g. of one timestep or sounding or a mean profile)
    """

    fig, ax1 = plt.subplots(figsize=(12, 12))
    altitude = "alt"
    ds_plot[variables[0]].plot(ax=ax1, y=altitude, color="navy", **kwargs)
    ax1.set_xlabel(axis_labels[0], color="navy")
    ax1.set_ylabel("altitude (m)", color="black")

    ax2 = ax1.twiny()
    ds_plot[variables[1]].plot(ax=ax2, y=altitude, color="lightblue", **kwargs)
    ax2.set_xlabel(axis_labels[1], color="lightblue")

    if height_labels:
        ax1.axhline(
            ds_plot[height_labels],
            color="navy",
            linewidth=2,
            alpha=1,
            label=height_labels,
        )

    ax1.set_title("")
    ax2.set_title("")
    ax1.grid()
    ax1.legend(loc="best")
    plt.tight_layout()
    plt.close()
    return fig
