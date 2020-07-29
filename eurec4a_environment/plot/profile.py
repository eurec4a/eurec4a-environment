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
    height_labels=None,
    cbar_label=None,
    cm="viridis",
    **kwargs
):
    # ensure we have time as a coordinate
    time = "launch_time"
    org_dim = None
    try:
        ds.isel(**{time: 0})
    except ValueError:
        org_dim = ds[time].dims[0]
        ds = ds.swap_dims({org_dim: time})

    var = ds[variable]

    varh = ds[height_labels]
    # var = ds[var_str].resample(launch_time=freq).mean(dim=time)
    fig, ax = plt.subplots(figsize=(20, 10))
    the_fig = var.plot.contourf(
        ax=ax, y=y, levels=10, cmap=cm, add_colorbar=False, robust=True
    )
    varh.plot(linestyle="None", marker="p", color="green", label=height_labels)
    ax.set_ylim(0, 18000)
    myFmt = mdates.DateFormatter("%m-%d")
    ax.xaxis.set_major_formatter(myFmt)
    fig.colorbar(the_fig, label=cbar_label)
    ax.set_xlabel("time")
    ax.set_ylabel("height / m")
    ax.legend()
    ax.xaxis.set_major_formatter(myFmt)
    fig.autofmt_xdate()
    ax.autoscale_view()
    plt.tight_layout()
    fig.savefig("ProfilePlot2D_{}.pdf".format(variable), bbox_inches="tight")
