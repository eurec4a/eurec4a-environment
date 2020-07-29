import matplotlib

if __name__ == "__main__":  # noqa
    matplotlib.use("Agg")

import matplotlib.pyplot as plt

from ..variables.boundary_layer.mixed_layer_height import calc_peak_RH
from ..source_data import open_joanne_dataset


def plot_mixed_layer_heights(
    ds, x="time", values=["z_RHmax"], altitude="height", rh="rh", **kwargs
):
    figure, ax = plt.subplots(figsize=(6, 4))
    for v in values:
        if v == "z_RHmax":
            da_h = calc_peak_RH(ds=ds, altitude=altitude, rh=rh)
            da_h.plot(ax=ax, x=x, **kwargs)

    return ax


if __name__ == "__main__":
    # hard-coded plot of JOANNE level3 data for now
    ds = open_joanne_dataset()
    ds = ds.swap_dims(dict(sounding="time"))
    ax = plot_mixed_layer_heights(
        ds=ds, rh="rh", altitude="height", x="time", linestyle="", marker="."
    )
    fn = "mixed_layer_heights.png"
    plt.savefig(fn)
    print(f"Saved plot to {fn}")
