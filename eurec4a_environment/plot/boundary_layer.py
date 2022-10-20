import matplotlib

if __name__ == "__main__":  # noqa
    matplotlib.use("Agg")

import matplotlib.pyplot as plt

from .. import nomenclature as nom
from ..variables.boundary_layer.mixed_layer_height import calc_peak_RH
from ..source_data import open_joanne_dataset


def plot_mixed_layer_heights(
    ds, x="time", values=["z_RHmax"], altitude=nom.ALTITUDE, rh=nom.RELATIVE_HUMIDITY, **kwargs
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
    ds = ds.swap_dims(dict(sonde_id="time"))
    ax = plot_mixed_layer_heights(
        ds=ds, rh=nom.RELATIVE_HUMIDITY, altitude=nom.ALTITUDE, x="time", linestyle="", marker="."
    )
    fn = "mixed_layer_heights.png"
    plt.savefig(fn)
    print(f"Saved plot to {fn}")
