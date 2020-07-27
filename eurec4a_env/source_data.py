from pathlib import Path
from datetime import datetime, timedelta

from tqdm import tqdm
import requests
import xarray as xr
import yaml


JOANNE_LEV3_URL = "https://owncloud.gwdg.de/index.php/s/uy2eBvEI2pHRVKf/download?path=%2FLevel_3&files=EUREC4A_JOANNE_Dropsonde-RD41_Level_3_v0.5.7-alpha%2B0.g45fe69d.dirty.nc"

HALO_FLIGHT_DECOMP_DATE_FORMAT = "%Y%m%d"
HALO_FLIGHT_DECOMP_URL = "https://raw.githubusercontent.com/eurec4a/halo-flight-phase-separation/master/flight_phase_files/EUREC4A_HALO_Flight-Segments_{date}_v1.0.yaml"

HALO_FLIGHT_DAYS = [
    "2020-01-22",
    "2020-01-24",
    "2020-01-26",
    "2020-01-28",
    "2020-01-31",
    "2020-02-02",
    "2020-02-05",
    "2020-02-07",
    "2020-02-09",
    "2020-02-11",
    "2020-02-13",
    "2020-02-15",
]


def _download_file(url, block_size=8192):
    """
    Download a remote file and show a progress bar during download
    Based on https://stackoverflow.com/a/16696317/271776 and
    https://stackoverflow.com/a/37573701/271776
    """
    local_filename = url.split("/")[-1]
    if "=" in local_filename:
        local_filename = local_filename.split("=")[-1]

    # put files in data/ locally
    local_filename = Path("data") / local_filename

    if Path(local_filename).exists():
        pass
    else:
        local_filename.parent.mkdir(exist_ok=True, parents=True)
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            total_size_in_bytes = int(r.headers.get("content-length", 0))
            progress_bar = tqdm(total=total_size_in_bytes, unit="iB", unit_scale=True)
            with open(local_filename, "wb") as f:
                for chunk in r.iter_content(block_size):
                    progress_bar.update(len(chunk))
                    f.write(chunk)
    return local_filename


def open_joanne_dataset(level=3):
    """
    Quick-hack routine to download whole JOANNE dataset to local path and open it with xarray
    This will be replaced with intake-based OPeNDAP download once JOANNE is on AERIS
    """
    if not level == 3:
        raise NotImplementedError(level)
    filename = _download_file(JOANNE_LEV3_URL)
    ds = xr.open_dataset(filename)
    return ds.sortby("launch_time")


def _open_halo_decomp_dataset(date):
    date_str = date.strftime(HALO_FLIGHT_DECOMP_DATE_FORMAT)
    url = HALO_FLIGHT_DECOMP_URL.format(date=date_str)
    filename = _download_file(url)
    with open(filename) as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def get_halo_flight_legs(legtype="all"):
    """
    Turn the HALO flight legs files into a xarray Dataset
    """
    segment_variables_to_keep = {"start", "end", "segment_id", "name"}
    datasets = []
    for date_str in HALO_FLIGHT_DAYS:
        t = datetime.strptime(date_str, "%Y-%m-%d")
        try:
            flightinfo = _open_halo_decomp_dataset(date=t)
        except requests.exceptions.HTTPError:
            # some days don't have measurements, should probably handle this better...
            pass
        t += timedelta(hours=24)
        dss_segments = []
        segments = flightinfo["segments"]
        if legtype != "all":
            segments = filter(lambda s: legtype in s["kinds"], segments)

        for segment in segments:
            ds_segment = xr.Dataset(coords=dict(segment=segment["segment_id"]))
            for v in segment_variables_to_keep:
                ds_segment[v] = segment[v]
            dss_segments.append(ds_segment)

        if len(dss_segments) > 0:
            ds_flight = xr.concat(dss_segments, dim="segment")
            ds_flight["flight_num"] = flightinfo["name"]
            datasets.append(ds_flight)
    return xr.concat(datasets, dim="segment")
