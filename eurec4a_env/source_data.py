from pathlib import Path

from tqdm import tqdm
import requests
import xarray as xr


JOANNE_LEV3_URL = "https://owncloud.gwdg.de/index.php/s/uy2eBvEI2pHRVKf/download?path=%2FLevel_3&files=EUREC4A_JOANNE_Dropsonde-RD41_Level_3_v0.5.7-alpha%2B0.g45fe69d.dirty.nc"


def _download_file(url, block_size=8192):
    """
    Download a remote file and show a progress bar during download
    Based on https://stackoverflow.com/a/16696317/271776 and
    https://stackoverflow.com/a/37573701/271776
    """
    local_filename = url.split('/')[-1]
    if "=" in local_filename:
        local_filename = local_filename.split('=')[-1]
    if Path(local_filename).exists():
        pass
    else:
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            total_size_in_bytes = int(r.headers.get('content-length', 0))
            progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)
            with open(local_filename, 'wb') as f:
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
    return xr.open_dataset(filename)
