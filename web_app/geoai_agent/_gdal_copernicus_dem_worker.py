"""
Runs inside QGIS's bundled Python environment ONLY (python-qgis.bat, which
carries a full GDAL install) -- never imported by the main process. Same
streamed-COG-mosaic-clip-resample operation as _gdal_dem_worker.py (see
that module's docstring for the full explanation of the technique), just
against the Copernicus GLO-30 global DEM instead of USGS 3DEP -- this
project's elevation source outside the continental US, where 3DEP has no
coverage at all.

Public AWS Open Data bucket (registry.opendata.aws/copernicus-dem),
Requester Pays (no cost under the Public Dataset Program, but every
request must carry an explicit acknowledgment header or S3 returns 404 --
checked directly: the exact same GET without this header/config fails).
GDAL's /vsis3/ virtual filesystem (not /vsicurl/, unlike the USGS worker)
handles this natively via the AWS_REQUEST_PAYER config option, with
AWS_NO_SIGN_REQUEST=YES so no actual AWS credentials are needed for this
publicly-readable-but-requester-pays bucket.

Tile naming confirmed live (not assumed from documentation, which is
stale/misleading here -- the public docs describe a "_30_" product code
that returns 404; the real, working code is "_10_"):
    Copernicus_DSM_COG_10_{N|S}{lat:02d}_00_{E|W}{lon:03d}_00_DEM/
      Copernicus_DSM_COG_10_{N|S}{lat:02d}_00_{E|W}{lon:03d}_00_DEM.tif
1-degree tiles, lower-left-corner-labeled (tile "N51" spans 51-52N, unlike
USGS's upper-corner "n39w077" spans 38-39N convention) -- confirmed against
a known London-area tile. "GLO-30 Public" coverage excludes a small subset
of tiles for specific countries not yet released by the Copernicus
Programme; a 404 for a real land tile most likely means that, not a naming
error -- the caller should fall back to Open-Meteo in that case, same as
any other failure here.

Usage:
    python-qgis.bat _gdal_copernicus_dem_worker.py <out.npz> <lat_min> <lat_max> <lon_min> <lon_max> <nx> <ny>

Writes out.npz with the same shape/convention as _gdal_dem_worker.py's
output (values: (ny, nx) float64, row 0 = north; tiles: names used).
"""
import math
import sys

import numpy as np
from osgeo import gdal

gdal.UseExceptions()

_BUCKET = "copernicus-dem-30m"
_NODATA_SENTINEL_CEILING = -1000.0


def _tile_name(lat_deg: int, lon_deg: int) -> str:
    ns = "N" if lat_deg >= 0 else "S"
    ew = "E" if lon_deg >= 0 else "W"
    return f"Copernicus_DSM_COG_10_{ns}{abs(lat_deg):02d}_00_{ew}{abs(lon_deg):03d}_00_DEM"


def _tiles_for_bounds(lat_min: float, lat_max: float, lon_min: float, lon_max: float) -> list[str]:
    """Copernicus GLO-30 1-degree tiles are lower-left-corner-labeled --
    tile 'N51E000' spans [51, 52) latitude x [0, 1) longitude. Enumerates
    every such tile overlapping the bounding box (almost always exactly
    one for a town/neighborhood-scale box, but more if it straddles a
    degree boundary or crosses the prime meridian/equator)."""
    lat_start, lat_end = math.floor(lat_min), math.floor(lat_max - 1e-9)
    lon_start, lon_end = math.floor(lon_min), math.floor(lon_max - 1e-9)
    tiles = []
    for lat_deg in range(lat_start, lat_end + 1):
        for lon_deg in range(lon_start, lon_end + 1):
            tiles.append(_tile_name(lat_deg, lon_deg))
    return tiles


def main() -> None:
    out_path = sys.argv[1]
    lat_min, lat_max, lon_min, lon_max = (float(v) for v in sys.argv[2:6])
    nx, ny = int(sys.argv[6]), int(sys.argv[7])

    # Requester-Pays anonymous access -- confirmed live these two config
    # options together are what makes /vsis3/ resolve without real AWS
    # credentials (AWS_NO_SIGN_REQUEST alone still gets a Requester-Pays
    # bucket's objects rejected; AWS_REQUEST_PAYER alone without also
    # disabling signing tries to sign the request with no credentials and
    # fails differently).
    gdal.SetConfigOption("AWS_NO_SIGN_REQUEST", "YES")
    gdal.SetConfigOption("AWS_REQUEST_PAYER", "requester")

    tiles = _tiles_for_bounds(lat_min, lat_max, lon_min, lon_max)
    urls = [f"/vsis3/{_BUCKET}/{t}/{t}.tif" for t in tiles]

    src = gdal.Open(urls[0]) if len(urls) == 1 else gdal.BuildVRT("", urls)

    clipped = gdal.Translate(
        "", src, format="MEM",
        projWin=[lon_min, lat_max, lon_max, lat_min],  # ulx, uly, lrx, lry
        projWinSRS="EPSG:4326",
        width=nx, height=ny, resampleAlg="bilinear",
    )
    band = clipped.GetRasterBand(1)
    nodata = band.GetNoDataValue()
    arr = band.ReadAsArray().astype(np.float64)
    if nodata is not None:
        arr[arr == nodata] = np.nan
    arr[arr < _NODATA_SENTINEL_CEILING] = np.nan

    np.savez(out_path, values=arr, tiles=np.array(tiles))


if __name__ == "__main__":
    main()
