"""
Runs inside QGIS's bundled Python environment ONLY (python-qgis.bat, which
carries a full GDAL install) -- never imported by the main process. Streams
a USGS 3DEP 1/3 arc-second (~10m) DEM COG directly from its S3-hosted
GeoTIFF via GDAL's /vsicurl/ virtual filesystem (no full download),
mosaics adjacent 1-degree tiles if the requested bounding box spans more
than one, clips to the exact requested extent, and resamples straight onto
the requested analysis grid -- all in one gdal.Translate call.

This is the same operation as gdal:cliprasterbyextent (the QGIS Processing
algorithm used in the reference pipeline this mirrors, D:\\PythonProject\\new
\\qgis_slope_college_park.py), just invoked directly through the GDAL
Python bindings instead of QGIS's Processing wrapper around that same GDAL
call -- identical result, one less framework layer (no QgsApplication /
Processing.initialize() needed for a raster-only operation).

Usage:
    python-qgis.bat _gdal_dem_worker.py <out.npz> <lat_min> <lat_max> <lon_min> <lon_max> <nx> <ny>

Writes out.npz with:
- values: (ny, nx) float64 array, row 0 = north edge of the source raster
  (GDAL's native top-to-bottom row order; the caller flips this to match
  the rest of the codebase's south-to-north row convention)
- tiles: the USGS 3DEP tile name(s) actually used
"""
import math
import sys

import numpy as np
from osgeo import gdal

gdal.UseExceptions()

_TILE_URL_TEMPLATE = (
    "/vsicurl/https://prd-tnm.s3.amazonaws.com/StagedProducts/Elevation/13/"
    "TIFF/current/{tile}/USGS_13_{tile}.tif"
)
# Below this, treat the pixel as missing data rather than a real elevation --
# a clip window that pokes slightly outside the source tile's actual extent
# (e.g. a bounding box edge landing a fraction of a degree past the nominal
# 1-degree tile boundary) gets filled with GDAL's NoData sentinel, which for
# this dataset is a large negative value, not always exactly matching
# GetNoDataValue() after a MEM-format Translate.
_NODATA_SENTINEL_CEILING = -1000.0


def _tiles_for_bounds(lat_min: float, lat_max: float, lon_min: float, lon_max: float) -> list[str]:
    """USGS 3DEP 1-degree tiles are named 'n{N}w{W}' where the tile spans
    latitude [N-1, N] and longitude [-W, -(W-1)] -- e.g. 'n39w077' covers
    38-39N, 77-76W. Enumerates every such tile overlapping the bounding
    box (almost always exactly one for a town/neighborhood-scale box, but
    more if it straddles a degree boundary)."""
    south_start, south_end = math.floor(lat_min), math.floor(lat_max - 1e-9)
    west_start, west_end = math.floor(lon_min), math.floor(lon_max - 1e-9)
    tiles = []
    for south in range(south_start, south_end + 1):
        for west in range(west_start, west_end + 1):
            tiles.append(f"n{south + 1}w{abs(west):03d}")
    return tiles


def main() -> None:
    out_path = sys.argv[1]
    lat_min, lat_max, lon_min, lon_max = (float(v) for v in sys.argv[2:6])
    nx, ny = int(sys.argv[6]), int(sys.argv[7])

    tiles = _tiles_for_bounds(lat_min, lat_max, lon_min, lon_max)
    urls = [_TILE_URL_TEMPLATE.format(tile=t) for t in tiles]

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
