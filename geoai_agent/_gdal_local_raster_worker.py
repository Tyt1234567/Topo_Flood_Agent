"""
Runs inside QGIS's bundled Python environment ONLY (python-qgis.bat, which
carries a full GDAL install) -- never imported by the main process. See
dem_cog_source.py's module docstring for why this project bridges out to a
subprocess for any raster work.

Clips, reprojects, and resamples a LOCAL GeoTIFF a user uploaded (as opposed
to _gdal_dem_worker.py's USGS 3DEP S3 tiles) directly onto the requested
lat/lon analysis grid. Uses gdal.Warp rather than gdal.Translate/projWin:
an arbitrary uploaded raster can't be assumed to already be in EPSG:4326
(unlike the curated USGS COG source), so the source's native CRS (whatever
it is, read from the file's own metadata) is reprojected on the fly as part
of the same call, rather than requiring the uploader to pre-reproject.

Usage:
    python-qgis.bat _gdal_local_raster_worker.py <in.tif> <out.npz> \
        <lat_min> <lat_max> <lon_min> <lon_max> <nx> <ny>

Writes out.npz with "values": (ny, nx) float64 array, row 0 = north edge
(GDAL's native top-to-bottom row order; the caller flips this to match the
rest of the codebase's south-to-north row convention).
"""
import sys

import numpy as np
from osgeo import gdal

gdal.UseExceptions()

# A generic uploaded raster's nodata convention isn't known in advance the
# way the curated USGS tiles' is -- GetNoDataValue() (checked below) covers
# the declared case; this is just a backstop for the occasional file that
# uses an absurd sentinel without declaring it as nodata.
_SENTINEL_CEILING = -1e30


def main() -> None:
    in_path = sys.argv[1]
    out_path = sys.argv[2]
    lat_min, lat_max, lon_min, lon_max = (float(v) for v in sys.argv[3:7])
    nx, ny = int(sys.argv[7]), int(sys.argv[8])

    src = gdal.Open(in_path)
    if src is None:
        raise RuntimeError(f"GDAL could not open '{in_path}' as a raster -- is it a valid GeoTIFF?")
    if src.GetProjection() == "" and src.GetGeoTransform() == (0.0, 1.0, 0.0, 0.0, 0.0, 1.0):
        raise RuntimeError(
            "Uploaded raster has no georeferencing (no CRS/geotransform) -- "
            "it needs to be a real GeoTIFF with spatial reference info, not a plain image."
        )

    warped = gdal.Warp(
        "", src, format="MEM",
        outputBounds=(lon_min, lat_min, lon_max, lat_max),
        outputBoundsSRS="EPSG:4326",
        dstSRS="EPSG:4326",
        width=nx, height=ny,
        resampleAlg="bilinear",
    )
    if warped is None:
        raise RuntimeError("GDAL Warp failed -- the uploaded raster may not overlap the requested area.")

    band = warped.GetRasterBand(1)
    nodata = band.GetNoDataValue()
    arr = band.ReadAsArray().astype(np.float64)
    if nodata is not None:
        arr[arr == nodata] = np.nan
    arr[arr < _SENTINEL_CEILING] = np.nan

    np.savez(out_path, values=arr)


if __name__ == "__main__":
    main()
