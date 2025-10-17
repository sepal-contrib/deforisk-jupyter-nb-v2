from pathlib import Path
from typing import Literal
import geopandas as gpd

import rioxarray
import fiona
from shapely.geometry import shape

import numpy as np
import rasterio


def calculate_utm_rioxarray(
    input_tif_file_path: str | Path,
    output_mode: Literal["str", "int"] = "str",
) -> str | int | None:
    """
    Estimate the UTM CRS of a GeoTIFF file using rioxarray.

    Parameters
    ----------
    input_tif_file_path : str | Path
        The path to the raster file.
    output_mode : Literal["str", "int"], default="str"
        How to return the CRS.  ``"str"`` returns an EPSG string (e.g.,
        ``EPSG:32633``); ``"int"`` returns the numeric EPSG code.

    Returns
    -------
    str | int | None
        The CRS representation or ``None`` if it could not be estimated.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If an unsupported `output_mode` is supplied.
    RuntimeError
        If rioxarray fails to open the raster.

    Examples
    --------
    >>> calculate_utm_rioxarray("sample.tif")
    'EPSG:32633'
    >>> calculate_utm_rioxarray("sample.tif", output_mode="int")
    32633
    """
    # Validate path
    path = Path(input_tif_file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Raster file {input_tif_file_path!r} does not exist")

    try:
        raster = rioxarray.open_rasterio(path, chunks="auto", cache=False, lock=False)
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(f"Failed to open raster {path!s}") from exc

    try:
        utm_crs = raster.rio.estimate_utm_crs()
        if utm_crs is None:
            return None

        if output_mode == "str":
            return utm_crs.to_string()
        elif output_mode == "int":
            epsg_code = utm_crs.to_epsg()  # may still be None
            return epsg_code
        else:  # pragma: no cover
            raise ValueError(f"output_mode must be 'str' or 'int', got {output_mode!r}")
    finally:
        # Ensure the dataset is closed to free resources
        raster.close()


def get_centroid(shapefile_path):
    """
    Get the centroid of the first feature in a shapefile using Shapely and Fiona.

    Parameters:
        shapefile_path (str): Path to the input shapefile.

    Returns:
        tuple: A tuple containing the latitude and longitude of the centroid.
    """
    # Open the shapefile
    with fiona.open(shapefile_path, "r") as shapefile:
        # Get the first feature
        first_feature = next(iter(shapefile))

        # Convert the feature geometry to a Shapely geometry object
        geom = shape(first_feature["geometry"])

        # Calculate the centroid
        centroid = geom.centroid

        # Get the coordinates of the centroid
        longitude, latitude = centroid.x, centroid.y

    return (longitude, latitude)


def get_utm_proj_str_from_lat_lon(lon, lat):
    """
    Given a longitude, latitude in WGS84, return the EPSG code as a string
    for the corresponding UTM or UPS projection.

    - UTM: EPSG:326xx (Northern) or EPSG:327xx (Southern)
    - UPS: EPSG:5041 (North, >84°N), EPSG:5042 (South, <–80°S)

    Handles special cases for Norway and Svalbard.
    """
    # UPS zones for polar regions
    if lat >= 84:
        return "EPSG:5041"  # UPS North
    elif lat <= -80:
        return "EPSG:5042"  # UPS South

    # Special cases for Norway and Svalbard
    if lat > 55 and lat < 64 and lon > 2 and lon < 6:
        zone_number = 32
    elif lat > 71 and lon >= 6 and lon < 9:
        zone_number = 31
    elif lat > 71 and ((lon >= 9 and lon < 12) or (lon >= 18 and lon < 21)):
        zone_number = 33
    elif lat > 71 and ((lon >= 21 and lon < 24) or (lon >= 30 and lon < 33)):
        zone_number = 35
    else:
        zone_number = int((lon + 180) / 6) + 1

    if lat >= 0:
        epsg_code = 32600 + zone_number  # Northern Hemisphere
    else:
        epsg_code = 32700 + zone_number  # Southern Hemisphere

    return f"EPSG:{epsg_code}"


def process_forest_loss(input1_path, input2_path, output_path):
    # Open the input rasters
    with rasterio.open(input1_path) as src1:
        input1 = src1.read(1)
        bounds1 = src1.bounds
        profile = src1.profile
        nodata1 = src1.nodata

    with rasterio.open(input2_path) as src2:
        input2 = src2.read(1)
        bounds2 = src2.bounds
        nodata2 = src2.nodata

    # Check if the bounds of input1 are equal to or larger than those of input2
    if not (
        bounds1.left <= bounds2.left
        and bounds1.right >= bounds2.right
        and bounds1.top >= bounds2.top
        and bounds1.bottom <= bounds2.bottom
    ):
        raise ValueError(
            "The bounds of input1 must be equal to or larger than those of input2."
        )

    # Create masks for valid data
    valid_mask = (input1 != nodata1) & (input2 != nodata2)

    # Initialize output with nodata (255)
    output = np.full(input1.shape, 255, dtype=np.uint8)

    # Set values based on conditions:
    # 1 where input1 == 1 and input2 == 0
    # 0 where input1 == 1 and input2 == 1
    # nodata (255) for all other cases

    # Create condition for 1s: input1 == 1 AND input2 == 0
    condition_1 = (input1 == 1) & (input2 == 0)

    # Create condition for 0s: input1 == 1 AND input2 == 1
    condition_0 = (input1 == 1) & (input2 == 1)

    # Apply conditions only where both inputs are valid
    output[valid_mask & condition_1] = 0
    output[valid_mask & condition_0] = 1

    # Update the profile for the output raster
    profile.update(dtype=rasterio.uint8, compress="deflate", nodata=255)

    # Write the output raster
    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(output, 1)


def reproject_shapefile(
    input_path: str,
    output_path: str,
    target_crs: str,
) -> gpd.GeoDataFrame:
    """
    Reprojects a shapefile to a target CRS.

    Args:
        input_path (str): Path to input shapefile
        target_crs (str): Target CRS (e.g., "EPSG:4326")
        output_path (str, optional): Path to save reprojected shapefile. If None, returns GeoDataFrame.

    Returns:
        gpd.GeoDataFrame: Reprojected GeoDataFrame
    """
    gdf = gpd.read_file(input_path)
    gdf = gdf.to_crs(target_crs)
    if output_path:
        gdf.to_file(output_path)
    return None


def xr_reproject(
    raster_path: str = None,
    geobox=None,
    resampling_method="nearest",
    output_path: str = None,
    **rasterio_kwargs,
):
    """
    Rasterizes a vector shapefile into a raster array.

    This function provides unified functionality for both binary and unique ID rasterization.

    Parameters
    ----------
    raster_path : str
        Path to the input shapefile containing vector data.
    geobox : odc.geo.geobox.GeoBox
        The spatial template defining the shape, coordinates, dimensions, and transform
        of the output raster.
    crs : str or CRS object, optional
        If ``geobox``'s coordinate reference system (CRS) cannot be
        determined, provide a CRS using this parameter.
        (e.g. 'EPSG:3577').
    output_path : string, optional
        Provide an optional string file path to export the rasterized
        data as a GeoTIFF file.
    **rasterio_kwargs :
        A set of keyword arguments to ``rasterio.features.rasterize``.
        Can include: 'all_touched', 'merge_alg', 'dtype'.

    Returns
    -------
    da_rasterized : xarray.DataArray
        The rasterized vector data.
    """

    import geopandas as gpd
    import rasterio
    import xarray
    from odc.geo import xr

    # Read the raster
    raster_array = rioxarray.open_rasterio(
        raster_path,
        chunks="auto",
        cache=False,
        lock=False,
    )

    # Convert numpy array to a full xarray.DataArray
    # and set array name if supplied
    da_reprojected = xr.xr_reproject(
        src=raster_array,
        how=geobox,
        resampling=resampling_method,
    )

    da_reprojected.rio.to_raster(
        output_path,
        driver="GTiff",
        compress="DEFLATE",
        predictor=2,
        bigtiff="YES",
        tiled=True,
    )

    # Explicitly close references – not strictly required but tidy.
    del raster_array
    del da_reprojected
