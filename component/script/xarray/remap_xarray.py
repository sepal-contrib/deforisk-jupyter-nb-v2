"""
Utilities for converting categorical GeoTIFFs into binary rasters.

The :func:`remap_categorical_to_binary` routine reads a GeoTIFF, remaps all
pixels whose values belong to the *one_values* list to **1**, those that belong
to the *zero_values* list to **0**, and assigns a user‑supplied No‑Data value
to every other pixel.  The implementation relies exclusively on :mod:`xarray`
and :mod:`dask` operations (via rioxarray), so the computation is lazy and
memory‑efficient even for terabyte‑scale rasters.

The function preserves all geospatial metadata (CRS, transform, bounds) and
keeps the original No‑Data mask intact.
"""

import pathlib
from typing import Iterable, Union

import rioxarray as rxr
import xarray as xr


def remap_categorical_to_binary(
    input_path: str | pathlib.Path,
    output_path: str | pathlib.Path,
    one_values: Iterable[Union[int, float]],
    zero_values: Iterable[Union[int, float]],
    nodata_value: Union[int, float] | None = None,
) -> None:
    """
    Convert a categorical GeoTIFF to a binary raster.

    The routine reads *input_path*, remaps all pixels whose values are in
    ``one_values`` to **1**, those that appear in ``zero_values`` to **0**,
    and assigns the supplied ``nodata_value`` to every other pixel.  The
    operation is performed lazily using :mod:`xarray`/:mod:`dask`, so it
    can handle very large rasters without exhausting RAM.

    Parameters
    ----------
    input_path:
        Path (string or pathlib.Path) to the source GeoTIFF.
    output_path:
        Destination path for the binary raster.  Existing files are
        overwritten.
    one_values:
        Iterable of numeric values that should be mapped to **1** in the
        output.  The order does not matter; all occurrences will be set to 1.
    zero_values:
        Iterable of numeric values that should be mapped to **0** in the
        output.  These are evaluated only for pixels *not* already mapped
        to 1, so ``one_values`` takes precedence if there is overlap.
    nodata_value:
        Numeric value to use as No‑Data for every pixel that does not
        belong to either list.  If ``None``, the original No‑Data value
        from *input_path* is preserved.

    Returns
    -------
    None

    Raises
    ------
    FileNotFoundError
        If *input_path* does not exist.
    ValueError
        If either ``one_values`` or ``zero_values`` is empty.
    """
    # --------------------------------------------------------------------
    # 1. Validate the input file and prepare the source DataArray
    # --------------------------------------------------------------------
    src_fp = pathlib.Path(input_path)
    if not src_fp.is_file():
        raise FileNotFoundError(f"Input file does not exist: {src_fp}")

    one_vals = list(one_values)
    zero_vals = list(zero_values)

    if not one_vals:
        raise ValueError("`one_values` must contain at least one element.")
    if not zero_vals:
        raise ValueError("`zero_values` must contain at least one element.")

    # Load the raster as a masked xarray.DataArray
    src_da = rxr.open_rasterio(
        src_fp,
        masked=True,
        chunks="auto",
    )

    # --------------------------------------------------------------------
    # 2. Build boolean masks for the two categories
    # --------------------------------------------------------------------
    cond_one: xr.DataArray = src_da.isin(one_vals)
    cond_zero: xr.DataArray = src_da.isin(zero_vals)

    # --------------------------------------------------------------------
    # 3. Construct the binary raster with nested xarray.where calls.
    #
    #     result = where(cond_one, 1,
    #                    where(cond_zero, 0, nodata_value))
    #
    # This keeps the original No‑Data mask automatically because
    # `da.isin(...)` inherits the mask from *da*.
    # --------------------------------------------------------------------
    nodata_to_use: Union[int, float] = (
        nodata_value if nodata_value is not None else src_da.rio.nodata
    )

    binary_array = xr.where(
        cond_one,
        1,
        xr.where(cond_zero, 0, nodata_to_use),
    ).astype("uint8")

    # --------------------------------------------------------------------
    # Wrap the array in an xarray.DataArray, preserving coordinates and
    # attributes.
    # --------------------------------------------------------------------
    binary_da = xr.DataArray(
        data=binary_array,
        dims=src_da.dims,
        coords=src_da.coords,
        # attrs=src_da.attrs,
    )

    # Preserve geospatial metadata explicitly (crs, transform, bounds).
    binary_da.rio.write_crs(src_da.rio.crs)
    binary_da.rio.write_transform(src_da.rio.transform(), inplace=True)

    # Explicitly set the No‑Data value in the output raster
    binary_da = binary_da.rio.write_nodata(nodata_to_use)

    # --------------------------------------------------------------------
    # 4. Write the result to disk
    # --------------------------------------------------------------------
    out_fp = pathlib.Path(output_path)

    # --------------------------------------------------------------------
    # Write the result
    # --------------------------------------------------------------------
    from dask.distributed import Lock

    binary_da.rio.to_raster(
        out_fp,
        driver="GTiff",
        compress="DEFLATE",
        bigtiff="YES",
        tiled=True,
        lock=Lock("rio"),
    )

    # ------------------------------------------------------------------------
    # Example usage (uncomment and adapt paths if you want to run it):
    #
    # >>> remap_categorical_to_binary(
    # ...     input_path="input.tif",
    # ...     output_path="output_binary.tif",
    # ...     one_values=[1, 2],          # e.g. forest & water
    # ...     zero_values=[3, 4, 5],      # e.g. urban, agriculture, bare soil
    # ...     nodata_value=-9999,
    # ... )
    #
    # The function will create *output_binary.tif* containing only values
    # 0, 1 and the specified No‑Data value, while keeping the CRS,
    # transform, bounds and any other metadata from the source file.
    # ------------------------------------------------------------------------
