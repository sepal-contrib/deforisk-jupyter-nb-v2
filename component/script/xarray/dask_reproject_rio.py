# --------------------------------------------------------------
# dask_reproject_rio.py
# --------------------------------------------------------------

"""
Utility that wraps raster reprojection in a Dask task.

The wrapper follows the same style as ``export_raster_with_dask``:
  * The input raster is read only on the worker, so the client
    only has to pass the file name (no large data is sent).
  * All arguments are typed and forwarded verbatim.
  * If the output file already exists and `overwrite=False`, a warning
    is logged and a dummy future is returned.

The function returns a :class:`dask.distributed.Future` that resolves
to ``None`` the caller can use it for side effects or to chain
further tasks.
"""

# ------------------------------------------------------------------
# Imports
# ------------------------------------------------------------------
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

import dask
from dask.distributed import Client, Future, Lock, get_client


# ------------------------------------------------------------------
# Public helper – the Dask entry point
# ------------------------------------------------------------------
def reproject_raster_rio_with_dask(
    input_file: str = None,
    target_epsg: int | str = None,
    resampling_method: str = "nearest",
    output_file: str = None,
    resolution: float = 30.0,
    overwrite: bool = False,
    **kwargs: Any,  # forwarded to the worker (unused but kept for API parity)
) -> Future:
    """
    Reproject a raster using rioxarray/odc.geo on a Dask worker.

    Parameters
    ----------
    input_file : str
        Path to the source raster (must be accessible from each worker).
    output_file : str
        Destination path for the reprojected raster.
    target_epsg : int | str
        Target EPSG code (e.g. ``4326`` or ``"EPSG:4326"``).
    resolution : float
        Output pixel size in target units (default 30 m).
    resampling_method : str
        Resampling algorithm accepted by ``xr_reproject`` (`nearest`,
        `bilinear`, etc.).
    overwrite : bool, optional
        Skip the task if ``output_file`` exists and this is False.
    **kwargs
        Any additional keyword arguments are forwarded to the worker
        for future‑proofing and API consistency.

    Returns
    -------
    dask.distributed.Future
        Future that resolves to ``None`` once reprojection has finished.
    """
    # 1. Skip if output already exists
    if not overwrite and Path(output_file).exists():
        logging.warning(
            f"File {output_file} already exists and overwrite=False – "
            "skipping rioxarray warp."
        )
        # Use get_client() to get the current client if none provided
        current_client = get_client()
        return current_client.submit(lambda: None)

    # 2. Submit the worker function to a Dask worker
    return _reproject_raster_worker(
        input_file,
        output_file,
        target_epsg,
        resolution,
        resampling_method,
        **kwargs,
    )


# ------------------------------------------------------------------
# Helper that actually performs the reprojection on a worker
# ------------------------------------------------------------------
def _reproject_raster_worker(
    input_file: str,
    output_file: str,
    target_epsg: int | str | Any,
    resolution: float = 30.0,
    resampling_method: str = "nearest",
) -> None:
    """
    Reprojects a raster file to a specified EPSG code using
    rioxarray + odc.geo.xr.xr_reproject and writes the result with
    DEFLATE compression.

    Parameters
    ----------
    input_file : str
        Path to the input raster.
    output_file : str
        Destination path for the reprojected raster.
    target_epsg : int | str
        Target EPSG code (e.g. ``4326`` or ``"EPSG:4326"``).
    resolution : float
        Output pixel size in target units.
    resampling_method : str
        Resampling algorithm (`near`, `bilinear`, etc.).

    Returns
    -------
    None
    """
    # Import lazily – this keeps the worker lightweight if it never runs.
    import rioxarray
    from odc.geo.xr import xr_reproject

    # ------------------------------------------------------------------
    # 1. Load the raster on the worker
    # ------------------------------------------------------------------
    raster = rioxarray.open_rasterio(
        input_file,
        chunks="auto",  # let XArray/Dask decide optimal chunking
        cache=False,  # avoid keeping an in‑memory copy of the raw dataset
        lock=False,  # we will handle write locking explicitly below
    )

    # ------------------------------------------------------------------
    # 2. Reproject with odc.geo.xr
    # ------------------------------------------------------------------
    reproj = xr_reproject(
        src=raster,
        how=target_epsg,
        resolution=resolution,
        resampling=resampling_method,
    )

    # ------------------------------------------------------------------
    # 3. Write out the result – use a Dask lock to avoid concurrent writes.
    # ------------------------------------------------------------------
    # Grab the local client so that we can create a distributed Lock
    from dask.distributed import Lock

    reproj.rio.to_raster(
        output_file,
        driver="GTiff",
        compress="DEFLATE",
        predictor=2,
        bigtiff="YES",
        tiled=True,
        lock=Lock("rio"),
    )

    # Explicitly close references – not strictly required but tidy.
    del raster
    del reproj
