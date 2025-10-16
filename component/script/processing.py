from typing import TYPE_CHECKING, List
import numpy as np
from osgeo import gdal
import rasterio
import rioxarray
import xarray as xr

if TYPE_CHECKING:
    from component.script.project import Project
    from component.script.variables import LocalRasterVar


def reproject_raster_gdal_warp(
    input_file: str,
    output_file: str,
    target_epsg: str,
    resolution: int | float = 30,
    resampling_method: str = "near",
) -> None:
    """
    Reprojects a raster file to a specified EPSG code using GDAL and saves it with DEFLATE compression.

    Parameters:
    input_file (str): The path to the input raster file.
    output_file (str): The path where the reprojected raster file will be saved.
    target_epsg (str): The EPSG code of the target coordinate reference system (e.g., 'EPSG:4326').
    resolution (int | float): Target resolution in the units of the target CRS. Default is 30.
    resampling_method (str): Resampling algorithm ('near', 'bilinear', 'cubic', etc.). Default is 'near'.

    Returns:
    None
    """
    import os

    # Enable GDAL exceptions to catch errors
    gdal.UseExceptions()

    # Open the input dataset
    dataset = gdal.Open(input_file)
    if not dataset:
        raise FileNotFoundError(f"Input file {input_file} not found.")

    # Get projection from the original raster
    src_proj = dataset.GetProjection()

    # For small files, don't use BIGTIFF as it can cause issues
    file_size_mb = os.path.getsize(input_file) / (1024 * 1024)
    creation_options = ["COMPRESS=DEFLATE", "PREDICTOR=2"]
    if file_size_mb > 2000:  # Only use BIGTIFF for files larger than 2GB
        creation_options.append("BIGTIFF=YES")

    param = gdal.WarpOptions(
        warpOptions=["overwrite"],
        srcSRS=src_proj,
        dstSRS=target_epsg,
        targetAlignedPixels=True,
        resampleAlg=resampling_method,
        xRes=resolution,
        yRes=resolution,
        multithread=True,
        creationOptions=creation_options,
    )

    # Perform reprojection
    result = gdal.Warp(output_file, input_file, format="GTiff", options=param)

    if result is None:
        raise RuntimeError("gdal.Warp() failed - check input file and parameters")

    # Close datasets
    result = None
    dataset = None


def xr_rasterize(
    shapefile_path: str = None,
    geobox=None,
    crs=None,
    output_path: str = None,
    mode: str = "binary",
    **rasterio_kwargs,
):
    """
    Rasterizes a vector shapefile into a raster array.

    This function provides unified functionality for both binary and unique ID rasterization.

    Parameters
    ----------
    shapefile_path : str
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
    mode : str, optional
        Rasterization mode: 'binary' or 'unique'.
        - 'binary': Creates a boolean raster with 1s and 0s (default)
        - 'unique': Creates a raster with unique integer IDs for each feature
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
    from odc.geo import xr

    # Read the shapefile
    gdf = gpd.read_file(filename=shapefile_path, engine="fiona")

    # Reproject vector data to raster's CRS
    gdf_reproj = gdf.to_crs(crs=geobox.crs)

    # Handle different modes
    if mode == "binary":
        # Binary mode: rasterize into a boolean array with 1s and 0s
        shapes = gdf_reproj.geometry
        values = [1] * len(gdf_reproj)  # All features set to 1
        shapes_and_values = list(zip(shapes, values))

    elif mode == "unique":
        # Unique ID mode: rasterize using unique integer IDs for each feature
        shapes = gdf_reproj.geometry
        # Create unique integer IDs starting from 1
        values = list(range(1, len(gdf_reproj) + 1))
        shapes_and_values = list(zip(shapes, values))

    else:
        raise ValueError("Mode must be either 'binary' or 'unique'")

    # Rasterize shapes into a numpy array
    im = rasterio.features.rasterize(
        shapes=shapes_and_values if mode == "unique" else shapes,
        out_shape=geobox.shape,
        transform=geobox.transform,
        dtype="uint8",
        **rasterio_kwargs,
    )

    # Convert numpy array to a full xarray.DataArray
    # and set array name if supplied
    da_rasterized = xr.wrap_xr(im=im, gbox=geobox)

    da_rasterized.rio.to_raster(
        output_path,
        driver="GTiff",
        compress="DEFLATE",
        predictor=2,
        bigtiff="YES",
        tiled=True,
    )

    # Explicitly close references – not strictly required but tidy.
    del im
    del da_rasterized


def distance_to_edge_gdal_no_mask(
    input_file,
    dist_file,
    values=0,
    nodata=0,
    max_distance_value=4294967295,
    input_nodata=True,
    verbose=False,
):
    """Computes the shortest distance to given pixel values in a raster,
    while preserving the original nodata mask in the output."""

    # Read input file
    src_ds = gdal.Open(input_file)
    srcband = src_ds.GetRasterBand(1)

    # Create raster of distance
    drv = gdal.GetDriverByName("GTiff")
    dst_ds = drv.Create(
        dist_file,
        src_ds.RasterXSize,
        src_ds.RasterYSize,
        1,
        gdal.GDT_UInt32,
        ["COMPRESS=DEFLATE", "PREDICTOR=2", "BIGTIFF=YES"],
    )
    dst_ds.SetGeoTransform(src_ds.GetGeoTransform())
    dst_ds.SetProjection(src_ds.GetProjection())
    dstband = dst_ds.GetRasterBand(1)

    # Use_input_nodata
    ui_nodata = "YES" if input_nodata else "NO"

    # Compute distance
    val = "VALUES=" + str(values)
    use_input_nodata = "USE_INPUT_NODATA=" + ui_nodata
    max_distance = "MAXDIST=" + str(max_distance_value)
    distance_nodata = "NODATA=" + str(nodata)
    cb = gdal.TermProgress_nocb if verbose else 0
    gdal.ComputeProximity(
        srcband,
        dstband,
        [val, use_input_nodata, max_distance, distance_nodata, "DISTUNITS=GEO"],
        callback=cb,
    )

    # Set nodata value
    dstband.SetNoDataValue(max_distance_value)

    # Flush to disk
    dstband.FlushCache()
    dst_ds.FlushCache()

    # Clean up
    srcband = None
    dstband = None
    del src_ds, dst_ds


def process_forest_loss_xarray(input1_path, input2_path, output_path):
    # Open the input rasters
    input1 = rioxarray.open_rasterio(
        input1_path,
        chunks="auto",
        cache=False,
        lock=False,
    ).squeeze()
    input2 = rioxarray.open_rasterio(
        input2_path,
        chunks="auto",
        cache=False,
        lock=False,
    ).squeeze()

    # Check bounds properly - extract bounds tuple values
    bounds1 = input1.rio.bounds()
    bounds2 = input2.rio.bounds()

    if not (
        bounds1[0] <= bounds2[0]  # left
        and bounds1[2] >= bounds2[2]  # right
        and bounds1[3] >= bounds2[3]  # top
        and bounds1[1] <= bounds2[1]  # bottom
    ):
        raise ValueError(
            "The bounds of input1 must be equal to or larger than those of input2."
        )

    # Create masks for valid data
    nodata1 = input1.rio.nodata
    nodata2 = input2.rio.nodata
    valid_mask = (input1 != nodata1) & (input2 != nodata2)

    # Create output based on conditions using xarray operations
    output = xr.where(
        valid_mask & (input1 == 1) & (input2 == 0),
        0,  # condition 0: input1 == 1 and input2 == 0
        xr.where(
            valid_mask & (input1 == 1) & (input2 == 1),
            1,  # condition 1: input1 == 1 and input2 == 1
            255,  # nodata for all other cases
        ),
    ).astype("uint8")

    # Set proper metadata
    output.rio.write_nodata(255, inplace=True)
    output.rio.write_crs(input1.rio.crs, inplace=True)
    output.rio.write_transform(input1.rio.transform(), inplace=True)

    output.rio.to_raster(
        output_path,
        driver="GTiff",
        compress="DEFLATE",
        predictor=2,
        bigtiff="YES",
        tiled=True,
    )


def generate_deforestation_raster(
    raster_1: "LocalRasterVar",
    raster_2: "LocalRasterVar",
    raster_3: "LocalRasterVar",
    project: "Project",
):
    """
    Generate a deforestation raster from three input rasters.

    Parameters:
    - raster1_path: Path to the first raster file (period 1).
    - raster2_path: Path to the second raster file (period 2).
    - raster3_path: Path to the third raster file (period 3).
    - output_path: Path to save the output raster file.
    """
    # Import here to avoid circular dependency
    from component.script.variables import LocalRasterVar

    # Open the input rasters
    with (
        rasterio.open(raster_1.path) as src1,
        rasterio.open(raster_2.path) as src2,
        rasterio.open(raster_3.path) as src3,
    ):
        # Read the data into numpy arrays
        raster1 = src1.read(1)
        raster2 = src2.read(1)
        raster3 = src3.read(1)

        # Create an output array initialized with NoData value (0)
        output_raster = np.zeros_like(raster1, dtype=np.uint8)

        # Set the values based on deforestation periods
        output_raster[(raster1 == 1) & (raster2 == 0)] = (
            1  # Deforestation in period 1-2
        )
        output_raster[(raster2 == 1) & (raster3 == 0)] = (
            2  # Deforestation in period 2-3
        )
        # Set the remaining forest value only where no deforestation has been marked
        output_raster[(output_raster == 0) & (raster3 == 1)] = (
            3  # Remaining forest in period 3
        )

    # Define the metadata for the output raster
    meta = src1.meta
    meta.update({"count": 1, "dtype": np.uint8, "nodata": 0, "compress": "deflate"})

    # extract years data from each of the rasters
    years = [raster_1.year, raster_2.year, raster_3.year]

    output_name = "defostack" + "_".join(map(str, years)) + ".tif"
    output_path = str(project.folders.data_raw_folder / output_name)

    # Write the output raster to a file
    with rasterio.open(output_path, "w", **meta) as dst:
        dst.write(output_raster, 1)

    return LocalRasterVar(
        name="defostack",
        path=output_path,
        raster_type="categorical",
        project=project,
        tags=["deforestation"],
    )


def get_forest_loss_calculated(
    project: "Project", forest_layers: List["LocalRasterVar"]
) -> List["LocalRasterVar"]:
    """
    Calculate forest loss between different time periods.

    Creates three forest loss rasters:
    1. Loss between year[0] and year[1]
    2. Loss between year[0] and year[2]
    3. Loss between year[1] and year[2]

    Parameters
    ----------
    project : Project
        The project containing configuration and years
    forest_layers : List[LocalRasterVar]
        List of forest raster layers, one for each year in project.years
        Should be in chronological order

    Returns
    -------
    List[LocalRasterVar]
        Three LocalRasterVar objects for the generated forest loss rasters
    """
    # Import here to avoid circular dependency
    from component.script.variables import LocalRasterVar
    from pathlib import Path

    years = project.years

    # Validate input
    if len(forest_layers) != len(years):
        raise ValueError(
            f"Number of forest layers ({len(forest_layers)}) must match "
            f"number of years ({len(years)})"
        )

    if len(years) < 3:
        raise ValueError(f"At least 3 years are required, got {len(years)}")

    # Sort forest layers by year (assuming they correspond to project.years in order)
    # Extract paths from LocalRasterVar objects
    sorted_raster_files = [str(layer.path) for layer in forest_layers]

    # Define output filenames with meaningful names
    forest_loss1_filename = str(
        project.folders.data_raw_folder / f"forest_loss_{years[0]}_{years[1]}.tif"
    )
    forest_loss2_filename = str(
        project.folders.data_raw_folder / f"forest_loss_{years[0]}_{years[2]}.tif"
    )
    forest_loss3_filename = str(
        project.folders.data_raw_folder / f"forest_loss_{years[1]}_{years[2]}.tif"
    )

    print(f"Calculating forest loss between {years[0]} and {years[1]}...")
    process_forest_loss_xarray(
        sorted_raster_files[0], sorted_raster_files[1], forest_loss1_filename
    )

    print(f"Calculating forest loss between {years[0]} and {years[2]}...")
    process_forest_loss_xarray(
        sorted_raster_files[0], sorted_raster_files[2], forest_loss2_filename
    )

    print(f"Calculating forest loss between {years[1]} and {years[2]}...")
    process_forest_loss_xarray(
        sorted_raster_files[1], sorted_raster_files[2], forest_loss3_filename
    )

    print("✓ Forest loss calculation complete!")

    # Create LocalRasterVar objects for the output files
    from component.script.variables import RasterType

    forest_loss_vars = [
        LocalRasterVar(
            name=f"forest_loss_{years[0]}_{years[1]}",
            path=Path(forest_loss1_filename),
            raster_type=RasterType.categorical,
            project=project,
            tags=["forest_loss"],
        ),
        LocalRasterVar(
            name=f"forest_loss_{years[0]}_{years[2]}",
            path=Path(forest_loss2_filename),
            raster_type=RasterType.categorical,
            project=project,
            tags=["forest_loss"],
        ),
        LocalRasterVar(
            name=f"forest_loss_{years[1]}_{years[2]}",
            path=Path(forest_loss3_filename),
            raster_type=RasterType.categorical,
            project=project,
            tags=["forest_loss"],
        ),
    ]

    return forest_loss_vars


def display_raster(name, path):
    import matplotlib.pyplot as plt
    import rasterio

    # Open the raster file using rasterio
    with rasterio.open(path) as src:
        # Read the first band of the raster data
        raster_data = src.read(1)

        # Create a plot for the raster data
        plt.figure(figsize=(8, 8))
        plt.imshow(raster_data, cmap="viridis")  # You can choose a different colormap
        plt.colorbar()  # Optional, to add a color bar to the plot

        # Add the name of the dataset as a title
        plt.title(f"Raster Layer: {name}")

        # Show the plot
        plt.show()
