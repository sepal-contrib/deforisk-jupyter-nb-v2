from pathlib import Path
from enum import Enum
from typing import Any, List, Optional, Union, TYPE_CHECKING
import ee
import geemap
import rioxarray
import odc.geo.xr  # Registers .odc accessor for xarray/rioxarray do not remove
from pydantic import BaseModel, field_validator, model_validator, ConfigDict
from pydantic import Field

from component.script.gee.ee_raster_export import download_ee_image
from component.script.processing import (
    display_raster,
    distance_to_edge_gdal_no_mask,
    reproject_raster_gdal_warp,
    xr_rasterize,
)
from component.script.utilities.file_helpers import copy_and_rename_file

if TYPE_CHECKING:
    from component.script.project import Project


class DataType(str, Enum):
    vector = "vector"
    raster = "raster"


class RasterizationMethod(str, Enum):
    binary = "binary"
    unique = "unique"


class RasterType(str, Enum):
    continuous = "continuous"
    categorical = "categorical"


class PostProcessing(str, Enum):
    edge = "edge"
    dist = "dist"


# ==== LocalVectorVar ====


# ==== LocalRasterVar ====
class LocalRasterVar(Variable):
    """
    Local filesystem-based raster variable.
    - Handles raster data (.tif, etc.)
    - Can be reprojected and post-processed
    - Use add_as_raw() or add_as_processed() to register to project
    """

    path: Path
    data_type: DataType = Field(default=DataType.raster, frozen=True)
    raster_type: RasterType
    post_processing: List[PostProcessing] = []
    default_crs: Optional[str] = None
    default_resolution: Optional[float] = None

    def show(self):
        display_raster(self.name, self.path)

    def add_as_raw(self, auto_save: bool = True) -> "LocalRasterVar":
        """
        Add this variable to the project's raw_variables collection.

        Raw variables are typically unprocessed data downloaded from sources
        like Google Earth Engine or other data providers.

        Parameters
        ----------
        auto_save : bool, optional
            If True (default), automatically saves the project after adding the variable.

        Returns
        -------
        LocalRasterVar
            Returns self for method chaining.

        Raises
        ------
        ValueError
            If the variable is not associated with a project.

        Examples
        --------
        >>> var = LocalRasterVar(name="dem", ..., project=project)
        >>> var.add_as_raw()  # Auto-saves by default
        >>> var.add_as_raw(auto_save=False)  # Skip auto-save
        """
        if self.project is None:
            raise ValueError(
                "Cannot add to project: this variable is not associated with a project. "
                "Please set the 'project' parameter when creating the variable."
            )

        self.project.raw_variables[self.name] = self
        print(f"✓ Added '{self.name}' to raw variables")

        if auto_save:
            self.project.save()

        return self

    def add_as_processed(self, auto_save: bool = True) -> "LocalRasterVar":
        """
        Add this variable to the project's processed variables collection.

        Processed variables are typically derived from raw variables through
        operations like reprojection, rasterization, or post-processing.

        Parameters
        ----------
        auto_save : bool, optional
            If True (default), automatically saves the project after adding the variable.

        Returns
        -------
        LocalRasterVar
            Returns self for method chaining.

        Raises
        ------
        ValueError
            If the variable is not associated with a project.

        Examples
        --------
        >>> reprojected = raw_var.reproject(...).add_as_processed()  # Auto-saves
        >>> reprojected = raw_var.reproject(...).add_as_processed(auto_save=False)
        """
        if self.project is None:
            raise ValueError(
                "Cannot add to project: this variable is not associated with a project. "
                "Please set the 'project' parameter when creating the variable."
            )

        self.project.variables[self.name] = self
        print(f"✓ Added '{self.name}' to processed variables")

        if auto_save:
            self.project.save()

        return self

    def download(self):
        """Copy from the default path to the project."""
        copy_and_rename_file(self.path)

    def get_base_geobox(self):
        """
        Get the geobox from this raster file to use as a spatial reference.

        Returns
        -------
        odc.geo.geobox.GeoBox
            The geobox defining spatial properties of this raster.
        """
        if not self.path.exists():
            raise FileNotFoundError(f"Raster file not found: {self.path}")

        raster_array = rioxarray.open_rasterio(
            str(self.path),
            chunks="auto",
            cache=False,
            lock=False,
        )

        # Return GeoBox using the .odc accessor (registered by odc.geo.xr import)
        return raster_array.odc.geobox

    def reproject(
        self,
        target_epsg: str,
        resolution: Optional[float] = None,
        resampling: Optional[str] = None,
        **kwargs,
    ) -> "LocalRasterVar":
        """
        Reproject this raster layer to a target EPSG coordinate system.
        Returns a new LocalRasterVar with the reprojected data.

        Parameters
        ----------
        target_epsg : str
            Target EPSG code (e.g., 'EPSG:4326' or '4326').
        resolution : float, optional
            Target resolution. If None, uses default_resolution or 30.
        resampling : str, optional
            Resampling method. If None, automatically selects based on raster_type:
            - categorical: 'near' (nearest neighbor)
            - continuous: 'bilinear'
        **kwargs
            Additional keyword arguments to pass to reproject_raster_gdal_warp.

        Returns
        -------
        LocalRasterVar
            A new LocalRasterVar instance with the reprojected data.
        """
        # Normalize EPSG format
        if not target_epsg.startswith("EPSG:"):
            target_epsg = f"EPSG:{target_epsg}"

        # Determine output path using project folders
        output_folder = (
            self.project.folders.processed_data_folder if self.project else Path.cwd()
        )
        output_folder.mkdir(parents=True, exist_ok=True)
        output_path = output_folder / f"{self.name}_reprojected.tif"

        # Determine resolution
        _resolution = resolution or self.default_resolution or 30

        # Auto-select resampling method based on raster_type if not specified
        if resampling is None:
            if self.raster_type == RasterType.categorical:
                resampling = "near"
            elif self.raster_type == RasterType.continuous:
                resampling = "bilinear"
            else:
                resampling = "near"  # default fallback

        # Reproject
        reproject_raster_gdal_warp(
            input_file=str(self.path),
            output_file=str(output_path),
            target_epsg=target_epsg,
            resolution=_resolution,
            resampling_method=resampling,
            **kwargs,
        )

        # Create and return new LocalRasterVar using model_construct to bypass validation
        return LocalRasterVar.model_construct(
            name=f"{self.name}_reprojected",
            raster_type=self.raster_type,
            path=output_path,
            default_crs=target_epsg,
            default_resolution=_resolution,
            post_processing=self.post_processing,
            project=self.project,
            data_type=DataType.raster,
            active=True,
        )

    def apply_post_processing(self, post_process: PostProcessing) -> "LocalRasterVar":
        """
        Apply a post-processing step to this raster and create a new LocalRasterVar.

        Parameters
        ----------
        post_process : PostProcessing
            Post-processing step to apply ('edge' or 'dist').

        Returns
        -------
        LocalRasterVar
            A new LocalRasterVar with the post-processing applied.
        """
        if post_process == PostProcessing.edge:
            return self._create_post_var(
                self.path,
                suffix="edge",
                raster_type=RasterType.categorical,
            )
        elif post_process == PostProcessing.dist:
            return self._create_post_var(
                self.path,
                suffix="dist",
                raster_type=RasterType.continuous,
            )
        else:
            raise ValueError(f"Unknown post-processing step: {post_process}")

    def _apply_post(self, raster_path: Path) -> List["LocalRasterVar"]:
        """
        Create new LocalRasterVars for each post-processing step defined in self.post_processing.
        These are registered automatically via model_post_init.

        Returns
        -------
        List[LocalRasterVar]
            List of new LocalRasterVar instances for post-processing outputs.
        """
        post_vars = []
        for step in self.post_processing:
            if step == PostProcessing.edge:
                post_var = self._create_post_var(
                    raster_path, suffix="edge", raster_type=RasterType.categorical
                )
                post_vars.append(post_var)
            elif step == PostProcessing.dist:
                post_var = self._create_post_var(
                    raster_path, suffix="dist", raster_type=RasterType.continuous
                )
                post_vars.append(post_var)
        return post_vars

    def _create_post_var(
        self, base_path: Path, suffix: str, raster_type: RasterType
    ) -> "LocalRasterVar":
        """
        Create a new LocalRasterVar for post-processing output and execute the processing.

        Parameters
        ----------
        base_path : Path
            The base raster path to use as input.
        suffix : str
            Suffix to add to the variable name and file (e.g., 'edge', 'dist').
        raster_type : RasterType
            The type of raster (continuous or categorical).

        Returns
        -------
        LocalRasterVar
            A new LocalRasterVar instance.
        """
        # Determine output path
        output_folder = self.project.folders.processed_data_folder
        output_folder.mkdir(parents=True, exist_ok=True)
        output_path = output_folder / f"{self.name}_{suffix}.tif"

        # Perform the actual post-processing
        if suffix == "edge":
            # Edge detection: creates a binary raster with edges marked
            # Uses scipy's binary_erosion to detect edges
            import rasterio
            import numpy as np
            from scipy.ndimage import binary_erosion

            # Read the input raster
            with rasterio.open(base_path) as src:
                data = src.read(1)
                profile = src.profile.copy()

                # Create binary mask (features are non-zero)
                binary_mask = data > 0

                # Erode the mask
                eroded = binary_erosion(binary_mask, structure=np.ones((3, 3)))

                # Edge is the difference between original and eroded
                edges = binary_mask.astype(np.uint8) - eroded.astype(np.uint8)

                # Update profile for output
                profile.update(
                    dtype=rasterio.uint8, compress="deflate", predictor=2, bigtiff="yes"
                )

                # Write the edge raster
                with rasterio.open(output_path, "w", **profile) as dst:
                    dst.write(edges, 1)

        elif suffix == "dist":
            # Distance calculation: calculates distance from features
            distance_to_edge_gdal_no_mask(
                input_file=str(base_path),
                dist_file=str(output_path),
                values=0,  # Calculate distance from 0 values (non-feature pixels)
                nodata=0,
                max_distance_value=4294967295,
                input_nodata=True,
                verbose=False,
            )

        # Create and return the new LocalRasterVar
        post_var = LocalRasterVar.model_construct(
            name=f"{self.name}_{suffix}",
            raster_type=raster_type,
            path=output_path,
            project=self.project,
            default_crs=self.default_crs,
            default_resolution=self.default_resolution,
            data_type=DataType.raster,
            active=True,
            post_processing=self.post_processing + [PostProcessing(suffix)],
        )

        return post_var

    def to_gee_var(self) -> ee.Image:
        """
        Convert a local raster to a Google Earth Engine Image.

        Note: This requires uploading to GEE assets first.

        Raises
        ------
        NotImplementedError
            Raster upload to GEE requires manual asset upload.
        """
        if not self.path.exists():
            raise FileNotFoundError(f"Local file not found: {self.path}")

        raise NotImplementedError(
            "Converting local raster to GEE Image requires uploading to GEE assets. "
            "Please use GEE's asset upload tools or geemap.upload_to_gee() manually."
        )

    def use_as_base_raster(self, auto_save: bool = True) -> "LocalRasterVar":
        """
        Set this raster as the base raster for the project.

        The base raster is used as a spatial reference for reprojecting and warping
        other layers to match its extent, resolution, and CRS.

        Parameters
        ----------
        auto_save : bool, optional
            If True (default), automatically saves the project after setting the base raster.

        Returns
        -------
        LocalRasterVar
            Returns self for method chaining.

        Raises
        ------
        ValueError
            If the variable is not associated with a project.
        FileNotFoundError
            If the raster file doesn't exist.

        Examples
        --------
        >>> dem.use_as_base_raster()  # Auto-saves by default
        >>> dem.use_as_base_raster(auto_save=False)  # Skip auto-save
        """
        if self.project is None:
            raise ValueError(
                "Cannot set as base raster: this variable is not associated with a project. "
                "Please ensure the variable has a project reference."
            )

        if not self.path.exists():
            raise FileNotFoundError(
                f"Cannot set as base raster: raster file not found at {self.path}"
            )

        # Set this raster as the project's base raster
        self.project.base_raster = self

        print(
            f"✓ Set '{self.name}' as base raster for project '{self.project.project_name}'"
        )
        print(f"  Path: {self.path}")
        print(f"  CRS: {self.default_crs or 'Not specified'}")
        print(f"  Resolution: {self.default_resolution or 'Not specified'}")

        if auto_save:
            self.project.save()

        return self


class GEEVar(Variable):
    """
    Google Earth Engine-backed variable.
    - Accepts either an asset id (gee_image as str or path as asset id) or an ee.Image object.
    - Optional local output path; if missing → defaults to CWD/<name>[_year].tif
    """

    path: Optional[Union[Path, str]] = None  # local target OR asset id
    gee_images: Optional[List[Union[str, Any]]] = Field(default=None, repr=False)
    default_scale: Optional[float] = None
    default_crs: Optional[str] = None

    # Optional metadata that will be used when converting to LocalVar
    raster_type: Optional[RasterType] = None  # for raster data
    rasterization_method: Optional[RasterizationMethod] = None  # for vector data
    post_processing: List[PostProcessing] = []  # for raster data

    @model_validator(mode="after")
    def _chk_source(self):
        if self.gee_images is not None:
            return self
        if isinstance(self.path, str) and (
            self.path.startswith("users/") or self.path.startswith("projects/")
        ):
            return self
        raise ValueError(
            "GEEVar needs `gee_images` (ee.Image or asset id str list) or `path` set to an asset id."
        )

    def _download(
        self,
        overwrite: bool = False,
    ) -> List[Path]:
        """
        Internal method to download GEE data to local file(s).

        Parameters
        ----------
        overwrite : bool, optional
            Whether to overwrite existing files (default: False).

        Returns
        -------
        List[Path]
            List of paths to the downloaded files.
        """
        output_path: Path = None
        extensions = {"vector": ".shp", "raster": ".tif"}

        # Ensure gee_images is set
        if not self.gee_images:
            raise ValueError("gee_images must be provided for download.")

        # Get the output folder
        output_folder = self.project.folders.data_raw_folder
        output_folder.mkdir(parents=True, exist_ok=True)

        local_paths = []

        # Process images

        output_path = output_folder / f"{self.name}"

        output_path = output_path.with_suffix(extensions[self.data_type])
        local_paths.append(output_path)

        if overwrite or not output_path.exists():

            if self.data_type == DataType.vector:
                geemap.ee_export_vector(
                    self.gee_images[0],
                    output_path,
                    selectors=["gaul0_name", "iso3_code"],
                    keep_zip=False,
                    timeout=600,
                    verbose=False,
                )

            elif self.data_type == DataType.raster:
                download_ee_image(
                    self.gee_images[0],
                    output_path,
                    scale=self.default_scale or 30,
                    crs=self.default_crs or "EPSG:4326",
                    region=self.aoi.geometry(),
                    overwrite=True,
                    unmask_value=255,
                    nodata_value=255,
                )
        elif output_path.exists():
            print(f"{output_path} already exists. Skipping download.")

        # Verify all files were downloaded
        for local_path in local_paths:
            if not local_path.exists():
                raise FileNotFoundError(
                    f"Download failed: file does not exist at {local_path}."
                )

        return local_paths

    def to_local_vector(
        self,
        overwrite: bool = False,
    ) -> Union["LocalVectorVar", List["LocalVectorVar"]]:
        """
        Download from GEE and convert to LocalVectorVar(s).
        Only works for vector data types.

        Parameters
        ----------
        overwrite : bool, optional
            Whether to overwrite existing files (default: False).

        Returns
        -------
        LocalVectorVar or List[LocalVectorVar]
            A LocalVectorVar instance if single image, or a list of LocalVectorVar instances
            if multiple images were downloaded.

        Raises
        ------
        ValueError
            If data_type is not 'vector'.
        """
        if self.data_type != DataType.vector:
            raise ValueError(
                f"to_local_vector() can only be used with vector data types, got '{self.data_type}'"
            )

        # Download the vector data
        local_paths = self._download(overwrite=overwrite)

        # Create LocalVectorVar instances for each downloaded file
        local_vars = []
        for i, local_path in enumerate(local_paths):

            local_var = LocalVectorVar.model_construct(
                name=self.name,
                rasterization_method=self.rasterization_method,
                path=local_path,
                default_crs=self.default_crs,
                project=self.project,
                data_type=DataType.vector,
                active=True,
                tags=self.tags.copy(),  # Copy tags from GEEVar
                year=self.year,
            )
            local_vars.append(local_var)

        # Return single instance if only one, otherwise return list
        return local_vars[0] if len(local_vars) == 1 else local_vars

    def to_local_raster(
        self,
        raster_type: Optional[RasterType] = None,
        rasterization_method: Optional[RasterizationMethod] = None,
        base: Optional["LocalRasterVar"] = None,
        post_processing: Optional[List[PostProcessing]] = None,
        overwrite: bool = False,
        **rasterize_kwargs,
    ) -> Union["LocalRasterVar", List["LocalRasterVar"]]:
        """
        Download from GEE and convert to LocalRasterVar(s).

        - For raster data: downloads directly and returns LocalRasterVar(s).
        - For vector data: downloads as LocalVectorVar(s), then rasterizes to LocalRasterVar(s).

        Parameters
        ----------
        raster_type : RasterType, optional
            Required if data_type is 'raster' and not set in GEEVar.
            If not provided, uses self.raster_type.
        rasterization_method : RasterizationMethod, optional
            Required if data_type is 'vector' and not set in GEEVar.
            Used when rasterizing vector data.
        base : LocalRasterVar, optional
            Required if data_type is 'vector'. The base raster to use as spatial reference
            for rasterization.
        post_processing : List[PostProcessing], optional
            Post-processing steps to apply (only for rasters).
            If not provided, uses self.post_processing.
        overwrite : bool, optional
            Whether to overwrite existing files (default: False).
        **rasterize_kwargs
            Additional keyword arguments passed to LocalVectorVar.rasterize().

        Returns
        -------
        LocalRasterVar or List[LocalRasterVar]
            A LocalRasterVar instance if single image, or a list of LocalRasterVar instances
            if multiple images were downloaded.

        Raises
        ------
        ValueError
            If required parameters are missing based on data_type.
        """
        if self.data_type == DataType.vector:
            # For vector data: download as LocalVectorVar, then rasterize
            if base is None:
                raise ValueError(
                    "base (LocalRasterVar) must be provided when converting a vector GEEVar to raster"
                )

            # Use provided rasterization_method or fall back to self's value
            _rasterization_method = rasterization_method or self.rasterization_method
            if _rasterization_method is None:
                raise ValueError(
                    "rasterization_method must be provided either as parameter or set in GEEVar when converting vector to raster"
                )

            # Download as LocalVectorVar(s)
            local_vectors = self.to_local_vector(overwrite=overwrite)

            # Ensure we have a list to iterate over
            if not isinstance(local_vectors, list):
                local_vectors = [local_vectors]

            # Rasterize each LocalVectorVar to create LocalRasterVar(s)
            local_rasters = []
            for local_vector in local_vectors:
                local_raster = local_vector.rasterize(
                    base=base,
                    rasterization_method=_rasterization_method,
                    **rasterize_kwargs,
                )
                local_rasters.append(local_raster)

            # Return single instance if only one, otherwise return list
            return local_rasters[0] if len(local_rasters) == 1 else local_rasters

        else:  # DataType.raster
            # For raster data: download directly
            local_paths = self._download(overwrite=overwrite)

            # Use provided raster_type or fall back to self's value
            _raster_type = raster_type or self.raster_type
            if _raster_type is None:
                raise ValueError(
                    "raster_type must be provided either as parameter or set in GEEVar when converting a raster GEEVar to LocalRasterVar"
                )

            # Use provided post_processing or fall back to self's value
            _post_processing = (
                post_processing if post_processing is not None else self.post_processing
            )

            # Create LocalRasterVar instances for each downloaded file
            local_vars = []
            for i, local_path in enumerate(local_paths):

                local_var = LocalRasterVar.model_construct(
                    name=self.name,
                    raster_type=_raster_type,
                    post_processing=_post_processing,
                    path=local_path,
                    default_crs=self.default_crs,
                    default_resolution=self.default_scale,
                    project=self.project,
                    data_type=DataType.raster,
                    active=True,
                    tags=self.tags.copy(),  # Copy tags from GEEVar
                    year=self.year,
                )
                local_vars.append(local_var)

            # Return single instance if only one, otherwise return list
            return local_vars[0] if len(local_vars) == 1 else local_vars


# Rebuild models after Project is imported to resolve forward references
try:
    from component.script.project import Project

    # Rebuild Variable classes first
    Variable.model_rebuild()
    LocalVectorVar.model_rebuild()
    LocalRasterVar.model_rebuild()
    GEEVar.model_rebuild()

    # Then rebuild Project to ensure it sees the updated Variable classes
    Project.model_rebuild()
except ImportError:
    pass  # Project not yet available
