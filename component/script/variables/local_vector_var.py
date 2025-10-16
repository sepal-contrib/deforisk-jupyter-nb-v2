from pathlib import Path
from typing import Optional

import ee
import geemap
from pydantic import Field
from component.script.processing import xr_rasterize
from component.script.utilities.file_helpers import copy_and_rename_file
from component.script.variables.models import DataType, RasterType, RasterizationMethod
from component.script.variables.variable import Variable
from component.script.variables.local_raster_var import LocalRasterVar


class LocalVectorVar(Variable):
    """
    Local filesystem-based vector variable.
    - Handles vector data (.shp, .geojson, etc.)
    - Can be rasterized to create LocalRasterVar
    - Use add_as_raw() or add_as_processed() to register to project
    """

    path: Path
    data_type: DataType = Field(default=DataType.vector, frozen=True)
    rasterization_method: Optional[RasterizationMethod] = None
    default_crs: Optional[str] = None

    def add_as_raw(self, auto_save: bool = True) -> "LocalVectorVar":
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
        LocalVectorVar
            Returns self for method chaining.

        Raises
        ------
        ValueError
            If the variable is not associated with a project.
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

    def add_as_processed(self, auto_save: bool = True) -> "LocalVectorVar":
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
        LocalVectorVar
            Returns self for method chaining.

        Raises
        ------
        ValueError
            If the variable is not associated with a project.
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

    def rasterize(
        self,
        base: "LocalRasterVar",
        rasterization_method: Optional[RasterizationMethod] = None,
        **kwargs,
    ) -> "LocalRasterVar":
        """
        Rasterize this vector layer using a base raster as spatial reference.

        Parameters
        ----------
        base : LocalRasterVar
            A LocalRasterVar to use as spatial reference.
        rasterization_method : RasterizationMethod, optional
            How to rasterize the vector data. If not provided, uses self.rasterization_method.
            Must be provided either here or when creating the LocalVectorVar.
        **kwargs
            Additional keyword arguments to pass to xr_rasterize.

        Returns
        -------
        LocalRasterVar
            A new LocalRasterVar instance.
        """
        if not isinstance(base, LocalRasterVar):
            raise ValueError("base must be a LocalRasterVar instance")

        # Use provided rasterization_method or fall back to self's value
        _rasterization_method = rasterization_method or self.rasterization_method
        if _rasterization_method is None:
            raise ValueError(
                "rasterization_method must be provided either as parameter or set in LocalVectorVar"
            )

        # Get geobox from base raster
        geobox = base.get_base_geobox()

        # Determine output path using project folders
        output_folder = (
            self.project.folders.processed_data_folder if self.project else Path.cwd()
        )
        output_folder.mkdir(parents=True, exist_ok=True)
        output_path = output_folder / f"{self.name}_rasterized.tif"

        # Map RasterizationMethod to mode parameter
        mode_mapping = {
            RasterizationMethod.binary: "binary",
            RasterizationMethod.unique: "unique",
        }
        mode = mode_mapping.get(_rasterization_method, "binary")

        # Rasterize
        xr_rasterize(
            shapefile_path=str(self.path),
            geobox=geobox,
            output_path=str(output_path),
            mode=mode,
            **kwargs,
        )

        # Create and return LocalRasterVar
        raster_type = (
            RasterType.categorical if mode == "unique" else RasterType.continuous
        )

        return LocalRasterVar.model_construct(
            name=f"{self.name}_rasterized",
            raster_type=raster_type,
            path=output_path,
            default_crs=self.default_crs,
            project=self.project,
            data_type=DataType.raster,
            active=True,
        )

    def to_gee_var(self) -> ee.FeatureCollection:
        """
        Convert this local vector to a Google Earth Engine FeatureCollection.

        Returns
        -------
        ee.FeatureCollection
            An Earth Engine FeatureCollection.

        Raises
        ------
        FileNotFoundError
            If the local file does not exist.
        """
        if not self.path.exists():
            raise FileNotFoundError(f"Local file not found: {self.path}")

        return geemap.shp_to_ee(str(self.path))


# Rebuild models after Project is imported to resolve forward references
try:
    from component.script.project import Project

    # Rebuild Variable classes first
    Variable.model_rebuild()
    LocalVectorVar.model_rebuild()
    LocalRasterVar.model_rebuild()

    # Then rebuild Project to ensure it sees the updated Variable classes
    Project.model_rebuild()
except ImportError:
    pass  # Project not yet available
