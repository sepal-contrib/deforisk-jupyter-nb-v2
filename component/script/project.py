import json
from typing import Dict, List, Optional, Union, Any
from collections.abc import Iterable
from pathlib import Path
from box import Box
from pydantic import BaseModel, Field, ConfigDict
from component.script.variables import LocalVectorVar, LocalRasterVar
from component.script.variables.models import DataType

root_folder: Path = Path.cwd().parent
downloads_folder = root_folder / "data"
downloads_folder.mkdir(parents=True, exist_ok=True)


def _stringify_paths(obj: Any) -> Any:
    """Recursively convert pathlib.Path objects to str for JSON serialization."""
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {k: _stringify_paths(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        t = type(obj)
        return t(_stringify_paths(v) for v in obj)
    return obj


class Project(BaseModel):
    """
    A Pydantic model representing a deforestation risk analysis project.

    Stores project metadata, variables, and manages folder structure.
    Can be serialized to/from JSON for persistence.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    project_name: str
    years: Optional[List[int]] = Field(
        default=None,
        description="Deprecated: Years are now auto-discovered from variables",
    )
    raw_variables: Dict[str, Union["LocalVectorVar", "LocalRasterVar"]] = Field(
        default_factory=dict
    )
    processed_variables: Dict[str, Union["LocalVectorVar", "LocalRasterVar"]] = Field(
        default_factory=dict
    )
    base_raster: Optional["LocalRasterVar"] = None

    @staticmethod
    def _ensure_model_schemas() -> None:
        """Ensure Pydantic forward references between Project and variable models are resolved."""
        from component.script.variables import (
            LocalVectorVar,
            LocalRasterVar,
            GEEVar,
        )
        from component.script.variables.variable import Variable

        # Rebuild variable models first so they know about Project
        types_namespace = {"Project": Project}

        Variable.model_rebuild(_types_namespace=types_namespace)
        LocalVectorVar.model_rebuild(_types_namespace=types_namespace)
        LocalRasterVar.model_rebuild(_types_namespace=types_namespace)
        GEEVar.model_rebuild(_types_namespace=types_namespace)

        # Finally rebuild Project to include the updated variable schemas
        project_namespace = {
            "LocalVectorVar": LocalVectorVar,
            "LocalRasterVar": LocalRasterVar,
            "Variable": Variable,
        }

        Project.model_rebuild(_types_namespace=project_namespace)

    @property
    def folders(self) -> Box:
        """Initialize and return project folder structure."""
        return self.initialize_folders()

    @property
    def raw_vars(self) -> Dict[str, Union["LocalVectorVar", "LocalRasterVar"]]:
        """Alias for raw variables."""
        return self.raw_variables

    @property
    def processed_vars(self) -> Dict[str, Union["LocalVectorVar", "LocalRasterVar"]]:
        """Alias for processed variables."""
        return self.processed_variables

    @property
    def variables(
        self,
    ) -> Dict[str, Dict[str, Union["LocalVectorVar", "LocalRasterVar"]]]:
        """Return both raw and processed variables as two separate collections.

        The resulting dictionary is structured as::

            {
                "raw": {
                    "var1": <LocalVar>,
                    "var2": <LocalVar>,
                    ...
                },
                "processed": {
                    "var3": <LocalVar>,
                    "var4": <LocalVar>,
                    ...
                }
            }

        This provides clear separation between raw and processed datasets.
        """
        print("project.variables → {'raw': {...}, 'processed': {...}} view")
        return {
            "raw": dict(self.raw_variables),
            "processed": dict(self.processed_variables),
        }

    def add_variable(self, variable: Union["LocalVectorVar", "LocalRasterVar"]) -> None:
        """
        Add a variable to the project's processed variables collection.

        Note: This method is kept for backward compatibility.
        Prefer using variable.add_as_raw() or variable.add_as_processed() instead.

        Parameters
        ----------
        variable : LocalVectorVar | LocalRasterVar
            The variable to add to the project
        """
        print(f"Adding variable: {variable.name}")
        self.processed_variables[variable.name] = variable

    def get_variable(
        self, name: str, year: Optional[int] = None, source: str = "processed"
    ) -> Optional[Union["LocalVectorVar", "LocalRasterVar"]]:
        """
        Get a variable by name and optional year.

        Parameters
        ----------
        name : str
            Variable name (e.g., 'towns', 'forest_gfc')
        year : int, optional
            Year for temporal variables. If None, returns static variable.
        source : str, optional
            Variable source: 'processed' (default) or 'raw'

        Returns
        -------
        LocalVectorVar | LocalRasterVar | None
            The variable if found, None otherwise
        """
        variables = (
            self.processed_variables if source == "processed" else self.raw_variables
        )

        # Construct storage key
        storage_key = f"{name}_{year}" if year else name

        # Try storage key lookup (fast path)
        if storage_key in variables:
            return variables[storage_key]

        return None

    def get_all_instances(
        self, name: str, source: str = "processed"
    ) -> List[Union["LocalVectorVar", "LocalRasterVar"]]:
        """
        Get all year instances of a variable by name.

        Parameters
        ----------
        name : str
            Variable name (e.g., 'towns', 'forest_gfc')
        source : str, optional
            Variable source: 'processed' (default) or 'raw'

        Returns
        -------
        List[LocalVectorVar | LocalRasterVar]
            List of all variable instances with matching name
        """
        variables = (
            self.processed_variables if source == "processed" else self.raw_variables
        )
        return [var for var in variables.values() if var.name == name]

    def is_temporal(self, name: str, source: str = "processed") -> bool:
        """
        Check if a variable has multiple year instances (is temporal).

        Parameters
        ----------
        name : str
            Variable name
        source : str, optional
            Variable source: 'processed' (default) or 'raw'

        Returns
        -------
        bool
            True if variable has multiple years, False otherwise
        """
        instances = self.get_all_instances(name, source)
        unique_years = set(var.year for var in instances if var.year is not None)
        return len(unique_years) > 1

    def get_variable_years(self, name: str, source: str = "processed") -> List[int]:
        """
        Get list of available years for a variable.

        Parameters
        ----------
        name : str
            Variable name
        source : str, optional
            Variable source: 'processed' (default) or 'raw'

        Returns
        -------
        List[int]
            Sorted list of years for the variable
        """
        instances = self.get_all_instances(name, source)
        years = sorted(set(var.year for var in instances if var.year is not None))
        return years

    def get_available_years(self, source: str = "all") -> List[int]:
        """
        Get all unique years across all variables in the project.

        This automatically discovers years from the variables that have been added,
        rather than relying on a manually specified years list.

        Parameters
        ----------
        source : str, optional
            Variable source: 'processed' (default), 'raw', or 'all'

        Returns
        -------
        List[int]
            Sorted list of all unique years found in variables

        Examples
        --------
        >>> project.get_available_years()  # All years from all variables
        [2015, 2020, 2024]
        >>> project.get_available_years(source='raw')  # Only from raw variables
        [2015, 2020, 2024]
        """
        years_set = set()

        if source in ("processed", "all"):
            for var in self.processed_variables.values():
                if var.year is not None:
                    years_set.add(var.year)

        if source in ("raw", "all"):
            for var in self.raw_variables.values():
                if var.year is not None:
                    years_set.add(var.year)

        return sorted(years_set)

    def list_unique_variable_names(self, source: str = "processed") -> List[str]:
        """
        Get list of unique variable names.

        Parameters
        ----------
        source : str, optional
            Variable source: 'processed' (default) or 'raw'

        Returns
        -------
        List[str]
            Sorted list of unique variable names (e.g., ['altitude', 'towns', 'forest_gfc'])
        """
        variables = (
            self.processed_variables if source == "processed" else self.raw_variables
        )
        unique_names = sorted(set(var.name for var in variables.values()))
        return unique_names

    def save(self, filename: Optional[str] = None) -> Path:
        """
        Save the project to a JSON file in the project folder.

        Parameters
        ----------
        filename : str, optional
            Custom filename for the project file. If None, uses '{project_name}_project.json'

        Returns
        -------
        Path
            Path to the saved JSON file
        """
        # Ensure schemas are up-to-date before serializing any variables
        self._ensure_model_schemas()

        if filename is None:
            filename = f"{self.project_name}_project.json"

        project_folder = self.folders.project_folder
        project_folder.mkdir(parents=True, exist_ok=True)

        save_path = project_folder / filename

        # Prepare data for serialization
        data = {
            "project_name": self.project_name,
            "raw_variables": {},
            "processed_variables": {},
        }

        # Only include years if explicitly set (for backward compatibility)
        if self.years is not None:
            data["years"] = self.years

        # Serialize raw variables
        for var_name, var in self.raw_variables.items():
            data["raw_variables"][var_name] = var.model_dump(mode="json")

        # Serialize processed variables
        for var_name, var in self.processed_variables.items():
            data["processed_variables"][var_name] = var.model_dump(mode="json")

        # Serialize base_raster if it exists
        if self.base_raster is not None:
            data["base_raster"] = self.base_raster.model_dump(mode="json")

        # Write to file
        save_path.write_text(
            json.dumps(data, indent=4, ensure_ascii=False, default=str),
            encoding="utf-8",
        )

        print(f"Project saved to: {save_path}")

        return save_path

    @classmethod
    def load(cls, project_name: str, filename: Optional[str] = None) -> "Project":
        """
        Load a project from a JSON file.

        Parameters
        ----------
        project_name : str
            Name of the project (used to locate the project folder)
        filename : str, optional
            Custom filename for the project file. If None, uses '{project_name}_project.json'

        Returns
        -------
        Project
            Loaded project instance with all variables
        """
        # Ensure schemas are up-to-date before instantiating variables
        cls._ensure_model_schemas()

        from component.script.variables import LocalVectorVar, LocalRasterVar

        if filename is None:
            filename = f"{project_name}_project.json"

        project_folder = downloads_folder / project_name
        load_path = project_folder / filename

        if not load_path.exists():
            raise FileNotFoundError(f"Project file not found: {load_path}")

        # Load JSON data
        data = json.loads(load_path.read_text(encoding="utf-8"))

        # Create project instance without variables first
        project = cls(project_name=data["project_name"], years=data.get("years"))

        # Reconstruct raw variables
        for var_name, var_data in data.get("raw_variables", {}).items():
            # Convert Path strings back to Path objects
            if "path" in var_data and var_data["path"]:
                var_data["path"] = Path(var_data["path"])

            # Determine which class to use based on data_type
            if var_data.get("data_type") == "vector":
                var = LocalVectorVar(**var_data)
            elif var_data.get("data_type") == "raster":
                var = LocalRasterVar(**var_data)
            else:
                raise ValueError(
                    f"Unknown data_type for variable {var_name}: {var_data.get('data_type')}"
                )

            # Set project reference and add to raw_variables
            var.project = project
            project.raw_variables[var_name] = var

        # Reconstruct processed variables
        for var_name, var_data in data.get("processed_variables", {}).items():
            # Convert Path strings back to Path objects
            if "path" in var_data and var_data["path"]:
                var_data["path"] = Path(var_data["path"])

            # Determine which class to use based on data_type
            if var_data.get("data_type") == "vector":
                var = LocalVectorVar(**var_data)
            elif var_data.get("data_type") == "raster":
                var = LocalRasterVar(**var_data)
            else:
                raise ValueError(
                    f"Unknown data_type for variable {var_name}: {var_data.get('data_type')}"
                )

            # Set project reference and add to processed_variables
            var.project = project
            project.processed_variables[var_name] = var

        # Reconstruct base_raster if it exists
        if "base_raster" in data and data["base_raster"]:
            base_data = data["base_raster"]

            # Convert Path strings back to Path objects
            if "path" in base_data and base_data["path"]:
                base_data["path"] = Path(base_data["path"])

            # Create the base raster variable
            project.base_raster = LocalRasterVar(**base_data)
            # Set project reference
            project.base_raster.project = project

        print(f"Project loaded from: {load_path}")
        print(f"Loaded {len(project.processed_variables)} processed variables")
        return project

    def reproject_and_match_all(
        self,
        target_epsg: Optional[str] = None,
        resolution: Optional[float] = None,
        source: str = "raw",
        add_to_processed: bool = True,
        auto_save: bool = True,
        **reproject_kwargs,
    ) -> Dict[str, "LocalRasterVar"]:
        """
        Reproject all raster variables in the project.

        Parameters
        ----------
        target_epsg : str, optional
            Target EPSG code. If None, uses base_raster's CRS (base_raster must be set).
        resolution : float, optional
            Target resolution. If None, uses base_raster's resolution (base_raster must be set).
        source : str, optional
            Which variables to reproject: 'raw' or 'processed' (default: 'raw').
        add_to_processed : bool, optional
            Whether to add reprojected variables to processed collection (default: True).
        auto_save : bool, optional
            Whether to auto-save after each reprojection (default: True).
        **reproject_kwargs
            Additional arguments passed to LocalRasterVar.reproject().

        Returns
        -------
        Dict[str, LocalRasterVar]
            Dictionary of reprojected variables {name: LocalRasterVar}.

        Raises
        ------
        ValueError
            If base_raster is not set when target_epsg or resolution is None.
            If source is not 'raw' or 'processed'.

        Examples
        --------
        >>> # Reproject all raw variables to base raster's CRS
        >>> project.reproject_all()

        >>> # Reproject to specific CRS
        >>> project.reproject_all(target_epsg="EPSG:32618", resolution=30)
        """
        from component.script.variables import LocalRasterVar

        # Determine source collection
        if source == "raw":
            source_vars = self.raw_variables
        elif source == "processed":
            source_vars = self.processed_variables
        else:
            raise ValueError(f"source must be 'raw' or 'processed', got '{source}'")

        # Determine target CRS and resolution
        if target_epsg is None or resolution is None:
            if self.base_raster is None:
                raise ValueError(
                    "base_raster must be set when target_epsg or resolution is not provided. "
                    "Use variable.use_as_base_raster() to set a base raster first."
                )

        # Reproject all active raster variables
        reprojected_vars = {}
        skipped_count = 0

        for var_key, var in source_vars.items():
            # Check data_type instead of isinstance to handle module reloads
            if var.data_type == DataType.raster:
                print(f"\n📍 Reprojecting '{var_key}'...")
                reprojected = var.reproject_and_match(
                    geobox=self.base_raster.get_base_geobox(),
                )

                if add_to_processed:
                    reprojected.add_as_processed(auto_save=auto_save)

                # Use storage key pattern (name_year or just name)
                storage_key = (
                    f"{reprojected.name}_{reprojected.year}"
                    if reprojected.year
                    else reprojected.name
                )
                reprojected_vars[storage_key] = reprojected

        print(f"\n✅ Reprojected {len(reprojected_vars)} raster variables")
        if skipped_count > 0:
            print(f"   ({skipped_count} inactive variables skipped)")
        return reprojected_vars

    def rasterize_all(
        self,
        source: str = "raw",
        add_to_processed: bool = True,
        auto_save: bool = True,
        **rasterize_kwargs,
    ) -> Dict[str, "LocalRasterVar"]:
        """
        Rasterize all vector variables in the project using the base raster.

        Parameters
        ----------
        source : str, optional
            Which variables to rasterize: 'raw' or 'processed' (default: 'raw').
        add_to_processed : bool, optional
            Whether to add rasterized variables to processed collection (default: True).
        auto_save : bool, optional
            Whether to auto-save after each rasterization (default: True).
        **rasterize_kwargs
            Additional arguments passed to LocalVectorVar.rasterize().

        Returns
        -------
        Dict[str, LocalRasterVar]
            Dictionary of rasterized variables {name: LocalRasterVar}.

        Raises
        ------
        ValueError
            If base_raster is not set.
            If source is not 'raw' or 'processed'.

        Examples
        --------
        >>> # Set base raster first
        >>> dem.use_as_base_raster()

        >>> # Rasterize all raw vector variables
        >>> project.rasterize_all()
        """
        from component.script.variables import LocalVectorVar

        # Check base raster is set
        if self.base_raster is None:
            raise ValueError(
                "base_raster must be set before rasterizing. "
                "Use variable.use_as_base_raster() to set a base raster first."
            )

        # Determine source collection
        if source == "raw":
            source_vars = self.raw_variables
        elif source == "processed":
            source_vars = self.processed_variables
        else:
            raise ValueError(f"source must be 'raw' or 'processed', got '{source}'")

        # Rasterize all active vector variables
        rasterized_vars = {}
        skipped_count = 0

        for var_key, var in source_vars.items():
            # Skip inactive variables
            if not var.active:
                print(f"⏭️  Skipping '{var_key}' (inactive)")
                skipped_count += 1
                continue

            # Check data_type instead of isinstance to handle module reloads
            if var.data_type == DataType.vector:
                print(f"\n🗺️  Rasterizing '{var_key}'...")
                rasterized = var.rasterize(base=self.base_raster, **rasterize_kwargs)

                if add_to_processed:
                    rasterized.add_as_processed(auto_save=auto_save)

                # Use storage key pattern (name_year or just name)
                storage_key = (
                    f"{rasterized.name}_{rasterized.year}"
                    if rasterized.year
                    else rasterized.name
                )
                rasterized_vars[storage_key] = rasterized

        print(f"\n✅ Rasterized {len(rasterized_vars)} vector variables")
        if skipped_count > 0:
            print(f"   ({skipped_count} inactive variables skipped)")
        return rasterized_vars

    def list_variables(
        self,
        source: str = "processed",
        **filters: Any,
    ) -> Dict[str, Union["LocalVectorVar", "LocalRasterVar"]]:
        """Return variables from the requested collection applying simple filters.

        Parameters
        ----------
        source : str, optional
            Collection to inspect: 'processed' (default), 'raw', or 'both'.
        **filters : dict
            Attribute filters evaluated as equality checks. Iterables (except
            strings/bytes) are treated as lists of acceptable values. Callables
            are invoked with the attribute value and must return True to keep it.
        """

        if source == "processed":
            candidates = self.processed_vars
        elif source == "raw":
            candidates = self.raw_vars
        elif source == "both":
            # Combine both, with processed overriding raw for duplicate names
            candidates = {}
            candidates.update(self.raw_vars)
            candidates.update(self.processed_vars)
        else:
            raise ValueError("source must be 'processed', 'raw', or 'both'")

        def matches(var: Union["LocalVectorVar", "LocalRasterVar"]) -> bool:
            for attr, expected in filters.items():
                if not hasattr(var, attr):
                    raise AttributeError(
                        f"Variable '{var.name}' has no attribute '{attr}'"
                    )

                value = getattr(var, attr)

                if callable(expected):
                    if not expected(value):
                        return False
                elif isinstance(expected, Iterable) and not isinstance(
                    expected, (str, bytes, bytearray)
                ):
                    if value not in expected:
                        return False
                else:
                    if value != expected:
                        return False

            return True

        if not filters:
            return dict(candidates)

        return {name: var for name, var in candidates.items() if matches(var)}

    def filter_by_tags(
        self,
        tags: Union[str, List[str]],
        match_all: bool = False,
        look_up_in: Optional[str] = None,
        **filters,
    ) -> Dict[str, Union["LocalVectorVar", "LocalRasterVar"]]:
        """
            Filter variables by tags.

            Parameters
            ----------
            tags : str or List[str]
                Tag(s) to filter by. Can be a single tag string or a list of tags.
            match_all : bool, optional
                If True, variables must have ALL specified tags (AND logic).
                If False (default), variables must have AT LEAST ONE tag (OR logic).
            look_up_in : str, optional
                Collection to search: 'processed' (default), 'raw', or 'both'.
            **filters : keyword arguments
                Additional filter criteria (same as list_variables).

            Returns
            -------
            Dict[str, Variable]
                Dictionary of variables that match the tag criteria and any additional filters.

            Examples
            --------
            >>> # Get all variables with 'climate' tag
        >>> project.filter_by_tags('climate')

        >>> # Get variables with either 'roads' OR 'infrastructure' tag
        >>> project.filter_by_tags(['roads', 'infrastructure'])

            >>> # Get active variables with BOTH 'climate' AND 'temperature' tags
            >>> project.filter_by_tags(['climate', 'temperature'], match_all=True, active=True)

            >>> # Get raw raster variables with 'elevation' tag
            >>> project.filter_by_tags('elevation', look_up_in='raw', data_type='raster')
        """
        # Normalize tags to a list
        if isinstance(tags, str):
            tags = [tags]

        if look_up_in is None:
            look_up_in = filters.pop("source", "processed")

        # Get all variables matching the basic filters
        variables = self.list_variables(source=look_up_in, **filters)

        # Filter by tags
        result = {}
        for var_name, var in variables.items():
            if match_all:
                # Variable must have ALL specified tags
                if all(tag in var.tags for tag in tags):
                    result[var_name] = var
            else:
                # Variable must have AT LEAST ONE of the specified tags
                if any(tag in var.tags for tag in tags):
                    result[var_name] = var

        return result

    def filter_by_attrs(
        self,
        source: str = "processed",
        **attrs,
    ) -> Dict[str, Union["LocalVectorVar", "LocalRasterVar"]]:
        """
        Filter variables by any attribute(s).

        This is an alias for list_variables() with a more explicit name for filtering.
        You can filter by any variable attribute such as year, name, active status,
        data_type, tags, or any custom attributes.

        Parameters
        ----------
        source : str, optional
            Collection to search: 'processed' (default), 'raw', or 'both'.
        **attrs : keyword arguments
            Attribute filters evaluated as equality checks. You can use:
            - Simple values: year=2020, active=True, data_type='raster'
            - Lists of acceptable values: year=[2019, 2020, 2021]
            - Callable functions: year=lambda y: y >= 2015
            - Tags (special): tags=["tag1"] or tags="tag1" checks if ANY tag matches (OR logic)

        Returns
        -------
        Dict[str, Variable]
            Dictionary of variables that match all specified attribute criteria.

        Examples
        --------
        >>> # Filter by year
        >>> project.filter_by_attrs(year=2020)

        >>> # Filter by multiple years
        >>> project.filter_by_attrs(year=[2019, 2020, 2021])

        >>> # Filter by tags (checks if variable has ANY of these tags)
        >>> project.filter_by_attrs(source="raw", tags=["town"])
        >>> project.filter_by_attrs(tags=["town", "city"])  # Has either "town" OR "city"

        >>> # Filter by name pattern (using callable)
        >>> project.filter_by_attrs(name=lambda n: 'forest' in n.lower())

        >>> # Filter active raster variables from 2020
        >>> project.filter_by_attrs(year=2020, active=True, data_type='raster')

        >>> # Filter by year range (using callable)
        >>> project.filter_by_attrs(year=lambda y: y >= 2015 and y <= 2020)

        >>> # Search in raw variables
        >>> project.filter_by_attrs(source='raw', year=2019)

        >>> # Search in both raw and processed
        >>> project.filter_by_attrs(source='both', active=True)

        Notes
        -----
        - String and bytes values are compared for exact equality.
        - Iterable values (lists, tuples, sets) are treated as "value in list" checks.
        - Tags are special: tags=["tag1", "tag2"] checks if variable has ANY of these tags.
        - Callable values are invoked with the attribute value and must return True.
        - All filters must match for a variable to be included (AND logic).
        """
        # Special handling for tags attribute
        if "tags" in attrs:
            tags_filter = attrs.pop("tags")

            # Normalize tags to a list
            if isinstance(tags_filter, str):
                tags_filter = [tags_filter]

            # Use filter_by_tags for tag filtering, then apply other filters
            if tags_filter and isinstance(tags_filter, (list, tuple)):
                # Get variables matching the tags
                result = self.filter_by_tags(tags_filter, look_up_in=source)

                # If there are additional filters, apply them
                if attrs:
                    filtered_result = {}
                    for var_name, var in result.items():
                        # Check if variable matches all other filters
                        matches = True
                        for attr, expected in attrs.items():
                            if not hasattr(var, attr):
                                matches = False
                                break

                            value = getattr(var, attr)

                            if callable(expected):
                                if not expected(value):
                                    matches = False
                                    break
                            elif isinstance(expected, Iterable) and not isinstance(
                                expected, (str, bytes, bytearray)
                            ):
                                if value not in expected:
                                    matches = False
                                    break
                            else:
                                if value != expected:
                                    matches = False
                                    break

                        if matches:
                            filtered_result[var_name] = var

                    return filtered_result
                else:
                    return result

        # No tag filtering, use standard list_variables
        return self.list_variables(source=source, **attrs)

    def reset(
        self,
        source: str = "processed",
        auto_save: bool = True,
        confirm: bool = True,
    ) -> int:
        """
        Remove all variables from the specified collection.

        This clears out all variables from either the raw or processed collection,
        useful for cleaning up or starting fresh without deleting the entire project.

        Parameters
        ----------
        source : str, optional
            Which collection to reset: 'processed' (default), 'raw', or 'both'.
        auto_save : bool, optional
            If True (default), automatically saves the project after reset.
        confirm : bool, optional
            If True (default), shows a warning message before clearing.
            Set to False to skip confirmation (useful in scripts).

        Returns
        -------
        int
            Number of variables removed.

        Raises
        ------
        ValueError
            If source is not 'processed', 'raw', or 'both'.

        Examples
        --------
        >>> # Reset processed variables (default)
        >>> project.reset()

        >>> # Reset raw variables
        >>> project.reset(source='raw')

        >>> # Reset both collections
        >>> project.reset(source='both')

        >>> # Reset without confirmation (in automated scripts)
        >>> project.reset(confirm=False, auto_save=False)

        Notes
        -----
        This does NOT delete the actual files on disk, only removes the
        variable references from the project. Use with caution as this
        operation cannot be undone unless you reload the project from disk.
        """
        if source not in ["processed", "raw", "both"]:
            raise ValueError("source must be 'processed', 'raw', or 'both'")

        removed_count = 0

        if source == "processed" or source == "both":
            count = len(self.processed_variables)
            if confirm and count > 0:
                print(
                    f"⚠️  WARNING: About to remove {count} processed variable(s) from project"
                )

            self.processed_variables.clear()
            removed_count += count

            if count > 0:
                print(f"✓ Removed {count} processed variable(s)")

        if source == "raw" or source == "both":
            count = len(self.raw_variables)
            if confirm and count > 0:
                print(
                    f"⚠️  WARNING: About to remove {count} raw variable(s) from project"
                )

            self.raw_variables.clear()
            removed_count += count

            if count > 0:
                print(f"✓ Removed {count} raw variable(s)")

        if removed_count == 0:
            print(f"ℹ️  No variables to remove from '{source}' collection")
        else:
            print(f"\n✅ Total removed: {removed_count} variable(s)")

        if auto_save and removed_count > 0:
            self.save()

        return removed_count

    def create_model_folder(self, model: str, test_name: Optional[str] = None) -> Path:

        # Create the folder path
        project_folder = downloads_folder / self.project_name
        model_folder = project_folder / model

        # Add test_name as subfolder if provided
        if test_name:
            model_folder = model_folder / test_name

        model_folder.mkdir(parents=True, exist_ok=True)

        # Create a meaningful key for project.folders
        if test_name:
            folder_key = f"{model}_{test_name}"
        else:
            folder_key = model

        # Save to project.folders (reinitialize to update the Box object)
        folders = self.initialize_folders()
        folders[folder_key] = model_folder

        print(f"✅ Created model folder: {model_folder}")
        print(f"📁 Saved as: project.folders.{folder_key}")
        return model_folder

    def initialize_folders(self, step=None, it_name=""):

        if step and not it_name:
            raise ValueError(
                "A suffix must be provided when a specific step is specified."
            )

        it_name = f"{it_name}_" if it_name else it_name

        project_folder = downloads_folder / self.project_name
        project_folder.mkdir(parents=True, exist_ok=True)

        folders = {
            "data_raw_folder": project_folder / "data_raw",
            "glm_model_folder": project_folder / "glm_model",
            "processed_data_folder": project_folder / "data",
            "sampling_folder": project_folder / "far_samples",
            "rmj_mw": project_folder / "rmj_mw",
            "plots_folder": project_folder / "plots",
            "rmj_bm": project_folder / f"{it_name}rmj_bm",
            "glm_model": project_folder / f"{it_name}far_glm",
            "icar_model": project_folder / f"{it_name}far_icar",
            "rf_model": project_folder / f"{it_name}far_rf",
        }

        if step:
            folder = folders.get(step)
            folder.mkdir(parents=True, exist_ok=True)

        else:
            for folder in folders.values():
                folder.mkdir(parents=True, exist_ok=True)

        folders.update(
            {
                "root_folder": root_folder,
                "downloads_folder": downloads_folder,
                "project_folder": project_folder,
            }
        )

        # Return a Box object for dot notation access
        return Box(folders)


# Rebuild Project model after Variable classes are imported to resolve forward references
try:
    from component.script.variables import (
        Variable,
        LocalVectorVar,
        LocalRasterVar,
        GEEVar,
    )

    # Rebuild Variable classes first to ensure they're fully defined
    Variable.model_rebuild()
    LocalVectorVar.model_rebuild()
    LocalRasterVar.model_rebuild()
    GEEVar.model_rebuild()

    # Then rebuild Project
    Project.model_rebuild()
except ImportError:
    pass  # Variables not yet available
