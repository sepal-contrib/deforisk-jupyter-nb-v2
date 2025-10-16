from component.script.config import get_period_dicts
from component.script.utilities.file_filter import (
    filter_files_by_keywords,
    list_files_by_extension,
)


def create_periods_dict(processed_data_folder, forest_source, years):

    input_raster_files = list_files_by_extension(
        processed_data_folder, [".tiff", ".tif"]
    )
    input_raster_files

    forest_change_file = filter_files_by_keywords(
        input_raster_files,
        ["forest", "loss", forest_source] + [str(y) for y in years],
        False,
        ["distance", "edge"],
    )[0]
    forest_change_file

    forest_yearly_files = filter_files_by_keywords(
        input_raster_files, ["forest", forest_source], False, ["loss", "edge"]
    )
    forest_yearly_files

    forest_edge_files = filter_files_by_keywords(
        input_raster_files, ["forest", forest_source, "edge"], False
    )
    forest_edge_files

    raster_subj_file = filter_files_by_keywords(input_raster_files, ["subj"])[0]
    raster_subj_file

    return get_period_dicts(forest_yearly_files, forest_edge_files)
