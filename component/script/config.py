from pathlib import Path
from component.script.utilities.file_filter import filter_files_by_keywords

project_name = "test"

forest_source = "gfc"  ##gfc, tmf, local
tree_cover_threshold = 10
years = [2015, 2020, 2024]
string_years = [str(num) for num in years]

params = {
    "sampling": {
        "n_samples": 10000,
        "random_seed": 1,
        "spatial_cell_size_km": 10,
        "adapt": True,
    },
    "mw_model": {
        "win_size_list": [5, 11, 21],
        "block_rows": 256,
    },
    "far_glm": {
        "random_seed": 1,
    },
    "far_icar": {
        "random_seed": 1,
        "csize": 10,
        "prior_vrho": -1,
        "mcmc": 10000,
        "thin": 1,
        "beta_start": -99,
        "csize_interpolate": 0.1,
    },
    "far_rf": {
        "random_seed": 1,
        "n_trees": 100,
    },
}

period_dict = {
    "calibration": {
        "period": "calibration",
        "train_period": "calibration",
        "initial_year": years[0],
        "final_year": years[1],
        "defor_value": 1,
        "time_interval": years[1] - years[0],
    },
    "validation": {
        "period": "validation",
        "train_period": "calibration",
        "initial_year": years[1],
        "final_year": years[2],
        "defor_value": 1,
        "time_interval": years[2] - years[1],
    },
    "historical": {
        "period": "historical",
        "train_period": "historical",
        "initial_year": years[0],
        "final_year": years[2],
        "defor_value": [1, 2],
        "time_interval": years[2] - years[0],
    },
    "forecast": {
        "period": "forecast",
        "train_period": "historical",
        "initial_year": years[0],
        "final_year": years[2],
        "defor_value": [1, 2],
        "time_interval": years[2] - years[0],
    },
}


def get_period_dicts(forest_yearly_files, forest_edge_files):

    calibration_dict = {
        "period": "calibration",
        "train_period": "calibration",
        "initial_year": years[0],
        "final_year": years[1],
        "defor_value": 1,
        "time_interval": years[1] - years[0],
        "initial_year_forest": filter_files_by_keywords(
            forest_yearly_files, [str(years[0])]
        )[0],
        "initial_year_forest_edge": filter_files_by_keywords(
            forest_edge_files, [str(years[0])]
        )[0],
    }

    validation_dict = {
        "period": "validation",
        "train_period": "calibration",
        "initial_year": years[1],
        "final_year": years[2],
        "defor_value": 1,
        "time_interval": years[2] - years[1],
        "initial_year_forest": filter_files_by_keywords(
            forest_yearly_files, [str(years[1])]
        )[0],
        "initial_year_forest_edge": filter_files_by_keywords(
            forest_edge_files, [str(years[1])]
        )[0],
    }

    historical_dict = {
        "period": "historical",
        "train_period": "historical",
        "initial_year": years[0],
        "final_year": years[2],
        "defor_value": [1, 2],
        "time_interval": years[2] - years[0],
        "initial_year_forest": filter_files_by_keywords(
            forest_yearly_files, [str(years[0])]
        )[0],
        "initial_year_forest_edge": filter_files_by_keywords(
            forest_edge_files, [str(years[0])]
        )[0],
    }

    forecast_dict = {
        "period": "forecast",
        "train_period": "historical",
        "initial_year": years[0],
        "final_year": years[2],
        "defor_value": [1, 2],
        "time_interval": years[2] - years[0],
        "initial_year_forest": filter_files_by_keywords(
            forest_yearly_files, [str(years[2])]
        )[0],
        "initial_year_forest_edge": filter_files_by_keywords(
            forest_edge_files, [str(years[2])]
        )[0],
    }

    return calibration_dict, validation_dict, historical_dict, forecast_dict


def get_variable_independant_files(input_raster_files, period):
    # Define the period-independent variables and their associated files
    period_independant_variables = ["altitude", "slope", "pa", "subj"]
    altitude_files = filter_files(input_raster_files, ["altitude"], None, False)
    slope_files = filter_files(input_raster_files, ["slope"], None, False)
    wdpa_files = filter_files(input_raster_files, ["pa"], None, False)
    subj_files = filter_files(input_raster_files, ["subj"], None, False)

    # Define the rivers and roads variables and their associated files
    rivers_files = filter_files(
        input_raster_files, ["rivers", "reprojected", "distance"], None, True
    )
    road_files = filter_files(
        input_raster_files, ["roads", "reprojected", "distance"], None, True
    )

    # Define the period-dependent variables and their associated files
    period_dictionary = period_dict[period]
    initial_year = str(period_dictionary["initial_year"])
    final_year = str(period_dictionary["final_year"])
    exclude_year = ", ".join(
        map(
            str,
            set(years)
            - {period_dictionary["initial_year"], period_dictionary["final_year"]},
        )
    )
    forest_loss_files = filter_files(
        input_raster_files,
        [forest_source, initial_year, final_year, "forest", "loss"],
        [exclude_year, "edge"],
        True,
    )
    forest_edge_files = filter_files(
        input_raster_files,
        [forest_source, initial_year, "forest", "reprojected", "edge"],
        None,
        True,
    )
    town_files = filter_files(
        input_raster_files,
        [initial_year, "town", "reprojected", "distance"],
        None,
        True,
    )

    # Create a dictionary with variable types as keys and file paths as values
    variable_file_mapping = {
        "period": period_dictionary["period"],
        "altitude": altitude_files[0],
        "slope": slope_files[0],
        "pa": wdpa_files[0],
        "subj": subj_files[0],
        "dist_river": rivers_files[0],
        "dist_road": road_files[0],
        "dist_town": town_files[0],
        "fcc": forest_loss_files[0],
        "dist_edge": forest_edge_files[0],
    }
    return variable_file_mapping


def filter_files(input_files, filter_words, exclude_words=None, include_all=True):
    """
    Filters a list of files based on include and exclude words.
    Parameters:
        input_files (list): List of file paths to be filtered.
        filter_words (list): Words that must be present in the filenames for inclusion.
        exclude_words (list, optional): Words that must not be present in the filenames for exclusion. Defaults to None.
        include_all (bool, optional): If True, all filter words must be present in the filename. If False, at least one of the filter words must be present. Defaults to False.
    Returns:
        list: Filtered list of files.
    """
    # Ensure all words are lowercase for case-insensitive comparison
    filter_words = [word.lower() for word in filter_words]
    exclude_words = [word.lower() for word in (exclude_words or [])]

    if include_all:
        filtered_files = [
            file
            for file in input_files
            if all(word in Path(file).name.lower() for word in filter_words)
            and not any(
                exclude_word in Path(file).name.lower()
                for exclude_word in exclude_words
            )
        ]
    else:
        filtered_files = [
            file
            for file in input_files
            if any(word in Path(file).name.lower() for word in filter_words)
            and not any(
                exclude_word in Path(file).name.lower()
                for exclude_word in exclude_words
            )
        ]

    return filtered_files
