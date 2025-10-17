from pathlib import Path
import os
import shutil
from typing import Union


def list_files_by_extension_os(folder_path, file_extensions):  # -> list:
    """
    List all files with specified extensions in the given folder.

    Parameters:
    folder_path (str): The path to the folder where you want to search for files.
    file_extensions (list of str): A list of file extensions to search for (e.g., ['.shp', '.tif']).

    Returns:
    list: A list of file paths with the specified extensions.
    """
    matching_files = []
    try:
        # Check if the provided path is a directory
        if os.path.isdir(folder_path):
            # Iterate over all files in the directory
            for filename in os.listdir(folder_path):
                # Construct full file path
                file_path = os.path.join(folder_path, filename)
                # Check if the file has any of the specified extensions
                if any(filename.lower().endswith(ext) for ext in file_extensions):
                    matching_files.append(file_path)
        else:
            print(f"The provided path '{folder_path}' is not a directory.")
    except Exception as e:
        print(f"An error occurred: {e}")
    return matching_files


def list_files_by_extension(folder_path, file_extensions, recursive=False):
    """
    List all files with specified extensions in the given folder.
    Parameters:
    folder_path (str or Path): The path to the folder where you want to search for files.
    file_extensions (list of str): A list of file extensions to search for (e.g., ['.shp', '.tif']).
    recursive (bool): Whether to recursively search through subdirectories or not.
    Returns:
    list: A list of file paths with the specified extensions.
    """
    matching_files = []
    try:
        # Convert folder_path to Path object if it's a string
        folder_path = Path(folder_path)

        # Check if the provided path is a directory
        if folder_path.is_dir():
            for entry in folder_path.iterdir():
                if entry.is_file() and any(
                    entry.suffix.lower() == ext.lower() for ext in file_extensions
                ):
                    matching_files.append(str(entry))
                elif recursive and entry.is_dir():
                    # Recursively search subdirectories
                    matching_files.extend(
                        list_files_by_extension(entry, file_extensions, recursive)
                    )
        else:
            print(f"The provided path '{folder_path}' is not a directory.")
    except Exception as e:
        print(f"An error occurred: {e}")
    return matching_files


def filter_file_os(input_files, filter_words, exclude_words=None):
    """
    Filters a list of files based on include and exclude words.

    Parameters:
        input_files (list): List of file paths to be filtered.
        filter_words (list): Words that must be present in the filenames for inclusion.
        exclude_words (list, optional): Words that must not be present in the filenames for exclusion. Defaults to None.

    Returns:
        list: Filtered list of files.
    """
    # Ensure all words are lowercase for case-insensitive comparison
    filter_words = [word.lower() for word in filter_words]
    exclude_words = [word.lower() for word in (exclude_words or [])]

    filtered_files = [
        file
        for file in input_files
        if all(word in os.path.basename(file).lower() for word in filter_words)
        and not any(
            exclude_word in os.path.basename(file).lower()
            for exclude_word in exclude_words
        )
    ]

    return filtered_files


from pathlib import Path


def filter_files(input_files, filter_words, exclude_words=None):
    """
    Filters a list of files based on include and exclude words.

    Parameters:
        input_files (list): List of file paths to be filtered.
        filter_words (list): Words that must be present in the filenames for inclusion.
        exclude_words (list, optional): Words that must not be present in the filenames for exclusion. Defaults to None.

    Returns:
        list: Filtered list of files.
    """
    # Ensure all words are lowercase for case-insensitive comparison
    filter_words = [word.lower() for word in filter_words]
    exclude_words = [word.lower() for word in (exclude_words or [])]

    filtered_files = [
        file
        for file in input_files
        if all(word in Path(file).name.lower() for word in filter_words)
        and not any(
            exclude_word in Path(file).name.lower() for exclude_word in exclude_words
        )
    ]

    return filtered_files


from pathlib import Path


def generate_output_filename_change(i1: Path, i2: Path, change_keyword: str) -> Path:
    """
    Generate an output filename representing a change between two input rasters,
    inserting a user-defined keyword such as 'deforestation', 'gain', etc.

    Args:
        i1 (Path): First input file path
        i2 (Path): Second input file path
        change_keyword (str): Keyword to insert (e.g. 'deforestation', 'gain', 'change')

    Returns:
        Path: Output file path (e.g. '/data/YARI_deforestation_gfc_10_20152020.tif')
    """
    base_name_i1 = i1.stem
    base_name_i2 = i2.stem

    # --- Extract years ---
    def extract_year(name: str) -> str:
        for part in name.split("_"):
            if part.isdigit() and len(part) == 4:
                return part
        raise ValueError(f"Year not found in: {name}")

    year_i1 = extract_year(base_name_i1)
    year_i2 = extract_year(base_name_i2)
    year_start, year_end = sorted([year_i1, year_i2])

    # --- Remove years from the parts ---
    parts = [p for p in base_name_i1.split("_") if not (p.isdigit() and len(p) == 4)]

    # --- Replace the 2nd token with the keyword ---
    if len(parts) >= 2:
        parts[1] = change_keyword
    else:
        # If there's only one token, just insert the keyword after it
        parts.append(change_keyword)

    # --- Build the new filename ---
    new_base = "_".join(parts) + f"_{year_start}{year_end}.tif"
    return i1.parent / new_base


from pathlib import Path


def generate_output_filename_stack(
    i1: Path, i2: Path, i3: Path, change_keyword: str
) -> Path:
    """
    Generate an output filename based on three input file paths.

    Args:
        i1: First input file path
        i2: Second input file path
        i3: Third input file path
        change_keyword (str): Keyword to insert (e.g. 'deforestation', 'gain', 'change')


    Returns:
        Path object for the generated output filename
    """
    # Extract the base names from the input file paths
    base_name_i1 = i1.stem  # Gets filename without extension
    base_name_i2 = i2.stem  # Gets filename without extension
    base_name_i3 = i3.stem  # Gets filename without extension

    # Extract years (4-digit numbers)
    def extract_year(name):
        for part in name.split("_"):
            if part.isdigit() and len(part) == 4:
                return part
        raise ValueError(f"Year not found in: {name}")

    year_i1 = extract_year(base_name_i1)
    year_i2 = extract_year(base_name_i2)
    year_i3 = extract_year(base_name_i3)

    # Remove years from parts
    parts = [p for p in base_name_i1.split("_") if not (p.isdigit() and len(p) == 4)]

    # --- Replace the 2nd token with the keyword ---
    if len(parts) >= 2:
        parts[1] = change_keyword
    else:
        # If there's only one token, just insert the keyword after it
        parts.append(change_keyword)

    # Build new base name
    new_base = "_".join(parts) + f"_{year_i1}_{year_i2}_{year_i3}.tif"

    return i1.parent / new_base


def copy_and_rename_file(
    file_path: Union[str, Path], destination_path: Union[str, Path]
) -> Path:
    """
    Copy a file to a designated location and rename it. If the source is a shapefile (.shp),
    also copies all corresponding auxiliary files.

    Args:
        file_path (Union[str, Path]): The path to the source file to be copied.
        destination_path (Union[str, Path]): The full path including folder and new filename for the copied file.

    Returns:
        Path: The path to the newly copied file.

    Raises:
        FileNotFoundError: If the source file does not exist.
        PermissionError: If there are insufficient permissions to read the source file or write to destination.

    Example:
        >>> copy_and_rename_file('/path/to/source/file.shp', '/path/to/destination/renamed_file.shp')
        PosixPath('/path/to/destination/renamed_file.shp')
    """
    # Convert to Path objects
    source_file = Path(file_path)
    new_file_path = Path(destination_path)

    # Validate that source file exists
    if not source_file.exists():
        raise FileNotFoundError(f"Source file '{source_file}' does not exist.")

    # Ensure the destination directory exists
    new_file_path.parent.mkdir(parents=True, exist_ok=True)

    # Copy the main file to the destination with a new name
    shutil.copy2(source_file, new_file_path)

    # If source is a shapefile (.shp), also copy auxiliary files
    if source_file.suffix.lower() == ".shp":
        # Use pathlib to construct the glob pattern and iterate over auxiliary files
        aux_files = list(source_file.parent.glob(f"{source_file.stem}.*"))
        # print(f"Found {len(aux_files)} auxiliary files: {aux_files}")
        for aux_file in aux_files:
            new_aux_filename = f"{new_file_path.stem}{aux_file.suffix}"
            new_aux_path = new_file_path.parent / new_aux_filename
            shutil.copy2(aux_file, new_aux_path)
            print(f"Auxiliary file copied to {new_aux_path}")

    # print(f"File copied to {new_file_path}")
    return new_file_path
