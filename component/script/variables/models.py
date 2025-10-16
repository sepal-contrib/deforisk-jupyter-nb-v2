from enum import Enum


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
