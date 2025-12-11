import re


def extract_variables(formula: str, mode: str = "predictors") -> set:
    """
    Extract variable names from a Patsy-style formula, safely handling I(), scale(), C(), etc.

    Parameters
    ----------
    formula : str
        A Patsy formula string, e.g. 'I(1 - fcc) + trial ~ scale(altitude) + C(pa)'.
    mode : {'predictors', 'target', 'I', 'all'}
        - 'predictors': extract right-hand side variables (default)
        - 'target': extract left-hand side variables
        - 'I': extract variables only inside I() expressions on the LHS
        - 'all': extract all variables from both sides

    Returns
    -------
    set
        A set of raw variable names (unique, untransformed).

    Example
    -------
    >>> formula = "I(1-fcc) + trial ~ scale(altitude) + scale(dist_edge) + C(pa)"
    >>> extract_variables(formula)
    {'altitude', 'dist_edge', 'pa'}
    >>> extract_variables(formula, mode='target')
    {'fcc', 'trial'}
    >>> extract_variables(formula, mode='I')
    {'fcc'}
    >>> extract_variables(formula, mode='all')
    {'altitude', 'dist_edge', 'pa', 'fcc', 'trial'}
    """
    # --- Split formula ---
    parts = formula.split("~", 1)
    lhs = parts[0].strip()
    rhs = parts[1].strip() if len(parts) > 1 else ""

    # --- Determine which text to parse based on mode ---
    if mode == "I":
        target_expr = lhs
    elif mode == "target":
        target_expr = lhs
    elif mode == "predictors":
        target_expr = rhs
    elif mode == "all":
        target_expr = formula
    else:
        raise ValueError("mode must be one of: 'predictors', 'target', 'I', 'all'")

    raw_vars = set()

    # --- Match function-like patterns: scale(...), C(...), I(...), etc. ---
    func_pattern = r"[a-zA-Z_][a-zA-Z0-9_]*\(([^)]*)\)"
    matches = re.findall(func_pattern, target_expr)

    for expr in matches:
        tokens = re.split(r"[+\-*/\(\)\s]", expr)
        tokens = [t.strip() for t in tokens if t.strip()]
        for token in tokens:
            if re.match(r"^\d+(\.\d+)?$", token):  # skip numbers
                continue
            if token.lower() in {"i", "scale", "c", "poly", "bs", "cr"}:
                continue
            raw_vars.add(token)

    # --- Extract standalone variables ---
    standalone = re.findall(r"\b[a-zA-Z_][a-zA-Z0-9_]*\b", target_expr)
    for var in standalone:
        if var.lower() in {"i", "scale", "c", "poly", "bs", "cr"}:
            continue
        raw_vars.add(var)

    # --- Special handling for mode="I" ---
    if mode == "I":
        I_expressions = re.findall(r"I\((.*?)\)", lhs)
        I_vars = set()
        for expr in I_expressions:
            I_vars.update(re.findall(r"[A-Za-z_]\w*", expr))
        raw_vars = {v for v in raw_vars if v in I_vars}

    # --- Validate identifiers ---
    raw_vars = {v for v in raw_vars if re.match(r"^[A-Za-z_]\w*$", v)}

    return raw_vars


import pandas as pd
from patsy import dmatrices
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from component.script.dataset import Dataset


def get_design_info(patsy_formula, samples_file):
    """Get design info from patsy."""
    dataset = pd.read_csv(samples_file)
    dataset = dataset.dropna(axis=0)
    dataset["trial"] = 1
    y, x = dmatrices(patsy_formula, dataset, 0, "drop")
    y_design_info = y.design_info
    x_design_info = x.design_info
    return (y_design_info, x_design_info)


def generate_patsy_formula(dataset: "Dataset") -> str:
    """
    Generate a regression formula string with scaled continuous variables
    and categorical variables using Patsy-style syntax.

    Automatically uses the Dataset's target and features, classifying variables
    based on their raster_type attribute (continuous vs categorical).

    Parameters
    ----------
    dataset : Dataset
        Dataset instance with configured target and features

    Returns
    -------
    str
        Patsy formula string

    Example
    -------
    >>> dataset.set_target('fcc', year=2020)
    >>> dataset.set_features(['altitude', 'pa', 'dist_edge'])
    >>> generate_patsy_formula(dataset)
    "I(fcc) + trial ~ scale(altitude) + scale(dist_edge) + C(pa)"
    """
    # Validate dataset configuration
    if not dataset.target:
        raise ValueError("Dataset target not set. Use dataset.set_target() first.")
    if not dataset.features:
        raise ValueError("Dataset features not set. Use dataset.set_features() first.")

    dependent_variable = dataset.target.name

    # Print dataset configuration
    print(f"\n📊 Generating Patsy formula:")
    print(f"  Target: {dependent_variable}")
    print(f"  Features: {', '.join([f.name for f in dataset.features])}")

    continuous = []
    categorical = []

    for var in dataset.features:
        # Check if variable has raster_type attribute (LocalRasterVar)
        if hasattr(var, "raster_type") and var.raster_type:
            if var.raster_type == "continuous":
                continuous.append(var.name)
            elif var.raster_type == "categorical":
                categorical.append(var.name)
            else:
                # Default to continuous if raster_type is not set
                continuous.append(var.name)
        else:
            # Default to continuous if no raster_type attribute
            continuous.append(var.name)

    parts = []
    if continuous:
        parts += [f"scale({x})" for x in continuous]
    if categorical:
        parts += [f"C({x})" for x in categorical]

    # Print classification results
    if continuous:
        print(f"  Continuous: {', '.join(continuous)}")
    if categorical:
        print(f"  Categorical: {', '.join(categorical)}")

    rhs = " + ".join(parts) if parts else "1"  # intercept-only model if empty
    formula = f"I({dependent_variable}) + trial ~ {rhs}"

    print(f"\n✓ Formula: {formula}\n")

    return formula
