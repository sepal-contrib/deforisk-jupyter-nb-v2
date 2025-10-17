# import re


# def extract_raw_variables(formula: str) -> set:
#     """
#     Extract raw variable names from a Patsy-style formula,
#     safely handling I(), scale(), C(), and other transformations.

#     Example:
#         "I(1 - fcc) + trial ~ scale(altitude) + C(pa)"
#         → returns {'fcc', 'trial', 'altitude', 'pa'}
#     """
#     raw_vars = set()

#     # Pattern to match: any Patsy function with content inside parentheses
#     # We capture the inner part, then extract variables from it
#     pattern = r"[a-zA-Z_][a-zA-Z0-9_]*\(([^)]+)\)"

#     # Find all expressions like I(...), scale(...), C(...)
#     matches = re.findall(pattern, formula)

#     for expr in matches:
#         # Clean the expression: remove spaces, split by operators
#         # We want to extract only variable names (no constants or math)
#         tokens = re.split(r"[+\-*/\(\)\s]", expr)  # Split on common symbols
#         tokens = [t.strip() for t in tokens if t.strip()]

#         # Keep only valid identifiers that are not numbers/strings
#         for token in tokens:
#             # Skip numeric literals (e.g., '1', '2.3')
#             if re.match(r"^\d+(\.\d+)?$", token):
#                 continue
#             # Skip keywords like 'I', 'scale'
#             if token.lower() in {"i", "scale", "c", "poly", "bs", "cr"}:
#                 continue
#             raw_vars.add(token)

#     # Now extract standalone variables (not inside functions)
#     standalone = re.findall(r"\b[a-zA-Z_][a-zA-Z0-9_]*\b", formula)

#     for var in standalone:
#         if var.lower() not in {"i", "scale", "c"}:  # Skip Patsy keywords
#             raw_vars.add(var)

#     # Remove invalid tokens (e.g., '1-fcc' is not a column name)
#     raw_vars = {v for v in raw_vars if re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", v)}

#     return raw_vars


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


def get_design_info(patsy_formula, samples_file):
    """Get design info from patsy."""
    dataset = pd.read_csv(samples_file)
    dataset = dataset.dropna(axis=0)
    dataset["trial"] = 1
    y, x = dmatrices(patsy_formula, dataset, 0, "drop")
    y_design_info = y.design_info
    x_design_info = x.design_info
    return (y_design_info, x_design_info)


def generate_patsy_formula(
    dependent_variable: str,
    independent_variables_continuous: list[str] | None = None,
    independent_variables_categorical: list[str] | None = None,
) -> str:
    """
    Generate a regression formula string with scaled continuous variables
    and categorical variables using Patsy-style syntax.

    Example:
        generate_formula("y", ["age", "weight"], ["sex", "breed"])
        -> "I(y) + trial ~ scale(age) + scale(weight) + C(sex) + C(breed)"
    """
    independent_variables_continuous = independent_variables_continuous or []
    independent_variables_categorical = independent_variables_categorical or []

    parts = []
    if independent_variables_continuous:
        parts += [f"scale({x})" for x in independent_variables_continuous]
    if independent_variables_categorical:
        parts += [f"C({x})" for x in independent_variables_categorical]

    rhs = " + ".join(parts) if parts else "1"  # intercept-only model if empty
    formula = f"I({dependent_variable}) + trial ~ {rhs}"
    return formula
