from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Literal, Dict, List, Tuple

from component.script.variables.local_raster_var import LocalRasterVar

Period = Literal["calibration", "validation", "historical", "forecast"]

_CFG: Dict[Period, dict] = {
    "calibration": {
        "train_period": "calibration",
        "initial_idx": 0,
        "final_idx": 1,
        "defor_value": 1,
        "var_idx": 0,
    },
    "validation": {
        "train_period": "calibration",
        "initial_idx": 1,
        "final_idx": 2,
        "defor_value": 1,
        "var_idx": 1,
    },
    "historical": {
        "train_period": "historical",
        "initial_idx": 0,
        "final_idx": 2,
        "defor_value": [1, 2],
        "var_idx": 0,
    },
    "forecast": {
        "train_period": "historical",
        "initial_idx": 0,
        "final_idx": 2,
        "defor_value": [1, 2],
        "var_idx": 2,
    },
}


def _header(years: list[int], period: Period) -> dict:
    c = _CFG[period]
    iy, fy, vy = years[c["initial_idx"]], years[c["final_idx"]], years[c["var_idx"]]
    return {
        "period": period,
        "train_period": c["train_period"],
        "initial_year": iy,
        "final_year": fy,
        "defor_value": c["defor_value"],
        "time_interval": fy - iy,
        "var_year": vy,
    }


def _infer_policy(items: List[LocalRasterVar]) -> str:
    # Check if any item has a name pattern indicating it's an interval variable
    # (e.g., "forest_loss_2015_2020" contains two years separated by underscore)
    for v in items:
        if v.year is not None:
            # Check if name contains period pattern (start_year_end_year)
            name_parts = v.name.split("_")
            # Look for pattern with two consecutive year-like numbers
            for i in range(len(name_parts) - 1):
                if name_parts[i].isdigit() and name_parts[i + 1].isdigit():
                    if len(name_parts[i]) == 4 and len(name_parts[i + 1]) == 4:
                        return "by_interval"
            return "by_year"  # Has year but not interval pattern
    return "static"  # purely static


def create_period_dict_from_meta(
    years: list[int],
    period: Period,
    variables: Iterable[LocalRasterVar],
    *,
    strict: bool = True,
    default_forecast_fallback: str = "middle",
    key_overrides: Dict[str, str] | None = None,  # rename output keys if you want
):
    if len(years) < 3:
        raise ValueError("years must have at least 3 elements")
    if period not in _CFG:
        raise ValueError(f"Unknown period: {period}")

    hdr = _header(years, period)
    iy, fy, vy = hdr["initial_year"], hdr["final_year"], hdr["var_year"]

    # Group: tag for families; allow tagless STATIC (key = name or override)
    groups: Dict[str, List[LocalRasterVar]] = {}
    for v in variables:
        if not v.active:
            continue
        is_static = v.year is None
        if v.tags:
            fam = v.tags[0]
            key = key_overrides.get(fam, fam) if key_overrides else fam
        else:
            if is_static:
                base = v.name
                key = key_overrides.get(base, base) if key_overrides else base
            else:
                if strict:
                    raise ValueError(
                        f"Variable '{v.name}' is dynamic/interval but has no tags; set tags[0] as family"
                    )
                else:
                    continue
        groups.setdefault(key, []).append(v)

    # Pickers
    def pick_static(items: List[LocalRasterVar]) -> Optional[Path]:
        for it in items:
            if it.year is None:
                return it.path
        return None

    def pick_by_year(items: List[LocalRasterVar], target_year: int) -> Optional[Path]:
        for it in items:
            if it.year == target_year:
                return it.path
        return None

    def pick_by_interval(
        items: List[LocalRasterVar], start: int, end: int
    ) -> Optional[Path]:
        # For interval variables, we use the end year as the 'year' attribute
        # and the full period is encoded in the name (e.g., "forest_loss_2015_2020")
        for it in items:
            if it.year == end:
                # Check if name contains the period pattern
                if f"_{start}_{end}" in it.name or f"_{end}_{start}" in it.name:
                    return it.path
        return None

    out = dict(hdr)
    for key, items in groups.items():
        policy = _infer_policy(items)
        if policy == "static":
            chosen = pick_static(items)
        elif policy == "by_year":
            chosen = pick_by_year(items, vy)
            if (
                chosen is None
                and period == "forecast"
                and default_forecast_fallback == "middle"
            ):
                chosen = pick_by_year(items, years[1])
        elif policy == "by_interval":
            chosen = pick_by_interval(items, iy, fy)
        else:
            raise ValueError(f"Unknown policy '{policy}' for key '{key}'")

        if strict and chosen is None:
            need = f"year={vy}" if policy == "by_year" else f"interval=({iy},{fy})"
            raise ValueError(
                f"Missing variable for key='{key}' policy='{policy}' (needed {need})"
            )

        out[key] = chosen

    return out


def build_all_periods_from_meta(
    years: list[int], variables: Iterable[LocalRasterVar], **kwargs
):
    cal = create_period_dict_from_meta(years, "calibration", variables, **kwargs)
    val = create_period_dict_from_meta(years, "validation", variables, **kwargs)
    his = create_period_dict_from_meta(years, "historical", variables, **kwargs)
    frc = create_period_dict_from_meta(years, "forecast", variables, **kwargs)
    return {
        cal["period"]: cal,
        val["period"]: val,
        his["period"]: his,
        frc["period"]: frc,
    }
