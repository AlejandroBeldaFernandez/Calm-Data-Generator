"""
Shared mathematical utilities for drift propagation and function evaluation.

These are free functions used by DriftInjector, ScenarioInjector, and CausalEngine
to avoid code duplication.
"""
from typing import Callable, Dict, Optional, Union

import numpy as np
import pandas as pd


def propagate_numeric_drift(
    df: pd.DataFrame,
    rows: pd.Index,
    driver_col: str,
    delta_driver: np.ndarray,
    correlations: Union[pd.DataFrame, Dict, bool],
    driver_std: Optional[float] = None,
) -> pd.DataFrame:
    """
    Propagates the drift (delta) from a driver column to other correlated columns.

    Formula: Delta_Y = Correlation(X, Y) * (Std_Y / Std_X) * Delta_X

    Args:
        df: DataFrame to modify in-place.
        rows: Index labels of affected rows.
        driver_col: Column whose drift is being propagated.
        delta_driver: Array of deltas applied to driver_col (length = len(rows)).
        correlations: Correlation matrix (DataFrame or dict), or True to compute
                      from current data, or False/None to skip propagation.
        driver_std: Pre-computed std of driver_col. Computed from df if None.

    Returns:
        df (modified in-place).
    """
    if correlations is None or correlations is False:
        return df

    if isinstance(correlations, bool) and correlations:
        corr_matrix = df.corr()
    elif isinstance(correlations, (pd.DataFrame, dict)):
        corr_matrix = pd.DataFrame(correlations)
    else:
        return df

    if driver_col not in corr_matrix.index:
        return df

    if driver_std is None:
        driver_std = df[driver_col].std()
    if driver_std == 0 or np.isnan(driver_std):
        return df

    target_cols = [
        c
        for c in df.columns
        if c != driver_col and pd.api.types.is_numeric_dtype(df[c])
    ]

    for target_col in target_cols:
        if target_col not in corr_matrix.columns:
            continue

        rho = corr_matrix.loc[driver_col, target_col]
        if pd.isna(rho) or rho == 0:
            continue

        target_std = df[target_col].std()
        if target_std == 0 or np.isnan(target_std):
            continue

        factor = rho * (target_std / driver_std)
        delta_target = delta_driver * factor

        current_vals = df.loc[rows, target_col].to_numpy()
        df.loc[rows, target_col] = current_vals + delta_target

    return df


def apply_func(
    func_name: Union[str, Callable],
    params: dict,
    x: np.ndarray,
) -> np.ndarray:
    """
    Evaluates a named transformation function over array x.

    Supported names:
        "linear"      : slope * x + intercept
        "exponential" : scale * exp(rate * x)
        "power"       : scale * |x|^exponent
        "polynomial"  : np.poly1d(coeffs)(x)
        callable      : func_name(x)

    Args:
        func_name: Function name string or a callable.
        params: Dict of parameters for the function.
        x: Input array.

    Returns:
        Transformed array (same shape as x).

    Raises:
        ValueError: If func_name is an unknown string.
    """
    params = params or {}

    if callable(func_name):
        return np.asarray(func_name(x), dtype=float)

    if func_name == "linear":
        slope = params.get("slope", 1.0)
        intercept = params.get("intercept", 0.0)
        return slope * x + intercept

    if func_name == "exponential":
        scale = params.get("scale", 1.0)
        rate = params.get("rate", 1.0)
        return scale * np.exp(rate * x)

    if func_name == "power":
        scale = params.get("scale", 1.0)
        exponent = params.get("exponent", 2.0)
        return scale * np.abs(x) ** exponent

    if func_name == "polynomial":
        coeffs = params.get("coeffs", [1, 0])
        return np.poly1d(coeffs)(x)

    raise ValueError(
        f"Unknown func_name '{func_name}'. "
        "Supported: 'linear', 'exponential', 'power', 'polynomial', or a callable."
    )
