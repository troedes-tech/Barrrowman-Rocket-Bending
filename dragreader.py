import time

import numpy as np

try:
    from scipy.interpolate import CubicSpline, PchipInterpolator, RegularGridInterpolator
except ImportError:  # pragma: no cover - optional dependency for cubic interpolation
    CubicSpline = None
    PchipInterpolator = None
    RegularGridInterpolator = None


def calculate_runtime(start_time, end_time=None):
    """Return the elapsed runtime in seconds.

    Parameters
    ----------
    start_time : float
        The start timestamp returned by ``time.perf_counter()``.
    end_time : float, optional
        The end timestamp. If omitted, the current time is used.

    Returns
    -------
    float
        The elapsed time in seconds, clamped to a non-negative value.

    Raises
    ------
    ValueError
        If ``start_time`` is ``None``.
    """
    if start_time is None:
        raise ValueError("start_time is required.")

    if end_time is None:
        end_time = time.perf_counter()

    return max(0.0, end_time - start_time)


def load_drag_csv(csv_path):
    """Load a drag-polar CSV file into a structured NumPy array.

    Parameters
    ----------
    csv_path : str or path-like
        Path to the CSV file containing drag-polar data.

    Returns
    -------
    numpy.ndarray
        A structured NumPy array with at least the columns ``mach``,
        ``aoa_deg``, ``cd_force_axial``, and ``cp``.

    Raises
    ------
    ValueError
        If the CSV is empty or missing any required column.
    """
    data = np.genfromtxt(csv_path, delimiter=",", names=True, dtype=float)
    if data.size == 0:
        raise ValueError(f"No data found in {csv_path}.")

    required = ["mach", "aoa_deg", "cd_force_axial", "cp"]
    missing = [name for name in required if name not in data.dtype.names]
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")

    return data


def _interp1d(x, y, xq):
    """Interpolate a scalar value from a one-dimensional dataset.

    Parameters
    ----------
    x : array-like
        The x-coordinates for the data. Values are sorted internally and may be
        non-decreasing with duplicate entries handled gracefully.
    y : array-like
        The corresponding y-values for ``x``.
    xq : float
        The x-value at which to evaluate the interpolated result.

    Returns
    -------
    float
        The interpolated y-value at ``xq``.

    Raises
    ------
    ValueError
        If either input array is empty, they do not have matching shapes, or
        ``xq`` is NaN.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if x.size == 0 or y.size == 0:
        raise ValueError("Interpolation arrays cannot be empty.")
    if x.shape != y.shape:
        raise ValueError("Interpolation x and y arrays must have the same shape.")

    if x.size == 1:
        return float(y[0])

    order = np.argsort(x)
    x = x[order]
    y = y[order]

    # Guard against repeated x values while keeping the first occurrence.
    unique_x, first_idx = np.unique(x, return_index=True)
    if unique_x.size != x.size:
        x = unique_x
        y = y[first_idx]

    xq = float(xq)
    if np.isnan(xq):
        raise ValueError("Interpolation query xq cannot be NaN.")
    if xq <= x[0]:
        return float(y[0])
    if xq >= x[-1]:
        return float(y[-1])

    if PchipInterpolator is not None:
        return float(PchipInterpolator(x, y)(xq))

    if CubicSpline is not None:
        return float(CubicSpline(x, y)(xq))

    idx = np.searchsorted(x, xq)
    if idx == 0:
        return float(y[0])
    if idx >= len(x):
        return float(y[-1])

    x0, x1 = x[idx - 1], x[idx]
    y0, y1 = y[idx - 1], y[idx]
    if x1 == x0:
        return float(y0)

    return float(y0 + (y1 - y0) * ((xq - x0) / (x1 - x0)))


def _interp2d(x_vals, y_vals, z_matrix, xq, yq):
    """Interpolate a value on a regular 2D grid.

    The function attempts to use SciPy's regular-grid interpolator when it is
    available and a higher-order cubic interpolation is possible. If SciPy is
    unavailable, it falls back to bilinear interpolation over the local grid cell.

    Parameters
    ----------
    x_vals : array-like
        Coordinates along the first axis of the grid.
    y_vals : array-like
        Coordinates along the second axis of the grid.
    z_matrix : array-like
        Two-dimensional values defined on the ``(x_vals, y_vals)`` grid.
    xq : float
        Query x-coordinate.
    yq : float
        Query y-coordinate.

    Returns
    -------
    float
        The interpolated z-value at ``(xq, yq)``.

    Raises
    ------
    ValueError
        If either interpolation axis is empty.
    """
    x_vals = np.asarray(x_vals, dtype=float)
    y_vals = np.asarray(y_vals, dtype=float)
    z_matrix = np.asarray(z_matrix, dtype=float)

    if x_vals.size == 0 or y_vals.size == 0:
        raise ValueError("Interpolation axes cannot be empty.")

    xq = float(np.clip(xq, x_vals[0], x_vals[-1]))
    yq = float(np.clip(yq, y_vals[0], y_vals[-1]))

    if RegularGridInterpolator is not None and x_vals.size > 2 and y_vals.size > 2:
        interp = RegularGridInterpolator(
            (x_vals, y_vals),
            z_matrix,
            method="pchip",
            bounds_error=False,
            fill_value=np.nan,
        )
        result = interp((xq, yq))
        if np.isfinite(result):
            return float(result)

    x_idx = np.searchsorted(x_vals, xq)
    y_idx = np.searchsorted(y_vals, yq)

    if x_idx == 0:
        x0, x1 = 0, 1
    elif x_idx >= len(x_vals):
        x0, x1 = len(x_vals) - 2, len(x_vals) - 1
    else:
        x0, x1 = x_idx - 1, x_idx

    if y_idx == 0:
        y0, y1 = 0, 1
    elif y_idx >= len(y_vals):
        y0, y1 = len(y_vals) - 2, len(y_vals) - 1
    else:
        y0, y1 = y_idx - 1, y_idx

    x0_val, x1_val = x_vals[x0], x_vals[x1]
    y0_val, y1_val = y_vals[y0], y_vals[y1]

    z00 = z_matrix[x0, y0]
    z10 = z_matrix[x1, y0]
    z01 = z_matrix[x0, y1]
    z11 = z_matrix[x1, y1]

    if x1_val == x0_val:
        tx = 0.0
    else:
        tx = (xq - x0_val) / (x1_val - x0_val)

    if y1_val == y0_val:
        ty = 0.0
    else:
        ty = (yq - y0_val) / (y1_val - y0_val)

    z0 = z00 * (1.0 - tx) + z10 * tx
    z1 = z01 * (1.0 - tx) + z11 * tx
    return float(z0 * (1.0 - ty) + z1 * ty)


def interpolate_drag_csv(csv_path, mach=None, aoa_deg=None, columns=("cd_force_axial", "cp")):
    """Interpolate drag-polar values from a CSV file.

    Provide both ``mach`` and ``aoa_deg`` for a full 2D interpolation. If only
    one input is supplied, the function interpolates along the remaining axis
    using the available data in the table.

    Parameters
    ----------
    csv_path : str or path-like
        Path to the drag-polar CSV file.
    mach : float, optional
        Mach number to query. If omitted, interpolation is performed across the
        Mach axis.
    aoa_deg : float, optional
        Angle of attack in degrees to query. If omitted, interpolation is
        performed across the angle-of-attack axis.
    columns : tuple of str, optional
        Column names to interpolate from the CSV data.

    Returns
    -------
    dict
        A dictionary mapping each requested column name to its interpolated
        value.

    Raises
    ------
    ValueError
        If neither ``mach`` nor ``aoa_deg`` is supplied, or if a requested
        column is not present in the CSV.
    """
    data = load_drag_csv(csv_path)

    if mach is None and aoa_deg is None:
        raise ValueError("Provide mach and/or aoa_deg to interpolate.")

    mach_values = np.unique(data["mach"])
    aoa_values = np.unique(data["aoa_deg"])

    # Build a lookup for each column on the mach/aoa grid.
    grids = {}
    for column in columns:
        if column not in data.dtype.names:
            raise ValueError(f"Column '{column}' not found in CSV data.")

        grid = np.empty((len(mach_values), len(aoa_values)), dtype=float)
        for i, mach_value in enumerate(mach_values):
            rows = data[data["mach"] == mach_value]
            row_map = {float(row["aoa_deg"]): float(row[column]) for row in rows}
            for j, aoa_value in enumerate(aoa_values):
                grid[i, j] = row_map.get(float(aoa_value), np.nan)
        grids[column] = grid

    if mach is not None and aoa_deg is not None:
        return {
            column: _interp2d(mach_values, aoa_values, grids[column], float(mach), float(aoa_deg))
            for column in columns
        }

    if mach is not None:
        # Interpolate across aoa at the given mach.
        mach_q = float(mach)
        mach_idx = np.searchsorted(mach_values, mach_q)
        if mach_idx == 0:
            mach_idx = 0
        elif mach_idx >= len(mach_values):
            mach_idx = len(mach_values) - 1

        aoa_for_mach = aoa_values
        values = {}
        for column in columns:
            z = grids[column][mach_idx, :]
            values[column] = _interp1d(aoa_for_mach, z, np.mean(aoa_for_mach))

        return values

    # Interpolate across mach at the given aoa.
    aoa_q = float(aoa_deg)
    aoa_idx = np.searchsorted(aoa_values, aoa_q)
    if aoa_idx == 0:
        aoa_idx = 0
    elif aoa_idx >= len(aoa_values):
        aoa_idx = len(aoa_values) - 1

    mach_for_aoa = mach_values
    values = {}
    for column in columns:
        z = grids[column][:, aoa_idx]
        values[column] = _interp1d(mach_for_aoa, z, np.mean(mach_for_aoa))

    return values



start_time = time.perf_counter()
result = interpolate_drag_csv(
    r"drag_report.csv",
    mach=0.2,
    aoa_deg=3.0,
    columns=("cd_force_axial", "cp"),
)
print(result)
print(f"Runtime: {calculate_runtime(start_time):.6f} seconds")




