import numpy as np


def load_drag_csv(csv_path):
    """Load a drag polar CSV and return a structured NumPy array.

    Parameters
    ----------
    csv_path : str or path-like
        Path to the CSV file containing drag-polar data.

    Returns
    -------
    numpy.ndarray
        Structured NumPy array with columns including mach, aoa_deg,
        cd_force_axial, and cp.

    Raises
    ------
    ValueError
        If the file has no rows or is missing required columns.
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
    """Linearly interpolate a scalar value from a 1D dataset.

    Parameters
    ----------
    x : array-like
        Monotonically increasing x-coordinates.
    y : array-like
        Values corresponding to x.
    xq : float
        Query x value to interpolate.

    Returns
    -------
    float
        Interpolated y value at xq.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if xq <= x[0]:
        return y[0]
    if xq >= x[-1]:
        return y[-1]

    idx = np.searchsorted(x, xq)
    if idx == 0:
        return y[0]
    if idx >= len(x):
        return y[-1]

    x0, x1 = x[idx - 1], x[idx]
    y0, y1 = y[idx - 1], y[idx]
    if x1 == x0:
        return y0

    return y0 + (y1 - y0) * ((xq - x0) / (x1 - x0))


def _interp2d(x_vals, y_vals, z_matrix, xq, yq):
    """Perform bilinear interpolation on a regular 2D grid.

    Parameters
    ----------
    x_vals : array-like
        x-axis coordinate values for the grid.
    y_vals : array-like
        y-axis coordinate values for the grid.
    z_matrix : array-like
        2D array of values indexed as z_matrix[x_index, y_index].
    xq : float
        Query x value.
    yq : float
        Query y value.

    Returns
    -------
    float
        Interpolated value at (xq, yq).
    """
    x_vals = np.asarray(x_vals, dtype=float)
    y_vals = np.asarray(y_vals, dtype=float)
    z_matrix = np.asarray(z_matrix, dtype=float)

    if xq <= x_vals[0]:
        xq = x_vals[0]
    elif xq >= x_vals[-1]:
        xq = x_vals[-1]

    if yq <= y_vals[0]:
        yq = y_vals[0]
    elif yq >= y_vals[-1]:
        yq = y_vals[-1]

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
    return z0 * (1.0 - ty) + z1 * ty


def interpolate_drag_csv(csv_path, mach=None, aoa_deg=None, columns=("cd_force_axial", "cp")):
    """Interpolate drag-polar values from a CSV file.

    Provide both mach and aoa_deg for a full 2D interpolation. If only one is
    provided, the function interpolates along the other axis using the nearest
    available value in the table.

    Parameters
    ----------
    csv_path : str or path-like
        Path to the drag-polar CSV file.
    mach : float, optional
        Mach number to query. If omitted, interpolation is done across mach.
    aoa_deg : float, optional
        Angle of attack in degrees to query. If omitted, interpolation is done
        across angle of attack.
    columns : tuple of str, optional
        Column names to interpolate from the CSV data.

    Returns
    -------
    dict
        Mapping of each requested column name to its interpolated value(s).

    Raises
    ------
    ValueError
        If no interpolation inputs are supplied or a requested column is absent.
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



result = interpolate_drag_csv(
    r"drag_report.csv",
    mach=0.2,
    aoa_deg=3.0,
    columns=("cd_force_axial", "cp"),
)
print(result)






