"""Read a drag-coefficient CSV and interpolate Cd by Mach, AoA, and Reynolds."""

import csv
import math
import re
from pathlib import Path


_COLUMN_ALIASES = {
	"mach": {"mach", "machnumber"},
	"angle": {
		"aoa", "alpha", "angle", "angleofattack", "angleofattackdeg",
		"angleattack", "angleattackdeg",
	},
	"reynolds": {"re", "reynolds", "reynoldsnumber"},
	"cd": {"cd", "dragcoefficient", "coefficientofdrag"},
}


def _normalise_header(header):
	return re.sub(r"[^a-z0-9]", "", header.lower())


class DragTable:
	"""CSV-backed rectangular data set with trilinear Cd interpolation.

	The CSV must have columns identifying Mach, angle of attack, Reynolds number,
	and Cd. Inputs use the same units as the CSV. Values outside the data range
	are rejected rather than extrapolated.
	"""

	def __init__(self, rows):
		self.values = {}
		for mach, angle, reynolds, cd in rows:
			point = (float(mach), float(angle), float(reynolds))
			cd = float(cd)
			if not all(math.isfinite(value) for value in point + (cd,)):
				raise ValueError("CSV data must contain finite numbers")
			if point in self.values and self.values[point] != cd:
				raise ValueError("Conflicting Cd values for point {}".format(point))
			self.values[point] = cd

		if not self.values:
			raise ValueError("CSV contains no data rows")
		self.axes = tuple(
			sorted({point[index] for point in self.values}) for index in range(3)
		)

	@classmethod
	def from_csv(cls, filename):
		"""Load a CSV whose header names its Mach, AoA, Reynolds, and Cd columns."""
		with Path(filename).open("r", newline="", encoding="utf-8-sig") as stream:
			reader = csv.DictReader(stream)
			if not reader.fieldnames:
				raise ValueError("CSV must contain a header row")

			headers = {_normalise_header(name): name for name in reader.fieldnames}
			columns = {}
			for field, aliases in _COLUMN_ALIASES.items():
				matches = [headers[name] for name in aliases if name in headers]
				if not matches:
					raise ValueError(
						"CSV is missing the {} column (found: {})".format(
							field, ", ".join(reader.fieldnames)
						)
					)
				columns[field] = matches[0]

			rows = []
			for line_number, row in enumerate(reader, start=2):
				try:
					rows.append(
						tuple(
							float(row[columns[field]])
							for field in ("mach", "angle", "reynolds", "cd")
						)
					)
				except (TypeError, ValueError, KeyError) as error:
					raise ValueError(
						"Invalid numeric data on CSV line {}".format(line_number)
					) from error
		return cls(rows)

	@staticmethod
	def _bracket(axis, value, label):
		if not math.isfinite(value):
			raise ValueError("{} must be finite".format(label))
		if value < axis[0] or value > axis[-1]:
			raise ValueError("{} is outside the range in the CSV".format(label))
		if value in axis:
			return value, value
		for low, high in zip(axis, axis[1:]):
			if low < value < high:
				return low, high
		raise ValueError("Could not interpolate {}".format(label))

	def interpolate_cd(self, mach, angle_of_attack, reynolds):
		"""Return trilinearly interpolated Cd at the supplied conditions."""
		query = (float(mach), float(angle_of_attack), float(reynolds))
		labels = ("Mach", "angle of attack", "Reynolds number")
		bounds = tuple(
			self._bracket(axis, value, label)
			for axis, value, label in zip(self.axes, query, labels)
		)

		result = 0.0
		for m in set(bounds[0]):
			for angle in set(bounds[1]):
				for re_value in set(bounds[2]):
					point = (m, angle, re_value)
					if point not in self.values:
						raise ValueError(
							"CSV is missing a grid point needed for interpolation: {}".format(
								point
							)
						)
					weight = 1.0
					for value, (low, high), corner in zip(query, bounds, point):
						if low != high:
							weight *= (
								(high - value) / (high - low)
								if corner == low
								else (value - low) / (high - low)
							)
					result += weight * self.values[point]
		return result


def interpolate_cd(filename, mach, angle_of_attack, reynolds):
	"""Convenience function to load a CSV and interpolate one Cd value."""
	return DragTable.from_csv(filename).interpolate_cd(
		mach, angle_of_attack, reynolds
	)


def drag_force_vector(cd, velocity, density, area):
	"""Return the drag-force vector for Cd and a velocity vector."""
	velocity = tuple(float(component) for component in velocity)
	speed = math.sqrt(sum(component ** 2 for component in velocity))
	if speed == 0:
		return tuple(0.0 for _ in velocity)
	scale = -0.5 * float(density) * float(area) * float(cd) * speed
	return tuple(scale * component for component in velocity)

