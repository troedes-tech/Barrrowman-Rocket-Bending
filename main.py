"""Exercise aerodynamics.py and report rocket drag for selected flight inputs.

Example: python main.py --mach 0.6 --aoa 2 --reynolds 1000000
"""
import argparse
import csv
import inspect
import itertools
import math

import aerodynamics as aero_module
from aerodynamics import *  # noqa: F401,F403 - retain the module's public API
from components import fin, nosecone_vk, tube, rocket


def main():
    """Generate a drag report CSV for a representative rocket across Mach and AOA sweep ranges.

    The script builds a simple rocket geometry, configures a flight envelope, and
    evaluates axial force, normal force, moment, and center of pressure coefficients
    for each Mach and angle-of-attack combination. Results are written to
    ``drag_report.csv`` in the working directory.
    """
    radius = 0.155 / 2
    area = math.pi * radius**2
    nose = nosecone_vk(radius, 0.624)
    body = tube(radius, 3.18 - 0.624)
    fins = fin(root_chord=0.155, tip_chord=0.0775, span=0.0775,
               sweep_length=0.0775, thickness=0.003, location=3.18 - 0.624)
    parts = [nose, body]
    kv = 1.51*10**-5
    temperature = 273.15+25 #K
    mach = 0
    aoa = 0
    roughness = 0.8
    mach_range = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    aoa_range = [0, 2, 4, 6]

    # Keep the constructor convention used by the original example.
    aero = aerodynamics(area, mach, aoa, kv, temperature)

    # HPRs most recent rocket
    meridian = rocket(fins, parts, roughness)

    csv_path = "drag_report.csv"
    with open(csv_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            "mach",
            "aoa_deg",
            "cd_force_axial",
            "cd_force_normal",
            "cd_moment_total",
            "cp"
        ])

        for i in mach_range:
            for j in aoa_range:
                # Update the aerodynamics object with the current flight conditions
                aero.mach = i
                aero.alpha = math.radians(j)
                # Calculate and report the drag for the current conditions
                cd_force_axial = aero.cd_total(meridian)
                cd_force_normal = aero.C_n_total(meridian)
                cd_moment = aero.C_m_total(meridian)
                cp = aero.cp(meridian)

                writer.writerow([
                    i,
                    j,
                    cd_force_axial,
                    cd_force_normal,
                    cd_moment,
                    cp
                ])


if __name__ == "__main__":
    main()
