from aerodynamics import *
from components import *
import numpy as np

nosecone_meridian = nosecone_vk(0.155/2, 0.624)
tube_meridian = tube(0.155/2, 3.18-0.624)
meridian_parts = [nosecone_meridian, tube_meridian]

AoA = 2*np.pi/180
Aref = np.pi*(0.155/2)**2
aero = aerodynamics(Aref)

for part in meridian_parts:
    print(f"Part: {part.__class__.__name__}")
    print(f"Normal Force Coefficient (C_na): {aero.C_na(part, AoA)}")
    print(f"Moment Coefficient (C_ma): {aero.C_ma(part, AoA)}")
    print(f"Lift Coefficient (C_lift): {aero.C_lift(part, AoA)}")
    print(f"Lift Location (X_lift): {aero.X_lift(part)}")
    print(f"Center of Pressure (X_n): {aero.X_n(part)}\n")