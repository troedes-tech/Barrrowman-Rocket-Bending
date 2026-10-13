"""Exercise aerodynamics.py and report rocket drag for selected flight inputs.

Example: python main.py --mach 0.6 --aoa 2 --reynolds 1000000
"""
import argparse
import inspect
import math

import aerodynamics as aero_module
from aerodynamics import *  # noqa: F401,F403 - retain the module's public API
from components import fin, nosecone_vk, tube


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mach", type=float, default=0.6, help="Freestream Mach number")
    parser.add_argument("--aoa", type=float, default=2.0, help="Angle of attack in degrees")
    parser.add_argument("--reynolds", type=float, default=1_000_000.0)
    parser.add_argument("--density", type=float, default=1.225, help="Air density (kg/m^3)")
    parser.add_argument("--temperature", type=float, default=285.0, help="Air temperature (K)")
    return parser.parse_args()


def invoke(function, values):
    """Call a function only when values for its required parameters are known."""
    try:
        signature = inspect.signature(function)
        args, kwargs = [], {}
        for parameter in signature.parameters.values():
            if parameter.kind in (parameter.VAR_POSITIONAL, parameter.VAR_KEYWORD):
                continue
            key = parameter.name.lower().replace("_", "")
            if key not in values:
                if parameter.default is inspect.Parameter.empty:
                    return False, f"no test value for {parameter.name}"
                continue
            if parameter.kind == parameter.POSITIONAL_ONLY:
                args.append(values[key])
            else:
                kwargs[parameter.name] = values[key]
        return True, function(*args, **kwargs)
    except Exception as error:
        return False, f"{type(error).__name__}: {error}"


def main():
    options = parse_args()
    radius = 0.155 / 2
    area = math.pi * radius**2
    nose = nosecone_vk(radius, 0.624)
    body = tube(radius, 3.18 - 0.624)
    fins = fin(root_chord=0.155, tip_chord=0.0775, span=0.0775,
               sweep_length=0.0775, thickness=0.003, location=3.18 - 0.624)
    parts = [nose, body]
    alpha = math.radians(options.aoa)

    # Keep the constructor convention used by the original example.
    aero = aerodynamics(area, options.mach, 0.5, options.density, options.temperature)
    values = {
        "part": nose, "component": nose, "meridianpart": nose, "parts": parts,
        "nose": nose, "nosecone": nose, "body": body, "tube": body,
        "fin": fins, "fins": fins, "meridianfin": fins, "meridianfins": fins,
        "alpha": alpha, "aoa": alpha, "angleofattack": alpha,
        "mach": options.mach, "re": options.reynolds,
        "reynolds": options.reynolds, "reynoldsnumber": options.reynolds,
        "rho": options.density, "density": options.density,
        "temperature": options.temperature, "temp": options.temperature,
        "area": area, "aref": area, "referencearea": area,
    }
    print(f"Conditions: Mach={options.mach:g}, AoA={options.aoa:g} deg, "
          f"Re={options.reynolds:g}")

    print("\nModule-level aerodynamics functions:")
    for name, function in inspect.getmembers(aero_module, callable):
        if (name.startswith("_") or inspect.isclass(function)
                or getattr(function, "__module__", None) != aero_module.__name__):
            continue
        ok, result = invoke(function, values)
        print(f"{name}: {'PASS' if ok else 'SKIP/FAIL'} — {result}")

    print("\nAerodynamics instance methods:")
    for name, method in inspect.getmembers(aero, callable):
        if name.startswith("_"):
            continue
        # Methods accepting a rocket part are tried with both meridian parts.
        result = None
        for part in parts:
            test_values = dict(values, part=part, component=part, meridianpart=part)
            ok, result = invoke(method, test_values)
            if ok:
                break
        else:
            ok, result = invoke(method, values)
        print(f"{name}: {'PASS' if ok else 'SKIP/FAIL'} — {result}")

    # Prefer a drag force function; otherwise use a drag coefficient and D=q*S*Cd.
    candidates = []
    for name, function in inspect.getmembers(aero, callable):
        compact = name.lower().replace("_", "")
        if not name.startswith("_") and ("drag" in compact or compact.startswith("cd")):
            candidates.append((name, function))
    for name, function in inspect.getmembers(aero_module, callable):
        compact = name.lower().replace("_", "")
        if (not name.startswith("_") and not inspect.isclass(function)
                and getattr(function, "__module__", None) == aero_module.__name__
                and ("drag" in compact or compact.startswith("cd"))):
            candidates.append((name, function))

    drag_name = drag_value = None
    for name, function in candidates:
        ok, result = invoke(function, values)
        if ok and isinstance(result, (int, float)) and math.isfinite(result):
            drag_name, drag_value = name, float(result)
            break
    if drag_value is None:
        print("\nTotal drag unavailable: aerodynamics.py exposes no usable drag/Cd function.")
    elif "cd" in drag_name.lower().replace("_", "") or "coefficient" in drag_name.lower():
        speed = options.mach * math.sqrt(1.4 * 287.05 * options.temperature)
        dynamic_pressure = 0.5 * options.density * speed**2
        total_drag = dynamic_pressure * area * drag_value
        print(f"\nDrag coefficient ({drag_name}): {drag_value:.6g}")
        print(f"Estimated total rocket drag: {total_drag:.6g} N")
    else:
        print(f"\nTotal rocket drag ({drag_name}): {drag_value:.6g} N")


if __name__ == "__main__":
    main()
