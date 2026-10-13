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
from components import fin, nosecone_vk, tube


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mach", type=float, default=0.6, help="Freestream Mach number")
    parser.add_argument("--aoa", type=float, default=2.0, help="Angle of attack in degrees")
    parser.add_argument("--reynolds", type=float, default=1_000_000.0)
    parser.add_argument("--density", type=float, default=1.225, help="Air density (kg/m^3)")
    parser.add_argument("--temperature", type=float, default=285.0, help="Air temperature (K)")
    parser.add_argument("--csv", default="cd_sweep.csv", help="Output CSV path")
    parser.add_argument("--mach-values", default="0.2,0.4,0.6,0.8,1.0",
                        help="Comma-separated Mach values for the CSV sweep")
    parser.add_argument("--aoa-values", default="0,2,4,6,8",
                        help="Comma-separated angles of attack in degrees")
    parser.add_argument("--reynolds-values", default="100000,500000,1000000,5000000",
                        help="Comma-separated Reynolds numbers")
    return parser.parse_args()


def parse_values(text):
    return [float(value.strip()) for value in text.split(",") if value.strip()]


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

    # Export a vector of Cd results, one row for each combination of inputs.
    sweep_rows = []
    cd_candidates = [
        (name, function) for name, function in candidates
        if "cd" in name.lower().replace("_", "")
        or "coefficient" in name.lower()
    ]
    for mach, aoa_degrees, reynolds in itertools.product(
            parse_values(options.mach_values), parse_values(options.aoa_values),
            parse_values(options.reynolds_values)):
        sweep_aero = aerodynamics(area, mach, 0.5, options.density, options.temperature)
        sweep_values = dict(values)
        sweep_values.update({
            "mach": mach, "alpha": math.radians(aoa_degrees),
            "aoa": math.radians(aoa_degrees),
            "angleofattack": math.radians(aoa_degrees),
            "re": reynolds, "reynolds": reynolds,
            "reynoldsnumber": reynolds,
        })
        result_cd = None
        for name, function in cd_candidates:
            # Prefer instance methods so coefficients use this row's Mach number.
            sweep_methods = inspect.getmembers(sweep_aero, callable)
            method = next((method for method_name, method in sweep_methods
                           if method_name == name), None)
            candidates_for_row = ([ (name, method) ] if method is not None else [])
            candidates_for_row.extend(
                (candidate_name, candidate) for candidate_name, candidate in cd_candidates
                if candidate_name == name and getattr(candidate, "__module__", None) == aero_module.__name__
            )
            for _, function in candidates_for_row:
                ok, result = invoke(function, sweep_values)
                if ok and isinstance(result, (int, float)) and math.isfinite(result):
                    result_cd = float(result)
                    break
            if result_cd is not None:
                break
        sweep_rows.append((mach, aoa_degrees, reynolds, result_cd))

    with open(options.csv, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(("Mach", "AoA_deg", "Reynolds", "Cd"))
        writer.writerows(sweep_rows)
    print(f"\nWrote {len(sweep_rows)} Cd sweep rows to {options.csv}")


if __name__ == "__main__":
    main()
