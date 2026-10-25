"""Load a drag-coefficient CSV and interpolate Cd by Mach and AoA."""
import argparse,csv; from pathlib import Path; import numpy as np; from scipy.interpolate import LinearNDInterpolator
_n=lambda v:"".join(ch.lower() for ch in str(v) if ch.isalnum())

def _g(g):
    """Convert a Mach/AoA lookup dictionary into NumPy arrays for interpolation."""
    p=[]; v=[]
    [p.append((float(m),float(a))) or v.append(float(cd)) for m,am in g.items() for a,cd in am.items()]
    return (np.empty((0,2),float),np.empty((0,),float)) if not p else (np.asarray(p,float),np.asarray(v,float))

def _m(path):
    """Parse a CSV where the first column is Mach and the remaining columns are AoA values."""
    rows=[]
    with Path(path).open("r",newline="",encoding="utf-8-sig") as s:
        for r in csv.reader(s):
            if any(c.strip() for c in r): rows.append([c.strip() for c in r])
    if not rows: return [],[],{}
    h=rows[0]; aoa=[]
    for c in h[1:]:
        try: aoa.append(float(c))
        except ValueError: pass
    g={}
    for r in rows[1:]:
        if len(r)<len(aoa)+1: continue
        try: m=float(r[0]); v=[float(x) for x in r[1:len(aoa)+1]]
        except ValueError: continue
        g[m]={a:cd for a,cd in zip(aoa,v)}
    return sorted(g), sorted({a for v in g.values() for a in v}), g

def _p(path):
    """Parse a CSV with explicit Mach, AoA, and drag-coefficient columns."""
    with Path(path).open("r",newline="",encoding="utf-8-sig") as s:
        r=csv.DictReader(s)
        if r.fieldnames is None: return [],[],{}
        f=[n for n in r.fieldnames if n is not None]; hm={_n(n):n for n in f}
        k1=next((hm[a] for a in ("mach","machnumber") if a in hm),None); k2=next((hm[a] for a in ("aoa","alpha","angle","angleofattack","angleattack") if a in hm),None); k3=next((hm[a] for a in ("cd","dragcoefficient","coefficientofdrag") if a in hm),None)
        g={}
        for row in r:
            try: m=float(row[k1]); a=float(row[k2]); c=float(row[k3])
            except (KeyError,TypeError,ValueError): continue
            g.setdefault(m,{})[a]=c
    return ([],[],{}) if not g else (sorted(g),sorted({a for v in g.values() for a in v}),g)

class DragTable:
    """Load a drag table from CSV data and interpolate Cd for arbitrary Mach and AoA values."""
    def __init__(self,csv_path):
        p=Path(csv_path)
        if not p.exists(): raise FileNotFoundError(f"CSV not found: {p}")
        m=_m(p)
        self.mach_values,self.aoa_values,self.grid=(m[0],m[1],m[2]) if m[2] else _p(p)
        self.points,self.values=_g(self.grid); self._interp=LinearNDInterpolator(self.points,self.values,fill_value=float("nan"))
    def interpolate(self,m,a):
        """Return the drag coefficient interpolated for the specified Mach and AoA."""
        m=float(m); a=float(a)
        if m in self.grid and a in self.grid[m]: return float(self.grid[m][a])
        return float(self._interp((m,a)))

def interpolate_cd(csv_path,m,a):
    """Convenience wrapper that loads a drag table and interpolates a single Cd value."""
    return DragTable(csv_path).interpolate(m,a)

def main():
    """Command-line entry point for interpolating Cd from a CSV file."""
    p=argparse.ArgumentParser(description="Interpolate a drag coefficient from a Mach/AoA CSV")
    p.add_argument("csv_path",help="Path to the CSV file"); p.add_argument("mach",type=float,help="Mach number to evaluate"); p.add_argument("aoa",type=float,help="Angle of attack in degrees")
    a=p.parse_args(); print(f"Cd = {interpolate_cd(a.csv_path,a.mach,a.aoa):.12g}")
if __name__=="__main__": main()


