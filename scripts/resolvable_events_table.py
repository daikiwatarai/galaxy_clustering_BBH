#!/usr/bin/env python3
"""Table I of the letter: expected events per year that reach the host-resolvable scale.

Two criteria are computed, because the draft has used both:

  per_event : each Fisher event is evaluated on its OWN cylinder (R_perp, L_par),
              counting it if sum_h' Nbar_{h'|h}(theta_cyl) < 1  (mean neighbours < 1).
              No fiducial aspect ratio enters.
  fixed_V   : the event's cylinder volume is compared with V_res^(h), the volume at
              which a cylinder of FIXED aspect ratio A = L_par/R_perp reaches one mean
              neighbour.  This is what "V < V_res" in the table header means.

The two differ because the forecast cylinders do not share one A.

Rates are all-sky Fisher detections per year (median rate) times f_sky = 0.34, the same
normalisation as the f_Q forecast.  Geometry: data/geometry_z0to1.npz (R_h, L_h in h^-1 Mpc;
V in Mpc^3), or the directory given by HOSTINFER_DATA.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import brentq

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import cylinder_host_type_forecast as C  # noqa: E402

GEOM = Path(os.environ.get("HOSTINFER_DATA", ROOT / "data")) / "geometry_z0to1.npz"
OUT = ROOT / "results/resolvable_events_table.json"
NETS = [("O5_HLVK30", "O5"), ("POSTO5_FIDUCIAL_HLIVK", "post-O5"), ("ET+2CE", "3G")]
FSKY = 0.34
H_LITTLE = 0.70
A_FID = 5.0


def neighbour_table():
    """log(sum_h' nu_{h'|h}) on a log (R, L) grid; the smallest cells stay above the
    corner where the extrapolated cross term turns the mean negative."""
    lr = np.linspace(np.log(0.015), np.log(700.0), 90)
    ll = np.linspace(np.log(0.06), np.log(4500.0), 90)
    tab = {}
    for h in ("Q", "SF"):
        T = np.array([[C.means(np.exp(x), np.exp(y), "model")[h]["nu"].sum() for y in ll]
                      for x in lr])
        assert np.all(T > 0), f"non-positive neighbour mean on the grid for host {h}"
        tab[h] = RegularGridInterpolator((lr, ll), np.log(T), bounds_error=False,
                                         fill_value=None)
    return tab


def v_res(host, A):
    """Volume [Mpc^3] at which a cylinder of aspect ratio A has one mean neighbour."""
    def f(Vh):
        R = (Vh / (2 * np.pi * A)) ** (1 / 3)
        return C.means(R, A * R, "model")[host]["nu"].sum() - 1.0
    return brentq(f, 1e-6, 1e5) / H_LITTLE**3


def main() -> None:
    g = np.load(GEOM)
    tab = neighbour_table()
    vres = {h: v_res(h, A_FID) for h in ("Q", "SF")}
    out = {"f_sky": FSKY, "A_fid": A_FID, "V_res_Mpc3": vres, "networks": {}}
    print(f"V_res at A={A_FID:.0f}: Q {vres['Q']:.3f}, SF {vres['SF']:.3f} Mpc^3\n")
    print(f"{'network':8s} | {'per-event Q':>11s} {'per-event SF':>12s} | "
          f"{'V<V_res Q':>10s} {'V<V_res SF':>10s}   (events/yr, f_sky={FSKY})")
    for key, lab in NETS:
        R, L, V = g[key + "__R_h"], g[key + "__L_h"], g[key + "__V"]
        w = float(g[key + "__ndot"]) / len(R) * FSKY
        p = np.column_stack([np.log(R), np.log(L)])
        nb = {h: np.exp(tab[h](p)) for h in ("Q", "SF")}
        row = {
            "per_event": {h: float(np.sum(nb[h] < 1.0) * w) for h in ("Q", "SF")},
            "fixed_V": {h: float(np.sum(V < vres[h]) * w) for h in ("Q", "SF")},
            "n_events_sampled": int(len(R)),
        }
        out["networks"][lab] = row
        print(f"{lab:8s} | {row['per_event']['Q']:11.3f} {row['per_event']['SF']:12.3f} | "
              f"{row['fixed_V']['Q']:10.3f} {row['fixed_V']['SF']:10.3f}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
