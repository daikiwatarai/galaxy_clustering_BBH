#!/usr/bin/env python3
"""How much of the lambda_Q forecast comes from events beyond the clustering sample's redshift?

The SF/Q correlation functions are measured in SDSS at 0.02 < z < 0.067 and held fixed out
to z = 1 in the forecast.  This reruns the 1-yr all-Q mocks of fq_posterior_forecast.py with
the SAME seeds and random-number sequence (so the all-event limits reproduce the paper), and
recomputes each mock's posterior using only the events below / above a redshift cut.

Reported per network: median over mocks of the one-sided 90% lower credible limit on
lambda_Q, for all events and for redshift-selected subsets, plus the share of the summed
log-likelihood ratio sum_i [ln p(N_i|Q) - ln p(N_i|SF)] carried by z <= Z_SDSS events.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import fq_posterior_forecast as F  # noqa: E402

Z_SDSS = 0.067                      # upper edge of the SDSS bright0 clustering sample
CUTS = [Z_SDSS, 0.1, 0.2, 0.5]
T, F0, N_MOCK = 1.0, 1.0, 100
OUT = ROOT / "results/fq_forecast_redshift_contribution.json"


def lower_limit(ll, q=0.10):
    p = np.exp(ll - ll.max())
    c = np.r_[0.0, np.cumsum(0.5 * (p[1:] + p[:-1]) * np.diff(F.FG))]
    return float(np.interp(q, c / c[-1], F.FG))


def main() -> None:
    G, tb = F.geometry(), F.nu_xi_tables()
    out = {"z_sdss": Z_SDSS, "cuts": CUTS, "networks": {}}
    for k, key in enumerate(F.KEYS):
        seed = 20260913 + 1000 * k + int(10 * T)            # as in fq_posterior_forecast.main
        Rh, Lh, V, z = (G[key + "__" + q] for q in ("R_h", "L_h", "V", "z"))
        use = V <= F.VMAX
        mean_n = T * float(G[key + "__ndot"]) / int(G[key + "__nsample"]) * F.FSKY * use.sum()
        Rh, Lh, V, z = Rh[use], Lh[use], V[use], z[use]
        rng = np.random.default_rng(seed)
        lim = {"all": []} | {f"z<={c}": [] for c in CUTS} | {f"z>{Z_SDSS}": []}
        share, frac_n = [], []
        for m in range(N_MOCK):
            n = rng.poisson(mean_n); i = rng.integers(0, len(V), n)
            isQ = rng.random(n) < F0
            lq, ls = F.event_loglikes(Rh[i], Lh[i], V[i], isQ, tb, rng)
            zi = z[i]
            lim["all"].append(lower_limit(F.loglike_grid(lq, ls)))
            for c in CUTS:
                s = zi <= c
                lim[f"z<={c}"].append(lower_limit(F.loglike_grid(lq[s], ls[s])))
            s = zi > Z_SDSS
            lim[f"z>{Z_SDSS}"].append(lower_limit(F.loglike_grid(lq[s], ls[s])))
            lr = np.clip(lq - ls, -50, 50)                      # host-only events are +-inf
            share.append(lr[zi <= Z_SDSS].sum() / lr.sum())
            frac_n.append(np.mean(zi <= Z_SDSS))
        row = {kk: float(np.median(v)) for kk, v in lim.items()}
        row["share_of_loglike_ratio_from_z<=z_sdss"] = float(np.median(share))
        row["fraction_of_events_z<=z_sdss"] = float(np.median(frac_n))
        row["mean_events"] = float(mean_n)
        out["networks"][F.NAMES[key]] = row
        print(f"{F.NAMES[key]:8s} all {row['all']:.3f} | " +
              "  ".join(f"z<={c}: {row[f'z<={c}']:.3f}" for c in CUTS) +
              f" | z>{Z_SDSS}: {row[f'z>{Z_SDSS}']:.3f} | events z<={Z_SDSS}: "
              f"{100*row['fraction_of_events_z<=z_sdss']:.1f}%  logLR share: "
              f"{100*row['share_of_loglike_ratio_from_z<=z_sdss']:.0f}%", flush=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
