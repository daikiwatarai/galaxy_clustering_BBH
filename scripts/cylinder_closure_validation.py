#!/usr/bin/env python3
"""Item 6: does the two-point closure improve as the cylinder is elongated?

Reads the SDSS cylinder counts and, for every (R_perp, L_par) on the grid, forms

    Sigma^{B,model}(h) = diag(nu^model) + Nbar^model Nbar^model Xi^model   (pure forecast)
    Sigma^{B,recal}(h) = diag(mu^emp)   + Nbar^emp  Nbar^emp  Xi^emp       (2-pt input removed)

    Xi^emp_ij = [Cov_rand(N_i,N_j) - delta_ij Nbar_i] / (Nbar_i Nbar_j)

and reports Cov_host,emp / Sigma as a function of A = L_par/R_perp and of L_par.
Errors are a 50-region k-means jackknife applied to the galaxy and random centres
together.  A ratio of 1 means the closure is exact.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cylinder_host_type_forecast as C

ROOT = Path(__file__).resolve().parent.parent
SRC = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(os.environ.get("HOSTINFER_DATA", ROOT / "data"))
OUT = ROOT / "results"
OUT.mkdir(parents=True, exist_ok=True)
LOG = []


def P_(s=""):
    print(s, flush=True)
    LOG.append(s)


def est(nq_g, ns_g, isf_g, nq_r, ns_r):
    """empirical moments and the recalibrated closure, for one subset."""
    R = np.column_stack([nq_r, ns_r]).astype(float)
    Nbar, Crand = R.mean(0), np.cov(R.T)
    XI = (Crand - np.diag(Nbar)) / np.outer(Nbar, Nbar)
    o = {"Nbar": Nbar, "Xi": XI, "Crand": Crand}
    for h, m in (("Q", ~isf_g), ("SF", isf_g)):
        H = np.column_stack([nq_g[m], ns_g[m]]).astype(float)
        mu, Cv = H.mean(0), np.cov(H.T)
        o[f"mu_{h}"] = mu
        o[f"C_{h}"] = Cv
        o[f"S_{h}"] = np.diag(mu) + np.outer(Nbar, Nbar) * XI      # B: unconditional
        o[f"D_{h}"] = np.diag(mu) + np.outer(mu, mu) * XI          # D: conditional mean
    return o


def main():
    d = np.load(SRC / "cyl_counts.npz")
    RE, LE = d["R_edges"], d["L_edges"]
    isf, greg, rreg = d["is_sf"], d["region"], d["rregion"]
    Hg, Hr, Ag, Ar = d["H_gal"], d["H_rand"], d["acc_gal"], d["acc_rand"]
    NJ = int(max(greg.max(), rreg.max())) + 1

    P_("=" * 104)
    P_("ITEM 6 -- EMPIRICAL CLOSURE VALIDATION IN CYLINDERS")
    P_("=" * 104)
    P_("\n1. MEAN CHECK.  Measured cylinder counts vs the model, and (at L=40 only) vs the")
    P_("   DIRECT published wp.  A cylinder with L_par = pi_max needs exactly the wp kernel,")
    P_("   so at L=40 this is a like-for-like comparison with no deprojection and no RSD model.")
    P_("%6s %6s %8s %6s | %-25s | %-25s | %8s"
       % ("R_per", "L_par", "V", "A", "Nbar_Q|Q  emp/model/ratio",
          "Nbar_SF|SF emp/model/ratio", "emp/direct"))

    rows = []
    for i, Rp in enumerate(RE):
        for j, L in enumerate(LE):
            gi = np.flatnonzero(Ag[i, j])
            ri = np.flatnonzero(Ar[i, j])
            if len(gi) < 500 or len(ri) < 500:
                continue
            e = est(Hg[gi, 0, i, j], Hg[gi, 1, i, j], isf[gi],
                    Hr[ri, 0, i, j], Hr[ri, 1, i, j])
            V = 2 * np.pi * Rp ** 2 * L
            mm = C.means(float(Rp), float(L), "model")
            XIm = C.Xi_matrix(float(Rp), float(L))
            Nm = mm["Nbar"]
            Sm = {h: np.diag(mm[h]["nu"]) + np.outer(Nm, Nm) * XIm for h in ("Q", "SF")}
            dr = None
            if abs(L - C.PIMAX) < 1e-9:
                dd = C.means(float(Rp), float(L), "direct")
                dr = (e["mu_Q"][0] / dd["Q"]["nu"][0], e["mu_SF"][1] / dd["SF"]["nu"][1])
            P_("%6.1f %6.1f %8.4g %6.2f | %8.3f %8.3f %7.3f | %8.3f %8.3f %7.3f | %8s"
               % (Rp, L, V, L / Rp, e["mu_Q"][0], mm["Q"]["nu"][0],
                  e["mu_Q"][0] / mm["Q"]["nu"][0], e["mu_SF"][1], mm["SF"]["nu"][1],
                  e["mu_SF"][1] / mm["SF"]["nu"][1],
                  "-" if dr is None else "%.3f/%.3f" % dr))

            # jackknife on the two ratio families
            def f(k):
                gm, rm = greg[gi] != k, rreg[ri] != k
                if min(gm.sum(), rm.sum()) < 200:
                    return None
                ee = est(Hg[gi[gm], 0, i, j], Hg[gi[gm], 1, i, j], isf[gi[gm]],
                         Hr[ri[rm], 0, i, j], Hr[ri[rm], 1, i, j])
                return np.array([ee["C_Q"][0, 0] / ee["S_Q"][0, 0],
                                 ee["C_SF"][1, 1] / ee["S_SF"][1, 1],
                                 ee["C_Q"][0, 0] / ee["D_Q"][0, 0],
                                 ee["C_SF"][1, 1] / ee["D_SF"][1, 1],
                                 ee["Xi"][0, 0] / XIm[0, 0]])
            v = [f(k) for k in range(NJ)]
            v = np.array([x for x in v if x is not None])
            err = np.sqrt((len(v) - 1) / len(v) * ((v - v.mean(0)) ** 2).sum(0))
            cen = np.array([e["C_Q"][0, 0] / e["S_Q"][0, 0],
                            e["C_SF"][1, 1] / e["S_SF"][1, 1],
                            e["C_Q"][0, 0] / e["D_Q"][0, 0],
                            e["C_SF"][1, 1] / e["D_SF"][1, 1],
                            e["Xi"][0, 0] / XIm[0, 0]])
            raw = (np.var(np.column_stack([Hg[gi, 0, i, j]])[~isf[gi]])
                   / np.var(Hr[ri, 0, i, j]))
            rows.append(dict(Rp=float(Rp), L=float(L), V=float(V), A=float(L / Rp),
                             n_gal=int(len(gi)), n_rand=int(len(ri)),
                             B_Q=cen[0], B_SF=cen[1], D_Q=cen[2],
                             D_SF=cen[3], raw_varratio=float(raw), xi_ratio=cen[4], err=err.tolist(),
                             mu_Q=e["mu_Q"].tolist(), mu_SF=e["mu_SF"].tolist(),
                             Xi_emp=e["Xi"].tolist(), Xi_model=XIm.tolist(),
                             C_Q=e["C_Q"].tolist(), C_SF=e["C_SF"].tolist(),
                             S_Q=e["S_Q"].tolist(), S_SF=e["S_SF"].tolist(),
                             dr=dr))

    P_("\n2. TWO-POINT INPUT.  Xi_QQ empirical / model (random centres; 1 = the frozen")
    P_("   power law describes this catalogue's cylinder variance).")
    P_("%6s %6s %6s | %14s" % ("R_per", "L_par", "A", "Xi_QQ emp/model"))
    for r in rows:
        P_("%6.1f %6.1f %6.2f | %8.3f +- %-6.3f" % (r["Rp"], r["L"], r["A"],
                                                     r["xi_ratio"], r["err"][4]))

    P_("\n3. THE ANSWER.  Cov_host,emp / Sigma, ordered by aspect ratio A.")
    P_("   B: clustering block uses the UNCONDITIONAL Nbar_i Nbar_j Xi (current fiducial)")
    P_("   D: clustering block uses the HOST-CONDITIONED mu_i mu_j Xi")
    P_("%6s %6s %6s %8s | %-19s %-19s | %-19s %-19s"
       % ("A", "R_per", "L_par", "V", "B  Q (QQ)", "B  SF (SFSF)",
          "D  Q (QQ)", "D  SF (SFSF)"))
    for r in sorted(rows, key=lambda x: x["A"]):
        P_("%6.2f %6.1f %6.1f %8.4g | %7.2f +- %-8.2f %7.2f +- %-8.2f | %7.2f +- %-8.2f %7.2f +- %-8.2f"
           % (r["A"], r["Rp"], r["L"], r["V"], r["B_Q"], r["err"][0],
              r["B_SF"], r["err"][1], r["D_Q"], r["err"][2],
              r["D_SF"], r["err"][3]))

    (OUT / "closure_validation.json").write_text(json.dumps(rows, indent=1))

    # -------------------------------------------------------------- figure 3
    fig, ax = plt.subplots(1, 3, figsize=(16.2, 4.7))
    Rlist = sorted({r["Rp"] for r in rows})
    cols = plt.cm.plasma(np.linspace(0.05, 0.8, len(Rlist)))
    for k, (key, ek, ttl) in enumerate(
            (("B_Q", 0, r"(a) current closure $\Sigma^B$: $\bar N_i\bar N_j\Xi_{ij}$"),
             ("D_Q", 2, r"(b) conditional closure $\Sigma^D$: $\nu_i\nu_j\Xi_{ij}$"))):
        a = ax[k]
        for c, Rp in zip(cols, Rlist):
            sel = [r for r in rows if r["Rp"] == Rp]
            sel.sort(key=lambda r: r["L"])
            a.errorbar([r["L"] for r in sel], [r[key] for r in sel],
                       [r["err"][ek] for r in sel], marker="o", ms=4, lw=1.9,
                       capsize=2, color=c, label=r"$R_\perp=%g$" % Rp)
        a.axhline(1, color="k", lw=1.4, ls="--")
        a.set_xscale("log"); a.set_yscale("log")
        a.set_xlabel(r"$L_\parallel\ [h^{-1}$Mpc]")
        a.set_ylabel(r"$\mathrm{Var}_{\rm emp}(N_Q|Q)\,/\,\Sigma_{QQ}$")
        a.set_ylim(0.12, 22)
        a.set_title(ttl + "\n1 = closure exact", fontsize=10)
        a.legend(fontsize=8, frameon=False, ncol=2)
    a = ax[2]
    for c, Rp in zip(cols, Rlist):
        sel = [r for r in rows if r["Rp"] == Rp]
        sel.sort(key=lambda r: r["L"])
        a.errorbar([r["L"] for r in sel], [r["xi_ratio"] for r in sel],
                   [r["err"][4] for r in sel], marker="s", ms=4, lw=1.9, capsize=2, color=c)
    a.axhline(1, color="k", lw=1.4, ls="--")
    a.set_xscale("log"); a.set_xlabel(r"$L_\parallel\ [h^{-1}$Mpc]")
    a.set_ylabel(r"$\Xi_{QQ}^{\rm emp}/\Xi_{QQ}^{\rm model}$")
    a.set_title("(c) two-point input: the cylinder removes\nthe redshift-space mismatch", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "fig3_closure_validation.png", dpi=150)
    fig.savefig(OUT / "fig3_closure_validation.pdf")
    (OUT / "closure_validation_log.txt").write_text("\n".join(LOG) + "\n")
    P_(f"\nwrote {OUT}/fig3_closure_validation.png, closure_validation.json")


if __name__ == "__main__":
    main()
