#!/usr/bin/env python3
"""Measured vs PLN+Sigma^D neighbour-count distribution, as a function of cylinder volume.

Shows the median and the 16-84% width only.  Facets by host type; within a panel
the comparison is measured (points) against model (band), so what is being judged
is encoded by mark type, not by hue.

Only the (R_perp, L_par) pairs present in results/sdss_cylinder_counts can be
used, so "A = 5" in practice means the A = L/R in [3.3, 8] subset -- the same
range the other cylinder figures use.

Counts exclude the central host, matching nu in the forecast module.

Two different uncertainties appear here and must not be confused:
  * the error bars in the top row are the 16-84% SPREAD of the cylinder-to-
    cylinder count distribution -- the thing the model has to reproduce;
  * the error bars in the bottom row are the statistical uncertainty on the
    measured percentile, from a 50-region jackknife.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import cylinder_host_type_forecast as C  # noqa: E402

SRC = Path(os.environ.get("HOSTINFER_DATA", ROOT / "data")) / "cyl_counts.npz"
OUT = ROOT / "figures"
H_LITTLE = 0.70
A_LO, A_HI = 3.3, 8.0
L_MIN = 30.0                  # --select lpar: every cell with L_par >= L_MIN, any A
NMC = 400_000
SEED = 20260919
QS = [16, 50, 84]

INK = "#1A1A1A"
MODEL = "#3E6E9E"
BAND = "#A9C6DE"
PC = [("#2166AC", "v", "16%"), (INK, "o", "median"), ("#B2182B", "^", "84%")]


def model_percentiles(Rp, L, host, rng):
    """PLN + Sigma^D draw of the total neighbour count."""
    nu = C.means(Rp, L, "model")[host]["nu"]
    S = np.log1p(C.Xi_matrix(Rp, L))       # Sigma^D closure => S = ln(1 + Xi)
    x = (-0.5 * np.diag(S))[None, :] \
        + rng.standard_normal((NMC, 2)) @ np.linalg.cholesky(S).T
    return np.percentile(rng.poisson(nu[None, :] * np.exp(x)).sum(1), QS)


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--select", choices=("aspect", "lpar"), default="aspect",
                    help="aspect: A in [A_LO, A_HI]; lpar: L_par >= L_MIN (the regime "
                         "in which the closure is validated)")
    a = ap.parse_args()
    d = np.load(SRC)
    RE, LE = d["R_edges"], d["L_edges"]
    H, acc, isf, greg = d["H_gal"], d["acc_gal"], d["is_sf"], d["region"]
    NJ = int(greg.max()) + 1
    rng = np.random.default_rng(SEED)

    cells = []
    for i, Rp in enumerate(RE):
        for j, L in enumerate(LE):
            keep = (A_LO <= L / Rp <= A_HI) if a.select == "aspect" else (L >= L_MIN)
            if keep:
                cells.append((i, j, float(Rp), float(L), L / Rp,
                              2 * np.pi * Rp**2 * L / H_LITTLE**3))
    cells.sort(key=lambda c: c[5])
    what = (f"A in [{A_LO}, {A_HI}]" if a.select == "aspect"
            else f"L_par >= {L_MIN:.0f} h^-1 Mpc")
    print(f"{len(cells)} cells with {what}")

    fig, axg = plt.subplots(2, 2, figsize=(10.0, 6.4), sharex=True,
                            gridspec_kw=dict(height_ratios=[2.1, 1.0],
                                             hspace=0.08, wspace=0.06))
    axg[0][1].sharey(axg[0][0])
    axg[1][1].sharey(axg[1][0])
    store = {}

    for ax, (host, title) in zip(axg[0], [("Q", "quenched host"),
                                          ("SF", "star-forming host")]):
        V, emp, mod, eerr = [], [], [], []
        for i, j, Rp, L, A, Vmpc in cells:
            sel = acc[i, j] & ((~isf) if host == "Q" else isf)
            ntot = (H[sel, 0, i, j] + H[sel, 1, i, j]).astype(float)
            reg = greg[sel]
            emp.append(np.percentile(ntot, QS))
            jk = np.array([np.percentile(ntot[reg != r], QS)
                           for r in range(NJ) if (reg != r).sum() > 100])
            eerr.append(np.sqrt((len(jk) - 1) / len(jk)
                                * ((jk - jk.mean(0))**2).sum(0)))
            mod.append(model_percentiles(Rp, L, host, rng))
            V.append(Vmpc)
            print(f"  {host:2s} R={Rp:4.1f} L={L:4.1f} A={A:4.2f} "
                  f"V={Vmpc:9.0f} n={sel.sum():6d}  "
                  f"med emp={emp[-1][1]:7.1f} mod={mod[-1][1]:7.1f}")
        V, emp, mod, eerr = (np.array(a) for a in (V, emp, mod, eerr))
        store[host] = (V, emp, mod, eerr)

        first = host == "Q"
        ax.fill_between(V, mod[:, 0], mod[:, 2], color=BAND, lw=0,
                        label="PLN model, 16--84%" if first else None)
        ax.plot(V, mod[:, 1], color=MODEL, lw=1.6,
                label="PLN model, median" if first else None)
        Lc = np.array([c[3] for c in cells])
        groups = ([(np.ones(len(V), bool), "o", "SDSS, median and 16--84%")]
                  if a.select == "aspect" else
                  [(Lc == Lv, mk, f"SDSS, $L_\\parallel={Lv:.0f}\\,h^{{-1}}$Mpc")
                   for Lv, mk in ((30.0, "o"), (40.0, "s"))])
        for m, mk, lab in groups:
            ax.errorbar(V[m], emp[m, 1],
                        yerr=[emp[m, 1] - emp[m, 0], emp[m, 2] - emp[m, 1]],
                        fmt=mk, ms=5.5, mfc="white", mec=INK, ecolor=INK,
                        elinewidth=1.2, capsize=2.5, mew=1.3, ls="none", zorder=5,
                        label=lab if first else None)
        ax.set_yscale("log")
        ax.set_title(f"({'a' if first else 'b'}) {title}", fontsize=11)
        ax.grid(alpha=0.25, lw=0.5)
        ax.set_axisbelow(True)

    for ax, host in zip(axg[1], ("Q", "SF")):
        V, emp, mod, eerr = store[host]
        for k, (col, mk, lab) in enumerate(PC):
            ax.errorbar(V, emp[:, k] / mod[:, k], yerr=eerr[:, k] / mod[:, k],
                        fmt=mk + "-", color=col, ms=5, lw=1.2, mew=0.8,
                        elinewidth=1.0, capsize=2,
                        mfc="white" if k == 1 else col,
                        label=lab if host == "Q" else None)
        ax.axhline(1.0, color=INK, lw=0.9)
        ax.axhspan(0.9, 1.1, color=INK, alpha=0.06, lw=0)
        ax.set_xscale("log")
        ax.set_xlabel(r"$V\ [{\rm Mpc}^3]$")
        ax.grid(alpha=0.25, lw=0.5)
        ax.set_axisbelow(True)

    axg[1][0].set_ylim(0.82, 1.32)
    axg[1][0].set_ylabel("SDSS / model")
    axg[1][0].legend(fontsize=8, ncol=3, loc="upper center", framealpha=0.95,
                     columnspacing=1.1, handletextpad=0.4)
    for ax in (axg[0][1], axg[1][1]):
        plt.setp(ax.get_yticklabels(), visible=False)
    axg[0][0].set_ylabel(r"neighbouring galaxies $N_{\rm tot}$")
    axg[0][0].legend(fontsize=8.5, loc="upper left", framealpha=0.95)
    fig.suptitle("Neighbour-count distribution: SDSS against the PLN model, " +
                 ("$A=L_\\parallel/R_\\perp = 3.3$--$8$" if a.select == "aspect"
                  else "$L_\\parallel\\geq%.0f\\,h^{-1}$Mpc" % L_MIN), fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    stem = ("count_distribution_vs_volume" if a.select == "aspect"
            else f"count_distribution_Lpar{L_MIN:.0f}")
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{stem}.{ext}", dpi=150)
    print(f"\nwrote {OUT / (stem + '.png')}\n")

    print("measured/model, with 50-region jackknife on the measured percentile")
    for host in ("Q", "SF"):
        V, emp, mod, eerr = store[host]
        for k, q in enumerate(QS):
            rat = emp[:, k] / mod[:, k]
            er = np.where(eerr[:, k] > 0, eerr[:, k] / mod[:, k], np.nan)
            z = (rat - 1.0) / er
            print(f"  {host:2s} P{q:3d}: med={np.median(rat):.3f} "
                  f"range=[{rat.min():.3f},{rat.max():.3f}]  "
                  f"median |z|={np.nanmedian(np.abs(z)):4.2f}  "
                  f"within 1 sigma: {int(np.nansum(np.abs(z) < 1))}/{len(rat)}")


if __name__ == "__main__":
    main()
