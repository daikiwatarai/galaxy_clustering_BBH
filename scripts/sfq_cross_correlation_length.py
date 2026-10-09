#!/usr/bin/env python3
"""Correlation lengths r0 for the SF/Q auto- AND cross-correlation.

The auto pieces are already fitted elsewhere; this refits them here so that all
four curves share one estimator, one fit range and one pi_max, and then adds the
cross term, which nothing has fitted so far.

The cross term is not measured.  It is reconstructed from the exact binary
partition identity, applied at the w_p level rather than the xi level because
w_p is linear in xi and the published errors live on w_p:

    w_all = f_SF^2 w_SF + f_Q^2 w_Q + 2 f_SF f_Q w_X
    =>  w_X = (w_all - f_SF^2 w_SF - f_Q^2 w_Q) / (2 f_SF f_Q)

and then the same truncated power law is fitted to w_X,

    w_p(rp) = 2 int_0^pimax (sqrt(rp^2 + pi^2)/r0)^-gamma dpi

over 0.3 < rp < 20 h^-1 Mpc, pi_max = 40 h^-1 Mpc.

Two things this cannot do, and does not pretend to:

  * The identity needs all three curves to come from ONE parent, mask and
    estimator.  w_SF and w_Q are Watson+2015 A1/A2; w_all is Hearin+2014
    Table 1, which results/sfq_input_audit flags as NOT unambiguously "all"
    (Table 3, RED, is byte-identical to Table 1 in arXiv v3 and in MNRAS 444,
    729).  If Table 1 is red rather than all, the reconstruction is void.  The
    r <= 1 check below is what stands between us and that failure mode.
  * No uncertainty on r0 or gamma, following the convention of
    sdss_split_auto_correlation.py.  A sensitivity scan over f_SF is reported
    instead, because f_SF is derived from the parent, not published.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from cylinder_host_type_forecast import F_Q, F_SF, NPARENT   # the bright0 parent

WP_JSON = Path(os.environ.get("HOSTINFER_DATA", ROOT / "data")) / "wp_measured.json"
OUTDIR = ROOT / "results" / "sfq_correlation_length"

PIMAX = 40.0
FIT_LO, FIT_HI = 0.3, 20.0
H_LITTLE = 0.70

_GL_X, _GL_W = np.polynomial.legendre.leggauss(256)


def wp_powerlaw(rp, r0, gamma, pimax=PIMAX):
    rp = np.atleast_1d(np.asarray(rp, float))
    pi = 0.5 * pimax * (_GL_X + 1.0)
    r = np.hypot(rp[:, None], pi[None, :])
    return pimax * ((r / r0) ** (-gamma) @ _GL_W)


def fit_powerlaw(rp, wp, sigma, lo=FIT_LO, hi=FIT_HI, pimax=PIMAX):
    from scipy.optimize import curve_fit

    m = (rp >= lo) & (rp <= hi) & np.isfinite(wp) & (wp > 0)
    p, _ = curve_fit(lambda x, r0, g: wp_powerlaw(x, r0, g, pimax),
                     rp[m], wp[m], p0=[6.0, 1.8], sigma=sigma[m],
                     absolute_sigma=True, maxfev=20000)
    resid = wp[m] - wp_powerlaw(rp[m], p[0], p[1], pimax)
    chi2 = ((resid / sigma[m]) ** 2).sum() / (m.sum() - 2)
    return float(p[0]), float(p[1]), float(chi2), int(m.sum())


def main() -> None:
    d = json.loads(WP_JSON.read_text())

    rp = np.array(d["rp"])
    wp = {k: np.array(v) for k, v in d["wp"].items()}
    sig = {k: np.array(v) for k, v in d["sig"].items()}
    f_SF, f_Q = F_SF, F_Q

    # ---- reconstruct the cross term on the measured rp grid -----------------
    num = wp["all"] - f_SF**2 * wp["A"] - f_Q**2 * wp["B"]
    wp_X = num / (2.0 * f_SF * f_Q)
    # treating the three as independent OVERSTATES sigma_X: they share a parent
    # and are positively correlated, which cancels in a difference.  Only the
    # shape across rp moves the best fit, so this is a safe weighting choice.
    sig_X = np.sqrt(sig["all"] ** 2 + f_SF**4 * sig["A"] ** 2
                    + f_Q**4 * sig["B"] ** 2) / (2.0 * f_SF * f_Q)

    fits = {}
    for key, lab in [("all", "all"), ("A", "SF x SF"), ("B", "Q x Q")]:
        r0h, g, c2, n = fit_powerlaw(rp, wp[key], sig[key])
        fits[lab] = dict(r0_hinvMpc=r0h, r0_Mpc=r0h / H_LITTLE, gamma=g,
                         chi2_dof=c2, nbin=n)
    r0h, g, c2, n = fit_powerlaw(rp, wp_X, sig_X)
    fits["SF x Q"] = dict(r0_hinvMpc=r0h, r0_Mpc=r0h / H_LITTLE, gamma=g,
                          chi2_dof=c2, nbin=n)

    # ---- physical admissibility: cross-correlation coefficient --------------
    # Cauchy-Schwarz bounds xi_X <= sqrt(xi_SF xi_Q), and integrating along the
    # line of sight preserves it, so w_X <= sqrt(w_SF w_Q) must hold on the
    # MEASURED bins.  That is the test that can actually fail.  The same ratio
    # built from the three fitted power laws is only a consistency check on the
    # fits: three independent power laws with different gamma cannot satisfy the
    # partition identity at every r, so it drifts by construction.
    mfit = (rp >= FIT_LO) & (rp <= FIT_HI)
    r_meas = wp_X / np.sqrt(wp["A"] * wp["B"])
    r_sig = r_meas * np.sqrt((sig_X / wp_X) ** 2
                             + 0.25 * (sig["A"] / wp["A"]) ** 2
                             + 0.25 * (sig["B"] / wp["B"]) ** 2)
    # how many sigma above 1 the worst bin sits; <1 sigma means the bound holds
    excess = np.where(r_meas > 1.0, (r_meas - 1.0) / r_sig, 0.0)

    rgrid = np.geomspace(FIT_LO, FIT_HI, 60)

    def xi(lab):
        return (rgrid / fits[lab]["r0_hinvMpc"]) ** (-fits[lab]["gamma"])

    rcoef = xi("SF x Q") / np.sqrt(xi("SF x SF") * xi("Q x Q"))

    # deterministic-bias benchmark: equal gamma => r0_X = sqrt(r0_SF r0_Q)
    r0_geom = np.sqrt(fits["SF x SF"]["r0_hinvMpc"] * fits["Q x Q"]["r0_hinvMpc"])

    # ---- sensitivity to f_SF (derived, not published) -----------------------
    scan = []
    for fs in [f_SF - 0.05, f_SF - 0.02, f_SF, f_SF + 0.02, f_SF + 0.05]:
        fq = 1.0 - fs
        wx = (wp["all"] - fs**2 * wp["A"] - fq**2 * wp["B"]) / (2 * fs * fq)
        sx = np.sqrt(sig["all"] ** 2 + fs**4 * sig["A"] ** 2
                     + fq**4 * sig["B"] ** 2) / (2 * fs * fq)
        a, b, _, _ = fit_powerlaw(rp, wx, sx)
        scan.append(dict(f_SF=fs, r0_hinvMpc=a, r0_Mpc=a / H_LITTLE, gamma=b))

    # ---- report -------------------------------------------------------------
    print(f"parent: N={NPARENT}  f_SF={f_SF:.4f}  f_Q={f_Q:.4f}")
    print(f"fit: {FIT_LO}-{FIT_HI} h^-1 Mpc, pi_max={PIMAX}, published diagonal errors\n")
    print(f"{'':10s} {'r0 [h-1 Mpc]':>13s} {'r0 [Mpc]':>10s} {'gamma':>7s} "
          f"{'chi2/dof':>9s} {'nbin':>5s}")
    for lab in ["all", "SF x SF", "Q x Q", "SF x Q"]:
        f = fits[lab]
        print(f"{lab:10s} {f['r0_hinvMpc']:13.3f} {f['r0_Mpc']:10.3f} "
              f"{f['gamma']:7.3f} {f['chi2_dof']:9.3f} {f['nbin']:5d}")

    print(f"\nsqrt(r0_SF r0_Q) = {r0_geom:.3f} h^-1 Mpc  "
          f"(equal-gamma deterministic-bias benchmark)")
    print("\nCauchy-Schwarz on the MEASURED bins, w_X / sqrt(w_SF w_Q):")
    print(f"  range over the fit bins: {r_meas[mfit].min():.3f} - "
          f"{r_meas[mfit].max():.3f}")
    print(f"  worst bin sits {excess[mfit].max():.2f} sigma above 1  -> "
          + ("consistent with r <= 1" if excess[mfit].max() < 1.0
             else "GENUINE VIOLATION, the inputs are inconsistent"))
    for x, v, s in zip(rp[mfit], r_meas[mfit], r_sig[mfit]):
        print(f"    rp={x:7.3f}  r={v:.3f} +- {s:.3f}")
    print(f"\nsame ratio from the three fitted power laws (consistency of the "
          f"fits, not a measurement): {rcoef.min():.3f} - {rcoef.max():.3f}")
    print(f"  at 0.3 / 1 / 5 / 20 h^-1 Mpc: "
          + " / ".join(f"{np.interp(x, rgrid, rcoef):.3f}" for x in [0.3, 1, 5, 20]))

    print("\nsensitivity of the cross fit to f_SF:")
    for s in scan:
        print(f"  f_SF={s['f_SF']:.3f}  r0={s['r0_hinvMpc']:7.3f} h^-1 Mpc "
              f"({s['r0_Mpc']:6.3f} Mpc)  gamma={s['gamma']:.3f}")

    OUTDIR.mkdir(parents=True, exist_ok=True)
    (OUTDIR / "correlation_lengths.json").write_text(json.dumps(dict(
        provenance=dict(
            wp_SF_Q="Watson+2015 (arXiv:1403.1578) Tables A1, A2",
            wp_all="Hearin+2014 (arXiv:1310.6747) Table 1 -- AMBIGUOUS, see "
                   "results/sfq_input_audit: Table 3 (RED) is identical to it",
            f_from="results/sfq_localization_watson/watson_results.json"),
        config=dict(pimax=PIMAX, fit_lo=FIT_LO, fit_hi=FIT_HI, h=H_LITTLE,
                    weighting="published diagonal errors"),
        f_SF=f_SF, f_Q=f_Q, fits=fits,
        r0_geometric_mean_hinvMpc=r0_geom,
        cross_coefficient_measured=dict(
            rp_hinvMpc=rp[mfit].tolist(), r=r_meas[mfit].tolist(),
            sigma=r_sig[mfit].tolist(),
            r_min=float(r_meas[mfit].min()), r_max=float(r_meas[mfit].max()),
            max_excess_sigma=float(excess[mfit].max()),
            consistent_with_unity=bool(excess[mfit].max() < 1.0)),
        cross_coefficient_from_fits=dict(
            r_min=float(rcoef.min()), r_max=float(rcoef.max()),
            r_grid_hinvMpc=rgrid.tolist(), r_of_r=rcoef.tolist()),
        f_SF_sensitivity=scan,
        wp_cross_reconstructed=dict(rp_hinvMpc=rp.tolist(), wp=wp_X.tolist(),
                                    sigma_independent_upper=sig_X.tolist()),
    ), indent=2))
    print(f"\nwrote {OUTDIR / 'correlation_lengths.json'}")


if __name__ == "__main__":
    main()
