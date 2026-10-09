#!/usr/bin/env python3
"""Host-type distinguishability with a CYLINDRICAL localization volume.

A full reformulation.  No spherical result is reused and no spherical closed form
is imported: the sphere expressions 3/(3-g)(R/r0)^-g and 3 J(g) (R/r0)^-g are
geometry-specific and are simply wrong for a cylinder.

GEOMETRY
    r_perp < R_perp,   |s_par| < L_par,   V = 2 pi R_perp^2 L_par
    A = L_par / R_perp                       (aspect ratio; A=1 is compact)

MEAN COUNTS -- built from the projected correlation function directly
    Nbar_{h'|h} = nbar_h' V + 2 pi nbar_h' \\int_0^{R_perp} dr_perp r_perp wp^{(L)}_{hh'}

    Two branches, kept strictly separate:

    [DIRECT]  L_par = pi_max = 40 h^-1 Mpc.  The published wp IS
              \\int_{-40}^{+40} ds_par xi^s(r_perp, s_par) measured in REDSHIFT
              space, which is exactly the kernel a redshift-space cylinder of
              L_par = 40 needs.  No deprojection, no power law, no RSD model.
              This is the only branch that is a measurement.

    [MODEL]   any L_par.  wp^{(L)}(r_perp) = 2 \\int_0^L dpi xi(sqrt(r_perp^2+pi^2))
              with the frozen REAL-space power laws.  Real space, so it is biased
              for L_par below the finger-of-god scale (~10-20 h^-1 Mpc); the
              measured suppression there is a factor 0.2-0.5.  Flagged everywhere.

    At L_par = 40 the two branches must agree; that agreement is the fit check.

COVARIANCE -- the cylinder window's own pair kernel, integrated numerically
    Sigma^{(h),B}_ij = delta_ij nu_i^(h) + nbar_i nbar_j \\int_V\\int_V xi_ij(x1-x2)

    The double integral is reduced with the window autocorrelation
    G_V(r_perp,s_par) = A2(r_perp) T1(s_par), which FACTORISES for a cylinder:
        A2(d)  = 2 R^2 arccos(d/2R) - (d/2) sqrt(4R^2-d^2)     (disc-disc overlap)
        T1(s)  = 2L - |s|                                       (segment overlap)
    so with (r_perp,s_par) = (r sin t, r cos t),
        P(r) = 4 pi r^2 \\int_0^{pi/2} sin t dt A2(r sin t) T1(r cos t),
        \\int_V\\int_V xi = \\int_0^{rmax} dr P(r) xi(r),   \\int P dr = V^2.
    P(r) also says exactly which separations the covariance samples, which is what
    decides whether the 0.3-20 h^-1 Mpc wp fit range is respected.

    The covariance cannot be built from the published wp: it needs xi at all
    separations, so it always uses the power-law fits.  Stated, not hidden.

Units: h^-1 Mpc throughout (r0 in h^-1 Mpc, nbar in (h^-1 Mpc)^-3).  No h enters.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.special import hyp2f1
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results/cylinder_hosttype"
OUT.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------------------- frozen inputs
import os
_DATA = Path(os.environ.get("HOSTINFER_DATA", ROOT / "data"))
_WP = next(p for p in (_DATA / "wp_measured.json", ROOT / "scripts/_wp_data.json")
           if p.exists())        # see DATA.md
D = json.loads(_WP.read_text())
RP = np.array(D["rp"])                       # h^-1 Mpc, 15 bins 0.1217 - 17.1579
WP = {k: np.array(v) for k, v in D["wp"].items()}     # all / A=SF / B=Q
SIG = {k: np.array(v) for k, v in D["sig"].items()}
PIMAX = float(D["pimax"])                    # 40 h^-1 Mpc
FIT_LO, FIT_HI = float(D["fit_lo"]), float(D["fit_hi"])   # 0.3, 20 h^-1 Mpc

# power-law fits, primitive units (h^-1 Mpc).  SSFR_REF is quoted in physical Mpc
# with h=0.70 by the wp script; r0[h^-1 Mpc] = r0[Mpc] * 0.70.
H_WP = 0.70
R0 = {"all": D["ssfr_ref"]["all"][0] * H_WP,
      "SS": D["ssfr_ref"]["A"][0] * H_WP,
      "QQ": D["ssfr_ref"]["B"][0] * H_WP}
GA = {"all": D["ssfr_ref"]["all"][1], "SS": D["ssfr_ref"]["A"][1], "QQ": D["ssfr_ref"]["B"][1]}

F_SF, F_Q = 0.5230905030025009, 0.4769094969974990
NPARENT = 74771                              # bright0 w_p parent: 39112 SF + 35659 Q
NBAR = 0.012702939655410006                  # (h^-1 Mpc)^-3, survey volume at Om0=0.27
N_ = {"Q": F_Q * NBAR, "SF": F_SF * NBAR}

TRAPZ = getattr(np, "trapezoid", np.trapz)   # numpy>=2 renamed trapz

LOG: list[str] = []


def P_(s=""):
    print(s, flush=True)
    LOG.append(s)


# ----------------------------------------------------------------- wp: data branch
def wp_cross_data():
    """wp_QSF from the binary-partition identity applied bin-by-bin to the DATA.
    wp is linear in xi, so the identity carries over unchanged."""
    w = (WP["all"] - F_SF ** 2 * WP["A"] - F_Q ** 2 * WP["B"]) / (2 * F_SF * F_Q)
    s = np.sqrt(SIG["all"] ** 2 + (F_SF ** 2 * SIG["A"]) ** 2
                + (F_Q ** 2 * SIG["B"]) ** 2) / (2 * F_SF * F_Q)
    return w, s


WP_D = {"QQ": WP["B"], "SFSF": WP["A"], "QSF": wp_cross_data()[0]}
SIG_D = {"QQ": SIG["B"], "SFSF": SIG["A"], "QSF": wp_cross_data()[1]}


def _ll(lr, lw, sgn, r):
    x = np.log(np.atleast_1d(r))
    y = np.interp(x, lr, lw)
    lo = x < lr[0]
    hi = x > lr[-1]
    if lo.any():
        m = (lw[1] - lw[0]) / (lr[1] - lr[0])
        y[lo] = lw[0] + m * (x[lo] - lr[0])
    if hi.any():
        m = (lw[-1] - lw[-2]) / (lr[-1] - lr[-2])
        y[hi] = lw[-1] + m * (x[hi] - lr[-1])
    return sgn * np.exp(y)


# ----------------------------------------------------------------- wp: model branch
def xi_pl(r, key):
    return (np.asarray(r) / R0[key]) ** (-GA[key])


def xi_pair(r, i, j):
    """xi_ij from the same binary-partition identity used everywhere in the project."""
    if (i, j) in (("Q", "Q"),):
        return xi_pl(r, "QQ")
    if (i, j) in (("SF", "SF"),):
        return xi_pl(r, "SS")
    return (xi_pl(r, "all") - F_SF ** 2 * xi_pl(r, "SS")
            - F_Q ** 2 * xi_pl(r, "QQ")) / (2 * F_SF * F_Q)


def _wp_pl(rperp, L, key):
    """2 int_0^L dpi (sqrt(rp^2+pi^2)/r0)^-gamma, in closed form.
    int_0^L (rp^2+pi^2)^{-g/2} dpi = rp^{1-g} x 2F1(1/2, g/2; 3/2; -x^2), x = L/rp.
    Verified against quad to machine precision."""
    g, r0 = GA[key], R0[key]
    rp = np.asarray(rperp, float)
    x = L / rp
    return 2.0 * r0 ** g * rp ** (1 - g) * x * hyp2f1(0.5, g / 2, 1.5, -x * x)


def wp_model(rperp, L, i, j):
    """wp^{(L)}(rperp) for any pair, via the binary-partition identity (linear in xi)."""
    if (i, j) == ("Q", "Q"):
        return _wp_pl(rperp, L, "QQ")
    if (i, j) == ("SF", "SF"):
        return _wp_pl(rperp, L, "SS")
    return (_wp_pl(rperp, L, "all") - F_SF ** 2 * _wp_pl(rperp, L, "SS")
            - F_Q ** 2 * _wp_pl(rperp, L, "QQ")) / (2 * F_SF * F_Q)


# ----------------------------------------------------------------- mean counts
def clustering_count_integral(Rp, L, i, j, branch, nr=800):
    """2 pi int_0^Rp dr r wp^{(L)}_ij(r)  -- the excess count per unit nbar."""
    r = np.geomspace(1e-4, Rp, nr)
    key = {("Q", "Q"): "QQ", ("SF", "SF"): "SFSF"}.get((i, j), "QSF")
    w = _ll(np.log(RP), np.log(np.abs(WP_D[key])), np.sign(WP_D[key][0]), r) \
        if branch == "direct" else wp_model(r, L, i, j)
    return 2 * np.pi * TRAPZ(r * w, r)


def means(Rp, L, branch):
    """(mu, nu, xibar) for both hosts.  nu = neighbour means; mu adds the host itself."""
    V = 2 * np.pi * Rp ** 2 * L
    out = {}
    for h in ("Q", "SF"):
        nu, xb = {}, {}
        for i in ("Q", "SF"):
            exc = clustering_count_integral(Rp, L, h, i, branch)
            nu[i] = N_[i] * V + N_[i] * exc
            xb[i] = exc / V
        out[h] = dict(nu=np.array([nu["Q"], nu["SF"]]),
                      xibar=np.array([xb["Q"], xb["SF"]]),
                      mu=np.array([nu["Q"], nu["SF"]]) + (np.array([1.0, 0.0]) if h == "Q"
                                                          else np.array([0.0, 1.0])))
    out["V"] = V
    out["Nbar"] = np.array([N_["Q"] * V, N_["SF"] * V])
    return out


# ----------------------------------------------------------------- cylinder kernel
def A2(d, Rp):
    d = np.clip(np.asarray(d), 0, None)
    x = np.clip(d / (2 * Rp), 0, 1)
    return np.where(d < 2 * Rp,
                    2 * Rp ** 2 * np.arccos(x) - (d / 2) * np.sqrt(np.clip(4 * Rp ** 2 - d ** 2, 0, None)),
                    0.0)


def T1(s, L):
    s = np.abs(np.asarray(s))
    return np.where(s < 2 * L, 2 * L - s, 0.0)


def pair_pdf(r, Rp, L, nt=400):
    """P(r) = 4 pi r^2 int_0^{pi/2} sin t dt A2(r sin t) T1(r cos t);  int P dr = V^2."""
    t, wt = np.polynomial.legendre.leggauss(nt)
    t = 0.5 * np.pi * (0.5 * (t + 1.0))
    wt = wt * 0.5 * (0.5 * np.pi)
    r = np.atleast_1d(r)[:, None]
    g = np.sin(t)[None, :] * A2(r * np.sin(t)[None, :], Rp) * T1(r * np.cos(t)[None, :], L)
    return 4 * np.pi * (r[:, 0] ** 2) * (g * wt[None, :]).sum(axis=1)


def pair_grid(Rp, L, nr=4000, nt=400):
    rmax = np.hypot(2 * Rp, 2 * L)
    r = np.geomspace(1e-6 * min(Rp, L), rmax, nr)
    return r, pair_pdf(r, Rp, L, nt)


def double_integral(Rp, L, i, j, grid=None, nr=4000, nt=400):
    r, P = grid if grid is not None else pair_grid(Rp, L, nr, nt)
    return float(TRAPZ(P * xi_pair(r, i, j) * r, np.log(r)))


def Xi_matrix(Rp, L, nr=4000, nt=400):
    V = 2 * np.pi * Rp ** 2 * L
    g = pair_grid(Rp, L, nr, nt)
    M = np.zeros((2, 2))
    for a, i in enumerate(("Q", "SF")):
        for b, j in enumerate(("Q", "SF")):
            M[a, b] = double_integral(Rp, L, i, j, grid=g) / V ** 2
    return M


def pair_weight_outside_fit_range(Rp, L, i="Q", j="Q", nr=8000, nt=400):
    """fraction of the covariance integrand coming from r outside [0.3, 20] h^-1 Mpc."""
    r, P = pair_grid(Rp, L, nr, nt)
    f = P * xi_pair(r, i, j) * r
    tot = TRAPZ(f, np.log(r))
    lo = TRAPZ(np.where(r < FIT_LO, f, 0.0), np.log(r))
    hi = TRAPZ(np.where(r > FIT_HI, f, 0.0), np.log(r))
    return float(lo / tot), float(hi / tot)


def Sigma(Rp, L, branch, mode="B"):
    m = means(Rp, L, branch)
    XI = Xi_matrix(Rp, L)
    N = m["Nbar"]
    out = {}
    for h in ("Q", "SF"):
        nu, xb = m[h]["nu"], m[h]["xibar"]
        if mode == "B":
            S = np.diag(nu) + np.outer(N, N) * XI
        else:
            raise ValueError(mode)
        out[h] = S
    out["Xi"] = XI
    out["means"] = m
    return out


# ----------------------------------------------------------------- statistics
def kl(mp, Sp, mq, Sq):
    Si = np.linalg.inv(Sq)
    d = mq - mp
    return 0.5 * (np.trace(Si @ Sp) + float(d @ Si @ d) - 2
                  + np.log(np.linalg.det(Sq) / np.linalg.det(Sp)))


LOG10E = np.log10(np.e)


def distinguishability(Rp, L, branch, mode="B"):
    s = Sigma(Rp, L, branch, mode)
    m = s["means"]
    muQ, muS = m["Q"]["mu"], m["SF"]["mu"]
    dmu = muQ - muS
    Spool = 0.5 * (s["Q"] + s["SF"])
    D2 = float(dmu @ np.linalg.solve(Spool, dmu))
    lq = kl(muQ, s["Q"], muS, s["SF"]) * LOG10E
    ls = kl(muS, s["SF"], muQ, s["Q"]) * LOG10E
    return dict(D2=D2, D=np.sqrt(max(D2, 0)), l_Q=lq, l_SF=ls,
                dmu=dmu, dmu_env=m["Q"]["nu"] - m["SF"]["nu"],
                muQ=muQ, muSF=muS, Xi=s["Xi"], SQ=s["Q"], SSF=s["SF"],
                xibarQ=m["Q"]["xibar"], xibarSF=m["SF"]["xibar"], V=m["V"],
                spd=bool(min(np.linalg.eigvalsh(s["Q"]).min(),
                             np.linalg.eigvalsh(s["SF"]).min()) > 0))
