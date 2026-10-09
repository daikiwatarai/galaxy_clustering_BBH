#!/usr/bin/env python3
"""Forecast of the BBH host-population posterior p(f_Q | data) for O5, post-O5 and 3G (ET+2CE).

Per-event procedure (no binned or Fisher-matrix shortcut)
  1. theta^cyl from each simulated event's GW Fisher covariance
       R_perp = chi sqrt(dOmega_90 / pi),   dOmega_90 = 2 ln10 pi sqrt(det Sigma_(ra,dec)) |cos dec|
       L_par  = 1.6449 sigma_dL dchi/dD_L,  dchi/dD_L = (c/H) / (chi + (1+z) c/H)
  2. Q and SF counts in that cylinder around a Q or an SF host: bivariate Poisson-lognormal with
     the conditional-mean closure Sigma^D, host included (+1 in its own type).  nu and Xi come from
     the w_p power-law model in sfq_cylinder_plnD_figures.py, tabulated on a (R_perp, L_par) grid.
  3. Mock data set: Poisson(T * Ndot * f_sky) events drawn from the Fisher sample; each host is Q
     with probability f_Q,fid; counts drawn from the PLN of the true host type.
  4. ln L(f_Q) = sum_i ln[ f_Q p(N_i | Q) + (1 - f_Q) p(N_i | SF) ],  flat prior on [0, 1].
     "Expected" posterior: from the mean ln L over mocks.  Mock scatter: 16-84% over mocks.

Conventions (as Table I and multievent_forecast_hosttype.ipynb)
  z <= 1, network SNR >= 12, median GWTC-5 default BBH rate.
  DESI-like footprint, f_sky = 0.34 (~14,000 deg^2), applied as an event-rate factor; localization
  volumes are unchanged.  --footprint radec instead keeps only events whose sky position lies in an
  approximate DESI-like region (|b| > 20 deg; NGC -10 < dec < 77, SGC -20 < dec < 32), rescaled to
  the same f_sky, to test the rate-factor approximation.
  Events with V_cyl > 1e8 Mpc^3 are dropped (no information; outside the w_p fit range).
  Events with V_cyl < 0.05 Mpc^3 are treated in the host-only limit (< 1 event in 3 yr for 3G).
Idealised: perfect host-type classification, complete galaxy sample with the SDSS bright0 density
and clustering at all z <= 1.  The host is always inside the cylinder: a real analysis weights the counts
with the GW localization posterior, which accounts for hosts outside it, so this is not modelled.
The likelihood was compared with measured SDSS counts only for R_perp, L_par >= 2 h^-1 Mpc
(fq_boundary_sdss_calibration.py); smaller cylinders are an extrapolation of the model, and the json
also gives the limit without them.

usage:  python scripts/fq_posterior_forecast.py [--years 1 3] [--mocks 100] [--footprint rate|radec]
"""
from __future__ import annotations
import argparse, json, os, pickle, sys, time
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT); sys.path.insert(0, str(ROOT / "scripts"))
from pln_batch import S_of, logpmf_batch, sample_batch, Tables, sigmaD

# Fisher tables (~1 GB); only needed to rebuild data/geometry_z0to1.npz from scratch.
FISHER = Path(os.environ.get("HOSTINFER_FISHER", ROOT / "fisher_tables"))
RATES = Path(os.environ.get("HOSTINFER_RATES", FISHER / "z0to1_cumulative_curves.csv"))
FILES = {"O5_HLVK30": "updated_o5_posto5_20260824/gwtc5_default_bbh_n100000_z0to1_o5_updated_20260824_fisher_merged.csv",
         "POSTO5_FIDUCIAL_HLIVK": "updated_o5_posto5_20260824/gwtc5_default_bbh_n100000_z0to1_posto5_fiducial_20260824_fisher_merged.csv",
         "ET+2CE": "z0to1/gwtc5_default_bbh_n100000_z0to1_fisher_merged.csv"}
KEYS = tuple(FILES)
NAMES = {"O5_HLVK30": "O5", "POSTO5_FIDUCIAL_HLIVK": "Post-O5", "ET+2CE": "3G"}      # 3G = ET + 2 CE
COLORS = {"O5_HLVK30": "tab:orange", "POSTO5_FIDUCIAL_HLIVK": "tab:green", "ET+2CE": "tab:blue"}
DATA = Path(os.environ.get("HOSTINFER_DATA", ROOT / "data"))       # shipped inputs
OUT = ROOT / "results"
CACHE = OUT / "fq_forecast"                                        # what a rerun writes
FSKY = 0.34


def _input(name: str) -> Path:
    """the shipped data product, or the copy an earlier run left in the cache"""
    return DATA / name if (DATA / name).exists() else CACHE / name
F_GAL = 0.4769094969974990          # Q fraction of the bright0 galaxy population
VBER, VMAX, SNR_MIN = 0.05, 1e8, 12.0
R_TESTED = L_TESTED = 2.0           # h^-1 Mpc: smallest SDSS cylinder the likelihood was tested on
H0, OM0, h, C = 70.0, 0.27, 0.7, 299792.458
FG = np.unique(np.r_[np.linspace(1e-4, 0.99, 2000), 1.0 - np.geomspace(1e-7, 0.01, 800)])
trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz


def log(*a):
    print(*a, flush=True)


# ------------------------------------------------------------------------------------ inputs
def nu_xi_tables() -> Tables:
    path = _input("nu_xi_tables.npz")
    if not path.exists():
        import cylinder_host_type_forecast as C
        path = CACHE / "nu_xi_tables.npz"; path.parent.mkdir(parents=True, exist_ok=True)
        lR = np.linspace(np.log(0.02), np.log(300), 46); lL = np.linspace(np.log(0.02), np.log(3000), 52)
        NUQ = np.empty((len(lR), len(lL), 2)); NUS = np.empty_like(NUQ); XI = np.empty((len(lR), len(lL), 3))
        for a, r in enumerate(np.exp(lR)):
            for b, l in enumerate(np.exp(lL)):
                m = C.means(float(r), float(l), "model"); X = C.Xi_matrix(float(r), float(l))
                NUQ[a, b] = m["Q"]["nu"]; NUS[a, b] = m["SF"]["nu"]; XI[a, b] = [X[0, 0], X[0, 1], X[1, 1]]
            log("  nu/Xi table row %d/%d" % (a + 1, len(lR)))
        np.savez(path, lR=lR, lL=lL, NUQ=NUQ, NUS=NUS, XI=XI)
    return Tables(str(path))


def geometry() -> dict:
    path = _input("geometry_z0to1.npz")
    if path.exists():
        return dict(np.load(path))
    from astropy.cosmology import FlatLambdaCDM
    cos = FlatLambdaCDM(H0=H0, Om0=OM0)
    cols = ["network_key", "status", "network_snr", "redshift", "ra", "dec", "cov_ra_ra", "cov_dec_dec",
            "cov_ra_dec", "cov_dL_dL", "localization_volume_3d_90_mpc3"]
    rates = pd.read_csv(RATES); G = {}
    for key, f in FILES.items():
        d = pd.concat([ch[(ch.network_key == key) & (ch.status == "ok") & (ch.network_snr >= SNR_MIN)]
                       for ch in pd.read_csv(FISHER / f, usecols=cols, chunksize=200_000)])
        d = d[np.isfinite(d.localization_volume_3d_90_mpc3) & (d.localization_volume_3d_90_mpc3 > 0)]
        z = d.redshift.to_numpy(); chi = cos.comoving_distance(z).value; cH = C / cos.H(z).value
        L = 1.6449 * np.sqrt(d.cov_dL_dL.to_numpy()) * cH / (chi + (1 + z) * cH)
        det = np.clip((d.cov_ra_ra * d.cov_dec_dec - d.cov_ra_dec ** 2).to_numpy(), 0, None)
        dOm = 2 * np.log(10) * np.pi * np.sqrt(det) * np.abs(np.cos(d.dec.to_numpy()))
        R = chi * np.sqrt(dOm / np.pi)
        ok = np.isfinite(R) & np.isfinite(L) & (R > 0) & (L > 0)
        r = rates[rates.network_key == key].sort_values("v90_comoving_mpc3").iloc[-1]
        assert int(r.n_sample_le_v90) == len(d), (key, r.n_sample_le_v90, len(d))
        G.update({key + "__R_h": R[ok] * h, key + "__L_h": L[ok] * h, key + "__V": 2 * np.pi * R[ok] ** 2 * L[ok],
                  key + "__z": z[ok], key + "__ra": d.ra.to_numpy()[ok], key + "__dec": d.dec.to_numpy()[ok],
                  key + "__ndot": np.float64(r.n_median_1yr), key + "__nsample": np.int64(len(d))})
        log("  %-22s %6d events (SNR>=%g, z<=1), all-sky rate %.1f /yr" % (key, ok.sum(), SNR_MIN, r.n_median_1yr))
    np.savez(path, **G)
    return G


def desi_like(ra, dec):
    """Approximate DESI-like footprint; ra, dec in radians."""
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    b = SkyCoord(ra=ra * u.rad, dec=dec * u.rad, frame="icrs").galactic.b.deg
    dd = np.degrees(dec)
    return ((b > 20) & (dd > -10) & (dd < 77)) | ((b < -20) & (dd > -20) & (dd < 32))


def desi_like_fsky(n=2_000_000, seed=1):
    rng = np.random.default_rng(seed)
    return desi_like(rng.uniform(0, 2 * np.pi, n), np.arcsin(rng.uniform(-1, 1, n))).mean()


# ------------------------------------------------------------------------------------- mocks
def event_loglikes(r, l, v, isQ, tb, rng):
    """ln p(N | Q host) and ln p(N | SF host) for one realisation of each event's counts."""
    n = len(v); lq = np.full(n, -np.inf); ls = np.full(n, -np.inf)
    ber = v < VBER; lq[ber & isQ] = 0.0; ls[ber & ~isQ] = 0.0
    pl = ~ber
    if pl.any():
        nuQ, nuS, XI = tb(r[pl], l[pl])
        SQ, SS = S_of(nuQ, sigmaD(nuQ, XI)), S_of(nuS, sigmaD(nuS, XI))
        q = isQ[pl]
        N = sample_batch(np.where(q[:, None], nuQ, nuS), np.where(q[:, None, None], SQ, SS), rng) \
            + np.where(q[:, None], [[1, 0]], [[0, 1]])
        a = N - np.array([1, 0]); b = N - np.array([0, 1]); okq = (a >= 0).all(1); oks = (b >= 0).all(1)
        tq = np.full(pl.sum(), -np.inf); ts = np.full(pl.sum(), -np.inf)
        if okq.any(): tq[okq] = logpmf_batch(a[okq], nuQ[okq], SQ[okq])
        if oks.any(): ts[oks] = logpmf_batch(b[oks], nuS[oks], SS[oks])
        lq[pl] = tq; ls[pl] = ts
    return lq, ls


def loglike_grid(lq, ls, chunk=256):
    lf, l1f = np.log(FG), np.log1p(-FG); out = np.empty(len(FG))
    for k in range(0, len(FG), chunk):
        s = slice(k, k + chunk)
        out[s] = np.logaddexp(lf[s, None] + lq[None, :], l1f[s, None] + ls[None, :]).sum(1)
    return out


def run_mocks(G, tb, key, T, f0, n_mock, footprint, seed):
    Rh, Lh, V, z = (G[key + "__" + q] for q in ("R_h", "L_h", "V", "z"))
    use = V <= VMAX
    rate = T * float(G[key + "__ndot"]) / int(G[key + "__nsample"]) * FSKY
    if footprint == "radec":
        use &= desi_like(G[key + "__ra"], G[key + "__dec"]); rate /= desi_like_fsky()
    mean_n = rate * use.sum()
    Rh, Lh, V, z = Rh[use], Lh[use], V[use], z[use]
    rng = np.random.default_rng(seed)
    LL = np.empty((n_mock, len(FG))); LL05 = np.empty_like(LL); LLt = np.empty_like(LL)
    nev = np.empty(n_mock, int); ntest = np.empty(n_mock, int)
    for m in range(n_mock):
        n = rng.poisson(mean_n); i = rng.integers(0, len(V), n)
        isQ = rng.random(n) < f0
        lq, ls = event_loglikes(Rh[i], Lh[i], V[i], isQ, tb, rng)
        LL[m] = loglike_grid(lq, ls)
        near = z[i] <= 0.5
        LL05[m] = loglike_grid(lq[near], ls[near])
        tested = (Rh[i] >= R_TESTED) & (Lh[i] >= L_TESTED)       # outside the extrapolated small scales
        LLt[m] = loglike_grid(lq[tested], ls[tested])
        nev[m] = n; ntest[m] = tested.sum()
    return {"LL": LL, "LL_z05": LL05, "LL_tested": LLt, "n_events": nev, "n_tested": ntest, "mean_n": mean_n}


# ------------------------------------------------------------------------------ posteriors
def posterior(ll):
    p = np.exp(ll - ll.max()); return p / trapz(p, FG)


def cdf_one_minus_f(p):
    c = np.r_[0.0, np.cumsum(0.5 * (p[1:] + p[:-1]) * np.diff(FG))]
    return 1.0 - FG[::-1], 1.0 - c[::-1]          # x = 1 - f_Q,  P(1 - f_Q < x | data)


def upper_limit(p, q):
    x, y = cdf_one_minus_f(p); return float(np.interp(q, y, x))


def prob_below(p, f):
    return float(trapz(np.where(FG < f, p, 0.0), FG))


def summarise(res):
    S = {}
    for (key, T), r in res.items():
        pe = posterior(r["LL"].mean(0)); pm = [posterior(l) for l in r["LL"]]
        u95 = np.array([upper_limit(p, 0.95) for p in pm]); u68 = np.array([upper_limit(p, 0.68) for p in pm])
        pz = posterior(r["LL_z05"].mean(0))
        S["%s|%g" % (key, T)] = {
            "network": NAMES[key], "years": T, "mean_events_in_footprint": r["mean_n"],
            "ul68_expected": upper_limit(pe, 0.68), "ul95_expected": upper_limit(pe, 0.95),
            "ul68_mocks_median_16_84": [float(np.median(u68)), *map(float, np.percentile(u68, [16, 84]))],
            "ul95_mocks_median_16_84": [float(np.median(u95)), *map(float, np.percentile(u95, [16, 84]))],
            "P_fQ_below_galaxy_population_expected": prob_below(pe, F_GAL),
            "ul95_expected_z_le_0p5_events_only": upper_limit(pz, 0.95),
            "ul95_expected_tested_scales_only": upper_limit(posterior(r["LL_tested"].mean(0)), 0.95),
            "fraction_events_extrapolated": float(1 - r["n_tested"].sum() / max(r["n_events"].sum(), 1))}
    return S


# ------------------------------------------------------------------------------------ figure
def figure(res, years, f0, stem, logy=False):
    """Expected posterior p(f_Q | data) for each observing run.  Densities are NOT normalised, so the
    narrowing is visible; the scatter over mock data sets is reported in the json, not drawn."""
    os.environ["PATH"] = "/Library/TeX/texbin:" + os.environ["PATH"]
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"text.usetex": True, "font.family": "serif", "font.serif": ["Computer Modern Roman"],
                         "text.latex.preamble": r"\usepackage{amsmath}", "font.size": 10})
    fig, ax = plt.subplots(1, len(years), figsize=(3.7 * len(years), 3.3), sharey=True,
                           gridspec_kw={"wspace": 0.12}, squeeze=False)
    ax = ax[0]; top = 0.0
    for a, T in zip(ax, years):
        for key in KEYS:
            r = res[(key, T)]; col = COLORS[key]
            pe = posterior(r["LL"].mean(0)); u = upper_limit(pe, 0.95)
            a.plot(FG, pe, color=col, lw=1.8, label=r"%s: $f_Q>%s$" % (NAMES[key], "%.3g" % (1 - u)))
            top = max(top, pe.max())
        a.axvline(F_GAL, color="k", lw=0.9, ls=":")
        a.set_xlim(0, 1); a.grid(alpha=0.2)
        a.set_xlabel(r"$f_Q$ (quiescent fraction of BBH hosts)", fontsize=9.5)
        a.set_title(r"%g yr of observation, DESI-like footprint" % T, fontsize=10)
        a.legend(fontsize=7.5, loc="upper left", framealpha=0.93, edgecolor="0.8", handlelength=1.3,
                 title=r"95\% lower limit", title_fontsize=7.5)
    for a in ax:
        if logy:
            a.set_yscale("log"); a.set_ylim(1e-2, 3 * top); yl, yt = 1.5 * top, 0.55
        else:
            a.set_ylim(0, 1.12 * top); yl, yt = 1.07 * top, 0.5 * top
        a.plot([f0], [yl], marker="v", color="crimson", ms=5, clip_on=False)
        a.text(f0 - 0.03, yl, "truth", color="crimson", ha="right", va="center", fontsize=7.5)
        a.text(F_GAL + 0.015, yt, "galaxy population", rotation=90, ha="left", va="center", fontsize=7.5)
    ax[0].set_ylabel(r"$p(f_Q\,|\,\mathrm{data})$", fontsize=10)
    for e in ("png", "pdf"):
        fig.savefig(OUT / ("%s.%s" % (stem, e)), dpi=200, bbox_inches="tight")
    plt.close(fig)
    log("wrote %s/%s.png/pdf" % (OUT.relative_to(ROOT), stem))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=float, nargs="+", default=[1.0, 3.0])
    ap.add_argument("--mocks", type=int, default=100)
    ap.add_argument("--fq", type=float, default=0.999)
    ap.add_argument("--footprint", choices=["rate", "radec"], default="rate")
    ap.add_argument("--plot-only", action="store_true", help="redraw from the cached mocks")
    args = ap.parse_args()
    CACHE.mkdir(parents=True, exist_ok=True)
    tag = "fq_posterior_desi_fQ%g%s" % (args.fq, "" if args.footprint == "rate" else "_radec")
    if args.plot_only:
        res = pickle.load(open(CACHE / (tag + ".pkl"), "rb"))["res"]
        figure(res, args.years, args.fq, tag); figure(res, args.years, args.fq, tag + "_logy", logy=True)
        return
    tb, G = nu_xi_tables(), geometry()
    if args.footprint == "radec":
        log("approximate DESI-like region: f_sky = %.3f (rescaled to %.2f)" % (desi_like_fsky(), FSKY))
    res = {}
    for T in args.years:
        for k, key in enumerate(KEYS):
            t0 = time.time()
            res[(key, T)] = run_mocks(G, tb, key, T, args.fq, args.mocks, args.footprint,
                                      seed=20260913 + 1000 * k + int(10 * T) + (7 if args.footprint == "radec" else 0))
            log("  %-8s %g yr: %.0f events in footprint (V_cyl <= 1e8 Mpc^3), %d mocks, %.0fs"
                % (NAMES[key], T, res[(key, T)]["mean_n"], args.mocks, time.time() - t0))
    pickle.dump({"FG": FG, "res": res, "args": vars(args)}, open(CACHE / (tag + ".pkl"), "wb"))
    S = summarise(res)
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(S, open(OUT / (tag + ".json"), "w"), indent=1)
    log("\nf_Q,fid = %g, f_sky = %.2f (event-rate factor%s), z <= 1, SNR >= 12, flat prior"
        % (args.fq, FSKY, "" if args.footprint == "rate" else "; events selected by sky position"))
    log("upper limits on 1 - f_Q: expected posterior | mocks median [16-84%] | z<=0.5 events only | "
        "without R_perp or L_par < 2 h^-1 Mpc (fraction of events there)")
    for s in S.values():
        log("  %-12s %g yr  N=%5.0f   68%%: %.3g   95%%: %.3g | %.3g [%.3g, %.3g] | %.3g | %.3g (%.1f%%)   P(f_Q<%.3f)=%.1e"
            % (s["network"], s["years"], s["mean_events_in_footprint"], s["ul68_expected"], s["ul95_expected"],
               *s["ul95_mocks_median_16_84"], s["ul95_expected_z_le_0p5_events_only"],
               s["ul95_expected_tested_scales_only"], 100 * s["fraction_events_extrapolated"], F_GAL,
               s["P_fQ_below_galaxy_population_expected"]))
    figure(res, args.years, args.fq, tag); figure(res, args.years, args.fq, tag + "_logy", logy=True)


if __name__ == "__main__":
    main()
