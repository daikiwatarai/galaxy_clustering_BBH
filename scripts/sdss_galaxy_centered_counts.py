#!/usr/bin/env python3
"""Direct measurement of the host-conditioned (galaxy-centered) count distribution
P(N_Q, N_SF | host type) in SDSS, for validating the Gaussian + two-point closure
used in the GW host-type forecast.

WHAT THIS MEASURES
------------------
For every galaxy in the volume-limited parent, centred on that galaxy, count the
NEIGHBOURS of each type inside a comoving sphere of radius R (the host itself is
excluded).  The resulting empirical distribution contains all orders of clustering
and all discreteness, so it directly tests

  (a) the Gaussian likelihood      -> measured skewness / kurtosis, and the error
                                      rate of the Gaussian decision rule on real data
  (b) the O(xi) covariance closure -> measured Cov(N_i,N_j|h) vs
                                      Sigma^B = diag(nu) + Nbar_i Nbar_j Xi_ij
                                      the difference IS Nbar^2 (Z_hij - xibar xibar).

PARENT SAMPLE
-------------
The same NYU-VAGC `bright0` parent as scripts/sdss_split_auto_correlation.py, built by
its own build_parent(): 0.02 < z < 0.067, k+e corrected M_r < -19, matched to MPA-JHU
within 1.5", log10 M*/Msun > 9.8, split at log sSFR = -11.  74,727 galaxies with
f_SF = 0.5228, against the 74,774 and 0.5231 recorded for the wp measurement; the
residual 47 come from the MPA-JHU match.

UNITS
-----
Everything is in h^-1 Mpc with FlatLambdaCDM(H0=100, Om0=0.27), matching the wp
measurement exactly.  The model's nbar = 0.012781 h^3 Mpc^-3 and r0 values in
h^-1 Mpc are used as-is; COUNTS are invariant under the h convention, so no
conversion to physical Mpc is needed (V_phys = V_h / h^3 with h = 0.6766).

EDGE HANDLING
-------------
A sphere is used only if it lies entirely inside the survey:
  * radial:  d_C - R > d_C(0.02)  and  d_C + R < d_C(0.067)   (exact)
  * angular: >= FMIN of NSAMP volume-uniform points inside the sphere land in
             occupied pixels of an nside=128 footprint mask built from the FULL
             Yang SDSS7 catalogue (~90 gal/deg^2, ~19 per pixel), with isolated
             holes filled.
The mask is built from a catalogue independent of the volume-limited subsample's
density field, and the cut uses only geometry, never the counts being measured,
so rejection costs statistics but does not bias the measured distribution.

REDSHIFT SPACE
--------------
Positions are redshift-space.  Fingers of god smear the radial coordinate by up
to ~10 h^-1 Mpc, comparable to R, whereas the model's xibar comes from real-space
wp.  This is a real and unresolved systematic of the comparison and is flagged in
the output JSON.  A follow-up pass collapsing Yang groups to their group redshift
is the intended diagnostic; it is NOT done here.

OUTPUT
------
results/sdss_galaxy_centered_counts/
  counts.npz      per-centre: host type, jackknife region, d_C, mask fraction,
                  and (N_Q, N_SF) at every radius -> every downstream statistic
                  can be recomputed without rerunning
  summary.json    moments, measured covariance, the three model prescriptions
  run_log.txt
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------- frozen config
Z_LO, Z_HI = 0.02, 0.067
MR_MAX = -19.0
LOGM_MIN = 9.8
SSFR_SPLIT = -11.0
MATCH_ARCSEC = 1.5
OM0, H0 = 0.27, 100.0
H_LITTLE = 0.6766                 # Planck18, the model's h (counts are h-invariant)

# frozen model inputs -- byte-identical to host_type_strict_2pt_covariance.ipynb
R0_H = {"all": 5.972546, "SS": 4.614366, "QQ": 7.364902}      # h^-1 Mpc
GAMMA = {"all": 1.781478, "SS": 1.587474, "QQ": 1.950136}
NBAR_MODEL = 0.012781                                          # h^3 Mpc^-3
F_SF_MODEL = 0.5230905030025009

CASCADE_REF = {"mr_cut": 88705, "mpa_matched": 83729, "parent": 74774}

RADII = [1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0]   # h^-1 Mpc
NSIDE_MASK = 128
NSAMP = 800
FMIN = 0.99
NJACK = 50

LOG_LINES: list[str] = []


def log(msg: str = "") -> None:
    print(msg, flush=True)
    LOG_LINES.append(msg)


# ---------------------------------------------------------------- parent sample
def build_parent(base: Path):
    """The NYU-VAGC bright0 parent, identical to the one used for the wp fits.

    sdss_split_auto_correlation.build_parent validates every cascade stage against
    the recorded bright0 counts, so a convention error is a hard failure.  The full
    bright0 rows are returned as well; they are what the footprint mask is built from.
    """
    import argparse
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "sac", base / "scripts" / "sdss_split_auto_correlation.py")
    sac = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sac)

    raw = base / "data/galaxies/nyu_vagc/raw"
    paths = {"vagc_post": raw / "post_catalog.dr72bright0.fits",
             "vagc_kcorrect": raw / "kcorrect.none.petro.z0.10.fits",
             "mpa_raw": base / "data/galaxies/mpa_jhu_dr7/raw"}
    args = argparse.Namespace(absmag_source="auto", absmag_band=2,
                              allow_cascade_mismatch=True)

    ra_all, dec_all, _, _ = sac.load_vagc(paths["vagc_post"], paths["vagc_kcorrect"],
                                          args.absmag_source, args.absmag_band)
    log(f"  bright0 rows (footprint mask){'':11s}{len(ra_all):>8,}")
    ra, dec, z, logm, logssfr = sac.build_parent(args, paths)
    is_sf = logssfr > SSFR_SPLIT
    log(f"  f_SF = {is_sf.mean():.6f}   (model uses {F_SF_MODEL:.6f})")
    return ra, dec, z, is_sf, (ra_all, dec_all)


def comoving(z):
    from astropy.cosmology import FlatLambdaCDM
    return FlatLambdaCDM(H0=H0, Om0=OM0).comoving_distance(z).value   # h^-1 Mpc


def unit_vectors(ra_deg, dec_deg):
    r = np.radians(ra_deg)
    d = np.radians(dec_deg)
    cd = np.cos(d)
    return np.column_stack([cd * np.cos(r), cd * np.sin(r), np.sin(d)])


# ---------------------------------------------------------------- footprint mask
def build_mask(ra_all, dec_all, nside):
    import healpy as hp
    npix = hp.nside2npix(nside)
    pix = hp.ang2pix(nside, ra_all, dec_all, lonlat=True)
    occ = np.zeros(npix, bool)
    occ[np.unique(pix)] = True
    log(f"  raw footprint: {occ.sum():,} / {npix:,} pixels "
        f"({occ.sum() * hp.nside2pixarea(nside, degrees=True):.0f} deg^2)")
    # fill isolated holes: an empty pixel with >=6 of 8 occupied neighbours
    empty = np.flatnonzero(~occ)
    nb = hp.get_all_neighbours(nside, empty)            # (8, n)
    valid = nb >= 0
    nocc = np.where(valid, occ[np.clip(nb, 0, npix - 1)], False).sum(axis=0)
    filled = empty[nocc >= 6]
    occ[filled] = True
    log(f"  hole-filled: +{len(filled):,} pixels -> {occ.sum():,} occupied")
    return occ


def sphere_samples(n, rng):
    """n points uniform in the unit ball."""
    v = rng.normal(size=(n, 3))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    return v * rng.uniform(size=(n, 1)) ** (1.0 / 3.0)


def mask_fraction(centres_xyz, R, samples, occ, nside, chunk=4000):
    import healpy as hp
    out = np.empty(len(centres_xyz))
    for i in range(0, len(centres_xyz), chunk):
        c = centres_xyz[i:i + chunk]
        p = c[:, None, :] + R * samples[None, :, :]
        pix = hp.vec2pix(nside, p[..., 0], p[..., 1], p[..., 2])
        out[i:i + chunk] = occ[pix].mean(axis=1)
    return out


# ---------------------------------------------------------------- jackknife
def random_centres(n, occ, nside, d_lo, d_hi, rng):
    """Uniform-in-volume random centres inside the footprint and the shell."""
    import healpy as hp
    keep_v = []
    while sum(len(k) for k in keep_v) < n:
        v = rng.normal(size=(3 * n, 3))
        v /= np.linalg.norm(v, axis=1, keepdims=True)
        keep_v.append(v[occ[hp.vec2pix(nside, v[:, 0], v[:, 1], v[:, 2])]])
    v = np.concatenate(keep_v)[:n]
    u = rng.uniform(size=n)
    d = (d_lo ** 3 + u * (d_hi ** 3 - d_lo ** 3)) ** (1.0 / 3.0)
    return v * d[:, None], d


def jackknife_regions(vecs, njack, seed):
    from scipy.cluster.vq import kmeans2
    rng = np.random.default_rng(seed)
    sub = vecs[rng.choice(len(vecs), size=min(200_000, len(vecs)), replace=False)]
    cen, _ = kmeans2(sub, njack, minit="++", seed=seed, iter=40)
    return np.argmax(vecs @ cen.T, axis=1), cen


# ---------------------------------------------------------------- model
def _Jpair(g):
    return 2.0 ** (3 - g) / (3 - g) - 0.75 * 2.0 ** (4 - g) / (4 - g) \
        + (1 / 16.) * 2.0 ** (6 - g) / (6 - g)


def model_prediction(R, f_sf, nbar):
    """(mu, Sigma_B, Sigma_D) for both hosts, in the (Q, SF) order, at radius R [h^-1 Mpc]."""
    def xibar(k):
        g = GAMMA[k]
        return 3.0 / (3.0 - g) * (R / R0_H[k]) ** (-g)

    def Xipair(k):
        g = GAMMA[k]
        return 3.0 * _Jpair(g) * (R / R0_H[k]) ** (-g)

    f_q = 1.0 - f_sf

    def trip(fn):
        a, ss, qq = fn("all"), fn("SS"), fn("QQ")
        return qq, (a - f_sf ** 2 * ss - f_q ** 2 * qq) / (2 * f_sf * f_q), ss

    xQ, xX, xS = trip(xibar)
    XQ, XX, XS = trip(Xipair)
    V = 4 * np.pi / 3 * R ** 3
    N = np.array([f_q * nbar * V, f_sf * nbar * V])
    XI = np.array([[XQ, XX], [XX, XS]])
    out = {}
    for host, xb in (("Q", np.array([xQ, xX])), ("SF", np.array([xX, xS]))):
        nu = N * (1 + xb)                       # NEIGHBOUR mean (host excluded)
        out[host] = dict(
            nu=nu.tolist(),
            Sigma_B=(np.diag(nu) + np.outer(N, N) * XI).tolist(),
            Sigma_D=(np.diag(nu) + np.outer(nu, nu) * XI).tolist(),
            xibar=xb.tolist(),
        )
    out["Xi"] = XI.tolist()
    out["Nbar"] = N.tolist()
    return out


# ---------------------------------------------------------------- main
def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--base-dir", type=Path, default=Path("/home/daiki.watarai/BBH_population"))
    p.add_argument("--output-dir", default="results/sdss_galaxy_centered_counts")
    p.add_argument("--radii", type=float, nargs="+", default=RADII)
    p.add_argument("--nsamp", type=int, default=NSAMP)
    p.add_argument("--fmin", type=float, default=FMIN)
    p.add_argument("--njack", type=int, default=NJACK)
    p.add_argument("--nrandom", type=int, default=200_000,
                   help="random centres: measures nbar and the unconditioned "
                        "counts-in-cells covariance (prescription A)")
    p.add_argument("--seed", type=int, default=20260908)
    a = p.parse_args(argv)

    from scipy.spatial import cKDTree
    t0 = time.time()
    outdir = a.base_dir / a.output_dir
    outdir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(a.seed)

    log("=" * 78)
    log("SDSS galaxy-centered counts  P(N_Q, N_SF | host type)")
    log("=" * 78)
    log("building parent")
    ra, dec, z, is_sf, (ra_all, dec_all) = build_parent(a.base_dir)

    dc = comoving(z)
    d_lo, d_hi = comoving(np.array([Z_LO, Z_HI]))
    vecs = unit_vectors(ra, dec)
    xyz = vecs * dc[:, None]
    log(f"  comoving shell {d_lo:.2f} - {d_hi:.2f} h^-1 Mpc (depth {d_hi - d_lo:.1f})")

    log("building footprint mask")
    occ = build_mask(ra_all, dec_all, NSIDE_MASK)
    samples = sphere_samples(a.nsamp, rng)

    log(f"drawing {a.nrandom:,} random centres (footprint + shell, uniform in volume)")
    rxyz, rdc = random_centres(a.nrandom, occ, NSIDE_MASK, d_lo, d_hi, rng)

    log(f"jackknife: {a.njack} k-means regions")
    region, kcen = jackknife_regions(vecs, a.njack, a.seed)
    rvec = rxyz / np.linalg.norm(rxyz, axis=1, keepdims=True)
    rregion = np.argmax(rvec @ kcen.T, axis=1)
    log(f"  region sizes {np.bincount(region, minlength=a.njack).min():,} - "
        f"{np.bincount(region, minlength=a.njack).max():,}")

    tree_q = cKDTree(xyz[~is_sf])
    tree_s = cKDTree(xyz[is_sf])
    log(f"  trees built: N_Q = {int((~is_sf).sum()):,}  N_SF = {int(is_sf.sum()):,}")

    nQ = np.full((len(a.radii), len(ra)), -1, np.int32)
    nS = np.full((len(a.radii), len(ra)), -1, np.int32)
    accept = np.zeros((len(a.radii), len(ra)), bool)
    fmask = np.zeros((len(a.radii), len(ra)), np.float32)
    rnQ = np.full((len(a.radii), len(rdc)), -1, np.int32)
    rnS = np.full((len(a.radii), len(rdc)), -1, np.int32)
    raccept = np.zeros((len(a.radii), len(rdc)), bool)
    summary = {}

    log("")
    log("%6s %9s %8s | %-30s | %-30s" % ("R", "V", "n_used", "host Q: <N_Q> <N_SF>",
                                          "host SF: <N_Q> <N_SF>"))
    for ir, R in enumerate(a.radii):
        radial = (dc - R > d_lo) & (dc + R < d_hi)
        fr = np.zeros(len(ra), np.float32)
        idx = np.flatnonzero(radial)
        fr[idx] = mask_fraction(xyz[idx], R, samples, occ, NSIDE_MASK)
        ok = radial & (fr >= a.fmin)
        fmask[ir] = fr
        accept[ir] = ok

        cq = tree_q.query_ball_point(xyz[ok], R, return_length=True)
        cs = tree_s.query_ball_point(xyz[ok], R, return_length=True)
        cq = cq - (~is_sf[ok]).astype(np.int64)       # drop the host itself
        cs = cs - is_sf[ok].astype(np.int64)
        nQ[ir, ok] = cq
        nS[ir, ok] = cs

        rradial = (rdc - R > d_lo) & (rdc + R < d_hi)
        ridx = np.flatnonzero(rradial)
        rf = mask_fraction(rxyz[ridx], R, samples, occ, NSIDE_MASK)
        rok = ridx[rf >= a.fmin]
        rq = tree_q.query_ball_point(rxyz[rok], R, return_length=True).astype(float)
        rs = tree_s.query_ball_point(rxyz[rok], R, return_length=True).astype(float)
        rnQ[ir, rok] = rq.astype(np.int32)
        rnS[ir, rok] = rs.astype(np.int32)
        raccept[ir, rok] = True

        V = 4 * np.pi / 3 * R ** 3
        rec = {"R_hMpc": R, "V_hMpc3": V, "V_phys_Mpc3": V / H_LITTLE ** 3,
               "n_accepted": int(ok.sum()), "accept_frac": float(ok.mean()),
               "model": model_prediction(R, float(is_sf.mean()), NBAR_MODEL)}
        for host, m in (("Q", ~is_sf[ok]), ("SF", is_sf[ok])):
            N = np.column_stack([cq[m], cs[m]]).astype(float)
            if len(N) < 10:
                continue
            c = np.cov(N.T)
            d3 = ((N - N.mean(0)) ** 3).mean(0) / N.std(0) ** 3
            d4 = ((N - N.mean(0)) ** 4).mean(0) / N.std(0) ** 4 - 3.0
            rec[host] = {"n": int(len(N)), "mean": N.mean(0).tolist(),
                         "cov": c.tolist(), "skew": d3.tolist(), "exkurt": d4.tolist()}
        RN = np.column_stack([rq, rs])
        rec["random"] = {"n": int(len(rq)), "mean": RN.mean(0).tolist(),
                         "cov": np.cov(RN.T).tolist(),
                         "skew": (((RN - RN.mean(0)) ** 3).mean(0) / RN.std(0) ** 3).tolist(),
                         "nbar_measured": float(RN.sum(1).mean() / V),
                         "nbar_model": NBAR_MODEL}
        summary[f"R{R:g}"] = rec
        mq, ms = rec.get("Q", {}).get("mean", [np.nan] * 2), rec.get("SF", {}).get("mean", [np.nan] * 2)
        log("%6.1f %9.4g %8d | %11.3f %11.3f      | %11.3f %11.3f"
            % (R, V, int(ok.sum()), mq[0], mq[1], ms[0], ms[1]))

    log("")
    log("RANDOM centres: density normalisation and the unconditioned (A) covariance")
    log("%6s %8s | %9s %9s | %-23s" % ("R", "n_rand", "nbar_meas", "nbar_mod",
                                        "Cov_rand / Sigma^A  QQ  SFSF"))
    for R in a.radii:
        rc = summary[f"R{R:g}"]["random"]
        m = summary[f"R{R:g}"]["model"]
        N = np.array(m["Nbar"]); XI = np.array(m["Xi"])
        SA = np.diag(N) + np.outer(N, N) * XI
        cm = np.array(rc["cov"])
        log("%6.1f %8d | %9.6f %9.6f | %10.3f %10.3f"
            % (R, rc["n"], rc["nbar_measured"], rc["nbar_model"],
               cm[0, 0] / SA[0, 0], cm[1, 1] / SA[1, 1]))

    log("")
    log("measured vs model covariance (diagonal), ratio Cov_meas / Sigma^B")
    log("%6s | %-22s | %-22s" % ("R", "host Q:  QQ      SFSF", "host SF: QQ      SFSF"))
    for R in a.radii:
        rec = summary[f"R{R:g}"]
        row = []
        for host in ("Q", "SF"):
            if host not in rec:
                row += [np.nan, np.nan]
                continue
            cm = np.array(rec[host]["cov"])
            sb = np.array(rec["model"][host]["Sigma_B"])
            row += [cm[0, 0] / sb[0, 0], cm[1, 1] / sb[1, 1]]
        log("%6.1f | %10.3f %10.3f | %10.3f %10.3f" % (R, *row))

    np.savez_compressed(
        outdir / "counts.npz",
        radii=np.asarray(a.radii), is_sf=is_sf, region=region.astype(np.int16),
        dc=dc.astype(np.float32), ra=ra.astype(np.float32), dec=dec.astype(np.float32),
        n_q=nQ, n_sf=nS, accepted=accept, mask_frac=fmask)
    np.savez_compressed(
        outdir / "random_centres.npz", radii=np.asarray(a.radii),
        dc=rdc.astype(np.float32), xyz=rxyz.astype(np.float32),
        region=rregion.astype(np.int16), n_q=rnQ, n_sf=rnS, accepted=raccept)

    meta = {
        "parent": {"n": int(len(ra)), "f_SF": float(is_sf.mean()),
                   "reference_bright0": CASCADE_REF, "f_SF_model": F_SF_MODEL,
                   "source": "NYU-VAGC bright0, as in sdss_split_auto_correlation.py"},
        "config": {"z": [Z_LO, Z_HI], "MR_MAX": MR_MAX, "LOGM_MIN": LOGM_MIN,
                   "SSFR_SPLIT": SSFR_SPLIT, "cosmology": {"H0": H0, "Om0": OM0},
                   "h_model": H_LITTLE, "nbar_model": NBAR_MODEL,
                   "nside_mask": NSIDE_MASK, "nsamp": a.nsamp, "fmin": a.fmin,
                   "njack": a.njack, "seed": a.seed},
        "caveats": [
            "positions are REDSHIFT SPACE; the model xibar is real-space (from wp). "
            "Fingers of god are an unquantified systematic in this comparison.",
            "R > 10 h^-1 Mpc samples pairs beyond the 20 h^-1 Mpc wp fit range",
        ],
        "radii": summary,
        "runtime_s": time.time() - t0,
    }
    (outdir / "summary.json").write_text(json.dumps(meta, indent=1))
    (outdir / "run_log.txt").write_text("\n".join(LOG_LINES) + "\n")
    log("")
    log(f"wrote {outdir}/counts.npz, random_centres.npz, summary.json, run_log.txt")
    log(f"total {time.time() - t0:.1f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
