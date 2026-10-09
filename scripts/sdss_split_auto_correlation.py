#!/usr/bin/env python3
"""Projected auto-correlation wp(rp) of a binary-split VAGC bright0 sample.

Runs the same measurement for two splits of the SAME parent:

    --split ssfr    SF / Q at log10(sSFR/yr^-1) = -11      (reproduces the known answer)
    --split mstar   High / Low at the parent's median log10(M*/Msun)

The parent is the NYU-VAGC `bright0` sample, 0.02 < z < 0.067, k+e corrected
M_r < -19, matched to MPA-JHU at 1.5", log10(M*/Msun) > 9.8.  Every stage of the
cascade is checked against the counts recorded in
results/observed_density/radial_random_diagnostic.json; a mismatch is a hard
failure, because those counts pin down the k+e sign convention and the match
radius, neither of which is documented anywhere else.

wp is Landy-Szalay in (rp, pi) with pi_max = 40 h^-1 Mpc, 15 log bins over
0.1-20 h^-1 Mpc, comoving distances from FlatLCDM(Om=0.27, H0=100).  Randoms take
their angular positions from the VAGC random catalogue and their redshifts either
by resampling the data (`shuffled_z`) or ~ dV/dz (`uniform_volume`).

The power law is fitted with the finite-pi_max truncation included,

    wp(rp) = 2 \\int_0^pimax (sqrt(rp^2 + pi^2)/r0)^-gamma dpi

over 0.3 < rp < 20 h^-1 Mpc (12 bins).  THREE weightings are computed on every
run and all are stored; --fit-weights picks which one is reported as primary:

    jackknife   (default) the full leave-one-region-out covariance, C^-1 in the
                chi^2, which is what the SDSS analyses do.  Neighbouring rp bins
                share pairs and are strongly correlated, so the off-diagonal
                terms matter.  Regions come from k-means on the randoms.
    published   the diagonal errors the original SF/Q fits used -- Hearin+2014
                Table 1 for `all`, Watson+2015 A1/A2 for SF/Q -- because the
                jackknife was never rerun on the bright0 parent.  Kept because
                it is the only weighting under which --split ssfr can reproduce
                the recorded answer, so it stays a live check on the pipeline.
                For --split mstar, where no published errors exist, the
                FRACTIONAL errors sigma/wp are transferred, high mass taking the
                quenched curve and low mass the star-forming one.
    equal       unweighted, as a sensitivity floor.

Expect chi2/dof >> 1 under the jackknife: a single power law is not a good
description of SDSS wp once the errors are the measurement's own, a point
Zehavi+2011 make explicitly.  The published weighting's chi2/dof << 1 reflects
errors imported from a different sample, not a good fit.

No uncertainty on r0 or gamma is produced anywhere: best-fit values only.

The cross term is never fitted.  It follows from the binary-partition identity,
which holds because all three curves come from the same parent, mask and
estimator:

    xi_AB = (xi_all - f_A^2 xi_AA - f_B^2 xi_BB) / (2 f_A f_B)

Run `--inspect` first on a new machine: it prints the columns of every input file
and exits, so the VAGC paths can be settled without a measurement run.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from astropy.io import fits

# ---------------------------------------------------------------------------
# frozen configuration -- these values define the SF/Q measurement and must not
# drift, or the M* fits stop being comparable to it
# ---------------------------------------------------------------------------
Z_LO, Z_HI = 0.02, 0.067          # bright0 redshift window
MR_MAX = -19.0                    # k+e corrected M_r cut
LOGM_MIN = 9.8                    # parent stellar-mass floor
SSFR_SPLIT = -11.0                # log10(sSFR/yr^-1), SF above / Q below
MATCH_ARCSEC = 1.5                # MPA-JHU match radius
EVOLUTION = dict(q0=2.0, q1=-1.0, qz0=0.1)
OM0, H0 = 0.27, 100.0             # FlatLCDM, distances in h^-1 Mpc
H_LITTLE = 0.70                   # h used to report r0 in physical Mpc
PIMAX = 40.0                      # h^-1 Mpc
RP_MIN, RP_MAX, NRP = 0.1, 20.0, 15
FIT_LO, FIT_HI = 0.3, 20.0        # h^-1 Mpc; see memory on the Zehavi fit range
NRAND = 1_000_000

# parent cascade, from results/observed_density/radial_random_diagnostic.json
CASCADE_REF = {
    "vagc_rows": 567759,
    "z_window": 144503,
    "mr_cut": 88705,
    "mpa_matched": 83729,
    "parent": 74774,
}
# stored SF/Q answer, in physical Mpc; the --split ssfr validation target
SSFR_REF = {
    "all": (8.532209, 1.781478),
    "A": (6.591951, 1.587474),    # SF x SF
    "B": (10.521289, 1.950136),   # Q  x Q
}

SPLIT_LABELS = {
    "ssfr": ("SF", "Q"),
    "mstar": ("High", "Low"),
}

# The wp measurement the SF/Q fits were actually run against (the uniform-in-
# comoving-volume randoms branch) and the published diagonal errors used as the
# fit weights: Hearin+2014 Table 1 for `all`, Watson+2015 A1/A2 for SF/Q.
# Transcribed from notebooks/xi_realspace.ipynb cell 7.
PUB_RP = np.array([0.1217, 0.1730, 0.2468, 0.3510, 0.5006, 0.7120, 1.0139, 1.4446,
                   2.0577, 2.9321, 4.1742, 5.9412, 8.4639, 12.0460, 17.1579])
PUB_WP = {
    "all": np.array([406.51, 324.01, 250.13, 197.86, 153.52, 117.76, 85.86, 61.88,
                     46.39, 35.78, 27.67, 20.48, 14.27, 9.53, 5.34]),
    "A": np.array([170.73, 140.20, 107.65, 93.90, 71.28, 61.31, 47.43, 37.36,
                   29.93, 24.52, 20.00, 15.23, 10.74, 7.33, 4.03]),
    "B": np.array([937.05, 718.31, 545.93, 417.35, 317.21, 228.92, 155.63, 104.52,
                   73.21, 52.56, 38.02, 27.28, 18.84, 12.66, 7.32]),
}
PUB_SIG = {
    "all": np.array([18.80, 15.07, 14.57, 13.40, 11.28, 9.41, 7.57, 5.95,
                     5.09, 4.15, 3.49, 2.91, 2.19, 1.56, 1.28]),
    "A": np.array([9.52, 7.47, 5.66, 4.97, 3.61, 3.19, 2.62, 2.40,
                   2.38, 2.29, 2.11, 1.98, 1.55, 1.23, 1.06]),
    "B": np.array([55.17, 42.57, 40.38, 35.66, 28.37, 23.97, 18.02, 12.78,
                   10.07, 7.36, 5.69, 4.39, 3.18, 2.17, 1.66]),
}
# which published fractional-error curve weights which subsample.  For ssfr this
# is the identity and reproduces the published fits exactly.  For mstar the high
# half is treated as quenched-like and the low half as star-forming-like.
WEIGHT_SOURCE = {
    "ssfr": {"all": "all", "A": "A", "B": "B"},
    "mstar": {"all": "all", "A": "B", "B": "A"},
}


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------
# inputs
# ---------------------------------------------------------------------------
def inspect(paths: dict[str, Path]) -> int:
    """Print the columns of every input file and exit.  Costs one cluster round trip."""
    rc = 0
    for name, path in paths.items():
        print(f"\n=== {name}: {path}")
        if not path.exists():
            print("    MISSING")
            rc = 1
            continue
        with fits.open(path) as h:
            for i, hdu in enumerate(h):
                if getattr(hdu, "data", None) is None or hdu.data.dtype.names is None:
                    continue
                print(f"    HDU {i}: {len(hdu.data):,} rows")
                for col in hdu.columns:
                    print(f"      {col.name:24s} {col.format}")
    return rc


def _column(data, *names):
    """First present column among `names`, case-insensitively."""
    have = {n.upper(): n for n in (data.dtype.names or ())}
    for n in names:
        if n.upper() in have:
            return np.asarray(data[have[n.upper()]])
    raise KeyError(f"none of {names} present; available: {sorted(have)}")


def evolution_correction(z, q0, q1, qz0):
    """Blanton et al. luminosity evolution, brighter in the past."""
    return q0 * (1.0 + q1 * (z - qz0)) * (z - qz0)


def _absmag_table(post_path: Path, kcorr_path: Path | None, absmag_source: str):
    """The (n, nband) absolute-magnitude table for the bright0 rows, however it is stored.

    VAGC keeps `ABSMAG` either on the LSS post catalogue itself or in a kcorrect
    file that is indexed through `OBJECT_POSITION`; some builds ship a kcorrect
    file already row-matched to the post catalogue.  All three are accepted.
    """
    with fits.open(post_path) as h:
        d = h[1].data
        n_post = len(d)
        if absmag_source in ("auto", "post"):
            try:
                a = np.asarray(_column(d, "ABSMAG", "ABSM", "AMAG"), float)
                return (a if a.ndim > 1 else a[:, None]), "post catalogue"
            except KeyError:
                if absmag_source == "post":
                    raise SystemExit(
                        f"--absmag-source post, but {post_path.name} has no ABSMAG "
                        f"column.  Run --inspect to see what it does have.")
        try:
            position = np.asarray(_column(d, "OBJECT_POSITION", "OBJECT_INDEX"), np.int64)
        except KeyError:
            position = None

    if kcorr_path is None or not kcorr_path.exists():
        raise SystemExit(
            f"absolute magnitudes are not on {post_path.name}, so the kcorrect file "
            f"is required; --vagc-kcorrect pointed at {kcorr_path}.  Run --inspect.")
    with fits.open(kcorr_path) as h:
        a = np.asarray(_column(h[1].data, "ABSMAG"), float)
    if a.ndim == 1:
        a = a[:, None]
    if len(a) == n_post:
        return a, "kcorrect file, row-matched"
    if position is None:
        raise SystemExit(
            f"{kcorr_path.name} has {len(a):,} rows against {n_post:,} in "
            f"{post_path.name}, and there is no OBJECT_POSITION column to index it "
            f"with.  Run --inspect.")
    return a[position], "kcorrect file, via OBJECT_POSITION"


def load_vagc(post_path: Path, kcorr_path: Path | None, absmag_source: str, band: int):
    """RA, DEC, Z and the absolute-magnitude table of every bright0 row."""
    with fits.open(post_path) as h:
        d = h[1].data
        ra = np.asarray(_column(d, "RA"), float)
        dec = np.asarray(_column(d, "DEC"), float)
        z = np.asarray(_column(d, "Z", "ZDIST"), float)
    table, origin = _absmag_table(post_path, kcorr_path, absmag_source)
    log(f"  absolute magnitudes from the {origin}, {table.shape[1]} band(s)")
    if band >= table.shape[1]:
        raise SystemExit(f"--absmag-band {band} but the table has only "
                         f"{table.shape[1]} band(s)")
    return ra, dec, z, table


def diagnose_mr_stage(z, table, in_window, target):
    """Report which (band, evolution sign) reproduces the recorded M_r cut count.

    The band index and the sign of the evolution term are the two conventions that
    survive nowhere in the notebooks, and the recorded count 88705 pins both.  One
    failed run should therefore be enough to settle them, rather than a guessing
    round trip per convention.
    """
    log("  scanning conventions against the recorded count "
        f"{target:,}:")
    hits = []
    ev = evolution_correction(z, **EVOLUTION)
    for band in range(table.shape[1]):
        for sign, name in ((-1.0, "M - E(z)"), (+1.0, "M + E(z)"), (0.0, "M, no evolution")):
            mr = table[:, band] + sign * ev
            n = int((in_window & np.isfinite(mr) & (mr < MR_MAX)).sum())
            mark = "  <== matches" if n == target else ""
            log(f"    band {band}  {name:16s} {n:>8,}{mark}")
            if n == target:
                hits.append((band, name))
    if hits:
        log(f"  rerun with --absmag-band {hits[0][0]} (evolution: {hits[0][1]})")
    else:
        log("  nothing matches; the parent definition or the input file differs")
    return hits


def load_mpa(raw: Path):
    """MPA-JHU positions, log10 M* and log10 sSFR, deduplicated by PHOTOID on S/N."""
    with fits.open(raw / "gal_info_dr7_v5_2.fit.gz") as h:
        g = h[1].data
        ra = np.asarray(g["RA"], float)
        dec = np.asarray(g["DEC"], float)
        z = np.asarray(g["Z"], float)
        zw = np.asarray(g["Z_WARNING"], int)
        st = np.char.strip(np.asarray(g["SPECTROTYPE"], str))
        pid = np.unique(np.asarray(g["PHOTOID"], np.int64), axis=0,
                        return_inverse=True)[1].astype(np.int64)
        snr = np.asarray(g["SN_MEDIAN"], float)
        if snr.ndim > 1:
            snr = np.nanmedian(snr, axis=tuple(range(1, snr.ndim)))
    with fits.open(raw / "totlgm_dr7_v5_2.fit.gz") as h:
        logm = np.asarray(h[1].data["MEDIAN"], float)
    with fits.open(raw / "gal_totspecsfr_dr7_v5_2.fits.gz") as h:
        logssfr = np.asarray(h[1].data["MEDIAN"], float)

    sky = np.isfinite(ra) & np.isfinite(dec) & (np.abs(dec) <= 90)
    clean = sky & (zw == 0) & (st == "GALAXY") & np.isfinite(z) & (z > 0)
    order = np.lexsort((-np.nan_to_num(snr), pid))
    first = np.ones(len(z), bool)
    first[order[1:]] = pid[order[1:]] != pid[order[:-1]]

    def valid(v):
        return np.isfinite(v) & (v > -90) & (v != -1.0)

    keep = clean & first & valid(logm) & valid(logssfr)
    return ra[keep], dec[keep], logm[keep], logssfr[keep]


def build_parent(args, paths):
    """The bright0 parent, with every cascade stage checked against the known counts."""
    from astropy.coordinates import SkyCoord
    import astropy.units as u

    ra, dec, z, table = load_vagc(paths["vagc_post"], paths.get("vagc_kcorrect"),
                                  args.absmag_source, args.absmag_band)
    cascade = [("VAGC bright0 rows", len(ra))]

    sel = np.isfinite(z) & (z > Z_LO) & (z < Z_HI)
    cascade.append(("0.02 < z < 0.067", int(sel.sum())))

    mr = table[:, args.absmag_band] + evolution_correction(z, **EVOLUTION)
    n_mr = int((sel & np.isfinite(mr) & (mr < MR_MAX)).sum())
    if n_mr != CASCADE_REF["mr_cut"] and not args.allow_cascade_mismatch:
        log(f"  M_r cut gives {n_mr:,}, not {CASCADE_REF['mr_cut']:,}")
        diagnose_mr_stage(z, table, sel, CASCADE_REF["mr_cut"])
    sel &= np.isfinite(mr) & (mr < MR_MAX)
    cascade.append(("k+e corrected M_r < -19", int(sel.sum())))

    ra, dec, z = ra[sel], dec[sel], z[sel]

    mra, mdec, mlogm, mlogssfr = load_mpa(paths["mpa_raw"])
    idx, sep, _ = SkyCoord(ra * u.deg, dec * u.deg).match_to_catalog_sky(
        SkyCoord(mra * u.deg, mdec * u.deg))
    hit = sep.arcsec <= MATCH_ARCSEC
    cascade.append((f'matched to MPA-JHU ({MATCH_ARCSEC}")', int(hit.sum())))

    ra, dec, z = ra[hit], dec[hit], z[hit]
    logm, logssfr = mlogm[idx[hit]], mlogssfr[idx[hit]]

    keep = logm > LOGM_MIN
    cascade.append((f"log10(M*/Msun) > {LOGM_MIN} -> PARENT", int(keep.sum())))

    for label, n in cascade:
        log(f"  {label:38s} {n:>8,}")

    got = [n for _, n in cascade]
    want = list(CASCADE_REF.values())
    if got != want and not args.allow_cascade_mismatch:
        raise SystemExit(
            "parent cascade does not match the recorded SF/Q parent.\n"
            f"  got  {got}\n  want {want}\n"
            "These counts pin down the k+e sign convention, the absmag column and the\n"
            "match radius.  Fix the inputs, or pass --allow-cascade-mismatch if the\n"
            "divergence is understood and intended."
        )
    return ra[keep], dec[keep], z[keep], logm[keep], logssfr[keep]


# ---------------------------------------------------------------------------
# geometry and randoms
# ---------------------------------------------------------------------------
def comoving(z):
    from astropy.cosmology import FlatLambdaCDM
    return FlatLambdaCDM(H0=H0, Om0=OM0).comoving_distance(z).value  # h^-1 Mpc


def load_random_angles(path: Path, n: int, rng):
    with fits.open(path) as h:
        d = h[1].data
        ra = np.asarray(_column(d, "RA"), float)
        dec = np.asarray(_column(d, "DEC"), float)
    ok = np.isfinite(ra) & np.isfinite(dec) & (np.abs(dec) <= 90)
    ra, dec = ra[ok], dec[ok]
    pick = rng.integers(0, len(ra), size=n)
    return ra[pick], dec[pick]


def random_redshifts(scheme: str, z_data, n, rng):
    """`shuffled_z` resamples the observed n(z); `uniform_volume` draws ~ dV/dz."""
    if scheme == "shuffled_z":
        return rng.choice(z_data, size=n, replace=True)
    if scheme == "uniform_volume":
        d_lo, d_hi = comoving(np.array([Z_LO, Z_HI]))
        d = (d_lo**3 + (d_hi**3 - d_lo**3) * rng.random(n)) ** (1.0 / 3.0)
        grid = np.linspace(Z_LO, Z_HI, 4096)
        return np.interp(d, comoving(grid), grid)
    raise ValueError(scheme)


# ---------------------------------------------------------------------------
# estimator
# ---------------------------------------------------------------------------
def _ddrppi():
    """DDrppi_mocks with the leading positional arguments this Corrfunc wants.

    Corrfunc <=2.4 takes (autocorr, cosmology, nthreads, pimax, binfile); 2.5
    dropped `cosmology`.  We always pass comoving distances, so when the argument
    is present it is inert, but it still has to be there.
    """
    import inspect as _inspect
    from Corrfunc.mocks import DDrppi_mocks

    params = list(_inspect.signature(DDrppi_mocks).parameters)
    needs_cosmology = "cosmology" in params

    def call(autocorr, nthreads, pimax, **kw):
        lead = (autocorr, 1, nthreads, pimax) if needs_cosmology else (autocorr, nthreads, pimax)
        return DDrppi_mocks(*lead, **kw)

    return call


def _counts(ddrppi, nthreads, pimax, bins, s1, s2=None, want_rpavg=False):
    """(rp, pi) pair counts as an (nrp, npi) array; autocorrelation when s2 is None.

    Corrfunc counts ordered pairs, so an autocorrelation returns each distinct pair
    twice and excludes self-pairs.  Every jackknife identity below assumes exactly
    that convention.
    """
    if bins[0] <= 0.0:
        raise SystemExit(
            "the innermost rp bin edge must be > 0.  The jackknife identities count "
            "cross(A, all) with A a subset of all, which pairs every object with its "
            "own copy at zero separation; a bin reaching rp = 0 would collect those "
            "self-pairs and break the identity by 2|A|.")
    nrp, npi = len(bins) - 1, int(round(pimax))
    kw = dict(binfile=bins, is_comoving_dist=True, output_rpavg=want_rpavg)
    if s2 is None:
        res = ddrppi(1, nthreads, pimax, RA1=s1[0], DEC1=s1[1], CZ1=s1[2], **kw)
    else:
        res = ddrppi(0, nthreads, pimax, RA1=s1[0], DEC1=s1[1], CZ1=s1[2],
                     RA2=s2[0], DEC2=s2[1], CZ2=s2[2], **kw)
    n = np.asarray(res["npairs"], np.float64).reshape(nrp, npi)
    if not want_rpavg:
        return n
    return n, np.asarray(res["rpavg"], float).reshape(nrp, npi)


def landy_szalay_wp(dd, dr, rr, nd, nr, pimax):
    """wp(rp) from (rp, pi) counts, in Corrfunc's normalisation.

    xi = (DD/ND^2 - 2 DR/(ND NR) + RR/NR^2) / (RR/NR^2),  wp = 2 sum_pi xi dpi.
    For an autocorrelation D1R2 and D2R1 are the same count, hence the factor 2.
    """
    npi = dd.shape[1]
    n_dd = dd / (nd * nd)
    n_dr = dr / (nd * nr)
    n_rr = rr / (nr * nr)
    xi = np.where(n_rr > 0, (n_dd - 2.0 * n_dr + n_rr) / np.where(n_rr > 0, n_rr, 1.0), 0.0)
    return 2.0 * xi.sum(axis=1) * (pimax / npi)


def random_counts(rra, rdec, rdc, bins, pimax, nthreads, r_lab=None, njack=0):
    """RR, and the per-region pieces needed to rebuild it under the jackknife.

    RR depends only on the randoms, so it is identical for every subsample of the
    data.  It is also the dominant cost once the randoms outnumber the galaxies,
    which they must.  Computing it once per random scheme and handing it to each
    subsample removes a factor of three, and the same for the jackknife pieces.
    """
    ddrppi = _ddrppi()
    r_all = (rra, rdec, rdc)
    rr = _counts(ddrppi, nthreads, pimax, bins, r_all)
    if njack <= 0:
        return rr, None
    pieces = []
    for k in range(njack):
        rm = r_lab == k
        r_k = (rra[rm], rdec[rm], rdc[rm])
        pieces.append((_counts(ddrppi, nthreads, pimax, bins, r_k, r_all),
                       _counts(ddrppi, nthreads, pimax, bins, r_k),
                       int(rm.sum())))
    return rr, pieces


def measure_wp(ra, dec, dc, rra, rdec, rdc, bins, pimax, nthreads, rr=None):
    """Landy-Szalay wp(rp), plus rpavg and the DD pair counts per rp bin."""
    ddrppi = _ddrppi()
    d, r = (ra, dec, dc), (rra, rdec, rdc)
    dd, rpavg_2d = _counts(ddrppi, nthreads, pimax, bins, d, want_rpavg=True)
    dr = _counts(ddrppi, nthreads, pimax, bins, d, r)
    if rr is None:
        rr = _counts(ddrppi, nthreads, pimax, bins, r)

    wp = landy_szalay_wp(dd, dr, rr, len(ra), len(rra), pimax)
    tot = dd.sum(axis=1)
    rpavg = np.where(tot > 0, (rpavg_2d * dd).sum(axis=1) / np.maximum(tot, 1.0),
                     np.sqrt(bins[:-1] * bins[1:]))
    return wp, rpavg, tot.astype(np.int64), (dd, dr, rr)


# ---------------------------------------------------------------------------
# jackknife
# ---------------------------------------------------------------------------
def jackknife_regions(rra, rdec, njack, seed, nsample=200_000):
    """Compact, roughly equal-area sky regions, from k-means on the random catalogue.

    The randoms trace the mask, so clustering them puts the region boundaries where
    the survey actually is.  Centroids are Euclidean on the unit sphere, which is a
    good approximation to spherical k-means at these region sizes.
    """
    from scipy.cluster.vq import kmeans2

    rng = np.random.default_rng(seed)
    vec = _unit_vectors(rra, rdec)
    sub = vec[rng.choice(len(vec), size=min(nsample, len(vec)), replace=False)]
    try:
        centers, _ = kmeans2(sub, njack, minit="++", iter=50, seed=seed)
    except TypeError:                     # scipy < 1.7 has no `seed` keyword
        np.random.seed(seed)
        centers, _ = kmeans2(sub, njack, minit="++", iter=50)
    return centers


def _unit_vectors(ra, dec):
    a, d = np.radians(ra), np.radians(dec)
    return np.column_stack([np.cos(d) * np.cos(a), np.cos(d) * np.sin(a), np.sin(d)])


def assign_regions(ra, dec, centers, chunk=200_000):
    """Nearest-centroid label for every point, in chunks to bound memory."""
    vec = _unit_vectors(ra, dec)
    out = np.empty(len(vec), np.int32)
    for i in range(0, len(vec), chunk):
        out[i:i + chunk] = np.argmax(vec[i:i + chunk] @ centers.T, axis=1)
    return out


def jackknife_covariance(ra, dec, dc, rra, rdec, rdc, bins, pimax, nthreads,
                         d_lab, r_lab, njack, totals, rr_pieces=None):
    """Leave-one-region-out wp realisations and their covariance.

    Rather than recomputing every count njack times, each realisation is assembled
    from the full counts by the exact partition identity.  Writing A for the dropped
    region and B for the rest, with ordered pairs,

        BB = total - AA - 2 AB = total - 2 * cross(A, all) + auto(A)

    for the two autocorrelations, and for the data-random cross term

        D_B R_B = total - cross(D_A, R_all) - cross(D_all, R_A) + cross(D_A, R_A).

    Summed over regions the cross terms cost about one full pass each, so the whole
    jackknife is a small multiple of the base measurement rather than njack times it.
    """
    ddrppi = _ddrppi()
    dd_tot, dr_tot, rr_tot = totals
    d_all, r_all = (ra, dec, dc), (rra, rdec, rdc)
    nd, nr = len(ra), len(rra)
    wp_jk = np.empty((njack, len(bins) - 1))

    for k in range(njack):
        dm, rm = d_lab == k, r_lab == k
        d_k = (ra[dm], dec[dm], dc[dm])
        r_k = (rra[rm], rdec[rm], rdc[rm])
        nd_k, nr_k = int(dm.sum()), int(rm.sum())
        if nd_k == nd or nr_k == nr:
            raise SystemExit(f"jackknife region {k} would empty the sample "
                             f"({nd_k}/{nd} data, {nr_k}/{nr} randoms); lower --njack")

        dd_k = (dd_tot - 2.0 * _counts(ddrppi, nthreads, pimax, bins, d_k, d_all)
                + _counts(ddrppi, nthreads, pimax, bins, d_k))
        if rr_pieces is None:
            rr_cross = _counts(ddrppi, nthreads, pimax, bins, r_k, r_all)
            rr_auto = _counts(ddrppi, nthreads, pimax, bins, r_k)
        else:
            rr_cross, rr_auto, _ = rr_pieces[k]
        rr_k = rr_tot - 2.0 * rr_cross + rr_auto
        dr_k = (dr_tot - _counts(ddrppi, nthreads, pimax, bins, d_k, r_all)
                - _counts(ddrppi, nthreads, pimax, bins, d_all, r_k)
                + _counts(ddrppi, nthreads, pimax, bins, d_k, r_k))

        wp_jk[k] = landy_szalay_wp(dd_k, dr_k, rr_k, nd - nd_k, nr - nr_k, pimax)

    mean = wp_jk.mean(axis=0)
    dev = wp_jk - mean
    cov = (njack - 1.0) / njack * (dev.T @ dev)
    return wp_jk, cov


# ---------------------------------------------------------------------------
# power-law fit and deprojection
# ---------------------------------------------------------------------------
_GL_X, _GL_W = np.polynomial.legendre.leggauss(256)


def wp_powerlaw(rp, r0, gamma, pimax=PIMAX):
    """2 * int_0^pimax (sqrt(rp^2+pi^2)/r0)^-gamma dpi, by Gauss-Legendre."""
    rp = np.atleast_1d(np.asarray(rp, float))
    pi = 0.5 * pimax * (_GL_X + 1.0)
    r = np.hypot(rp[:, None], pi[None, :])
    return pimax * ((r / r0) ** (-gamma) @ _GL_W)


def published_sigma(source: str, rpavg, wp):
    """Published fractional error sigma/wp, interpolated in log rp and rescaled to wp.

    Only the shape across rp bins affects a best fit, so transferring the
    fractional curve is well defined even when the sample is not the published one.
    """
    frac = PUB_SIG[source] / PUB_WP[source]
    return wp * np.interp(np.log(rpavg), np.log(PUB_RP), frac)


def fit_powerlaw(rpavg, wp, sigma=None, lo=FIT_LO, hi=FIT_HI, pimax=PIMAX):
    """Least squares against the finite-pi_max model.

    `sigma` is either 1-D diagonal errors or a 2-D covariance matrix; a covariance
    engages the full C^-1 form, which is what the SDSS analyses use, because
    neighbouring rp bins share pairs and are strongly correlated.

    Returns (r0, gamma, chi2_dof).  chi2_dof is a goodness diagnostic, not an
    uncertainty on the parameters.  Note that any scalar rescaling of C -- a
    Hartlap-style factor, say -- leaves the best fit untouched and moves only this
    number, so no such correction is applied.
    """
    from scipy.optimize import curve_fit
    m = (rpavg >= lo) & (rpavg <= hi) & np.isfinite(wp) & (wp > 0)
    if m.sum() < 3:
        raise SystemExit(f"only {int(m.sum())} usable rp bins in [{lo}, {hi}]")
    sig = None
    if sigma is not None:
        sigma = np.asarray(sigma, float)
        sig = sigma[np.ix_(m, m)] if sigma.ndim == 2 else sigma[m]
    if sig is not None and sig.ndim == 2:
        # a jackknife covariance from N regions has rank at most N-1, so it is
        # singular whenever the fit uses as many bins as there are regions.  Say so
        # here rather than letting scipy fail on the Cholesky factorisation.
        nbin = int(m.sum())
        ev = np.linalg.eigvalsh(sig)
        if ev.min() <= 0 or ev.max() / ev.min() > 1e12:
            raise SystemExit(
                f"the covariance over {nbin} fit bins is singular or near-singular "
                f"(smallest eigenvalue {ev.min():.3e}, condition number "
                f"{ev.max() / max(ev.min(), 1e-300):.3e}).  A jackknife covariance "
                f"from N regions has rank at most N-1, so --njack must exceed {nbin} "
                f"and should comfortably exceed it; 50-100 is the norm.")
    p, _ = curve_fit(lambda x, r0, g: wp_powerlaw(x, r0, g, pimax),
                     rpavg[m], wp[m], p0=[6.0, 1.8], sigma=sig,
                     absolute_sigma=sig is not None, maxfev=20000)
    resid = wp[m] - wp_powerlaw(rpavg[m], p[0], p[1], pimax)
    if sig is None:
        chi2 = (resid**2).sum()
    elif sig.ndim == 2:
        chi2 = float(resid @ np.linalg.solve(sig, resid))
    else:
        chi2 = ((resid / sig)**2).sum()
    return float(p[0]), float(p[1]), float(chi2 / (m.sum() - 2))


def xi_from_fit(r, r0_phys, gamma):
    return (r / r0_phys) ** (-gamma)


def cross_from_identity(r, fits_phys, f_a, f_b):
    """xi_AB from the exact binary partition; returns (xi_AB, closure_error)."""
    xa = xi_from_fit(r, *fits_phys["A"])
    xb = xi_from_fit(r, *fits_phys["B"])
    xall = xi_from_fit(r, *fits_phys["all"])
    xab = (xall - f_a**2 * xa - f_b**2 * xb) / (2.0 * f_a * f_b)
    back = f_a**2 * xa + f_b**2 * xb + 2.0 * f_a * f_b * xab
    return xab, float(np.abs(back - xall).max() / np.abs(xall).max())


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------
def make_figures(outdir: Path, split, bins, samples, fits_phys, labels):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ca, cb = ("blue", "red") if split == "ssfr" else ("darkorange", "purple")
    colors = {"all": "black", "A": ca, "B": cb}

    fig, ax = plt.subplots(figsize=(6.0, 4.6))
    for key, name in (("all", "all"), ("A", labels[0]), ("B", labels[1])):
        s = samples[key]
        ax.plot(s["rpavg"], s["wp"], "o", ms=4, color=colors[key], label=name)
        grid = np.logspace(np.log10(FIT_LO), np.log10(FIT_HI), 200)
        r0h, g = fits_phys[key][0] * H_LITTLE, fits_phys[key][1]
        ax.plot(grid, wp_powerlaw(grid, r0h, g), "-", lw=1.4, color=colors[key])
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"$r_{\rm p}$  [$h^{-1}$ Mpc]")
    ax.set_ylabel(r"$w_{\rm p}(r_{\rm p})$  [$h^{-1}$ Mpc]")
    ax.set_title(f"{split} split, $\\pi_{{\\max}}={PIMAX:.0f}\\,h^{{-1}}$Mpc")
    ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(outdir / "fig1_wp.png", dpi=180)
    fig.savefig(outdir / "fig1_wp.pdf"); plt.close(fig)

    r = np.logspace(np.log10(FIT_LO / H_LITTLE), np.log10(FIT_HI / H_LITTLE), 400)
    f_a, f_b = samples["A"]["fraction"], samples["B"]["fraction"]
    xab, _ = cross_from_identity(r, fits_phys, f_a, f_b)
    fig, ax = plt.subplots(figsize=(6.0, 4.6))
    ax.plot(r, xi_from_fit(r, *fits_phys["A"]), color=ca, lw=2.4,
            label=rf"$\xi_{{\rm {labels[0]},{labels[0]}}}$")
    ax.plot(r, xi_from_fit(r, *fits_phys["B"]), color=cb, lw=2.4,
            label=rf"$\xi_{{\rm {labels[1]},{labels[1]}}}$")
    ax.plot(r, xab, color="green", lw=2.4, ls="--",
            label=rf"$\xi_{{\rm {labels[0]},{labels[1]}}}$ (identity)")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"$r$  [Mpc]"); ax.set_ylabel(r"$\xi(r)$")
    ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(outdir / "fig2_xi.png", dpi=180)
    fig.savefig(outdir / "fig2_xi.pdf"); plt.close(fig)


# ---------------------------------------------------------------------------
def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base-dir", type=Path, default=Path("/home/daiki.watarai/BBH_population"))
    p.add_argument("--split", choices=("ssfr", "mstar"), default="mstar")
    p.add_argument("--vagc-post", default="data/galaxies/nyu_vagc/raw/post_catalog.dr72bright0.fits")
    p.add_argument("--vagc-kcorrect", default="data/galaxies/nyu_vagc/raw/kcorrect.none.petro.z0.10.fits")
    p.add_argument("--vagc-random", default="data/galaxies/nyu_vagc/raw/random-0.dr72bright0.fits")
    p.add_argument("--mpa-raw", default="data/galaxies/mpa_jhu_dr7/raw")
    p.add_argument("--absmag-source", choices=("auto", "post", "kcorrect"), default="auto")
    p.add_argument("--absmag-band", type=int, default=2, help="0=u 1=g 2=r 3=i 4=z")
    p.add_argument("--output-dir", default=None)
    p.add_argument("--randoms", choices=("shuffled_z", "uniform_volume", "both"),
                   default="both")
    p.add_argument("--nrand", type=int, default=NRAND)
    p.add_argument("--fit-weights", choices=("jackknife", "published", "equal"),
                   default="jackknife",
                   help="which weighting drives the reported fit; all of them are "
                        "computed and stored either way")
    p.add_argument("--njack", type=int, default=100,
                   help="jackknife regions (0 disables the jackknife entirely)")
    p.add_argument("--nthreads", type=int, default=8)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--inspect", action="store_true",
                   help="print the columns of every input file and exit")
    p.add_argument("--allow-cascade-mismatch", action="store_true")
    a = p.parse_args(argv)

    paths = {
        "vagc_post": a.base_dir / a.vagc_post,
        "vagc_kcorrect": a.base_dir / a.vagc_kcorrect,
        "vagc_random": a.base_dir / a.vagc_random,
        "mpa_gal_info": a.base_dir / a.mpa_raw / "gal_info_dr7_v5_2.fit.gz",
        "mpa_totlgm": a.base_dir / a.mpa_raw / "totlgm_dr7_v5_2.fit.gz",
        "mpa_totspecsfr": a.base_dir / a.mpa_raw / "gal_totspecsfr_dr7_v5_2.fits.gz",
    }
    if a.inspect:
        return inspect(paths)
    paths["mpa_raw"] = a.base_dir / a.mpa_raw

    outdir = a.base_dir / (a.output_dir or f"results/{a.split}_autocorrelation")
    outdir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(a.seed)

    log("building parent")
    ra, dec, z, logm, logssfr = build_parent(a, paths)

    if a.split == "ssfr":
        is_a = logssfr > SSFR_SPLIT
        threshold = SSFR_SPLIT
    else:
        threshold = float(np.median(logm))
        is_a = logm > threshold
    labels = SPLIT_LABELS[a.split]
    n_a, n_b = int(is_a.sum()), int((~is_a).sum())
    f_a, f_b = n_a / len(is_a), n_b / len(is_a)
    log(f"split {a.split} at {threshold:.6f}: {labels[0]}={n_a:,} {labels[1]}={n_b:,} "
        f"(f_{labels[0]}={f_a:.6f})")

    dc = comoving(z)
    bins = np.logspace(np.log10(RP_MIN), np.log10(RP_MAX), NRP + 1)
    rra, rdec = load_random_angles(paths["vagc_random"], a.nrand, rng)
    schemes = ("shuffled_z", "uniform_volume") if a.randoms == "both" else (a.randoms,)

    d_lab = r_lab = None
    if a.njack > 0:
        if a.njack < 3 * (NRP - 3):
            log(f"WARNING: --njack {a.njack} is small for {NRP - 3} fit bins; the "
                f"covariance inverse will be noisy.  50-100 regions is the norm.")
        log(f"building {a.njack} jackknife regions from the randoms")
        centers = jackknife_regions(rra, rdec, a.njack, a.seed)
        d_lab, r_lab = assign_regions(ra, dec, centers), assign_regions(rra, rdec, centers)
        occ = np.bincount(r_lab, minlength=a.njack)
        log(f"  region occupancy: min {occ.min():,}  median {int(np.median(occ)):,}  "
            f"max {occ.max():,}")

    subsets = {"all": np.ones(len(z), bool), "A": is_a, "B": ~is_a}
    out = {
        "split": a.split,
        "threshold": threshold,
        "labels": {"A": labels[0], "B": labels[1]},
        "parent": {"N": len(z), "N_A": n_a, "N_B": n_b, "f_A": f_a, "f_B": f_b},
        "config": {
            "binfile": bins.tolist(), "pimax": PIMAX, "nrand": a.nrand,
            "cosmology": f"FlatLCDM Om={OM0} H0={H0:.0f} (h^-1 Mpc)",
            "fit_range_hinvMpc": [FIT_LO, FIT_HI], "h": H_LITTLE,
            "fit_weights": a.fit_weights, "njack": a.njack,
            "fit": ("least squares against the finite-pimax model.  jackknife uses the "
                    "full leave-one-region-out covariance; published transfers the "
                    "diagonal errors of Hearin+2014 Table 1 (all) and Watson+2015 "
                    "A1/A2 (SF/Q) as fractional errors; equal is unweighted"),
        },
    }

    for scheme in schemes:
        log(f"randoms: {scheme}")
        rdc = comoving(random_redshifts(scheme, z, a.nrand, rng))
        block, fits_phys = {}, {}
        for key, mask in subsets.items():
            wp, rpavg, npairs, totals = measure_wp(ra[mask], dec[mask], dc[mask],
                                                   rra, rdec, rdc, bins, PIMAX,
                                                   a.nthreads)
            entry = {
                "ND": int(mask.sum()),
                "fraction": float(mask.sum() / len(mask)),
                "wp": wp.tolist(), "rpavg": rpavg.tolist(),
                "npairs_DD": npairs.tolist(),
            }

            # every weighting is fitted every run: they cost microseconds, and
            # keeping the published one alive is what lets --split ssfr stay a
            # check on the pipeline even when the jackknife drives the answer
            source = WEIGHT_SOURCE[a.split][key]
            weightings = {
                "equal": None,
                "published": published_sigma(source, rpavg, wp),
            }
            if a.njack > 0:
                log(f"  {key:3s} jackknife, {a.njack} regions")
                wp_jk, cov = jackknife_covariance(
                    ra[mask], dec[mask], dc[mask], rra, rdec, rdc, bins, PIMAX,
                    a.nthreads, d_lab[mask], r_lab, a.njack, totals)
                weightings["jackknife"] = cov
                entry["jackknife"] = {
                    "njack": a.njack,
                    "sigma": np.sqrt(np.diag(cov)).tolist(),
                    "cov": cov.tolist(),
                    "wp_realisations": wp_jk.tolist(),
                }

            entry["fits"] = {}
            for wname, sigma in weightings.items():
                r0h, gamma, chi2 = fit_powerlaw(rpavg, wp, sigma)
                entry["fits"][wname] = {
                    "r0_hinvMpc": r0h, "r0_Mpc": r0h / H_LITTLE,
                    "gamma": gamma, "chi2_dof": chi2,
                }
            if a.fit_weights not in entry["fits"]:
                raise SystemExit(f"--fit-weights {a.fit_weights} needs --njack > 0")
            primary = entry["fits"][a.fit_weights]
            entry.update({"weight_source": source,
                          "r0_hinvMpc": primary["r0_hinvMpc"],
                          "r0_Mpc": primary["r0_Mpc"],
                          "gamma": primary["gamma"],
                          "chi2_dof": primary["chi2_dof"]})
            fits_phys[key] = (primary["r0_Mpc"], primary["gamma"])
            block[key] = entry

            log(f"  {key:3s} ND={int(mask.sum()):>7,}")
            for wname, f in entry["fits"].items():
                star = " <- primary" if wname == a.fit_weights else ""
                log(f"      {wname:10s} r0={f['r0_Mpc']:8.4f} Mpc  "
                    f"gamma={f['gamma']:7.4f}  chi2/dof={f['chi2_dof']:8.3f}{star}")

        r = np.logspace(np.log10(FIT_LO / H_LITTLE), np.log10(FIT_HI / H_LITTLE), 400)
        _, closure = cross_from_identity(r, fits_phys, f_a, f_b)
        block["cross"] = {"from": "binary-partition identity, not fitted",
                          "closure_error": closure}
        out[scheme] = block

        if a.split == "ssfr":
            # SSFR_REF was fitted with the published errors, so the comparison has to
            # be against that weighting whatever is driving the primary fit
            pub = {k: (block[k]["fits"]["published"]["r0_Mpc"],
                       block[k]["fits"]["published"]["gamma"]) for k in SSFR_REF}
            dev = {k: (abs(pub[k][0] - SSFR_REF[k][0]) / SSFR_REF[k][0],
                       abs(pub[k][1] - SSFR_REF[k][1]) / SSFR_REF[k][1])
                   for k in SSFR_REF}
            out[scheme]["validation"] = {
                k: {"r0_rel_dev": v[0], "gamma_rel_dev": v[1]} for k, v in dev.items()}
            worst = max(max(v) for v in dev.values())
            note = "" if scheme == "uniform_volume" else \
                "  (note: the published fits used the uniform_volume randoms)"
            log(f"  validation vs stored SF/Q fits: worst relative deviation "
                f"{worst:.4f}{note}")
            for k, v in dev.items():
                log(f"    {k:3s} r0 {pub[k][0]:8.4f} vs {SSFR_REF[k][0]:8.4f} "
                    f"({v[0]:+.2%})   gamma {pub[k][1]:7.4f} vs "
                    f"{SSFR_REF[k][1]:7.4f} ({v[1]:+.2%})")

        if scheme == schemes[0]:
            make_figures(outdir, a.split, bins, block, fits_phys, labels)

    path = outdir / f"{a.split}_autocorrelation.json"
    path.write_text(json.dumps(out, indent=1))
    log(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
