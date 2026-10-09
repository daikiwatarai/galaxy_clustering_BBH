#!/usr/bin/env python3
"""CYLINDER galaxy-centered counts in SDSS: the empirical test of the two-point
closure as a function of aspect ratio.

Item 6 of the cylinder reformulation.  Oguri & Lin (2015) argue that for a cylinder
long along the line of sight the higher-order contributions to the neighbour-count
covariance are diluted.  This measures that directly.

For every centre (a galaxy of known type, and separately a random point) the
neighbours are decomposed in the CENTRE'S OWN line-of-sight frame,

    s_par = (x_j - x_i) . nhat_i ,     r_perp = sqrt(|dx|^2 - s_par^2)

and histogrammed once into (r_perp bin, |s_par| bin).  A cumulative sum then gives
the counts for EVERY cylinder on the grid from a single pass over the pairs.

    R_perp edges : 2, 3, 5, 8, 12          h^-1 Mpc
    L_par  edges : 2, 5, 10, 20, 30, 40    h^-1 Mpc
    -> 30 geometries, V = 50 - 3.6e4 h^-3 Mpc^3, A = L/R = 0.17 - 20

Random centres are measured with the identical machinery so that the two-point
input can be recalibrated:

    Xi^emp_ij = [Cov_rand(N_i,N_j) - delta_ij Nbar_i] / (Nbar_i Nbar_j)

Without that step the ratio mixes a two-point input mismatch with the higher-order
term; the spherical run showed the two are comparable in size and pull opposite ways.

Parent sample, footprint mask and jackknife regions are unchanged from
scripts/sdss_galaxy_centered_counts.py (Yang DR7 SDSS7 reconstruction; the NYU-VAGC
bright0 files are not on this cluster).

Edge handling: a cylinder is used only if >= FMIN of NSAMP points drawn uniformly
inside it (in the centre's own frame) land in the footprint AND inside the redshift
shell.  Geometry only, never the counts being measured.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

R_EDGES = np.array([2.0, 3.0, 5.0, 8.0, 12.0])
L_EDGES = np.array([2.0, 5.0, 10.0, 20.0, 30.0, 40.0])
RMAX = float(np.hypot(R_EDGES[-1], L_EDGES[-1]))
NSAMP, FMIN, NJACK = 800, 0.99, 50

LOG: list[str] = []


def log(m=""):
    print(m, flush=True)
    LOG.append(m)


# --------------------------------------------------------------- local frame
def frames(vec):
    """orthonormal (e1, e2) perpendicular to each unit vector."""
    a = np.zeros_like(vec)
    a[:, 2] = 1.0
    bad = np.abs(vec[:, 2]) > 0.9
    a[bad] = 0.0
    a[bad, 0] = 1.0
    e1 = np.cross(vec, a)
    e1 /= np.linalg.norm(e1, axis=1, keepdims=True)
    e2 = np.cross(vec, e1)
    return e1, e2


def cylinder_samples(n, rng):
    u = rng.random(n)
    ph = rng.random(n) * 2 * np.pi
    z = rng.random(n) * 2 - 1
    return np.sqrt(u) * np.cos(ph), np.sqrt(u) * np.sin(ph), z      # unit cylinder


def accept(cxyz, nhat, e1, e2, Rp, L, samp, occ, nside, d_lo, d_hi, chunk=2000):
    """fraction of the cylinder inside footprint AND shell, per centre."""
    import healpy as hp
    a, b, c = samp
    out = np.empty(len(cxyz))
    for i in range(0, len(cxyz), chunk):
        C = cxyz[i:i + chunk]
        P = (C[:, None, :]
             + Rp * a[None, :, None] * e1[i:i + chunk][:, None, :]
             + Rp * b[None, :, None] * e2[i:i + chunk][:, None, :]
             + L * c[None, :, None] * nhat[i:i + chunk][:, None, :])
        d = np.linalg.norm(P, axis=2)
        ok = (d > d_lo) & (d < d_hi)
        ok &= occ[hp.vec2pix(nside, P[..., 0], P[..., 1], P[..., 2])]
        out[i:i + chunk] = ok.mean(axis=1)
    return out


# --------------------------------------------------------------- pair histogram
def cyl_hist(cxyz, nhat, gxyz, is_sf, gal_self_index, tree, chunk=1500):
    """per-centre 2-D histogram of neighbours in (r_perp bin, |s_par| bin), by type."""
    from scipy.spatial import cKDTree
    nR, nL = len(R_EDGES), len(L_EDGES)
    H = np.zeros((len(cxyz), 2, nR, nL), np.int32)
    for i0 in range(0, len(cxyz), chunk):
        sl = slice(i0, min(i0 + chunk, len(cxyz)))
        C, N = cxyz[sl], nhat[sl]
        pr = cKDTree(C).sparse_distance_matrix(tree, RMAX, output_type="ndarray")
        ii, jj = pr["i"], pr["j"]
        if gal_self_index is not None:                 # drop the host's self-pair
            keep = gal_self_index[sl][ii] != jj
            ii, jj = ii[keep], jj[keep]
        d = gxyz[jj] - C[ii]
        sp = np.einsum("ij,ij->i", d, N[ii])
        rp = np.sqrt(np.maximum((d * d).sum(1) - sp * sp, 0.0))
        ir = np.searchsorted(R_EDGES, rp, side="right")
        il = np.searchsorted(L_EDGES, np.abs(sp), side="right")
        m = (ir < nR) & (il < nL)
        flat = (((ii[m] * 2 + is_sf[jj[m]].astype(np.int64)) * nR + ir[m]) * nL + il[m])
        H[sl] += np.bincount(flat, minlength=len(C) * 2 * nR * nL
                             ).reshape(len(C), 2, nR, nL).astype(np.int32)
    return H.cumsum(axis=2).cumsum(axis=3)             # counts within (R_perp, L_par)


# --------------------------------------------------------------- main
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base-dir", type=Path, default=Path("/home/daiki.watarai/BBH_population"))
    p.add_argument("--output-dir", default="results/sdss_cylinder_counts")
    p.add_argument("--nrandom", type=int, default=150_000)
    p.add_argument("--seed", type=int, default=20260908)
    a = p.parse_args()

    import importlib.util
    from scipy.spatial import cKDTree
    spec = importlib.util.spec_from_file_location(
        "gcc", a.base_dir / "scripts/sdss_galaxy_centered_counts.py")
    G = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(G)

    t0 = time.time()
    out = a.base_dir / a.output_dir
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(a.seed)

    log("=" * 88)
    log("SDSS CYLINDER galaxy-centered counts")
    log("=" * 88)
    ra, dec, z, is_sf, (ra_all, dec_all) = G.build_parent(a.base_dir)
    dc = G.comoving(z)
    d_lo, d_hi = G.comoving(np.array([G.Z_LO, G.Z_HI]))
    vec = G.unit_vectors(ra, dec)
    xyz = vec * dc[:, None]
    e1, e2 = frames(vec)
    log(f"  shell {d_lo:.2f} - {d_hi:.2f} h^-1 Mpc;  max pair radius {RMAX:.2f}")

    occ = G.build_mask(ra_all, dec_all, G.NSIDE_MASK)
    region, kcen = G.jackknife_regions(vec, NJACK, a.seed)

    rxyz, rdc = G.random_centres(a.nrandom, occ, G.NSIDE_MASK, d_lo, d_hi, rng)
    rvec = rxyz / np.linalg.norm(rxyz, axis=1, keepdims=True)
    re1, re2 = frames(rvec)
    rregion = np.argmax(rvec @ kcen.T, axis=1)
    log(f"  {a.nrandom:,} random centres")

    tree = cKDTree(xyz)
    samp = cylinder_samples(NSAMP, rng)

    log("  pair histogram: galaxies ...")
    Hg = cyl_hist(xyz, vec, xyz, is_sf, np.arange(len(xyz)), tree)
    log(f"    done {time.time()-t0:.0f}s;  randoms ...")
    Hr = cyl_hist(rxyz, rvec, xyz, is_sf, None, tree)
    log(f"    done {time.time()-t0:.0f}s")

    log("")
    log("  acceptance per geometry (fraction of centres whose cylinder fits)")
    log("  %7s %7s %10s %7s | %9s %9s" % ("R_perp", "L_par", "V", "A", "gal", "rand"))
    Ag = np.zeros((len(R_EDGES), len(L_EDGES), len(xyz)), bool)
    Ar = np.zeros((len(R_EDGES), len(L_EDGES), len(rxyz)), bool)
    for i, Rp in enumerate(R_EDGES):
        for j, L in enumerate(L_EDGES):
            fg = accept(xyz, vec, e1, e2, Rp, L, samp, occ, G.NSIDE_MASK, d_lo, d_hi)
            fr = accept(rxyz, rvec, re1, re2, Rp, L, samp, occ, G.NSIDE_MASK, d_lo, d_hi)
            Ag[i, j] = fg >= FMIN
            Ar[i, j] = fr >= FMIN
            log("  %7.1f %7.1f %10.4g %7.2f | %9d %9d"
                % (Rp, L, 2 * np.pi * Rp ** 2 * L, L / Rp, Ag[i, j].sum(), Ar[i, j].sum()))

    np.savez_compressed(
        out / "cyl_counts.npz", R_edges=R_EDGES, L_edges=L_EDGES,
        is_sf=is_sf, region=region.astype(np.int16), rregion=rregion.astype(np.int16),
        H_gal=Hg, H_rand=Hr, acc_gal=Ag, acc_rand=Ar,
        dc=dc.astype(np.float32), rdc=rdc.astype(np.float32))
    (out / "run_log.txt").write_text("\n".join(LOG) + "\n")
    (out / "meta.json").write_text(json.dumps(dict(
        RMAX=RMAX, NSAMP=NSAMP, FMIN=FMIN, NJACK=NJACK, nrandom=a.nrandom,
        n_parent=int(len(ra)), f_SF=float(is_sf.mean()),
        shell=[float(d_lo), float(d_hi)], runtime_s=time.time() - t0), indent=1))
    log(f"\nwrote {out}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
