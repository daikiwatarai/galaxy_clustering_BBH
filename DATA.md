# Input data

Derived data products in `data/`, read by the notebooks and scripts.  Set `HOSTINFER_DATA` to
point elsewhere, keeping these file names.

| File | Size | Needed by |
|---|---|---|
| `cyl_counts.npz` | 17 MB | Fig. 4, covariance-closure validation |
| `cyl_counts_meta.json` | 232 B | describes the sample behind `cyl_counts.npz`; not read by the code |
| `geometry_z0to1.npz` | 6 MB | Fig. 2, Table I |
| `nu_xi_tables.npz` | 136 kB | `fq_posterior_forecast.py` (cache; rebuilt if absent) |
| `wp_measured.json` | 949 B | `cylinder_host_type_forecast.py`, hence Fig. 4 and the scripts |
| `wp_published.csv` | 1.5 kB | published values our fits are compared with; not read by the code |

All of them come from the NYU-VAGC `bright0` sample (0.02 < z < 0.067, k+e corrected
M_r < −19, matched to MPA-JHU DR7 within 1.5″, log₁₀(M*/M⊙) > 9.8, split into quenched and
star-forming at log₁₀(sSFR/yr⁻¹) = −11), except `geometry_z0to1.npz`, which comes from GWFish.

## `cyl_counts.npz`

Neighbour counts around every parent galaxy, and around 150 000 random points, in cylinders
aligned with each centre's line of sight.  Made by `scripts/sdss_cylinder_counts.py`.

| Array | Shape | Meaning |
|---|---|---|
| `R_edges`, `L_edges` | (5,), (6,) | `R_perp` = 2, 3, 5, 8, 12 and `L_par` = 2, 5, 10, 20, 30, 40 h⁻¹ Mpc |
| `H_gal`, `H_rand` | (74727, 2, 5, 6), (150000, …) | neighbours: [quenched, star-forming] × R × L, centre excluded |
| `acc_gal`, `acc_rand` | (5, 6, N) | footprint acceptance; a cylinder is usable only where True |
| `is_sf` | (74727,) | star-forming flag of the central galaxy |
| `region`, `rregion` | (N,) | 50 k-means jackknife labels |
| `dc`, `rdc` | (N,) | comoving distance [h⁻¹ Mpc] |

Parent size 74 727, `f_SF` = 0.523, shell 59.71–198.09 h⁻¹ Mpc, acceptance 0.99 of 800 points.

## `geometry_z0to1.npz`

One entry per simulated BBH detection for each network (`O5_HLVK30`,
`POSTO5_FIDUCIAL_HLIVK`, `ET+2CE`), as `<network>__<quantity>`: `R_h`, `L_h` [h⁻¹ Mpc], `V`
[Mpc³], `z`, `ra`, `dec`, and the scalars `ndot` (all-sky detections per year) and `nsample`.
Made by `geometry()` in `scripts/fq_posterior_forecast.py` from GWFish output (IMRPhenomXHM,
network SNR ≥ 12, z ≤ 1, 100 000 mergers from the GWTC-5 Default BBH population).

## `nu_xi_tables.npz`

`lR`, `lL`, `NUQ`, `NUS`, `XI`: host-conditioned mean neighbour counts and volume-averaged
two-point functions on a log grid.  Pure model output; the Fig. 2 notebook builds its own copy.

## `wp_measured.json`

`rp` (15 bins), `wp` and `sig` for the `all`, SF (`A`) and Q (`B`) samples, the fit
configuration and the resulting power laws (`ssfr_ref`).  Our own measurement on the `bright0`
parent with uniform-in-comoving-volume randoms; `sig` are the published diagonal uncertainties
used as fit weights.  The same values appear inline in the Fig. 3 notebook.

## `wp_published.csv`

Tabulated w_p(r_p) from Watson et al. (2015) Tables A1 and A2 for the SF and Q samples, and
Hearin et al. (2014) Table 1 for the full sample.
