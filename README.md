# galaxy_clustering_BBH

Code for *Galaxy Clustering Can Reveal Host-Galaxy Properties of Binary Black Hole Mergers*
([arXiv:2610.11045](https://arxiv.org/abs/2610.11045)): inferring the host-galaxy population of binary black hole mergers
from the galaxies inside gravitational-wave localization volumes, without identifying
individual hosts.

## Reproducing the paper

| Paper item | Run | Needs | Time |
|---|---|---|---|
| Fig. 1 host-conditioned environments | `notebooks/fig1_environments.ipynb` | — | 15 s |
| Fig. 2 forecast posteriors on the quenched-host fraction | `notebooks/fig2_fq_forecast.ipynb` | `geometry_z0to1.npz` | 1.5 min |
| Table I events reaching the host-resolvable scale | last section of the same notebook | `geometry_z0to1.npz` | 10 s |
| Fig. 3 and Table II projected correlation functions and fits | `notebooks/fig3_wp_fits.ipynb` | — | 3 s |
| Fig. 4 counts in SDSS cylinders vs the model | `notebooks/fig4_counts_vs_sdss.ipynb` | `cyl_counts.npz` | 5 s |
| Covariance-closure validation | `python scripts/cylinder_closure_validation.py` | `cyl_counts.npz` | 15 s |

Each notebook runs top to bottom and writes its own figure.

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
jupyter lab
```

Inputs are read from `data/` (see `DATA.md`); set `HOSTINFER_DATA` to use another directory.
Figures go to `figures/`, other script output to `results/`; both are git-ignored.

## Scripts

| File | What it does |
|---|---|
| `cylinder_host_type_forecast.py` | correlation functions, host-conditioned means, volume-averaged two-point functions, covariance closure |
| `pln_batch.py` | bivariate Poisson--lognormal likelihood by adaptive Gauss--Hermite quadrature |
| `fq_posterior_forecast.py` | mock realizations and the posterior on the quenched-host fraction |
| `resolvable_events_table.py` | Table I |
| `cylinder_closure_validation.py` | closure test against the measured SDSS counts |
| `sfq_cross_correlation_length.py` | correlation lengths of the auto- and cross-correlations |
| `plot_count_distribution_vs_volume.py` | Fig. 4 as a script |
| `fq_forecast_redshift_contribution.py` | which redshifts the forecast constraint comes from |
| `sdss_split_auto_correlation.py` | projected correlation functions of the SF/Q split |
| `sdss_cylinder_counts.py`, `sdss_galaxy_centered_counts.py` | SDSS host-centred cylinder counts |

The three SDSS measurements need the NYU-VAGC `bright0` and MPA-JHU DR7 catalogues, which are
not part of this repository; their outputs are shipped in `data/` so everything downstream runs
from a clone.  `geometry_z0to1.npz` likewise comes from GWFish Fisher tables (~1 GB), rebuilt by
`geometry()` in `fq_posterior_forecast.py` with `HOSTINFER_FISHER` and `HOSTINFER_RATES` set.

## Citation

Daiki Watarai, Hayato Imafuku, and Fumihiro Chuman, *Galaxy Clustering Can Reveal Host-Galaxy Properties of Binary Black Hole Mergers*,
[arXiv:2610.11045](https://arxiv.org/abs/2610.11045).
