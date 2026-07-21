# Accelerated Random-Sweep Gibbs Sampling for Gaussian Graphical Models via Dual Normal Factor Graphs

Code accompanying the paper:

> B. Khodabandeh and M. Molkaraie, "Accelerated Random-Sweep Gibbs Sampling for
> Gaussian Graphical Models via Dual Normal Factor Graphs," *IEEE Transactions
> on Information Theory* (under review), 2026.
> [[paper (link TBD)]()]

We study Gibbs sampling for Gaussian graphical models with a thin-membrane
prior, and show that sampling in a *dual* representation — obtained by taking
the Fourier transform of the local factors of the normal factor graph — gives
exact, topology-independent convergence rates and, empirically, dramatic
speed-ups over sampling in the original (primal) domain, at matching
`O(|E|)` per-sweep cost.

## Repository contents

| File | Description |
|---|---|
| `graphs.py` | GMRF model classes (torus, `k`-regular, complete bipartite, heterogeneous Watts–Strogatz), precision matrix construction, and the primal/dual samplers. |
| `sampler.py` | Numba-jitted Gibbs sampling kernels (primal and dual, `O(1)`-per-edge dual updates). |
| `divergences.py` | Optional closed-form Gaussian divergence utilities (KL, MMD, FID, W2). Not used in the paper's final figures; kept for reference. |
| `experiments.py` | Reproduces every figure in Section VII of the paper. Each experiment is a `run_*` function; see the table below. |

## Quickstart

```python
from graphs import GMRF_Torus
from experiments import run_convergence

run_convergence(GMRF_Torus(n=10, s=1.0, sigma=0.1),
                 label="torus10", sweeps=200, n_chains=10_000,
                 tag="torus10", normalize=False, random_init=True)
```

This draws `n_chains` independent primal and dual chains on a homogeneous
`10x10` torus, computes the unbiased squared-Frobenius error estimator of
Section VII-B, and writes both a `.pdf` figure and a `.csv` of the raw curve
data (for anyone who wants to replot in `pgfplots`/`tikz` instead).

## Reproducing the paper's figures

Every experiment writes its outputs to the current working directory. Run
each of the following independently (they do not share state):

| Figure | Call |
|---|---|
| Fig. 5 — variance-conservation identity (Prop. 2) | `run_woodbury_verification(GMRF_Torus(n=8, s=1.0, sigma=0.3), label="torus8", n_samples=100_000, n_bootstrap=10_000, tag="variance_conservation_torus8", show_labels=False)` |
| Fig. 6 — primal vs. dual convergence | `run_convergence(GMRF_Torus(n=10, s=1.0, sigma=0.1), label="torus10", sweeps=200, n_chains=10_000, tag="torus10", normalize=False, random_init=True)` |
| Fig. 7 — heterogeneous Watts–Strogatz ensemble | `run_convergence_band(ws_hetero_builder(mode="both", n=64, k=4, p=0.3), label="Watts-Strogatz (heterogeneous)", R=64, sweeps=100, n_chains=2000, metric="Eu", band="std", tag="ws_hetero_both", random_init=True, normalize=True)` |
| Fig. 8 — scan-order robustness | `run_scan_comparison(GMRF_Torus(n=10, s=1.0, sigma=0.3), label="torus10", scans=("random","permutation","fixed"), sweeps=100, n_chains=10_000, tag="scan_torus10")` |
| Fig. 9 — effect of degree `k` | `run_param_sweep(lambda k: GMRF_K_Regular(100, k, s=1.0, sigma=0.5), values=(2,4,6,8,16,32), value_label="k", sweeps=100, n_chains=20_000, metric="Eu", tag="kreg_N100", normalize=True)` |
| Fig. 10 — rate vs. `s/sigma` | `run_ratio_sweep(lambda s, sigma: GMRF_K_Regular(5, 4, s=s, sigma=sigma), N=5, sweeps=100, n_chains=5000, metric="mean", show_theory=True, tag="K5")` — writes `ratio_sweep_K5.csv`, rendered in the paper via `pgfplots` (see `paper/` for the `.tex` snippet). |
| Fig. 11 — scalability to a `100x100` torus | `run_large_marginal_torus(100, s=1.0, sigma=0.25, sweeps=50, n_chains=600, measure_every=1, normalize=True, random_init=True, seed=0, tag="torus100")` |

**Note:** `n_chains` in the table above matches what was used for the figures
in the paper and can take a while to run (Fig. 9, for instance, draws 20,000
independent chains per `k` value). Reduce `n_chains` for a quick sanity check
of the qualitative behavior; the underlying theory doesn't change, only the
Monte Carlo noise floor.

<!-- ## Citation

If you use this code, please cite:

```bibtex
@article{khodabandeh2026accelerated,
  title   = {Accelerated Random-Sweep Gibbs Sampling for Gaussian Graphical
             Models via Dual Normal Factor Graphs},
  author  = {Khodabandeh, Borna and Molkaraie, Mehdi},
  journal = {IEEE Transactions on Information Theory},
  year    = {2026},
  note    = {under review}
}
``` -->


## License

Released under the MIT License — see [LICENSE](LICENSE).
