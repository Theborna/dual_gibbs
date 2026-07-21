"""
experiments.py
==============
Reproduces the numerical experiments (Section VII) of:

  B. Khodabandeh and M. Molkaraie, "Accelerated Random-Sweep Gibbs Sampling
  for Gaussian Graphical Models via Dual Normal Factor Graphs."

See the repository README for a figure-by-figure list of which function to
call for each plot in the paper, and for setup instructions.

Convergence experiments for the dual-Gibbs paper (final driver).

What changed vs the old test driver:
  * Frobenius (squared) error instead of nuclear-norm relative error.
  * The M independent chains are measured ACROSS chains AT each sweep t
    (the marginal law at t), not time-averaged within a chain. Time-averaging
    bakes in the non-stationary burn-in and produces a misleading floor.
  * Two estimators of the squared error, both available:
        E_b  = || Sigma_hat - Sigma ||_F^2                       (biased)
        E_u  = tr( (Sigma_A - Sigma)^T (Sigma_B - Sigma) )        (unbiased)
    where A, B are two independent halves of the M chains. E_b has a positive
    floor = the MC variance of Sigma_hat; E_u is unbiased for ||Sigma_t-Sigma||_F^2
    and may go negative once the chain has reached the truth within noise.
  * For the dual sampler, "distance to the truth" is on the Woodbury-
    reconstructed primal covariance  Sigma = D_s - D_s Cov(x~) D_s, so primal
    and dual are compared against the same target (the true primal Sigma).
  * Every figure also writes a .csv of the raw (unnormalized) data for pgfplots.

Experiments (Section VIII of the paper):
  run_woodbury_verification(...)  VIII.A  identity verification (Prop. 2), 4 separate PDFs
  run_convergence(...)             VIII.B  primal-vs-dual on one graph (torus / WS-hetero)
  run_k_sweep(...)                 VIII.C  k-regular, dual rate is universal in k
  run_ratio_sweep(...)             VIII.C  -ln(r) vs s/sigma (dual stable, primal collapses)

Requires: graphs.py, sampler.py  (divergences.py optional).
"""

import csv
import time
import numpy as np
import numba
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator
from scipy.stats import gaussian_kde
from sklearn.linear_model import TheilSenRegressor

from graphs import (
    GMRF_Torus, GMRF_K_Regular, GMRF_CompleteBipartite,
    GMRF_Watts_Strogatz_Hetero,
)

# ----------------------------------------------------------------------
PRIMAL_C, DUAL_C = "#1f77b4", "#d62728"


def _set_style(usetex=False):
    """
    Matches IEEEtran journal typesetting: Times-like serif via STIX (bundled
    with matplotlib, no system-font dependency, text+math self-consistent --
    unlike naming "Times New Roman" directly, which silently falls back to
    DejaVu Serif for text on systems without it installed while math still
    renders in a different font). 10pt to match IEEEtran's default body size.
    pdf.fonttype=42 embeds real TrueType outlines (Type 3 is the matplotlib
    PDF default and some venues' PDF checkers reject it).
    Pass usetex=True only if you have a working local LaTeX and want true
    Computer Modern rendering instead.
    """
    plt.rcParams.update({
        "text.usetex": usetex,
        "font.family": "STIXGeneral" if not usetex else "serif",
        "font.serif": ["Computer Modern Roman"] if usetex else ["STIXGeneral"],
        "mathtext.fontset": "cm" if usetex else "stix",
        "font.size": 10,
        "axes.labelsize": 10,
        "axes.titlesize": 10,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "axes.linewidth": 0.8,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linestyle": ":",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


@numba.njit(cache=False)
def _seed_numba(seed):
    np.random.seed(seed)


def _save_csv(path, header, columns):
    """columns: list of 1D arrays, same length, in header order."""
    rows = np.column_stack(columns)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in rows:
            w.writerow([f"{v:.10g}" for v in r])
    print(f"  saved -> {path}")


def _one_indexed_ticks(ax, n, axis="both"):
    """
    Keep the SAME pixel positions (0-indexed, as imshow requires) but print
    1-indexed labels, so node 0 displays as "1" and node n-1 displays as "n"
    (e.g. 1 -> 64 instead of 0 -> 63).
    """
    positions = [0, n // 2, n - 1]
    labels = [p + 1 for p in positions]
    if axis in ("x", "both"):
        ax.set_xticks(positions)
        ax.set_xticklabels(labels)
    if axis in ("y", "both"):
        ax.set_yticks(positions)
        ax.set_yticklabels(labels)


# ----------------------------------------------------------------------
# VIII.A  Verification of the variance-conservation identity (Proposition 2)
# ----------------------------------------------------------------------
def measure_woodbury_identity(model, n_samples=100_000, thinning=10,
                              burnin=None, n_bootstrap=10_000, seed=0,
                              random_init=True):
    """
    Draws one long dual chain, reconstructs the primal covariance via
    Proposition 2, and bootstraps the normalised Frobenius residual of the
    identity  D_s^{-1/2} Cov(X) D_s^{-1/2} + D_s^{1/2} Cov(X~) D_s^{1/2} = I.

    Returns a dict with the three matrices to plot (Sigma_norm, Dual_norm,
    residual) and the bootstrap distribution `errors_frob`. Nothing here uses
    the M-chains machinery below; this is a single long dual chain, matching
    the identity-verification experiment in the paper (Fig. 5).
    """
    if burnin is None:
        burnin = n_samples // 4
    _seed_numba(seed)
    np.random.seed(seed)

    Sigma_true = np.linalg.inv(model.Q_primal.toarray())
    ds2 = model.s_nodes ** 2
    n_nodes = model.num_nodes

    print(f"  sampling dual chain ({n_samples} kept, {burnin} burn-in, thin={thinning})...")
    y = model.sample_dual(n_samples=n_samples + burnin, thinning=thinning,
                          mean_only=False, random_init=random_init)
    y = y[burnin:]
    x_tilde = y @ model.B.T.toarray()
    n_kept = x_tilde.shape[0]

    Ds_sqrt = np.sqrt(ds2)
    Ds_inv_sqrt = 1.0 / Ds_sqrt

    Cov_full = np.cov(x_tilde, rowvar=False)
    Sigma_norm = Ds_inv_sqrt[:, None] * Sigma_true * Ds_inv_sqrt[None, :]
    Dual_norm = Ds_sqrt[:, None] * Cov_full * Ds_sqrt[None, :]
    residual = (Sigma_norm + Dual_norm) - np.eye(n_nodes)

    print(f"  bootstrapping ({n_bootstrap} replicates)...")
    errors_frob = np.empty(n_bootstrap)
    for b in range(n_bootstrap):
        idx = np.random.choice(n_kept, size=n_kept, replace=True)
        Cov_boot = np.cov(x_tilde[idx], rowvar=False)
        Dual_boot = Ds_sqrt[:, None] * Cov_boot * Ds_sqrt[None, :]
        res_boot = (Sigma_norm + Dual_boot) - np.eye(n_nodes)
        errors_frob[b] = np.linalg.norm(res_boot, "fro") / np.sqrt(n_nodes)

    print(f"  normalised Frobenius residual: {errors_frob.mean():.4f} +/- {errors_frob.std():.4f}")
    return dict(Sigma_norm=Sigma_norm, Dual_norm=Dual_norm, residual=residual,
               errors_frob=errors_frob, n_nodes=n_nodes)


def plot_woodbury_identity(data, tag="woodbury", show_labels=True, usetex=False):
    """
    Produces FOUR SEPARATE PDFs (primal heatmap, dual heatmap, residual
    heatmap, bootstrap error distribution), each its own figure, so they can
    be combined as subfigures in LaTeX rather than as one matplotlib subplot
    grid. Node-index ticks are shown 1-indexed (1..n instead of 0..n-1).

    show_labels=False strips x/y axis labels and titles from all four panels
    (tick numbers and colorbars are kept, since those carry information a
    LaTeX caption/subcaption wouldn't otherwise supply).
    """
    _set_style(usetex=usetex)
    n_nodes = data["n_nodes"]
    norm = mcolors.SymLogNorm(linthresh=0.05, vmin=-1.0, vmax=1.0, base=10)
    cb_ticks = [-1, -0.5, -0.1, 0, 0.1, 0.5, 1]

    # (a) Primal: D_s^{-1/2} Cov(X) D_s^{-1/2}
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    im = ax.imshow(data["Sigma_norm"], cmap="RdBu_r", norm=norm)
    cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.locator = FixedLocator(cb_ticks)
    cb.update_ticks()
    _one_indexed_ticks(ax, n_nodes)
    if show_labels:
        ax.set_xlabel("Node Index")
        ax.set_ylabel("Node Index")
        ax.set_title(r"(a) Primal: $\mathbf{D}_s^{-1/2}\mathrm{Cov}(\mathbf{X})\mathbf{D}_s^{-1/2}$")
    plt.tight_layout()
    plt.savefig(f"{tag}_primal.pdf", bbox_inches="tight")
    print(f"  saved -> {tag}_primal.pdf")
    plt.show()

    # (b) Dual: D_s^{1/2} Cov(X~) D_s^{1/2}
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    im = ax.imshow(data["Dual_norm"], cmap="RdBu_r", norm=norm)
    cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.locator = FixedLocator(cb_ticks)
    cb.update_ticks()
    _one_indexed_ticks(ax, n_nodes)
    if show_labels:
        ax.set_xlabel("Node Index")
        ax.set_ylabel("Node Index")
        ax.set_title(r"(b) Dual: $\mathbf{D}_s^{1/2}\mathrm{Cov}(\tilde{\mathbf{X}})\mathbf{D}_s^{1/2}$")
    plt.tight_layout()
    plt.savefig(f"{tag}_dual.pdf", bbox_inches="tight")
    print(f"  saved -> {tag}_dual.pdf")
    plt.show()

    # (c) Residual: Delta = (Sum - I)   [own linear symmetric scale, not symlog]
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    res = data["residual"]
    lim = np.max(np.abs(res))
    im = ax.imshow(res, cmap="RdBu_r", vmin=-lim, vmax=lim)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    _one_indexed_ticks(ax, n_nodes)
    if show_labels:
        ax.set_xlabel("Node Index")
        ax.set_ylabel("Node Index")
        ax.set_title(r"(c) Residual: $\mathbf{\Delta} = (\mathrm{Sum}-\mathbf{I})$")
    plt.tight_layout()
    plt.savefig(f"{tag}_residual.pdf", bbox_inches="tight")
    print(f"  saved -> {tag}_residual.pdf")
    plt.show()

    # (d) Bootstrap error distribution
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    errs = data["errors_frob"]
    ax.hist(errs, bins=22, density=True, color="#639922", alpha=0.28,
           edgecolor="#3B6D11", linewidth=0.8, zorder=2)
    mu, sd = errs.mean(), errs.std()
    x_lo, x_hi = mu - 4.5 * sd, mu + 4.5 * sd
    x_kde = np.linspace(x_lo, x_hi, 400)
    kde = gaussian_kde(errs, bw_method="scott")
    y_kde = kde(x_kde)
    ax.plot(x_kde, y_kde, color="#3B6D11", lw=1.8, zorder=4)
    mask = (x_kde >= mu - sd) & (x_kde <= mu + sd)
    ax.fill_between(x_kde, y_kde, where=mask, color="#639922", alpha=0.14, zorder=1)
    ax.axvline(mu, color="#A32D2D", lw=1.8, ls="--", zorder=5,
              label=rf"Mean $={mu:.4f}$")
    ax.axvline(mu - sd, color="#A32D2D", lw=1.2, ls=":", zorder=5, alpha=0.75)
    ax.axvline(mu + sd, color="#A32D2D", lw=1.2, ls=":", zorder=5, alpha=0.75,
              label=rf"$\pm 1\,\mathrm{{std}} = {sd:.4f}$")
    if show_labels:
        ax.set_xlabel("Normalised Frobenius error")
        ax.set_ylabel("Density")
        ax.set_title(f"(d) Bootstrap error distribution ({len(errs)} replicates)")
    ax.legend(fontsize=9, framealpha=0.9, edgecolor="#cccccc", loc="upper right")
    ax.grid(True, alpha=0.2, linewidth=0.5)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(f"{tag}_dist.pdf", bbox_inches="tight")
    print(f"  saved -> {tag}_dist.pdf")
    plt.show()


def run_woodbury_verification(model, label, n_samples=100_000, thinning=10,
                              burnin=None, n_bootstrap=10_000, seed=0,
                              random_init=True, tag=None, show_labels=True,
                              usetex=False):
    """
    VIII.A orchestrator: samples, bootstraps, writes a CSV of the bootstrap
    errors, and produces the four separate PDFs (primal/dual/residual/dist).
    """
    tag = tag or f"woodbury_{label}"
    print(f"\n=== Woodbury identity verification: {label} ===")
    data = measure_woodbury_identity(model, n_samples=n_samples, thinning=thinning,
                                     burnin=burnin, n_bootstrap=n_bootstrap,
                                     seed=seed, random_init=random_init)
    _save_csv(f"{tag}_bootstrap_errors.csv", ["error"], [data["errors_frob"]])
    plot_woodbury_identity(data, tag=tag, show_labels=show_labels, usetex=usetex)
    return data


# ----------------------------------------------------------------------
# Core: run a group of chains, accumulate per-checkpoint sufficient stats
# (in primal/node space; dual edge samples are projected x~ = B y~).
# ----------------------------------------------------------------------
def _accumulate(model, mode, n_group, n_check, measure_every, random_init,
                batch_size=250, scan="random"):
    d = model.num_nodes
    S1 = np.zeros((n_check, d))
    S2 = np.zeros((n_check, d, d))
    B = model.B  # sparse |V| x |E|
    t0 = time.time()

    buf = []
    done = 0
    while done < n_group:
        b = min(batch_size, n_group - done)
        buf.clear()
        for _ in range(b):
            if mode == "primal":
                tr = model.sample_primal(n_samples=n_check, thinning=measure_every,
                                         mean_only=False, random_init=random_init, scan=scan)
                X = tr
            else:
                tr = model.sample_dual(n_samples=n_check, thinning=measure_every,
                                       mean_only=False, random_init=random_init, scan=scan)
                X = (B @ tr.T).T          # project to node space
            buf.append(X)
        Xb = np.stack(buf)                 # (b, n_check, d)
        S1 += Xb.sum(axis=0)
        S2 += np.einsum("bci,bcj->cij", Xb, Xb)
        done += b
    dt = time.time() - t0
    return S1, S2, n_group, dt


def _cov(S1, S2, n):
    mean = S1 / n
    cov = (S2 - n * np.einsum("ci,cj->cij", mean, mean)) / (n - 1)
    return mean, cov


def _reconstruct(cov, ds2, mode):
    if mode == "primal":
        return cov
    return np.diag(ds2)[None] - ds2[None, :, None] * cov * ds2[None, None, :]


def measure_curves(model, mode, sweeps, n_chains, measure_every=1,
                   random_init=True, seed=0, scan="random"):
    """
    Returns dict with raw (unnormalized) per-checkpoint arrays:
        sweep   : checkpoint sweep indices
        time    : wall-clock axis (linspace 0..sampling_time)
        Eb      : || Sigma_hat - Sigma ||_F^2            (biased)
        Eu      : tr((Sigma_A-Sigma)^T (Sigma_B-Sigma))  (unbiased)
        mean    : || mean across chains ||_2  (decays at the chain rate r)

    scan: 'random' (default; matches the paper's theory), 'permutation', or
          'fixed' -- see graphs.py:_build_scan_indices. Only 'random' is
          covered by Propositions 1-5; the others are for empirical
          comparison (cf. run_scan_comparison).
    """
    assert n_chains >= 4 and n_chains % 2 == 0, "n_chains must be even and >= 4"
    _seed_numba(seed)
    np.random.seed(seed)

    Sigma = np.linalg.inv(model.Q_primal.toarray())
    ds2 = model.s_nodes ** 2
    n_check = sweeps // measure_every + 1
    half = n_chains // 2

    S1A, S2A, nA, dtA = _accumulate(model, mode, half, n_check, measure_every, random_init, scan=scan)
    S1B, S2B, nB, dtB = _accumulate(model, mode, half, n_check, measure_every, random_init, scan=scan)

    mA, cA = _cov(S1A, S2A, nA)
    mB, cB = _cov(S1B, S2B, nB)
    mC, cC = _cov(S1A + S1B, S2A + S2B, nA + nB)

    SigA = _reconstruct(cA, ds2, mode)
    SigB = _reconstruct(cB, ds2, mode)
    SigC = _reconstruct(cC, ds2, mode)

    Eb = np.empty(n_check)
    Eu = np.empty(n_check)
    mean_norm = np.empty(n_check)
    for c in range(n_check):
        dC = SigC[c] - Sigma
        Eb[c] = np.sum(dC * dC)
        Eu[c] = np.sum((SigA[c] - Sigma) * (SigB[c] - Sigma))
        mean_norm[c] = np.linalg.norm(mC[c])

    sweep_axis = np.arange(n_check) * measure_every
    time_axis = np.linspace(0, dtA + dtB, n_check)
    return dict(sweep=sweep_axis, time=time_axis, Eb=Eb, Eu=Eu, mean=mean_norm)


# ----------------------------------------------------------------------
# VIII.B  Convergence: primal vs dual on a single graph
# ----------------------------------------------------------------------
def _symlog_thresh(*arrays):
    v = np.abs(np.concatenate([np.asarray(a).ravel() for a in arrays]))
    v = v[np.isfinite(v) & (v > 0)]
    return max(1e-12, 0.02 * np.nanmax(v)) if v.size else 1e-12


def _symlog_setup(ax, *arrays, top_pad=1.4, decades_below=None):
    """
    Applies a symlog y-scale with a THIN linear band around zero, then bounds
    the y-limits so the plot is filled rather than showing a big empty gap
    between the last log decade and 0.

    The key parameter is linthresh (the half-width of the linear region around
    0). If it is too large -- e.g. 2% of the max, which for curves normalized
    to start at 1 is ~0.02 -- then the whole [-0.02, 0.02] band is drawn
    linearly and, since most converged points sit at ~0, that band balloons to
    fill much of the axis. We instead put linthresh a couple of decades below
    the largest value, so the near-zero linear band is visually thin and the
    log region (where the actual geometric decay lives) gets the space.

    The lower limit is set symmetrically to -linthresh (just enough to show the
    noise crossing zero) and the upper limit a little above the max.
    """
    flat = np.concatenate([np.asarray(a).ravel() for a in arrays])
    finite = flat[np.isfinite(flat)]
    if not finite.size:
        ax.set_yscale("symlog", linthresh=1e-6)
        return 1e-6
    vmax = max(np.max(np.abs(finite)), 1e-12)
    if decades_below is None:
        decades_below = 3.0
    lt = vmax * 10.0 ** (-decades_below)      # thin linear band, ~3 decades below peak
    ax.set_yscale("symlog", linthresh=lt, linscale=0.5)
    ax.set_ylim(bottom=-lt, top=top_pad * vmax)
    return lt


def run_convergence(model, label, sweeps=200, n_chains=2000, measure_every=1,
                    random_init=True, normalize=False, seed=0, tag=""):
    """
    Single-panel convergence figure: E_u, the unbiased squared-Frobenius
    error, plotted as $\\hat{\\mathcal{E}}$ (the paper no longer reports the
    biased estimator E_b; it is still computed and written to the CSV, but
    not plotted). E_u may go negative near convergence, so the y-axis is
    symlog.
    """
    print(f"\n=== convergence: {label}  sweeps={sweeps} chains={n_chains} ===")
    dp = measure_curves(model, "primal", sweeps, n_chains, measure_every, random_init, seed)
    dd = measure_curves(model, "dual", sweeps, n_chains, measure_every, random_init, seed)
    print(f"  final Eb  primal={dp['Eb'][-1]:.4e}  dual={dd['Eb'][-1]:.4e}")
    print(f"  final Eu  primal={dp['Eu'][-1]:.4e}  dual={dd['Eu'][-1]:.4e}")

    base = f"convergence_{tag or label}"
    _save_csv(base + ".csv",
              ["sweep", "Eb_primal", "Eu_primal", "Eb_dual", "Eu_dual",
               "time_primal", "time_dual"],
              [dp["sweep"], dp["Eb"], dp["Eu"], dd["Eb"], dd["Eu"],
               dp["time"], dd["time"]])

    _set_style()

    def _norm(y):
        if not normalize:
            return y
        ref = y[np.isfinite(y) & (np.abs(y) > 0)]
        return y / ref[0] if ref.size else y

    sp = dp["sweep"]
    eu_p, eu_d = _norm(dp["Eu"]), _norm(dd["Eu"])

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(sp, eu_p, color=PRIMAL_C, ls="-", lw=1.8, label="Primal Gibbs")
    ax.plot(sp, eu_d, color=DUAL_C, ls="-", lw=1.8, label="Dual Gibbs (Ours)")
    _symlog_setup(ax, eu_p, eu_d)
    ax.set_xlabel("Sweeps")
    ax.set_ylabel(r"$\hat{\mathcal{E}}$" + (r" (norm.)" if normalize else ""))
    ax.legend(frameon=True, fontsize=9, loc="upper right")
    plt.tight_layout()
    plt.savefig(base + "_sweeps.pdf", bbox_inches="tight")
    print(f"  saved -> {base}_sweeps.pdf")
    plt.show()

    # ---- wall-clock version ----
    fig2, ax2 = plt.subplots(figsize=(7, 4.4))
    ax2.plot(dp["time"], eu_p, color=PRIMAL_C, ls="-", lw=1.8, label="Primal Gibbs")
    ax2.plot(dd["time"], eu_d, color=DUAL_C, ls="-", lw=1.8, label="Dual Gibbs (Ours)")
    _symlog_setup(ax2, eu_p, eu_d)
    ax2.set_xlabel("Wall-clock time (s)")
    ax2.set_ylabel(r"$\hat{\mathcal{E}}$" + (r" (norm.)" if normalize else ""))
    ax2.legend(frameon=True, fontsize=9)
    plt.tight_layout()
    plt.savefig(base + "_time.pdf", bbox_inches="tight")
    print(f"  saved -> {base}_time.pdf")
    plt.show()

    return dp, dd


# ----------------------------------------------------------------------
# EXPLORATORY: scan-order comparison (random / permutation / fixed)
# ----------------------------------------------------------------------
# NOTE: Propositions 1-5 are proved specifically for the RANDOM-SCAN sampler
# (Roberts & Sahu 1997, Theorem 2; Amit 1996) -- the algebraically-largest-
# eigenvalue rate formula does not apply as-is to 'permutation' or 'fixed'
# scans, which have a different (non-symmetric) one-sweep operator and no
# closed-form rate in this paper (cf. Section IX, Future Work). This
# experiment is purely empirical: it checks whether the dual advantage
# persists qualitatively under scan orders the theory does not cover. Treat
# it as a robustness check, not as evidence for a claim broader than
# random-sweep.
SCAN_COLORS = {"random": "#1f77b4", "permutation": "#2ca02c", "fixed": "#9467bd"}
SCAN_LABELS = {"random": "Random scan", "permutation": "Random permutation", "fixed": "Fixed sweep"}


def run_scan_comparison(model, label, scans=("random", "permutation", "fixed"),
                        sweeps=200, n_chains=2000, measure_every=1, metric="Eu",
                        normalize=True, random_init=True, seed=0, tag=None):
    """
    Compares primal-vs-dual convergence (colored by scan order, solid=primal/
    dashed=dual, matching the convention used throughout this file) across
    random, random-permutation, and fixed-order sweeps.
    """
    tag = tag or f"scan_{label}"
    print(f"\n=== scan comparison: {label}  scans={scans} ===")
    results = {}
    csv_head, csv_cols = ["sweep"], None
    for sc in scans:
        dp = measure_curves(model, "primal", sweeps, n_chains, measure_every,
                            random_init=random_init, seed=seed, scan=sc)
        dd = measure_curves(model, "dual", sweeps, n_chains, measure_every,
                            random_init=random_init, seed=seed, scan=sc)
        results[sc] = (dp, dd)
        if csv_cols is None:
            csv_cols = [dp["sweep"]]
        csv_head += [f"{metric}_primal_{sc}", f"{metric}_dual_{sc}"]
        csv_cols += [dp[metric], dd[metric]]
        print(f"  {sc}: final {metric} primal={dp[metric][-1]:.3e} dual={dd[metric][-1]:.3e}")
    _save_csv(f"{tag}.csv", csv_head, csv_cols)

    _set_style()
    fig, ax = plt.subplots(figsize=(7, 4.5))
    allv = []
    for sc in scans:
        dp, dd = results[sc]
        yp, yd = dp[metric].copy(), dd[metric].copy()
        if normalize:
            ref_p = yp[np.isfinite(yp) & (np.abs(yp) > 0)]
            ref_d = yd[np.isfinite(yd) & (np.abs(yd) > 0)]
            yp = yp / ref_p[0] if ref_p.size else yp
            yd = yd / ref_d[0] if ref_d.size else yd
        col = SCAN_COLORS.get(sc, "black")
        ax.plot(dp["sweep"], yp, color=col, ls="-", lw=1.8, label=SCAN_LABELS.get(sc, sc))
        ax.plot(dd["sweep"], yd, color=col, ls="--", lw=1.8)
        allv.extend([yp, yd])
    _symlog_setup(ax, *allv)
    ax.set_xlabel("Sweeps")
    ax.set_ylabel(r"$\hat{\mathcal{E}}$ (norm.)" if metric == "Eu" else "Error (norm.)")
    ax.set_title(f"Effect of scan order on convergence ({label})")
    leg1 = ax.legend(frameon=True, fontsize=9, loc="upper right", title="Scan order")
    handles = [Line2D([0], [0], color="k", ls="-", lw=1.5, label="Primal Gibbs"),
               Line2D([0], [0], color="k", ls="--", lw=1.5, label="Dual Gibbs (Ours)")]
    ax.add_artist(leg1)
    ax.legend(handles=handles, frameon=True, fontsize=9, loc="right")
    plt.tight_layout()
    plt.savefig(f"{tag}.pdf", bbox_inches="tight")
    print(f"  saved -> {tag}.pdf")
    plt.show()
    return results


# ----------------------------------------------------------------------
# VIII.C  connectivity / parameter sweep on a graph family
# ----------------------------------------------------------------------
def run_param_sweep(graph_fn, values, value_label="k", sweeps=200, n_chains=1500,
                    measure_every=1, metric="Eu", normalize=True, random_init=False,
                    seed=0, tag="sweep", title=None, zoom_sweeps=None,
                    inset_bbox=(0.64, 0.62, 0.33, 0.30)):
    """
    graph_fn(val) -> model.  Plots primal (solid) and dual (dashed) on ONE
    shared symlog axis, coloured by val.

    random_init defaults to False here (unlike most other functions in this
    file): the conditional-std random init starts noticeably closer to the
    target than a deterministic start, which compresses the dual's already-
    fast collapse into essentially one point -- too fast to show any
    k-dependent SHAPE even when the x-axis is zoomed. The deterministic
    start gives every chain the same real distance to travel, so the
    decay trajectory (and its k-ordering) is actually resolvable.

    Connectivity story: the primal rate worsens with connectivity, the dual
    EFFECTIVE rate improves with it (larger k -> larger Fiedler value
    lambda_2(L); Section VI.A, Table I). Both curves are normalized to start
    at 1, but the dual collapses to ~0 within the first several sweeps while
    the primal takes the full horizon -- so the dual's k-ordering only lives
    in that narrow EARLY-sweep window, invisible on the full x-range. An
    inset therefore zooms the X-AXIS (not y) onto the first `zoom_sweeps`
    sweeps, showing the dual curves only (primal is still ~flat there and
    would just add clutter), with a matplotlib zoom-indicator box connecting
    it back to the main plot.
    """
    print(f"\n=== {value_label}-sweep: {list(values)}  metric={metric}  tag={tag} ===")
    results, csv_cols, csv_head = [], [], []
    for val in values:
        m = graph_fn(val)
        dp = measure_curves(m, "primal", sweeps, n_chains, measure_every, seed=seed, random_init=random_init)
        dd = measure_curves(m, "dual", sweeps, n_chains, measure_every, seed=seed, random_init=random_init)
        results.append((val, dp, dd))
        if not csv_head:
            csv_head.append("sweep"); csv_cols.append(dp["sweep"])
        csv_head += [f"{metric}_primal_{value_label}{val}", f"{metric}_dual_{value_label}{val}"]
        csv_cols += [dp[metric], dd[metric]]
        print(f"  {value_label}={val}: final {metric} primal={dp[metric][-1]:.3e} "
              f"dual={dd[metric][-1]:.3e}")
    _save_csv(f"param_sweep_{tag}.csv", csv_head, csv_cols)

    if zoom_sweeps is None:
        zoom_sweeps = max(measure_every * 6, sweeps // 15)

    _set_style()
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(results)))

    def _norm1(y):
        if not normalize:
            return y
        ref = y[np.isfinite(y) & (np.abs(y) > 0)]
        return y / ref[0] if ref.size else y

    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    allv = []
    for (val, dp, dd), col in zip(results, colors):
        yp, yd = _norm1(dp[metric].copy()), _norm1(dd[metric].copy())
        sp = dp["sweep"]
        ax.plot(sp, yp, color=col, ls="-", lw=1.6, label=rf"${value_label}={val}$", zorder=3)
        ax.plot(sp, yd, color=col, ls="--", lw=1.6, zorder=3)
        allv.extend([yp, yd])
    _symlog_setup(ax, *allv)
    ax.set_xlabel("Sweeps")
    ax.set_ylabel(r"$\hat{\mathcal{E}}$" + (r" (norm.)" if normalize else ""))
    # Three non-colliding corners inside the axes: k-legend upper-left,
    # style-legend lower-left, inset top-right (added below). Keeping all
    # legends INSIDE the axes lets tight_layout handle them reliably --
    # an outside-axes legend gets clipped by tight_layout + bbox_inches.
    leg1 = ax.legend(frameon=True, fontsize=8, title=rf"${value_label}$",
                     loc="upper left", ncol=1, framealpha=0.9)
    handles = [Line2D([0], [0], color="k", ls="-", lw=1.5, label="Primal Gibbs"),
               Line2D([0], [0], color="k", ls="--", lw=1.5, label="Dual Gibbs (Ours)")]
    ax.add_artist(leg1)
    ax.legend(handles=handles, frameon=True, fontsize=8, loc="lower left", framealpha=0.9)

    # ---- inset: X-AXIS zoom onto the early sweeps, DUAL ONLY ----
    axins = ax.inset_axes(inset_bbox)
    allv_ins = []
    for (val, dp, dd), col in zip(results, colors):
        yd = _norm1(dd[metric].copy())
        sp = dd["sweep"]
        mask = sp <= zoom_sweeps
        axins.plot(sp[mask], yd[mask], color=col, ls="--", lw=1.6)
        allv_ins.append(yd[mask])
    axins.set_xlim(0, zoom_sweeps)
    _symlog_setup(axins, *allv_ins)
    axins.set_title("Dual, early sweeps", fontsize=8)
    axins.tick_params(labelsize=7)
    axins.grid(True, alpha=0.25, linewidth=0.4)
    ax.indicate_inset_zoom(axins, edgecolor="0.4")

    plt.tight_layout()
    plt.savefig(f"param_sweep_{tag}.pdf", bbox_inches="tight")
    print(f"  saved -> param_sweep_{tag}.pdf")
    plt.show()
    return results


# ----------------------------------------------------------------------
# VIII.C  ratio sweep: empirical decay rate vs s/sigma
# ----------------------------------------------------------------------
# def _decay_rate(curve, sweep, decades=2.0, floor_mult=3.0):
#     """
#     Positive decay rate -slope(log curve), fitted on the clean geometric window:
#     from t=0 down to the first point that is either `decades` below the initial
#     value or within `floor_mult` x the noise floor (estimated from the tail).
#     This adapts the window per configuration so the flat MC floor never enters
#     the slope fit -- essential for the s/sigma sweep where the rate spans orders
#     of magnitude. Returns nan if there is no clean descending segment.
#     """
#     y = np.asarray(curve, float)
#     x = np.asarray(sweep, float)
#     pos = y > 0
#     if pos.sum() < 4:
#         return np.nan
#     y0 = y[pos][0]
#     floor = np.median(np.abs(y[-max(3, len(y) // 5):]))
#     thresh = max(y0 * 10.0 ** (-decades), floor_mult * floor)
#     # contiguous early segment above thresh
#     end = np.argmax(~(y > thresh)) if np.any(~(y > thresh)) else len(y)
#     end = max(4, end)
#     xs, ys = x[:end], y[:end]
#     g = ys > 0
#     if g.sum() < 4:
#         return np.nan
#     slope = np.polyfit(xs[g], np.log(ys[g]), 1)[0]
#     return -slope

def _decay_rate_robust(curve, sweep, decades=2.0, floor_mult=3.0):
    y = np.asarray(curve, float)
    x = np.asarray(sweep, float)
    n = len(y)

    # --- noise floor ---
    tail = y[-max(3, n // 5):]
    floor = np.median(np.abs(tail))
    eps = max(floor_mult * floor, 1e-12)   # keep log safe

    # --- first reliable positive value (skip initial noise) ---
    pos = y > 0
    if pos.sum() < 4:
        return np.nan
    y0 = y[pos][0]

    # --- window where decay is still geometric (not floor‑limited) ---
    thresh = max(y0 * 10.0 ** (-decades), floor_mult * floor)
    if np.any(~(y > thresh)):
        end = np.argmax(~(y > thresh))  # first point below threshold
    else:
        end = n
    end = max(4, end)

    # --- prepare regression ---
    x_win = x[:end]
    y_win = y[:end]
    # only points that are clearly above noise
    mask = y_win > 0          # still exclude original negatives here
    if mask.sum() < 4:
        return np.nan

    logy = np.log(y_win[mask] + eps)   # shift only to avoid log(0)
    X = x_win[mask].reshape(-1, 1)

    # --- robust fit (Theil‑Sen) ---
    try:
        reg = TheilSenRegressor(random_state=0).fit(X, logy)
        slope = reg.coef_[0]
    except Exception:
        # fallback to ordinary least squares
        slope = np.polyfit(X.ravel(), logy, 1)[0]

    return -slope

_decay_rate = _decay_rate_robust

def run_ratio_sweep(graph_fn, ratios=None, sweeps=300, n_chains=1500,
                    metric="mean", decades=2.0, seed=42, tag="ratio",
                    title=None, show_theory=False):
    """
    graph_fn(s, sigma) -> model.  Fixes sigma=1, sweeps s = ratio.
    Plots the empirical decay rate -slope(log metric) vs s/sigma for primal/dual.
    metric='mean' (||mean across chains||) recovers the chain rate r most cleanly;
    'Eb'/'Eu' fit the squared-error decay instead (a constant multiple of ln r).

    show_theory overlays:
        primal   -ln r        = 1/(1 + k (s/sigma)^2)
        dual eff -ln r_eff     = (1 + lambda2 (s/sigma)^2) / (1 + 2 (s/sigma)^2)
    where k and the Fiedler value lambda2(L) are read from a representative model.
    The mean metric on the dual tracks x~ = B y~, i.e. the EFFECTIVE rate of
    Section VI.A, so the dual curve validates Table I (-> lambda2/2 as s/sigma->inf).
    """
    if ratios is None:
        ratios = np.geomspace(0.01, 100, 16)
    ratios = np.asarray(ratios, float)
    sigma = 1.0
    rp, rd = [], []
    mult_theory = 2.0
    if metric == "mean":
        mult_theory = 1.0
    print(f"\n=== ratio sweep: {tag}  ratios={list(ratios)}  metric={metric} ===")
    for ratio in ratios:
        m = graph_fn(s=sigma * np.sqrt(ratio), sigma=sigma / np.sqrt(ratio))
        # random_init=False is REQUIRED here: metric='mean' tracks the decay
        # of the cross-chain mean toward 0. With random_init=True, per-chain
        # initial values already average to ~0 across chains at t=0, so there
        # is no real trajectory left to fit -- _decay_rate ends up fitting
        # pure Monte Carlo noise. The deterministic init gives every chain
        # the same nonzero starting point, so the mean decays cleanly.
        dp = measure_curves(m, "primal", sweeps, n_chains, seed=seed, random_init=True)
        dd = measure_curves(m, "dual", sweeps, n_chains, seed=seed, random_init=True)
        sp = _decay_rate(dp[metric], dp["sweep"], decades) / mult_theory
        sd = _decay_rate(dd[metric], dd["sweep"], decades) / mult_theory
        rp.append(sp); rd.append(sd)
        print(f"  s/sigma={ratio:6.2f}  primal rate={sp:.4f}  dual rate={sd:.4f}")
    rp, rd = np.array(rp), np.array(rd)
    _save_csv(f"ratio_sweep_{tag}.csv",
              ["ratio", f"rate_primal", f"rate_dual"], [ratios, rp, rd])

    _set_style()
    fig, ax = plt.subplots(figsize=(7, 4.5))
    if show_theory:
        m0 = graph_fn(s=1.0, sigma=1.0)
        k = getattr(m0, "k", None)
        L = (m0.B @ m0.B.T).toarray()
        lam = np.sort(np.linalg.eigvalsh(L))
        lam2 = lam[1]
        rr = np.logspace(np.log10(ratios.min()), np.log10(ratios.max()), 300)
        if k is not None:
            ax.plot(rr, 1.0 / (1.0 + k * rr**2), color=PRIMAL_C, ls=":", lw=1.2,
                    alpha=0.8, label="Primal theory")
        ax.plot(rr, (1.0 + lam2 * rr**2) / (1.0 + 2.0 * rr**2), color=DUAL_C,
                ls=":", lw=1.2, alpha=0.8,
                label=r"Dual eff. theory ($\lambda_2=%.2f$)" % lam2)
    ax.plot(ratios, rp, color=PRIMAL_C, marker="o", ls="-", lw=1.5, ms=7,
            label="Primal empirical")
    ax.plot(ratios, rd, color=DUAL_C, marker="s", ls="-", lw=1.5, ms=7,
            label="Dual empirical")
    ax.set_xscale("log")
    ax.set_xlabel(r"$s/\sigma$")
    ax.set_ylabel(r"Empirical decay rate $-\,\mathrm{slope}\,\log\mathcal{E}$")
    ax.set_title(title or rf"Convergence rate vs $s/\sigma$ ($|\mathcal{{V}}|={N}$)")
    ax.legend(frameon=True, fontsize=9)
    plt.tight_layout()
    plt.savefig(f"ratio_sweep_{tag}.pdf", bbox_inches="tight")
    print(f"  saved -> ratio_sweep_{tag}.pdf")
    plt.show()
    return ratios, rp, rd


# ----------------------------------------------------------------------
# VIII.B  band over realizations (for the non-homogeneous figure)
# ----------------------------------------------------------------------
def ws_hetero_builder(mode="both", base_seed=0, n=64, k=6, p=0.3,
                      min_s=0.75, max_s=1.5, min_sigma=0.25, max_sigma=0.5):
    """
    Returns build(r) -> GMRF_Watts_Strogatz_Hetero, with the randomness controlled
    so each source can be ablated:
        'both'     : fresh topology AND fresh heterogeneous params each realization
        'params'   : FIXED topology, fresh params  (isolates parameter heterogeneity)
        'topology' : fresh topology, FIXED param draws (isolates graph randomness)
    """
    def build(r):
        if mode == "both":
            gseed, pseed = base_seed + r, base_seed + 1000 + r
        elif mode == "params":
            gseed, pseed = base_seed, base_seed + 1000 + r
        elif mode == "topology":
            gseed, pseed = base_seed + r, base_seed + 7
        else:
            raise ValueError(f"unknown mode {mode}")
        np.random.seed(pseed)          # controls the beta draws for s_nodes / sigma_edges
        return GMRF_Watts_Strogatz_Hetero(
            n=n, k=k, p=p, min_s=min_s, max_s=max_s,
            min_sigma=min_sigma, max_sigma=max_sigma, seed=gseed)
    return build


def run_convergence_band(build_fn, label, R=30, sweeps=200, n_chains=1000,
                         metric="Eu", band="iqr", normalize=False, seed=0, tag="band", random_init=True):
    """
    Convergence over R random realizations of a graph family. Each realization's
    curve is normalized by its own initial value (different realizations have
    different Sigma), then we plot the MEDIAN curve with a shaded band
    (band='iqr' -> 25-75 percentile, band='std' -> mean +/- 1 std), for primal
    vs dual. Answers the "is this a single cherry-picked graph?" concern.
    """
    print(f"\n=== convergence band: {label}  R={R}  sweeps={sweeps} chains={n_chains} "
          f"metric={metric} ===")
    P, D = [], []
    for r in range(R):
        m = build_fn(r)
        dp = measure_curves(m, "primal", sweeps, n_chains, seed=seed + r, random_init=random_init)
        dd = measure_curves(m, "dual", sweeps, n_chains, seed=seed + r, random_init=random_init)
        cp, cd = dp[metric].copy(), dd[metric].copy()
        if normalize:
            cp = cp / cp[0]; cd = cd / cd[0]
        P.append(cp); D.append(cd)
        if (r + 1) % 10 == 0:
            print(f"  {r+1}/{R} realizations done")
    sp = dp["sweep"]
    P, D = np.array(P), np.array(D)

    def _stats(M):
        if band == "std":
            mu = M.mean(0); sd = M.std(0); return mu, mu - sd, mu + sd
        med = np.median(M, 0)
        return med, np.percentile(M, 25, 0), np.percentile(M, 75, 0)

    mp, lp, up = _stats(P)
    md, ld, ud = _stats(D)

    _save_csv(f"convergence_band_{tag}.csv",
              ["sweep", "primal_med", "primal_lo", "primal_hi",
               "dual_med", "dual_lo", "dual_hi"],
              [sp, mp, lp, up, md, ld, ud])

    _set_style()
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.fill_between(sp, lp, up, color=PRIMAL_C, alpha=0.18, linewidth=0)
    ax.fill_between(sp, ld, ud, color=DUAL_C, alpha=0.18, linewidth=0)
    ax.plot(sp, mp, color=PRIMAL_C, ls="-", lw=1.8, label="Primal Gibbs")
    ax.plot(sp, md, color=DUAL_C, ls="-", lw=1.8, label="Dual Gibbs (Ours)")
    _symlog_setup(ax, mp, md, lp, ld, up, ud)
    ax.set_xlabel("Sweeps")
    ylab = r"$\hat{\mathcal{E}}$ (norm.)" if (metric == "Eu" and normalize) else \
           (r"$\hat{\mathcal{E}}$" if metric == "Eu" else "Error")
    ax.set_ylabel(ylab)
    ax.legend(frameon=True, fontsize=9)
    plt.tight_layout()
    plt.savefig(f"convergence_band_{tag}.pdf", bbox_inches="tight")
    print(f"  saved -> convergence_band_{tag}.pdf")
    plt.show()
    return sp, P, D


# ======================================================================
# LARGE GRAPHS: marginal-variance convergence (never forms anything |V|^2)
# ======================================================================
# For a large graph the full covariance Sigma is |V| x |V| and cannot be stored,
# so we track the diagonal of Sigma (the marginal variances) instead. The
# estimator keeps only per-node sum and sum-of-squares across the M chains
# (O(|V|) memory), and the ground truth is supplied in closed form, so nothing
# of size |V|^2 is ever materialized.

def true_marginal_variance_torus(m, s, sigma):
    """
    Exact per-node variance (Q^{-1})_{vv} for the homogeneous m x m torus.
    The torus is vertex-transitive, so every node has the same marginal variance
        v* = (1/m^2) sum_{a,b} 1 / (1/s^2 + lambda_{a,b}/sigma^2),
        lambda_{a,b} = (2 - 2 cos(2*pi a/m)) + (2 - 2 cos(2*pi b/m)).
    (Verified to machine precision against diag(Q^{-1}) for small m.)
    """
    a = np.arange(m)
    c = 2.0 - 2.0 * np.cos(2.0 * np.pi * a / m)      # 1D cycle Laplacian spectrum
    lam = c[:, None] + c[None, :]                     # torus = Cartesian product
    q = 1.0 / s ** 2 + lam / sigma ** 2
    return float(np.mean(1.0 / q))


def true_mean_marginal_variance(laplacian_eigs, s, sigma):
    """
    General fallback: average marginal variance (1/|V|) tr(Sigma) from the
    Laplacian eigenvalues, which needs only the spectrum of L (not Q^{-1}).
    For vertex-transitive graphs this equals every node's marginal variance.
    """
    q = 1.0 / s ** 2 + np.asarray(laplacian_eigs, float) / sigma ** 2
    return float(np.mean(1.0 / q))


def _accumulate_diag(model, mode, n_group, n_check, measure_every, random_init,
                     batch_size=1):
    """Per-node sum (S1) and sum-of-squares (SS): O(n_check * |V|) memory only."""
    d = model.num_nodes
    S1 = np.zeros((n_check, d))
    SS = np.zeros((n_check, d))
    B = model.B
    t0 = time.time()
    done = 0
    while done < n_group:
        b = min(batch_size, n_group - done)
        buf = []
        for _ in range(b):
            if mode == "primal":
                tr = model.sample_primal(n_samples=n_check, thinning=measure_every,
                                         mean_only=False, random_init=random_init)
                X = tr
            else:
                tr = model.sample_dual(n_samples=n_check, thinning=measure_every,
                                       mean_only=False, random_init=random_init)
                X = (B @ tr.T).T                      # project to node space
            buf.append(X)
        Xb = np.stack(buf)                            # (b, n_check, d)
        S1 += Xb.sum(axis=0)
        SS += (Xb * Xb).sum(axis=0)
        done += b
    return S1, SS, n_group, time.time() - t0


def _marg_var(S1, SS, n, ds2, mode):
    """Per-node marginal-variance estimate at each checkpoint -> (n_check, |V|)."""
    mean = S1 / n
    var = (SS - n * mean * mean) / (n - 1)            # variance of x (primal) or x~ (dual)
    if mode == "primal":
        return mean, var
    # dual diagonal reconstruction: Sigma_vv = s_v^2 - s_v^4 Var(x~_v)
    return mean, ds2[None, :] - (ds2 ** 2)[None, :] * var


def measure_marginal_curves(model, mode, sweeps, n_chains, true_diag,
                            measure_every=1, random_init=True, seed=0):
    """
    Same return signature as measure_curves, but the error is the squared
    Frobenius error on the DIAGONAL of Sigma (the marginal variances):
        Eb = sum_v (vhat_v - v*_v)^2                       (biased)
        Eu = sum_v (vhat_v^A - v*_v)(vhat_v^B - v*_v)       (unbiased, split halves)
    `true_diag` is the exact marginal-variance vector (length |V|). Nothing of
    size |V|^2 is formed, so this scales to very large graphs.
    """
    assert n_chains >= 4 and n_chains % 2 == 0, "n_chains must be even and >= 4"
    _seed_numba(seed)
    np.random.seed(seed)

    ds2 = model.s_nodes ** 2
    n_check = sweeps // measure_every + 1
    half = n_chains // 2
    td = np.asarray(true_diag, float)

    S1A, SSA, nA, dtA = _accumulate_diag(model, mode, half, n_check, measure_every, random_init)
    S1B, SSB, nB, dtB = _accumulate_diag(model, mode, half, n_check, measure_every, random_init)

    mA, vA = _marg_var(S1A, SSA, nA, ds2, mode)
    mB, vB = _marg_var(S1B, SSB, nB, ds2, mode)
    mC, vC = _marg_var(S1A + S1B, SSA + SSB, nA + nB, ds2, mode)

    dC = vC - td[None, :]
    Eb = np.sum(dC * dC, axis=1)
    Eu = np.sum((vA - td[None, :]) * (vB - td[None, :]), axis=1)
    mean_norm = np.linalg.norm(mC, axis=1)

    return dict(sweep=np.arange(n_check) * measure_every,
                time=np.linspace(0, dtA + dtB, n_check),
                Eb=Eb, Eu=Eu, mean=mean_norm)


def run_large_marginal_torus(m, s=1.0, sigma=0.25, sweeps=200, n_chains=600,
                             measure_every=1, normalize=False, random_init=True,
                             seed=0, tag=None):
    """
    Scalable convergence demonstration on a large m x m torus. Tracks the squared
    error of the estimated marginal variances against the exact closed-form value,
    so it runs for graphs (e.g. 100x100, |V|=10^4) where the full covariance
    cannot be stored. Plots Eu (solid) with faint Eb, and writes a CSV.
    """
    tag = tag or f"torus{m}"
    n = m * m
    v_star = true_marginal_variance_torus(m, s, sigma)
    true_diag = np.full(n, v_star)
    model = GMRF_Torus(n=m, s=s, sigma=sigma)
    print(f"\n=== large torus {m}x{m} (|V|={n})  true marginal variance v*={v_star:.6f} ===")
    dp = measure_marginal_curves(model, "primal", sweeps, n_chains, true_diag,
                                 measure_every, random_init, seed)
    dd = measure_marginal_curves(model, "dual", sweeps, n_chains, true_diag,
                                 measure_every, random_init, seed)
    print(f"  final Eb  primal={dp['Eb'][-1]:.4e}  dual={dd['Eb'][-1]:.4e}")
    print(f"  final Eu  primal={dp['Eu'][-1]:.4e}  dual={dd['Eu'][-1]:.4e}")

    base = f"marginal_convergence_{tag}"
    _save_csv(base + ".csv",
              ["sweep", "Eb_primal", "Eu_primal", "Eb_dual", "Eu_dual",
               "time_primal", "time_dual"],
              [dp["sweep"], dp["Eb"], dp["Eu"], dd["Eb"], dd["Eu"],
               dp["time"], dd["time"]])

    _set_style()

    def _norm(y):
        if not normalize:
            return y
        ref = y[np.isfinite(y) & (np.abs(y) > 0)]
        return y / ref[0] if ref.size else y

    sp = dp["sweep"]
    eu_p, eu_d = _norm(dp["Eu"]), _norm(dd["Eu"])
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(sp, eu_p, color=PRIMAL_C, ls="-", lw=1.8, label="Primal Gibbs")
    ax.plot(sp, eu_d, color=DUAL_C, ls="-", lw=1.8, label="Dual Gibbs (Ours)")
    _symlog_setup(ax, eu_p, eu_d)
    ax.set_xlabel("Sweeps")
    ax.set_ylabel(r"$\hat{\mathcal{E}}$" + (r" (norm.)" if normalize else ""))
    ax.legend(frameon=True, fontsize=9, loc="upper right")
    plt.tight_layout()
    plt.savefig(base + "_sweeps.pdf", bbox_inches="tight")
    print(f"  saved -> {base}_sweeps.pdf")
    plt.show()
    return dp, dd


# ----------------------------------------------------------------------
if __name__ == "__main__":
    # # VIII.A  Woodbury identity verification (4 separate PDFs: primal/dual/residual/dist)
    # run_woodbury_verification(
    #     GMRF_Torus(n=8, s=1.0, sigma=0.3), label="torus8",
    #     n_samples=100_000, n_bootstrap=10_000,
    #     tag="variance_conservation_torus8", show_labels=False)

    # # VIII.B  homogeneous 10x10 torus -- RAW SCALE (normalize=False): this is
    # #         the one figure that shows the real, un-rescaled error.
    # run_convergence(GMRF_Torus(n=10, s=1.0, sigma=0.1),
    #                 label="torus10", sweeps=200, n_chains=10_000,
    #                 tag="torus10", normalize=False, random_init=True)

    # # VIII.B  non-homogeneous, non-regular small-world graph: BAND over
    # #         realizations, normalized (per-curve, from this figure onward).
    # run_convergence_band(
    #     ws_hetero_builder(mode="both", n=64, k=4, p=0.3),
    #     label="Watts-Strogatz (heterogeneous)", R=64, sweeps=100, n_chains=2000,
    #     metric="Eu", band="std", tag="ws_hetero_both", random_init=True,
    #     normalize=True)

    # # VIII.C  rate vs s/sigma on a fully connected K5 -- CSV only needed here;
    # #         the actual figure in the paper is the pgfplots version.
    # run_ratio_sweep(lambda s, sigma: GMRF_K_Regular(5, 4, s=s, sigma=sigma),
    #                 sweeps=100, n_chains=20_000, metric="Eu", show_theory=True,
    #                 tag="K5_Eu", title=r"Convergence rate vs $s/\sigma$ (Fully connected $|\mathcal{V}|=5$)")

    # # Large 100x100 torus: marginal-variance convergence, normalized.
    # run_large_marginal_torus(100, s=1.0, sigma=0.25, sweeps=50, n_chains=600,
    #                          measure_every=1, normalize=True, random_init=True,
    #                          seed=0, tag="torus100")

    # # ---- not part of the current paper draft; left available, not run ----
    # # EXPLORATORY  scan-order robustness check (not covered by Props 1-5)
    # run_scan_comparison(GMRF_Torus(n=10, s=1.0, sigma=0.3),
    #                     label="torus10", scans=("random", "permutation", "fixed"),
    #                     sweeps=100, n_chains=10_000, tag="scan_torus10")

    # VIII.C  k-sweep: primal (left) vs dual (right), independent y-scales so
    #         the dual's opposite-direction ordering (improves with k) is visible
    run_param_sweep(lambda k: GMRF_K_Regular(64, k, s=1.0, sigma=0.25),
                    values=(2, 4, 8, 16, 32), value_label="k",
                    sweeps=100, n_chains=20000, metric="Eu", tag="kreg_N64_rand", normalize=True, random_init=True,
                    title=r"Effect of $k$ on convergence ($k$-regular, $|\mathcal{V}|=100$)")