"""Generate a 2x2 diagnostic figure for WDM orthogonality and decorrelation.

The figure summarises four related checks for a small WDM tiling with
``N = N_t * N_f = 64`` (``N_t = N_f = 8``):

1. Top-left -- ``G G^\\dagger`` indexed by the flattened pair ``(n, m)``.
   Equals ``sum_l g_{nm}[l] g_{pq}^*[l]``. For the interior channels
   ``m in {1, ..., N_f-1}`` this is the identity; for the DC and Nyquist
   edge channels ``m in {0, N_f}`` the diagonal is exactly ``1/2`` and
   there are additional off-diagonals at ``p = n +/- N_t / 2`` of magnitude
   ``1/2``, reflecting the packed real-valued storage of the edge bins.

2. Top-right -- ``G^\\dagger G`` indexed by the time/frequency-sample
   indices ``(l, l')``. Equals ``sum_{n,m} g_{nm}[l] g_{nm}^*[l']`` and is
   the ``N x N`` identity to machine precision, encoding perfect
   reconstruction of the forward-then-inverse round trip.

3. Bottom-left -- empirical correlation matrix of interior WDM coefficients
   for white stationary noise; off-diagonals are small.

4. Bottom-right -- empirical correlation matrix of interior WDM
   coefficients for colored stationary noise; off-diagonal structure
   becomes visible.
"""

from __future__ import annotations

import sys

import matplotlib.pyplot as plt
import numpy as np

import paths

REPO_ROOT = paths.root.parents[1]
REPO_SRC = REPO_ROOT / "src"
if str(REPO_SRC) not in sys.path:
    sys.path.insert(0, str(REPO_SRC))

from wdm_transform import TimeSeries


def _analysis_matrix(nt: int, nf: int, dt: float) -> np.ndarray:
    """Return ``A`` with rows indexed by ``alpha = n (N_f + 1) + m`` and columns
    indexed by the time-sample index ``l``. ``A[alpha, l]`` is the WDM
    coefficient produced by an impulse at sample ``l``; up to the FFT
    convention this is the discrete atom ``g_{nm}[l]``.
    """
    n_total = nt * nf
    columns = []
    for sample_index in range(n_total):
        impulse = np.zeros(n_total)
        impulse[sample_index] = 1.0
        coeffs = TimeSeries(impulse, dt=dt).to_wdm(nt=nt)
        columns.append(np.asarray(coeffs.coeffs).reshape(-1))
    return np.stack(columns, axis=1)


def _interior_indices(nt: int, nf: int) -> np.ndarray:
    indices = []
    for n in range(nt):
        for m in range(1, nf):
            indices.append(n * (nf + 1) + m)
    return np.asarray(indices, dtype=int)


def _coefficient_samples(
    nt: int, nf: int, dt: float, kind: str, nsamp: int = 4000
) -> np.ndarray:
    rng = np.random.default_rng(0)
    n_total = nt * nf
    freqs = np.fft.rfftfreq(n_total, d=dt)
    color_filter = 1.0 / np.sqrt(1.0 + (freqs / 0.06) ** 4)

    rows = []
    for _ in range(nsamp):
        samples = rng.normal(size=n_total)
        if kind == "colored":
            spectrum = np.fft.rfft(samples)
            samples = np.fft.irfft(spectrum * color_filter, n=n_total)
        coeffs = TimeSeries(samples, dt=dt).to_wdm(nt=nt)
        rows.append(np.asarray(coeffs.coeffs).reshape(-1))
    return np.stack(rows, axis=0)


def _correlation_matrix(samples: np.ndarray) -> np.ndarray:
    covariance = np.cov(samples, rowvar=False)
    scales = np.sqrt(np.diag(covariance))
    correlation = covariance / np.outer(scales, scales)
    return np.nan_to_num(correlation, nan=0.0, posinf=0.0, neginf=0.0)


def _edge_tick_positions(nt: int, nf: int) -> list[int]:
    """Flattened indices that start each ``n`` block, for guide lines."""
    return [n * (nf + 1) for n in range(nt + 1)]


def main() -> None:
    paths.figures.mkdir(parents=True, exist_ok=True)

    nt = 8
    nf = 8
    dt = 1.0
    n_total = nt * nf

    analysis = _analysis_matrix(nt, nf, dt)
    # NumPy's FFT convention leaves the raw analysis rows with norm sqrt(N).
    # Divide by N to display the Gram matrices for unit-norm discrete atoms.
    gg_dagger = (analysis @ analysis.T) / n_total
    g_dagger_g = (analysis.T @ analysis) / n_total

    white_corr = _correlation_matrix(
        _coefficient_samples(nt, nf, dt, kind="white")
    )
    colored_corr = _correlation_matrix(
        _coefficient_samples(nt, nf, dt, kind="colored")
    )

    gdg_residual = g_dagger_g - np.eye(n_total)
    # Edge channels are linearly dependent on interior channels, so their
    # empirical correlations include exact +/-1 entries that would dominate
    # the "off-diagonal" summary. Restrict the summary to interior pairs.
    interior = _interior_indices(nt, nf)
    white_interior = white_corr[np.ix_(interior, interior)]
    colored_interior = colored_corr[np.ix_(interior, interior)]
    white_offdiag = white_interior[~np.eye(interior.size, dtype=bool)]
    colored_offdiag = colored_interior[~np.eye(interior.size, dtype=bool)]

    plt.rcParams.update(
        {
            "font.size": 10.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 9.5), constrained_layout=True)
    panel_labels = ["(a)", "(b)", "(c)", "(d)"]
    for ax, label in zip(axes.flat, panel_labels):
        ax.set_box_aspect(1)
        ax.text(
            -0.18,
            1.02,
            label,
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=12,
            fontweight="bold",
        )

    # Top-left: G G^dagger (full packed array, exposes edge-channel structure).
    ax = axes[0, 0]
    image = ax.imshow(
        gg_dagger,
        origin="lower",
        cmap="viridis",
        vmin=0.0,
        vmax=1.0,
        interpolation="nearest",
        aspect="auto",
    )
    ax.set_title(
        r"$(GG^{\dagger})_{(n,m),(p,q)} = \sum_{\ell} g_{nm}[\ell]\, g^{*}_{pq}[\ell]$"
    )
    ax.set_xlabel(r"flattened index $\alpha = n(N_f+1)+m$")
    ax.set_ylabel(r"flattened index $\alpha = n(N_f+1)+m$")
    for tick in _edge_tick_positions(nt, nf):
        ax.axhline(tick - 0.5, color="white", linewidth=0.4, alpha=0.4)
        ax.axvline(tick - 0.5, color="white", linewidth=0.4, alpha=0.4)
    cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.02)
    cbar.set_label("inner product")

    # Top-right: G^dagger G (N x N identity to machine precision).
    ax = axes[0, 1]
    image = ax.imshow(
        g_dagger_g,
        origin="lower",
        cmap="viridis",
        vmin=0.0,
        vmax=1.0,
        interpolation="nearest",
        aspect="auto",
    )
    ax.set_title(
        r"$(G^{\dagger}G)_{\ell \ell'} = \sum_{n,m} g_{nm}[\ell]\, g^{*}_{nm}[\ell']$"
    )
    ax.set_xlabel(r"sample index $\ell'$")
    ax.set_ylabel(r"sample index $\ell$")
    ax.text(
        0.98,
        0.02,
        r"$\max |G^{\dagger}G - I| =$" + f" {np.max(np.abs(gdg_residual)):.1e}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none", "pad": 2.5},
    )
    cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.02)
    cbar.set_label("inner product")

    # Bottom-left: white-noise empirical correlation (interior channels).
    ax = axes[1, 0]
    image = ax.imshow(
        white_corr,
        origin="lower",
        cmap="coolwarm",
        vmin=-0.45,
        vmax=0.45,
        interpolation="nearest",
        aspect="auto",
    )
    ax.set_title("White-noise coefficient correlation")
    ax.set_xlabel(r"flattened index $\alpha = n(N_f+1)+m$")
    ax.set_ylabel(r"flattened index $\alpha = n(N_f+1)+m$")
    for tick in _edge_tick_positions(nt, nf):
        ax.axhline(tick - 0.5, color="black", linewidth=0.3, alpha=0.25)
        ax.axvline(tick - 0.5, color="black", linewidth=0.3, alpha=0.25)
    ax.text(
        0.98,
        0.02,
        r"interior mean $|\rho_{\rm off}| = $"
        + f"{np.mean(np.abs(white_offdiag)):.3f}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none", "pad": 2.5},
    )
    cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.02)
    cbar.set_label("correlation")

    # Bottom-right: colored-noise empirical correlation (interior channels).
    ax = axes[1, 1]
    image = ax.imshow(
        colored_corr,
        origin="lower",
        cmap="coolwarm",
        vmin=-0.45,
        vmax=0.45,
        interpolation="nearest",
        aspect="auto",
    )
    ax.set_title("Colored-noise coefficient correlation")
    ax.set_xlabel(r"flattened index $\alpha = n(N_f+1)+m$")
    ax.set_ylabel(r"flattened index $\alpha = n(N_f+1)+m$")
    for tick in _edge_tick_positions(nt, nf):
        ax.axhline(tick - 0.5, color="black", linewidth=0.3, alpha=0.25)
        ax.axvline(tick - 0.5, color="black", linewidth=0.3, alpha=0.25)
    ax.text(
        0.98,
        0.02,
        r"interior mean $|\rho_{\rm off}| = $"
        + f"{np.mean(np.abs(colored_offdiag)):.3f}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none", "pad": 2.5},
    )
    cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.02)
    cbar.set_label("correlation")

    fig.savefig(
        paths.figures / "wdm_orthogonality_decorrelation.pdf", bbox_inches="tight"
    )


if __name__ == "__main__":
    main()
