"""Generate a three-panel diagnostic figure for WDM orthogonality and decorrelation.

The figure summarizes three related checks for a small WDM tiling:

1. Left panel: Gram matrix of the interior analysis atoms. These are the
   channels with ``m = 1, ..., nf - 1``; in the packed real-valued layout the
   DC and Nyquist edge channels are stored redundantly, so the direct
   orthonormality check is most transparent on the interior set. For the
   sampled transform used here, the interior atoms are orthonormal to machine
   precision.

2. Middle panel: empirical correlation matrix of interior WDM coefficients for
   white stationary noise. This tests the usual decorrelation claim: for white
   noise, interior coefficients should be close to uncorrelated, so
   off-diagonal correlations remain small.

3. Right panel: empirical correlation matrix of interior WDM coefficients for
   colored stationary noise. This shows that once the input has non-flat power
   across frequency, coefficient correlations become more visible even though the
   transform still provides a localized time-frequency representation.
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


def _coefficient_samples(nt: int, nf: int, dt: float, kind: str, nsamp: int = 4000) -> np.ndarray:
    rng = np.random.default_rng(0)
    n_total = nt * nf
    freqs = np.fft.rfftfreq(n_total, d=dt)
    # A smooth low-pass shape produces clearly colored stationary noise.
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


def main() -> None:
    paths.figures.mkdir(parents=True, exist_ok=True)

    nt = 8
    nf = 8
    dt = 1.0

    keep = _interior_indices(nt, nf)
    analysis = _analysis_matrix(nt, nf, dt)
    interior_gram = analysis[keep, :] @ analysis[keep, :].T
    white_corr = _correlation_matrix(_coefficient_samples(nt, nf, dt, kind="white")[:, keep])
    colored_corr = _correlation_matrix(_coefficient_samples(nt, nf, dt, kind="colored")[:, keep])

    gram_residual = interior_gram - np.eye(interior_gram.shape[0])
    gram_offdiag = gram_residual[~np.eye(gram_residual.shape[0], dtype=bool)]
    white_offdiag = white_corr[~np.eye(white_corr.shape[0], dtype=bool)]
    colored_offdiag = colored_corr[~np.eye(colored_corr.shape[0], dtype=bool)]

    plt.rcParams.update(
        {
            "font.size": 10.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )

    fig, axes = plt.subplots(1, 3, figsize=(12.2, 4.1), constrained_layout=True)

    image = axes[0].imshow(
        interior_gram,
        origin="lower",
        cmap="viridis",
        vmin=0.0,
        vmax=1.0,
        interpolation="nearest",
    )
    axes[0].set_title(r"Interior Gram matrix $\langle g_i, g_j \rangle$")
    axes[0].set_xlabel("interior atom index")
    axes[0].set_ylabel("interior atom index")
    axes[0].text(
        0.98,
        0.02,
        r"$\max |G-I| =$" + f" {np.max(np.abs(gram_residual)):.1e}\n"
        + r"$\max |G_{ij}|_{i\neq j} =$"
        + f" {np.max(np.abs(gram_offdiag)):.1e}",
        transform=axes[0].transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none", "pad": 2.5},
    )
    cbar = fig.colorbar(image, ax=axes[0], fraction=0.046, pad=0.02)
    cbar.set_label("inner product")

    image = axes[1].imshow(
        white_corr,
        origin="lower",
        cmap="coolwarm",
        vmin=-0.45,
        vmax=0.45,
        interpolation="nearest",
    )
    axes[1].set_title("White-noise coefficient correlation")
    axes[1].set_xlabel("interior coefficient index")
    axes[1].set_ylabel("interior coefficient index")
    axes[1].text(
        0.98,
        0.02,
        r"mean $|\rho_{\rm off}|$" + f" = {np.mean(np.abs(white_offdiag)):.3f}",
        transform=axes[1].transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none", "pad": 2.5},
    )
    cbar = fig.colorbar(image, ax=axes[1], fraction=0.046, pad=0.02)
    cbar.set_label("correlation")

    image = axes[2].imshow(
        colored_corr,
        origin="lower",
        cmap="coolwarm",
        vmin=-0.45,
        vmax=0.45,
        interpolation="nearest",
    )
    axes[2].set_title("Colored-noise coefficient correlation")
    axes[2].set_xlabel("interior coefficient index")
    axes[2].set_ylabel("interior coefficient index")
    axes[2].text(
        0.98,
        0.02,
        r"mean $|\rho_{\rm off}|$" + f" = {np.mean(np.abs(colored_offdiag)):.3f}",
        transform=axes[2].transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none", "pad": 2.5},
    )
    cbar = fig.colorbar(image, ax=axes[2], fraction=0.046, pad=0.02)
    cbar.set_label("correlation")

    fig.savefig(paths.figures / "wdm_orthogonality_decorrelation.pdf", bbox_inches="tight")


if __name__ == "__main__":
    main()
