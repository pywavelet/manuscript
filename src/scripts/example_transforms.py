from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

import paths

from wdm_transform import TimeSeries


# ---------------------------------------------------------------------------
# Synthetic gravitational-wave-like signals
# ---------------------------------------------------------------------------

def _emri_like(times: np.ndarray, dt: float) -> np.ndarray:
    """Long-duration slowly-chirping EMRI-like waveform."""
    duration = times[-1] + dt
    u = times / duration
    # Slow frequency evolution typical of EMRIs
    phase = 2.0 * np.pi * (0.02 * times + 0.5 * u**2 * duration * 0.04)
    envelope = np.exp(-0.5 * ((u - 0.5) / 0.35) ** 2)
    return envelope * np.cos(phase)


def _vgb_like(times: np.ndarray, dt: float) -> np.ndarray:
    """Quasi-monochromatic verification galactic-binary-like signal."""
    # Two nearly-monochromatic sinusoids with slight frequency difference
    primary_frequency = 0.012
    secondary_frequency = 0.0135
    primary_amplitude, secondary_amplitude = 1.0, 0.45
    return primary_amplitude * np.cos(
        2.0 * np.pi * primary_frequency * times
    ) + secondary_amplitude * np.cos(
        2.0 * np.pi * secondary_frequency * times + 0.8
    )


def _mbhb_like(times: np.ndarray, dt: float) -> np.ndarray:
    """Massive-black-hole-binary-like signal: rapid inspiral and merger."""
    duration = times[-1] + dt
    u = times / duration
    # Frequency sweeps rapidly toward merger at u ~ 0.72
    tau = np.maximum(0.72 - u, 1e-3)
    freq = 0.005 + 0.10 * tau ** (-3.0 / 8.0)
    freq = np.minimum(freq, 0.45 / dt)
    phase = 2.0 * np.pi * np.cumsum(freq) * dt
    amplitude = tau ** (-1.0 / 4.0)
    amplitude /= amplitude.max()
    # Ringdown after merger
    ringdown = np.exp(-np.maximum(u - 0.72, 0.0) / 0.04)
    return amplitude * np.cos(phase) * ringdown


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _wdm_panel(
    ax: plt.Axes,
    signal: np.ndarray,
    dt: float,
    nt: int,
    label: str,
    cmap: str = "magma",
) -> None:
    series = TimeSeries(signal, dt=dt)
    coeffs = series.to_wdm(nt=nt)
    grid = np.abs(np.asarray(coeffs.coeffs)).T  # shape (nf+1, nt)
    ax.imshow(
        grid,
        origin="lower",
        aspect="auto",
        cmap=cmap,
        interpolation="nearest",
    )
    nf = grid.shape[0] - 1
    ax.set_yticks([0, nf // 2, nf])
    ax.set_yticklabels(["DC", f"{nf // 2}", "Ny"])
    ax.set_xlabel("time bin $n$")
    ax.set_ylabel("channel $m$")
    ax.set_title(label)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    paths.figures.mkdir(parents=True, exist_ok=True)

    # Common parameters
    nt = 64
    nf = 64
    dt = 1.0
    n_total = nt * nf

    times = np.arange(n_total) * dt

    signals = [
        (_emri_like(times, dt), "(a) EMRI-like"),
        (_vgb_like(times, dt), "(b) Galactic binary-like"),
        (_mbhb_like(times, dt), "(c) MBHB-like"),
    ]

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(13.0, 3.8),
        constrained_layout=True,
    )

    for ax, (signal, label) in zip(axes, signals):
        _wdm_panel(ax, signal, dt, nt, label)

    fig.savefig(paths.figures / "example_transforms.pdf", bbox_inches="tight")


if __name__ == "__main__":
    main()
