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


def toy_signal(times: np.ndarray, dt: float) -> np.ndarray:
    duration = times[-1] + dt
    u = times / duration

    narrowband = 0.35 * np.sin(2.0 * np.pi * 0.035 * times)
    chirp_envelope = np.exp(-0.5 * ((u - 0.62) / 0.12) ** 2)
    chirp_phase = 2.0 * np.pi * (6.0 * u + 22.0 * u**2) + 0.3
    chirp = chirp_envelope * np.cos(chirp_phase)

    return narrowband + chirp


def main() -> None:
    paths.figures.mkdir(parents=True, exist_ok=True)

    nt = 32
    nf = 32
    dt = 1.0
    n_total = nt * nf

    times = np.arange(n_total) * dt
    data = toy_signal(times, dt)

    series = TimeSeries(data, dt=dt)
    coeffs = series.to_wdm(nt=nt)

    freqs = np.fft.rfftfreq(n_total, d=dt)
    spectrum = np.abs(np.fft.rfft(data))
    grid = np.abs(np.asarray(coeffs.coeffs)).T

    fig = plt.figure(figsize=(8.0, 7.2), constrained_layout=True)
    axes = fig.subplot_mosaic(
        [["time"], ["fft"], ["wdm"]],
        height_ratios=[1.0, 1.0, 1.4],
    )

    ax = axes["time"]
    ax.plot(times, data, color="#1f77b4", linewidth=1.2)
    ax.set_title("Toy signal in the time domain")
    ax.set_xlabel("time sample $k$")
    ax.set_ylabel("$x[k]$")
    ax.set_xlim(times[0], times[-1])

    ax = axes["fft"]
    ax.plot(freqs, spectrum, color="#d95f02", linewidth=1.2)
    ax.set_title("The same signal in the Fourier domain")
    ax.set_xlabel("frequency")
    ax.set_ylabel(r"$|\tilde{x}(f)|$")
    ax.set_xlim(0.0, freqs[-1])

    ax = axes["wdm"]
    image = ax.imshow(
        grid,
        origin="lower",
        aspect="auto",
        cmap="magma",
        interpolation="nearest",
    )
    ax.set_title("Packed WDM coefficient magnitude")
    ax.set_xlabel("time bin $n$")
    ax.set_ylabel("channel $m$")
    ax.set_yticks([0, nf // 2, nf])
    ax.set_yticklabels(["0 (DC)", f"{nf // 2}", f"{nf} (Nyquist)"])
    cbar = fig.colorbar(image, ax=ax, pad=0.02)
    cbar.set_label(r"$|w_{nm}|$")

    fig.savefig(paths.figures / "wdm_overview.pdf", bbox_inches="tight")


if __name__ == "__main__":
    main()
