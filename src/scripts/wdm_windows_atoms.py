from __future__ import annotations

import sys

import matplotlib.pyplot as plt
import numpy as np

import paths

REPO_ROOT = paths.root.parents[1]
REPO_SRC = REPO_ROOT / "src"
if str(REPO_SRC) not in sys.path:
    sys.path.insert(0, str(REPO_SRC))

from wdm_transform import get_backend
from wdm_transform.windows import gnmf, phi_unit


def _positive_window_response(
    scaled_freq: np.ndarray,
    *,
    m: int,
    nf: int,
    a: float,
    d: float,
) -> np.ndarray:
    backend = get_backend("numpy")
    if m == 0:
        return np.asarray(phi_unit(backend, scaled_freq, a, d))
    if m == nf:
        return np.asarray(
            phi_unit(backend, scaled_freq - m, a, d)
            + phi_unit(backend, scaled_freq + m, a, d)
        )
    return np.asarray(
        (
            phi_unit(backend, scaled_freq - m, a, d)
            + phi_unit(backend, scaled_freq + m, a, d)
        )
        / np.sqrt(2.0)
    )


def main() -> None:
    paths.figures.mkdir(parents=True, exist_ok=True)

    backend = get_backend("numpy")
    a = 1.0 / 3.0
    d = 1.0
    nt = 32
    nf = 32
    dt = 1.0 / 256.0

    plt.rcParams.update(
        {
            "font.size": 10.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
        }
    )

    fig, axes = plt.subplots(1, 3, figsize=(11.8, 3.8), constrained_layout=True)

    scaled = np.linspace(-1.15, 1.15, 1400)
    phi = np.asarray(phi_unit(backend, scaled, a, d))
    ax = axes[0]
    ax.plot(scaled, phi, color="#0f766e", linewidth=2.6)
    ax.fill_between(scaled, 0.0, phi, color="#99f6e4", alpha=0.30)
    for x in (-a, a):
        ax.axvline(x, color="#0f766e", linewidth=1.0, linestyle="--", alpha=0.7)
    for x in (-(1.0 - a), 1.0 - a):
        ax.axvline(x, color="#b45309", linewidth=0.9, linestyle=":", alpha=0.7)
    ax.set_title(r"Base window $\tilde{\phi}$")
    ax.set_xlabel(r"normalized frequency $\omega / \Delta \Omega$")
    ax.set_ylabel("amplitude")
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-0.02, 1.05)
    ax.text(0.0, 0.96, "flat core", ha="center", va="top", fontsize=9)
    ax.text(-0.66, 0.09, "taper", ha="center", va="bottom", fontsize=9, color="#92400e")
    ax.text(0.66, 0.09, "taper", ha="center", va="bottom", fontsize=9, color="#92400e")
    ax.grid(True, alpha=0.15)

    ax = axes[1]
    scaled_pos = np.linspace(0.0, nf, 1800)
    channels = [0, 8, 16, nf]
    colors = ["#dc2626", "#16a34a", "#2563eb", "#7c3aed"]
    for m, color in zip(channels, colors):
        response = _positive_window_response(scaled_pos, m=m, nf=nf, a=a, d=d)
        ax.plot(scaled_pos, response, color=color, linewidth=2.2)
        peak = np.argmax(response)
        x_peak = scaled_pos[peak]
        y_peak = response[peak]
        label = "DC" if m == 0 else ("Nyquist" if m == nf else fr"$m={m}$")
        x_text = x_peak + (0.8 if m == 0 else (-1.0 if m == nf else 0.0))
        ha = "left" if m == 0 else ("right" if m == nf else "center")
        ax.text(x_text, min(y_peak + 0.06, 1.02), label, color=color, ha=ha, va="bottom", fontsize=9)
    ax.set_title("Shifted windows define the channels")
    ax.set_xlabel(r"positive-frequency channel coordinate")
    ax.set_ylabel("window amplitude")
    ax.set_xlim(0.0, nf)
    ax.set_ylim(-0.02, 1.05)
    ax.grid(True, alpha=0.15)

    ax = axes[2]
    n_fixed = 12
    m_fixed = 8
    n_total = nt * nf
    dt_block = nf * dt
    freqs = np.fft.fftfreq(n_total, d=dt)
    atom_freq = np.asarray(gnmf(backend, n_fixed, m_fixed, freqs, dt_block, nf, a, d))
    atom_time = np.fft.ifft(atom_freq)
    times = np.arange(n_total) * dt
    envelope = np.abs(atom_time)
    center = times[np.argmax(envelope)]
    mask = (times >= center - 0.75) & (times <= center + 0.75)
    ax.plot(times[mask], np.real(atom_time)[mask], color="#2563eb", linewidth=1.0)
    ax.plot(times[mask], envelope[mask], color="#111827", linewidth=2.0, alpha=0.95)
    ax.plot(times[mask], -envelope[mask], color="#111827", linewidth=2.0, alpha=0.45)
    ax.axvline(center, color="0.5", linestyle="--", linewidth=0.9, alpha=0.7)
    ax.set_title(fr"Localized atom for $(n={n_fixed},\,m={m_fixed})$")
    ax.set_xlabel("time")
    ax.set_ylabel("amplitude")
    ax.set_xlim(times[mask][0], times[mask][-1])
    ax.text(0.97, 0.93, r"$|g_{n,m}[k]|$", color="#111827", fontsize=9, ha="right", va="top", transform=ax.transAxes)
    ax.text(0.03, 0.08, r"$\Re\,g_{n,m}[k]$", color="#2563eb", fontsize=9, ha="left", va="bottom", transform=ax.transAxes)
    ax.grid(True, alpha=0.15)

    fig.savefig(paths.figures / "wdm_windows_atoms.pdf", bbox_inches="tight")


if __name__ == "__main__":
    main()
