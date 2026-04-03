from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

import paths

from wdm_transform import TimeSeries


def _emri_signal(times: np.ndarray, dt: float) -> np.ndarray:
    """Synthetic EMRI-like signal: slowly chirping with gradual frequency evolution."""
    T = times[-1] + dt
    u = times / T
    # Slow power-law frequency evolution (orbital inspiral)
    f0, f1 = 0.06, 0.14
    freq = f0 + (f1 - f0) * u ** 3
    phase = 2.0 * np.pi * np.cumsum(freq) * dt
    amplitude = 0.4 * (1.0 + 0.8 * u ** 2)
    return amplitude * np.sin(phase)


def _vgb_signal(times: np.ndarray) -> np.ndarray:
    """Synthetic VGB-like signal: superposition of quasi-monochromatic binaries."""
    binaries = [
        (0.05, 1.00),
        (0.10, 0.75),
        (0.18, 0.60),
        (0.26, 0.45),
        (0.34, 0.35),
    ]
    signal = np.zeros(len(times))
    for freq, amp in binaries:
        signal += amp * np.cos(2.0 * np.pi * freq * times)
    return signal


def _mbhb_signal(times: np.ndarray, dt: float) -> np.ndarray:
    """Synthetic MBHB-like signal: rapidly chirping inspiral-merger-ringdown."""
    T = times[-1] + dt
    t_merge = 0.85 * T
    inspiral = times < t_merge

    # Frequency ramps up rapidly toward merger
    f0, f1 = 0.02, 0.40
    tau = np.clip(1.0 - times / t_merge, 0.0, 1.0)
    freq = np.where(inspiral, f0 + (f1 - f0) * (1.0 - tau ** 0.375), f1)
    phase = 2.0 * np.pi * np.cumsum(freq) * dt

    # Amplitude grows during inspiral, decays during ringdown
    amp_insp = np.where(inspiral, 0.8 * (1.0 - tau) ** 0.5, 0.0)
    amp_ring = np.where(~inspiral, 0.8 * np.exp(-(times - t_merge) / (0.06 * T)), 0.0)
    amplitude = amp_insp + amp_ring

    return amplitude * np.sin(phase)


def main() -> None:
    paths.figures.mkdir(parents=True, exist_ok=True)

    nt = 64
    nf = 64
    dt = 1.0
    n_total = nt * nf
    times = np.arange(n_total) * dt

    signals = [
        ("(a) EMRI", _emri_signal(times, dt), "magma"),
        ("(b) Verification binaries", _vgb_signal(times), "viridis"),
        ("(c) MBHB", _mbhb_signal(times, dt), "inferno"),
    ]

    plt.rcParams.update(
        {
            "font.size": 10.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )

    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.2), constrained_layout=True)

    for ax, (label, data, cmap) in zip(axes, signals):
        series = TimeSeries(data, dt=dt)
        wdm = series.to_wdm(nt=nt)
        grid = np.abs(np.asarray(wdm.coeffs)).T

        image = ax.imshow(
            grid,
            origin="lower",
            aspect="auto",
            cmap=cmap,
            interpolation="nearest",
        )
        ax.set_title(label)
        ax.set_xlabel("time bin $n$")
        ax.set_ylabel("channel $m$")
        ax.set_yticks([0, nf // 2, nf])
        ax.set_yticklabels(["DC", f"{nf // 2}", "Nyq"])
        cbar = fig.colorbar(image, ax=ax, pad=0.02)
        cbar.set_label(r"$|W_{nm}|$")

    fig.savefig(paths.figures / "example_transforms.pdf", bbox_inches="tight")


if __name__ == "__main__":
    main()
