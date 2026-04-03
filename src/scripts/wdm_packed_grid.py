from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np

import paths


def main() -> None:
    paths.figures.mkdir(parents=True, exist_ok=True)

    nt = 8
    nf = 6

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(8.0, 5.8),
        constrained_layout=True,
        gridspec_kw={"height_ratios": [0.8, 1.6]},
    )

    ax = axes[0]
    channel_positions = np.arange(nf + 1)
    ax.hlines(0.0, 0.0, nf, color="0.3", linewidth=2.0)
    ax.vlines(channel_positions, -0.12, 0.12, color="0.3", linewidth=1.2)
    ax.scatter(channel_positions, np.zeros_like(channel_positions), s=55, color="#1f77b4", zorder=3)
    ax.scatter([0, nf], [0, 0], s=70, color="#c44e52", zorder=4)
    ax.text(0, 0.2, "DC\n$m=0$", ha="center", va="bottom", color="#c44e52")
    ax.text(nf, 0.2, "Nyquist\n$m=N_f$", ha="center", va="bottom", color="#c44e52")
    ax.text(nf / 2, -0.26, r"positive-frequency channel index $m = 0, \ldots, N_f$", ha="center")
    ax.set_xlim(-0.4, nf + 0.4)
    ax.set_ylim(-0.45, 0.55)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(r"Stored WDM channels span the index range $m=0,\ldots,N_f$")
    for spine in ax.spines.values():
        spine.set_visible(False)

    ax = axes[1]
    packed = np.ones((nf + 1, nt))
    packed[0, :] = 0
    packed[-1, :] = 2
    cmap = ListedColormap(["#c44e52", "#4c72b0", "#c44e52"])
    ax.imshow(packed, origin="lower", aspect="auto", interpolation="nearest", cmap=cmap)
    ax.set_title(r"Packed coefficient array has shape $N_t \times (N_f + 1)$")
    ax.set_xlabel("time bin $n$")
    ax.set_ylabel("channel $m$")
    ax.set_xticks([0, nt - 1])
    ax.set_xticklabels(["0", r"$N_t-1$"])
    ax.set_yticks([0, 1, nf - 1, nf])
    ax.set_yticklabels(["0", "1", r"$N_f-1$", r"$N_f$"])
    ax.text(nt / 2 - 0.5, 0, "DC edge channel", ha="center", va="center", color="white", fontsize=10)
    ax.text(nt / 2 - 0.5, nf, "Nyquist edge channel", ha="center", va="center", color="white", fontsize=10)
    ax.text(nt / 2 - 0.5, nf / 2, "interior channels", ha="center", va="center", color="white", fontsize=11)

    fig.savefig(paths.figures / "wdm_packed_grid.pdf", bbox_inches="tight")


if __name__ == "__main__":
    main()
