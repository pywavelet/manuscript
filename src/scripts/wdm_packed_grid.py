from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

import paths


def draw_frequency_panel(ax, nf: int, c_edge: str, c_fill: str, c_grid: str, c_outline: str, c_text: str) -> None:
    n_boxes = nf + 1
    ax.set_xlim(-0.6, n_boxes - 0.4)
    ax.set_ylim(-0.9, 0.9)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    y0 = 0.0
    for k in range(n_boxes):
        facecolor = c_edge if k in (0, n_boxes - 1) else c_fill
        edgecolor = c_outline if k in (0, n_boxes - 1) else c_grid
        ax.add_patch(
            Rectangle(
                (k - 0.34, y0 - 0.12),
                0.68,
                0.24,
                facecolor=facecolor,
                edgecolor=edgecolor,
                linewidth=0.9,
            )
        )

    ax.set_title(r"One-sided spectrum $\tilde{x}[k]$", pad=4)
    ax.text(0.0, -0.48, "DC", ha="center", va="top", color=c_edge, fontsize=9)
    ax.text(n_boxes - 1, -0.48, "Nyquist", ha="center", va="top", color=c_edge, fontsize=9)
    ax.text((n_boxes - 1) / 2, -0.72, r"$k=0,\ldots,N/2$", ha="center", va="top", color=c_text, fontsize=9.2)
    ax.text(2.1, y0, r"$\cdots$", ha="center", va="center", color=c_outline, fontsize=11)
    ax.text(n_boxes - 3.1, y0, r"$\cdots$", ha="center", va="center", color=c_outline, fontsize=11)
    ax.annotate(
        "",
        xy=(3.0, 0.34),
        xytext=(4.0, 0.34),
        arrowprops={"arrowstyle": "<->", "lw": 1.0, "color": c_outline},
    )
    ax.text(3.5, 0.42, r"$df$", ha="center", va="bottom", color=c_text, fontsize=9.2)


def draw_time_panel(ax, nt: int, c_fill: str, c_grid: str, c_outline: str, c_text: str) -> None:
    t = np.linspace(0.0, nt, 800)
    y = 0.32 * np.sin(2 * np.pi * 0.38 * t) + 0.14 * np.sin(2 * np.pi * 1.05 * t + 0.6)

    ax.set_xlim(0.0, nt)
    ax.set_ylim(-0.75, 0.75)
    ax.set_yticks([])
    ax.set_xticks([0.0, nt])
    ax.set_xticklabels(["0", r"$T_{\rm obs}$"])
    ax.tick_params(length=0)
    ax.spines["left"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_color("0.5")

    for n in range(nt):
        ax.add_patch(
            Rectangle(
                (n, -0.75),
                1.0,
                1.5,
                facecolor=c_fill,
                edgecolor="none",
                alpha=0.55 if n % 2 == 0 else 0.28,
                zorder=0,
            )
        )
        ax.axvline(n, color=c_grid, lw=0.75, zorder=1)

    ax.plot(t, y, color=c_outline, lw=1.25, zorder=3)
    ax.set_title(r"Time series $x(t)$", pad=4)

    n0 = max(1, nt // 2 - 1)
    ax.annotate(
        "",
        xy=(n0, -0.55),
        xytext=(n0 + 1.0, -0.55),
        arrowprops={"arrowstyle": "<->", "lw": 1.0, "color": c_outline},
    )
    ax.text(n0 + 0.5, -0.63, r"$dt$", ha="center", va="top", color=c_text)


def draw_w_grid(ax, nt: int, nf: int, c_edge: str, c_interior: str, c_grid: str, c_outline: str, c_text: str) -> None:
    ax.set_xlim(0.0, nt + 1.2)
    ax.set_ylim(0.0, nf + 1.0)

    for n in range(nt):
        for m in range(nf + 1):
            if m in (0, nf):
                facecolor = c_edge
                alpha = 0.18
            else:
                facecolor = c_interior
                alpha = 0.10

            ax.add_patch(
                Rectangle(
                    (n, m),
                    1.0,
                    1.0,
                    facecolor=facecolor,
                    edgecolor=c_grid,
                    linewidth=0.5,
                    alpha=alpha,
                )
            )

    ax.add_patch(
        Rectangle(
            (0.0, 0.0),
            nt,
            nf + 1,
            fill=False,
            edgecolor=c_outline,
            linewidth=1.2,
        )
    )

    ax.set_xlabel(r"time bin $n$")
    ax.set_ylabel(r"channel $m$")
    ax.set_xticks([0.5, nt - 0.5])
    ax.set_xticklabels(["0", r"$N_t-1$"])
    ax.set_yticks([0.5, 1.5, nf - 0.5, nf + 0.5])
    ax.set_yticklabels(["0", "1", r"$N_f-1$", r"$N_f$"])
    ax.tick_params(length=0)
    ax.set_title(r"Packed coefficients $W[n,m]$", pad=6)

    n0 = int(0.62 * nt)
    m0 = int(0.55 * nf)
    ax.add_patch(
        Rectangle(
            (n0, m0),
            1.0,
            1.0,
            facecolor=c_interior,
            edgecolor=c_outline,
            linewidth=1.4,
            alpha=0.84,
            zorder=5,
        )
    )

    ax.annotate(
        "",
        xy=(n0, m0 - 0.46),
        xytext=(n0 + 1.0, m0 - 0.46),
        arrowprops={"arrowstyle": "<->", "lw": 1.0, "color": c_outline},
    )
    ax.text(n0 + 0.5, m0 - 0.7, r"$\Delta T$", ha="center", va="top", color=c_text)

    ax.annotate(
        "",
        xy=(n0 - 0.46, m0),
        xytext=(n0 - 0.46, m0 + 1.0),
        arrowprops={"arrowstyle": "<->", "lw": 1.0, "color": c_outline},
    )
    ax.text(n0 - 0.74, m0 + 0.5, r"$\Delta F$", ha="center", va="center", rotation=90, color=c_text)

    ax.text(nt + 0.28, nf + 0.5, "Nyquist", ha="left", va="center", color=c_text, fontsize=10)
    ax.plot([nt, nt + 0.24], [nf + 0.5, nf + 0.5], color=c_outline, lw=1.0, clip_on=False)
    ax.text(nt + 0.28, 0.5, "DC", ha="left", va="center", color=c_text, fontsize=10)
    ax.plot([nt, nt + 0.24], [0.5, 0.5], color=c_outline, lw=1.0, clip_on=False)
    ax.text(nt + 0.28, nf * 0.50 + 0.25, r"$N=N_tN_f$", ha="left", va="center", color=c_text, fontsize=10.5)
    ax.text(
        nt + 0.28,
        nf * 0.50 - 0.45,
        r"$\Delta T\,\Delta F=\frac{1}{2}$",
        ha="left",
        va="center",
        color=c_text,
        fontsize=10.5,
    )

    for spine in ax.spines.values():
        spine.set_visible(False)


def draw_c_grid(ax, nt: int, nf: int, c_real: str, c_imag: str, c_grid: str, c_outline: str, c_text: str) -> None:
    ax.set_xlim(0.0, nt)
    ax.set_ylim(0.0, nf + 1.0)
    ax.set_xticks([0.5, nt - 0.5])
    ax.set_xticklabels(["0", r"$N_t-1$"])
    ax.set_yticks([0.5, nf + 0.5])
    ax.set_yticklabels(["0", r"$N_f$"])
    ax.tick_params(length=0)
    ax.set_title(r"Phase factor $C_{nm}$", pad=6)

    for n in range(nt):
        for m in range(nf + 1):
            even = (n + m) % 2 == 0
            facecolor = c_real if even else c_imag
            ax.add_patch(
                Rectangle(
                    (n, m),
                    1.0,
                    1.0,
                    facecolor=facecolor,
                    edgecolor=c_grid,
                    linewidth=0.5,
                    alpha=0.95,
                )
            )

    ax.add_patch(
        Rectangle(
            (0.0, 0.0),
            nt,
            nf + 1,
            fill=False,
            edgecolor=c_outline,
            linewidth=1.1,
        )
    )

    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.text(0.0, -0.85, r"$C_{nm}=1$ for even $n+m$", ha="left", va="top", color=c_text, fontsize=9.2)
    ax.text(0.0, -1.35, r"$C_{nm}=i$ for odd $n+m$", ha="left", va="top", color=c_text, fontsize=9.2)

    legend_y = nf + 0.55
    ax.add_patch(Rectangle((0.10, legend_y), 0.42, 0.24, facecolor=c_real, edgecolor=c_grid, linewidth=0.5))
    ax.text(0.62, legend_y + 0.12, r"$1$", ha="left", va="center", color=c_text, fontsize=9.5)
    ax.add_patch(Rectangle((1.25, legend_y), 0.42, 0.24, facecolor=c_imag, edgecolor=c_grid, linewidth=0.5))
    ax.text(1.78, legend_y + 0.12, r"$i$", ha="left", va="center", color=c_text, fontsize=9.5)


def main() -> None:
    paths.figures.mkdir(parents=True, exist_ok=True)

    nt = 10
    nf = 8

    c_edge = "#A85555"
    c_interior = "#5F7FB8"
    c_grid = "#D7D9DE"
    c_outline = "#3A3A3A"
    c_fill = "#F7F8FA"
    c_text = "#222222"
    c_real = "#EEF1F7"
    c_imag = "#D7E0F0"

    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "mathtext.fontset": "dejavuserif",
            "font.family": "DejaVu Sans",
        }
    )

    fig = plt.figure(figsize=(13.2, 5.2), constrained_layout=True)
    grid = fig.add_gridspec(
        2,
        3,
        width_ratios=[1.45, 2.7, 1.05],
        height_ratios=[1.0, 1.05],
    )

    ax_freq = fig.add_subplot(grid[0, 0])
    draw_frequency_panel(ax_freq, nf, c_edge, c_fill, c_grid, c_outline, c_text)

    ax_time = fig.add_subplot(grid[1, 0])
    draw_time_panel(ax_time, nt, c_fill, c_grid, c_outline, c_text)

    ax_w = fig.add_subplot(grid[:, 1])
    draw_w_grid(ax_w, nt, nf, c_edge, c_interior, c_grid, c_outline, c_text)

    ax_c = fig.add_subplot(grid[:, 2])
    draw_c_grid(ax_c, nt, nf, c_real, c_imag, c_grid, c_outline, c_text)

    fig.savefig(paths.figures / "wdm_packed_grid.pdf", bbox_inches="tight")


if __name__ == "__main__":
    main()
