from __future__ import annotations

from pathlib import Path
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

COL = {
    "text": "#222222",
    "text_dim": "#555555",
    "line": "#222222",
    "grid": "#D9DDE4",
    "light": "#F4F6F8",
    "edge": "#E9D6D6",
    "accent": "#777777",
    "accent_light": "#D7E0F0",
    "gray": "#777777",
}


def set_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 12,
            "axes.labelsize": 12,
            "mathtext.fontset": "dejavuserif",
            "font.family": "DejaVu Sans",
            "figure.dpi": 160,
            "savefig.dpi": 300,
        }
    )


def clean_axis(ax) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def save(fig, outpath: Path, transparent: bool = True) -> None:
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, bbox_inches="tight", pad_inches=0.08, transparent=transparent)
    plt.close(fig)


def phi_unit(freqs: np.ndarray, *, center: float = 0.0, a: float = 0.20) -> np.ndarray:
    """Compact Meyer-like schematic window."""
    b = 1.0 - 2.0 * a
    abs_f = np.abs(freqs - center)
    return np.select(
        [abs_f <= a, (abs_f > a) & (abs_f <= a + b)],
        [1.0, np.cos((np.pi / 2.0) * (abs_f - a) / b)],
        default=0.0,
    )


def save_xtilde_panel(
    outpath: Path,
    *,
    figsize: tuple[float, float] = (5.4, 2.2),
    transparent: bool = True,
) -> None:
    fig, ax = plt.subplots(figsize=figsize)
    # Increased top margin to prevent label overlap seen in image_54b741.png
    fig.subplots_adjust(left=0.08, right=0.95, top=0.82, bottom=0.22)
    clean_axis(ax)

    # Indices centered around 0
    # -3: -N*df/2, -2 & -1: inner bins, 0: center, 1: next bin, 3: (N/2-1)*df
    indices = [-3, -2, -1, 0, 1, 3]
    x0, x1 = -3.0, 3.0
    
    ax.set_xlim(x0 - 0.5, x1 + 0.5)
    ax.set_ylim(-0.8, 1.5)

    # Main horizontal axis
    ax.plot([x0, x1], [0, 0], color=COL["line"], lw=2.2, solid_capstyle="butt")

    for k in indices:
        # Tall stems for boundaries and zero frequency
        h = 0.35 if k in (x0, 0, x1) else 0.35
        ax.plot([k, k], [0, h], color=COL["line"], lw=2.2, solid_capstyle="butt")

    # Balanced ellipses for data continuity
    ax.text(-1.5, 0.12, r"$\dots$", ha="center", va="bottom", fontsize=18, color=COL["line"])
    ax.text(2.0, 0.12, r"$\dots$", ha="center", va="bottom", fontsize=18, color=COL["line"])

    # Frequency labels placed higher to avoid stem collision
    lbl_y = 0.45
    ax.text(0, lbl_y, r"$0$", ha="center", va="bottom", fontsize=16, color=COL["text"])
    ax.text(x0, lbl_y, r"$-N\Delta f/2$", ha="center", va="bottom", fontsize=14, color=COL["text"])
    ax.text(x1, lbl_y, r"$\left(\frac{N}{2}-1\right)\Delta f$", ha="center", va="bottom", fontsize=14, color=COL["text"])

    # Delta f indicator centered under the [0, 1] interval
    ax.annotate(
        "",
        xy=(0.08, -0.28),
        xytext=(0.92, -0.28),
        arrowprops=dict(arrowstyle="<->", lw=1.2, color=COL["line"]),
    )
    ax.text(0.5, -0.38, r"$\Delta f$", ha="center", va="top", fontsize=16, color=COL["text"])
    
    # Top-left variable label (Positioned to avoid overlap with -N*df/2)
    ax.text(0.0, 1.05, r"$\tilde{x}[\ell]$", ha="left", va="top", 
            transform=ax.transAxes, fontsize=24, color=COL["text"])

    save(fig, outpath, transparent=transparent)


def save_phi_panel(
    outpath: Path,
    *,
    n: int = 1600,
    a: float = 0.20,
    figsize: tuple[float, float] = (6.2, 2.7),
    transparent: bool = True,
) -> None:
    b = 1.0 - 2.0 * a
    shift = a + b / 2.0
    centers = np.array([0.0, 2.0 * shift, 4.0 * shift])

    f = np.linspace(-0.7, centers[-1] + 0.7, n)
    windows = [phi_unit(f, center=c, a=a) for c in centers]

    fig, ax = plt.subplots(figsize=figsize)
    fig.subplots_adjust(left=0.04, right=0.96, top=0.92, bottom=0.17)
    clean_axis(ax)
    ax.set_xlim(f.min() - 0.02, f.max() + 0.18)
    ax.set_ylim(-0.42, 1.32)

    ax.annotate(
        "",
        xy=(f.max() + 0.08, 0),
        xytext=(f.min(), 0),
        arrowprops=dict(arrowstyle="->", lw=1.2, color=COL["line"]),
    )
    ax.text(f.max() + 0.16, 0, r"$f$", ha="left", va="center", fontsize=15, color=COL["text"])

    ax.plot(f, windows[0], color=COL["gray"], lw=2.0, ls="--", alpha=0.75)
    ax.plot(f, windows[1], color=COL["line"], lw=2.4, alpha=1.0)
    ax.plot(f, windows[2], color=COL["gray"], lw=2.0, ls="--", alpha=0.55)

    labels = [r"$f_{m-1}$", r"$f_m$", r"$f_{m+1}$"]
    phi_labels = [r"$\tilde{\varphi}_{m-1}(f)$", r"$\tilde{\varphi}_{m}(f)$", r"$\tilde{\varphi}_{m+1}(f)$"]
    colors = [COL["gray"], COL["text"], COL["gray"]]
    for c, lab, plab, col in zip(centers, labels, phi_labels, colors):
        ax.plot([c, c], [0, 1.02], color=COL["grid"], lw=1.0, ls=":")
        ax.text(c, -0.11, lab, ha="center", va="top", fontsize=13, color=col)
        ax.text(c, 1.12, plab, ha="center", va="bottom", fontsize=14, color=col)

    ax.annotate(
        "",
        xy=(centers[0], -0.35),
        xytext=(centers[2], -0.35),
        arrowprops=dict(arrowstyle="<->", lw=1.0, linestyle="-", color=COL["accent"]),
    )
    ax.text(centers[1], -0.42, r"shift $\tilde{\varphi}(f)$ along $N_f$ channels", ha="center", va="top", fontsize=11, color=COL["accent"])

    save(fig, outpath, transparent=transparent)


def save_cnm_panel(
    outpath: Path,
    *,
    n_periods: int = 2,
    figsize: tuple[float, float] = (2.45, 2.55),
    transparent: bool = True,
) -> None:

    size = 2 * n_periods
    fig, ax = plt.subplots(figsize=figsize)
    
    # 1. ADD TICKS & LABELS
    # Using standard ticks keeps everything perfectly aligned
    ax.set_xticks(np.arange(size) + 0.5)
    ax.set_yticks(np.arange(size) + 0.5)
    ax.set_xticklabels(range(size), fontsize=10)
    ax.set_yticklabels(range(size), fontsize=10)
    
    # Label the axes with better spacing
    ax.set_xlabel(r"$n$", fontsize=12, labelpad=8)
    ax.set_ylabel(r"$m$", fontsize=12, labelpad=8)
    
    # Remove the tick marks themselves, keep labels
    ax.tick_params(length=0)

    # 2. DRAW THE GRID
    for n in range(size):
        for m in range(size):
            even = (n + m) % 2 == 0
            face = "white" if even else COL["accent_light"]
            ax.add_patch(Rectangle((n, m), 1, 1, facecolor=face, edgecolor=COL["grid"], lw=0.8))
            ax.text(n + 0.5, m + 0.5, r"$1$" if even else r"$i$", 
                    ha="center", va="center", fontsize=15, color=COL["text"])

    # 3. OPEN-ENDED INDICATORS (Dots)
    # Use a lighter color for dots so they don't draw too much focus
    dot_col = COL["grid"]
    dot_y = size + 0.2
    dot_x = size + 0.3
    
    for i in range(size):
        ax.text(i + 0.5, dot_y, r"$\vdots$", ha="center", va="bottom", color=dot_col)
        ax.text(dot_x, i + 0.5, r"$\dots$", ha="left", va="center", color=dot_col)
    
    # Diagonal dot to finish the look
    ax.text(dot_x, dot_y, r"$\ddots$", ha="left", va="bottom", color=dot_col)

    # 4. POLISH
    ax.set_title(r"$C_{nm}$", fontsize=22, pad=20)
    ax.set_aspect("equal")
    
    # Spines make the "L" shape you wanted
    ax.spines[:].set_visible(False)
    ax.spines["left"].set_visible(True)
    ax.spines["bottom"].set_visible(True)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)

    # Tight limits to keep the "dots" close to the grid
    ax.set_xlim(0, size + 0.8)
    ax.set_ylim(0, size + 0.8)
    save(fig, outpath, transparent=transparent)



def save_wnm_panel(
    outpath: Path,
    *,
    nt: int = 10,
    nf: int = 8,
    figsize: tuple[float, float] = (5.7, 5.2),
    transparent: bool = True,
) -> None:
    fig, ax = plt.subplots(figsize=figsize)
    fig.subplots_adjust(left=0.08, right=0.93, top=0.92, bottom=0.13)
    clean_axis(ax)
    ax.set_aspect("equal")
    ax.set_xlim(-1.15, nt + 2.55)
    ax.set_ylim(-1.55, nf + 2.15)

    for n in range(nt):
        for m in range(nf + 1):
            face = COL["edge"] if m in (0, nf) else COL["light"]
            alpha = 0.52 if m in (0, nf) else 0.78
            ax.add_patch(Rectangle((n, m), 1, 1, facecolor=face, edgecolor=COL["grid"], lw=0.55, alpha=alpha))

    ax.add_patch(Rectangle((0, 0), nt, nf + 1, facecolor="none", edgecolor=COL["line"], lw=1.2))

    n0 = int(0.62 * nt)
    m0 = int(0.52 * nf)
    ax.add_patch(Rectangle((n0, m0), 1, 1, facecolor=COL["accent"], edgecolor=COL["line"], lw=1.2, alpha=0.82, zorder=5))

    ax.annotate("", xy=(n0, m0 - 0.34), xytext=(n0 + 1, m0 - 0.34), arrowprops=dict(arrowstyle="<->", lw=1.0, color=COL["line"]), zorder=6)
    ax.text(n0 + 0.5, m0 - 0.56, r"$\Delta T$", ha="center", va="top", fontsize=13, color=COL["text"])

    ax.annotate("", xy=(n0 - 0.34, m0), xytext=(n0 - 0.34, m0 + 1), arrowprops=dict(arrowstyle="<->", lw=1.0, color=COL["line"]), zorder=6)
    ax.text(n0 - 0.57, m0 + 0.5, r"$\Delta F$", ha="right", va="center", fontsize=13, color=COL["text"])

    # ax.plot(n0 + 1.20, m0 + 0.55, marker="o", ms=2.8, color=COL["line"], zorder=7)
    # ax.annotate(r"$W[n,m]$", xy=(n0 + 1.20, m0 + 0.55), xytext=(nt + 0.55, m0 + 0.82), ha="left", va="center", fontsize=16, arrowprops=dict(arrowstyle="-", lw=1.0, color=COL["line"]), color=COL["text"])

    ax.text(0.5, -0.10, r"$0$", ha="center", va="top", fontsize=13, color=COL["text"])
    ax.text(nt - 0.5, -0.10, r"$N_t-1$", ha="center", va="top", fontsize=13, color=COL["text"])
    ax.annotate("", xy=(nt - 0.35, -0.88), xytext=(1.1, -0.88), arrowprops=dict(arrowstyle="->", lw=1.0, linestyle="-", color=COL["gray"]))
    ax.text(nt / 2, -1.08, r"time bins $n$ ($N_t$)", ha="center", va="top", fontsize=12, color=COL["text"])

    ax.text(-0.25, 0.5, r"$0$", ha="right", va="center", fontsize=13, color=COL["text"])
    ax.text(-0.25, nf + 0.5, r"$N_f$", ha="right", va="center", fontsize=13, color=COL["text"])
    ax.annotate("", xy=(-0.72, nf + 0.25), xytext=(-0.72, 1.0), arrowprops=dict(arrowstyle="->", lw=1.0, linestyle="-", color=COL["gray"]))
    ax.text(-0.92, (nf + 1) / 2, r"frequency bins $m$ ($N_f +1$)", ha="right", va="center", fontsize=13, color=COL["text"], rotation="vertical")
    # ax.text(0.05, nf + 1.16, r"channel $m$", ha="left", va="bottom", fontsize=12, color=COL["text"])

    ax.text(nt + 0.22, 0.5, "DC", ha="left", va="center", fontsize=11, color=COL["gray"])
    ax.text(nt + 0.22, nf + 0.5, "Nyquist", ha="left", va="center", fontsize=11, color=COL["gray"])
    ax.plot([nt, nt + 0.16], [0.5, 0.5], color=COL["gray"], lw=0.9)
    ax.plot([nt, nt + 0.16], [nf + 0.5, nf + 0.5], color=COL["gray"], lw=0.9)

    ax.text(nt / 2, nf + 1.62, r"Packed coefficients $W_{nm}$", ha="center", va="center", fontsize=23, color=COL["text"])
    # ax.text(nt / 2, -1.42, r"$(N_t,\;N_f+1)$", ha="center", va="top", fontsize=14, color=COL["text"])

    save(fig, outpath, transparent=transparent)




def main() -> None:
    parser = argparse.ArgumentParser(description="Generate modular WDM infographic panels as transparent PNGs.")
    parser.add_argument("--outdir", type=Path, default=Path("wdm_panels_clean"), help="Output directory for generated panels.")
    args = parser.parse_args()

    set_style()
    transparent = False  # Set to True for manuscript, False for presentation (white background)
    out = args.outdir
    out.mkdir(parents=True, exist_ok=True)

    save_xtilde_panel(out / "wdm_xtilde.png", transparent=transparent)
    save_phi_panel(out / "wdm_phi_windows.png", transparent=transparent)
    save_cnm_panel(out / "wdm_Cnm.png", transparent=transparent)
    save_wnm_panel(out / "wdm_Wnm.png", transparent=transparent)


    print(f"Saved WDM panels to: {out.resolve()}")


if __name__ == "__main__":
    main()
