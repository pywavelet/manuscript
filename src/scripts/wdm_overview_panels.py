from __future__ import annotations

from pathlib import Path
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import paths

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


# ─────────────────────────────────────────────────────────────────────────────
# Stage 1 — x̃[ℓ]
# Frequency axis with tick marks, ellipses, and a single dashed rectangle
# marking the N_t bins of channel m.
# ─────────────────────────────────────────────────────────────────────────────

def save_xtilde_panel(
    outpath: Path,
    *,
    figsize: tuple[float, float] = (4.0, 1.8),
    transparent: bool = True,
) -> None:
    fig, ax = plt.subplots(figsize=figsize)
    fig.subplots_adjust(left=0.05, right=0.95, top=0.72, bottom=0.18)
    clean_axis(ax)

    indices = [-3, -2, -1, 0, 1, 3]
    x0, x1 = -3.0, 3.0

    ax.set_xlim(x0 - 0.5, x1 + 0.5)
    ax.set_ylim(-0.60, 1.30)

    # Main horizontal axis
    ax.plot([x0, x1], [0, 0], color=COL["line"], lw=2.2, solid_capstyle="butt")

    for k in indices:
        ax.plot([k, k], [0, 0.28], color=COL["line"], lw=2.2, solid_capstyle="butt")

    # Ellipses — placed so neither falls inside the channel bracket
    ax.text(-1.5, 0.08, r"$\dots$", ha="center", va="bottom",
            fontsize=16, color=COL["line"])
    ax.text(2.3, 0.08, r"$\dots$", ha="center", va="bottom",
            fontsize=16, color=COL["line"])

    # Frequency labels — smaller to avoid crowding
    lbl_y = 0.34
    ax.text(0,  lbl_y, r"$0$", ha="center", va="bottom",
            fontsize=11, color=COL["text"])
    ax.text(x0, lbl_y, r"$-N\Delta f/2$", ha="center", va="bottom",
            fontsize=10, color=COL["text"])
    ax.text(x1, lbl_y, r"$\left(\frac{N}{2}-1\right)\Delta f$",
            ha="center", va="bottom", fontsize=10, color=COL["text"])

    # ── Channel m bracket: centred at -1 so it sits clearly left of x=1 ──
    # and well away from the right endpoint label
    ch_left   = -1.8
    ch_right  =  0.2
    ch_centre = (ch_left + ch_right) / 2   # = -0.8
    bracket_y = 0.52

    ax.plot([ch_left,  ch_left],  [0, bracket_y], color=COL["line"], lw=1.0, ls="--")
    ax.plot([ch_right, ch_right], [0, bracket_y], color=COL["line"], lw=1.0, ls="--")
    ax.plot([ch_left,  ch_right], [bracket_y, bracket_y], color=COL["line"], lw=1.0)

    serif_h = 0.05
    ax.plot([ch_left,  ch_left],  [bracket_y - serif_h, bracket_y], color=COL["line"], lw=1.0)
    ax.plot([ch_right, ch_right], [bracket_y - serif_h, bracket_y], color=COL["line"], lw=1.0)

    ax.text(ch_centre, bracket_y + 0.05, r"$N_t$", ha="center", va="bottom",
            fontsize=11, color=COL["text"])

    # Centre tick + label below axis
    ax.plot([ch_centre, ch_centre], [0, -0.10], color=COL["line"], lw=1.0)
    ax.text(ch_centre, -0.14, r"$mN_t/2$", ha="center", va="top",
            fontsize=10, color=COL["text"])

    # Panel letter inside top-left of axes (axes coords), variable label beside it
    ax.text(0.01, 0.97, r"$(a)$", ha="left", va="top",
            transform=ax.transAxes, fontsize=11, color=COL["text"],
            fontstyle="italic")
    ax.text(0.10, 0.99, r"$\tilde{x}[\ell]$", ha="left", va="top",
            transform=ax.transAxes, fontsize=16, color=COL["text"])

    save(fig, outpath, transparent=transparent)


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2 — φ̃[ℓ]
# Existing phi panel, unchanged.
# ─────────────────────────────────────────────────────────────────────────────

def save_phi_panel(
    outpath: Path,
    *,
    n: int = 1600,
    a: float = 0.20,
    figsize: tuple[float, float] = (2.8, 1.8),
    transparent: bool = True,
) -> None:
    b = 1.0 - 2.0 * a
    shift = a + b / 2.0
    centers = np.array([0.0, 2.0 * shift, 4.0 * shift])

    f = np.linspace(-0.7, centers[-1] + 0.7, n)
    windows = [phi_unit(f, center=c, a=a) for c in centers]

    fig, ax = plt.subplots(figsize=figsize)
    fig.subplots_adjust(left=0.04, right=0.90, top=0.82, bottom=0.28)
    clean_axis(ax)
    ax.set_xlim(f.min() - 0.02, f.max() + 0.18)
    ax.set_ylim(-0.55, 1.55)

    ax.annotate(
        "",
        xy=(f.max() + 0.08, 0),
        xytext=(f.min(), 0),
        arrowprops=dict(arrowstyle="->", lw=1.2, color=COL["line"]),
    )
    ax.text(f.max() + 0.16, 0, r"$\ell$", ha="left", va="center",
            fontsize=12, color=COL["text"])

    ax.plot(f, windows[0], color=COL["gray"], lw=2.0, ls="--", alpha=0.75)
    ax.plot(f, windows[1], color=COL["line"], lw=2.4, alpha=1.0)
    ax.plot(f, windows[2], color=COL["gray"], lw=2.0, ls="--", alpha=0.55)

    labels = [r"$(m{-}1)N_t/2$", r"$mN_t/2$", r"$(m{+}1)N_t/2$"]
    phi_labels = [
        r"$\tilde{\varphi}_{m-1}[\ell]$",
        r"$\tilde{\varphi}_{m}[\ell]$",
        r"$\tilde{\varphi}_{m+1}[\ell]$",
    ]
    colors = [COL["gray"], COL["text"], COL["gray"]]
    for c, lab, plab, col in zip(centers, labels, phi_labels, colors):
        ax.plot([c, c], [0, 1.02], color=COL["grid"], lw=1.0, ls=":")
        ax.text(c, -0.14, lab, ha="center", va="top", fontsize=8, color=col)
        ax.text(c, 1.08, plab, ha="center", va="bottom", fontsize=9, color=col)

    ax.annotate(
        "",
        xy=(centers[0], -0.44),
        xytext=(centers[2], -0.44),
        arrowprops=dict(arrowstyle="<->", lw=1.0, linestyle="-", color=COL["accent"]),
    )
    ax.text(centers[1], -0.52,
            r"shift along $N_f$ channels",
            ha="center", va="top", fontsize=8, color=COL["accent"])

    # Panel letter inside top-left corner, variable label beside
    ax.text(0.02, 0.97, r"$(b)$", ha="left", va="top",
            transform=ax.transAxes, fontsize=11, color=COL["text"],
            fontstyle="italic")
    ax.text(0.18, 0.99, r"$\tilde{\varphi}_m[\ell]$", ha="left", va="top",
            transform=ax.transAxes, fontsize=13, color=COL["text"])

    save(fig, outpath, transparent=transparent)


# ─────────────────────────────────────────────────────────────────────────────
# Stage 3 — x_m[n] = IFFT{ x̃[ℓ] · φ̃_m[ℓ] }
# Horizontal time axis, single Gaussian-enveloped cosine waveform,
# dashed envelope curves. Same proportions and line style as Stage 1.
# ─────────────────────────────────────────────────────────────────────────────

def save_atom_panel(
    outpath: Path,
    *,
    Nt: int = 48,
    m_channel: int = 5,
    figsize: tuple[float, float] = (4.0, 1.8),
    transparent: bool = True,
) -> None:
    fig, ax = plt.subplots(figsize=figsize)
    fig.subplots_adjust(left=0.05, right=0.93, top=0.82, bottom=0.18)
    clean_axis(ax)

    t = np.linspace(0, 1, 800)
    sigma = 0.18
    envelope = np.exp(-0.5 * ((t - 0.5) / sigma) ** 2)
    carrier = np.cos(2 * np.pi * m_channel * t)
    waveform = envelope * carrier

    ax.set_xlim(-0.08, 1.12)
    ax.set_ylim(-0.60, 1.10)

    ax.plot([0, 1], [0, 0], color=COL["line"], lw=1.6, solid_capstyle="butt")

    tick_h = 0.05
    ax.plot([0, 0], [-tick_h, tick_h], color=COL["line"], lw=1.6)
    ax.plot([1, 1], [-tick_h, tick_h], color=COL["line"], lw=1.6)

    ax.text(0,  -0.12, r"$0$",       ha="center", va="top", fontsize=11, color=COL["text"])
    ax.text(1,  -0.12, r"$N_t{-}1$", ha="center", va="top", fontsize=11, color=COL["text"])

    # Single arrow + label at tip — no separate n text that can merge
    ax.annotate(
        r"$n$",
        xy=(1.08, 0),
        xytext=(-0.06, 0),
        fontsize=12, color=COL["text"], ha="left", va="center",
        arrowprops=dict(arrowstyle="->", lw=1.2, color=COL["line"]),
    )

    ax.plot(t,  envelope, color=COL["gray"], lw=1.0, ls="--", alpha=0.55)
    ax.plot(t, -envelope, color=COL["gray"], lw=1.0, ls="--", alpha=0.55)
    ax.plot(t,  waveform, color=COL["line"], lw=1.8)

    # Panel letter + label inside top-left
    ax.text(0.01, 0.97, r"$(c)$", ha="left", va="top",
            transform=ax.transAxes, fontsize=11, color=COL["text"],
            fontstyle="italic")
    ax.text(0.10, 0.99, r"$x_m[n]$", ha="left", va="top",
            transform=ax.transAxes, fontsize=15, color=COL["text"])

    save(fig, outpath, transparent=transparent)


# ─────────────────────────────────────────────────────────────────────────────
# Stage 4 — C_{nm}
# Existing Cnm panel, unchanged.
# ─────────────────────────────────────────────────────────────────────────────

def save_cnm_panel(
    outpath: Path,
    *,
    n_periods: int = 2,
    figsize: tuple[float, float] = (2.8, 2.4),
    transparent: bool = True,
) -> None:
    size = 2 * n_periods
    fig, ax = plt.subplots(figsize=figsize)
    fig.subplots_adjust(left=0.14, right=0.88, top=0.80, bottom=0.18)

    ax.set_xticks(np.arange(size) + 0.5)
    ax.set_yticks(np.arange(size) + 0.5)
    ax.set_xticklabels(range(size), fontsize=10)
    ax.set_yticklabels(range(size), fontsize=10)

    ax.set_xlabel(r"$n$", fontsize=11, labelpad=6)
    ax.set_ylabel(r"$m$", fontsize=11, labelpad=6)
    ax.tick_params(length=0)

    for n in range(size):
        for m in range(size):
            even = (n + m) % 2 == 0
            face = "white" if even else COL["accent_light"]
            ax.add_patch(Rectangle(
                (n, m), 1, 1,
                facecolor=face, edgecolor=COL["grid"], lw=0.8,
            ))
            ax.text(n + 0.5, m + 0.5, r"$1$" if even else r"$i$",
                    ha="center", va="center", fontsize=13, color=COL["text"])

    dot_col = COL["grid"]
    dot_y = size + 0.2
    dot_x = size + 0.3

    for i in range(size):
        ax.text(i + 0.5, dot_y, r"$\vdots$", ha="center", va="bottom", color=dot_col)
        ax.text(dot_x, i + 0.5, r"$\dots$", ha="left", va="center", color=dot_col)

    ax.text(dot_x, dot_y, r"$\ddots$", ha="left", va="bottom", color=dot_col)

    ax.set_aspect("equal")
    ax.spines[:].set_visible(False)
    ax.spines["left"].set_visible(True)
    ax.spines["bottom"].set_visible(True)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)

    ax.set_xlim(0, size + 0.8)
    ax.set_ylim(0, size + 0.8)

    # Panel letter and title via fig.text so they sit snugly above the grid
    fig.text(0.03, 0.93, r"$(d)$", ha="left", va="top",
             fontsize=11, color=COL["text"], fontstyle="italic")
    fig.text(0.18, 0.95, r"$C_{nm}$", ha="left", va="top",
             fontsize=14, color=COL["text"])

    save(fig, outpath, transparent=transparent)


# ─────────────────────────────────────────────────────────────────────────────
# Stage 5 — W_{nm}
# Existing Wnm panel with w_{nm} callout on the highlighted cell.
# ─────────────────────────────────────────────────────────────────────────────

def save_wnm_panel(
    outpath: Path,
    *,
    nt: int = 7,
    nf: int = 5,
    figsize: tuple[float, float] = (4.5, 3.2),
    transparent: bool = True,
) -> None:
    fig, ax = plt.subplots(figsize=figsize)
    fig.subplots_adjust(left=0.18, right=0.82, top=0.88, bottom=0.18)
    clean_axis(ax)
    ax.set_aspect("equal")
    ax.set_xlim(-1.4, nt + 3.2)
    ax.set_ylim(-1.55, nf + 2.0)

    # Grid cells
    for n in range(nt):
        for m in range(nf + 1):
            face = COL["edge"] if m in (0, nf) else COL["light"]
            alpha = 0.52 if m in (0, nf) else 0.78
            ax.add_patch(Rectangle(
                (n, m), 1, 1,
                facecolor=face, edgecolor=COL["grid"], lw=0.55, alpha=alpha,
            ))

    # Outer border
    ax.add_patch(Rectangle(
        (0, 0), nt, nf + 1,
        facecolor="none", edgecolor=COL["line"], lw=1.2,
    ))

    # Highlighted cell
    n0 = int(0.55 * nt)
    m0 = int(0.50 * nf)
    ax.add_patch(Rectangle(
        (n0, m0), 1, 1,
        facecolor=COL["accent"], edgecolor=COL["line"],
        lw=1.2, alpha=0.82, zorder=5,
    ))

    # ΔT indicator
    ax.annotate(
        "", xy=(n0, m0 - 0.32), xytext=(n0 + 1, m0 - 0.32),
        arrowprops=dict(arrowstyle="<->", lw=1.0, color=COL["line"]),
        zorder=6,
    )
    ax.text(n0 + 0.5, m0 - 0.50, r"$\Delta T$", ha="center", va="top",
            fontsize=11, color=COL["text"])

    # ΔF indicator
    ax.annotate(
        "", xy=(n0 - 0.32, m0), xytext=(n0 - 0.32, m0 + 1),
        arrowprops=dict(arrowstyle="<->", lw=1.0, color=COL["line"]),
        zorder=6,
    )
    ax.text(n0 - 0.50, m0 + 0.5, r"$\Delta F$", ha="right", va="center",
            fontsize=11, color=COL["text"])

    # w_{nm} callout — anchored to right edge of highlighted cell
    ax.plot(n0 + 0.5, m0 + 0.5, marker="o", ms=2.5,
            color=COL["line"], zorder=7)
    ax.annotate(
        r"$w_{nm} = \mathrm{Re}[C^*_{nm}\,x_m[n]]$",
        xy=(n0 + 0.5, m0 + 0.5),
        xytext=(nt + 0.3, m0 + 0.5),
        ha="left", va="center", fontsize=10,
        arrowprops=dict(arrowstyle="-", lw=0.9, color=COL["line"]),
        color=COL["text"],
    )

    # x-axis labels
    ax.text(0.5,      -0.12, r"$0$",      ha="center", va="top", fontsize=11, color=COL["text"])
    ax.text(nt - 0.5, -0.12, r"$N_t{-}1$", ha="center", va="top", fontsize=11, color=COL["text"])
    ax.annotate(
        "", xy=(nt - 0.3, -0.80), xytext=(1.0, -0.80),
        arrowprops=dict(arrowstyle="->", lw=1.0, color=COL["gray"]),
    )
    ax.text(nt / 2, -1.05, r"time bins $n$", ha="center", va="top",
            fontsize=11, color=COL["text"])

    # y-axis labels — kept close to grid, rotated label via fig.text
    ax.text(-0.20, 0.5,      r"$0$",   ha="right", va="center", fontsize=11, color=COL["text"])
    ax.text(-0.20, nf + 0.5, r"$N_f$", ha="right", va="center", fontsize=11, color=COL["text"])
    ax.annotate(
        "", xy=(-0.55, nf + 0.2), xytext=(-0.55, 0.8),
        arrowprops=dict(arrowstyle="->", lw=1.0, color=COL["gray"]),
    )
    # Rotated y-axis label via fig.text so it never clips
    fig.text(0.03, 0.50, r"frequency bins $m$",
             ha="center", va="center", fontsize=10,
             color=COL["text"], rotation="vertical")

    # Edge channel labels
    ax.text(nt + 0.18, 0.5,      "DC",      ha="left", va="center", fontsize=10, color=COL["gray"])
    ax.text(nt + 0.18, nf + 0.5, "Nyquist", ha="left", va="center", fontsize=10, color=COL["gray"])
    ax.plot([nt, nt + 0.14], [0.5,      0.5     ], color=COL["gray"], lw=0.9)
    ax.plot([nt, nt + 0.14], [nf + 0.5, nf + 0.5], color=COL["gray"], lw=0.9)

    # Panel letter + title via fig.text
    fig.text(0.18, 0.95, r"$(e)$", ha="left", va="top",
             fontsize=11, color=COL["text"], fontstyle="italic")
    fig.text(0.28, 0.96, r"Packed coefficients $W_{nm}$", ha="left", va="top",
             fontsize=12, color=COL["text"])

    save(fig, outpath, transparent=transparent)


# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate WDM infographic panels."
    )
    parser.add_argument(
        "--outdir", type=Path, default=Path("wdm_panels"),
        help="Output directory.",
    )
    args = parser.parse_args()

    set_style()
    transparent = False
    out = paths.figures
    out.mkdir(parents=True, exist_ok=True)

    save_xtilde_panel(out / "wdm_xtilde.png", transparent=transparent)
    save_phi_panel(out / "wdm_phi_windows.png", transparent=transparent)
    save_atom_panel(out / "wdm_atom.png", transparent=transparent)
    save_cnm_panel(out / "wdm_Cnm.png", transparent=transparent)
    save_wnm_panel(out / "wdm_Wnm.png", transparent=transparent)

    print(f"Saved to: {out.resolve()}")
    print("  Stage 1: wdm_xtilde.png")
    print("  Stage 2: wdm_phi_windows.png")
    print("  Stage 3: wdm_atom.png")
    print("  Stage 4: wdm_Cnm.png")
    print("  Stage 5: wdm_Wnm.png")


if __name__ == "__main__":
    main()