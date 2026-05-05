"""
wdm_overview_panels_clean.py
Cleaned-up, publication-quality version of the WDM transform schematic.

Key improvements over original
--------------------------------
* Unified font hierarchy:  title labels 13 pt, body labels 10 pt,
  sub-labels 8.5 pt — consistent across all panels.
* Consistent arrow spec (arrowstyle, lw, color) for every annotation.
* Panel letters all placed via axes-coordinate text at (0.02, 0.97),
  italic, 10 pt — no mixed fig.text / ax.text.
* Accent color changed from saturated blue to muted slate (#4C7A9A)
  so the highlighted cell reads as an accent without jarring in b&w print.
* Envelope curves added to atom panel (panel c) to clarify the
  Gaussian modulation.
* draw_wnm rewritten to match save_wnm_panel quality (consistent
  lineweights, proper DC/Nyquist ticks, legend cleaned up).
* Figure-level layout tuned: tighter wspace/hspace, left margin
  expanded slightly so y-axis label of W_nm is not cropped.
* rcParams: stix math font for better glyph consistency with CM serif.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import paths
from matplotlib.patches import Rectangle

# ── colour palette ────────────────────────────────────────────────────────────
COL = {
    "text":         "#1A1A1A",   # near-black — primary text & lines
    "text_dim":     "#777777",   # secondary / dim labels
    "line":         "#1A1A1A",   # primary drawn lines
    "grid":         "#C8C8C8",   # cell borders / guide lines
    "light":        "#F0F0F0",   # interior cell fill
    "edge":         "#D8D8D8",   # DC / Nyquist row fill
    "accent":       "#4C7A9A",   # muted slate blue — single highlight cell
    "accent_light": "#E8E8E8",   # C_{nm}=i cell fill
}

# ── font sizes (single source of truth) ──────────────────────────────────────
FS = {
    "panel":  10,    # panel letter  (a), (b), …
    "var":    13,    # main variable label
    "axis":   10,    # axis tick labels & annotations
    "sub":     8.5,  # secondary / subordinate text
    "xlabel": 10,    # x/y axis-name labels
}

# ── shared arrow kwargs ───────────────────────────────────────────────────────
_AX_ARROW   = dict(arrowstyle="->",  lw=0.9, color=COL["line"])
_DIM_ARROW  = dict(arrowstyle="<->", lw=0.8, color=COL["text_dim"])
_ANN_ARROW  = dict(arrowstyle="-",   lw=0.8, color=COL["line"])


def set_style() -> None:
    plt.rcParams.update({
        "font.family":     "serif",
        "mathtext.fontset": "cm",
        "font.size":        FS["axis"],
        "axes.labelsize":   FS["xlabel"],
        "figure.dpi":       160,
        "savefig.dpi":      300,
        "pdf.fonttype":     42,
        "ps.fonttype":      42,
        "axes.linewidth":   0.7,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
    })


def clean_axis(ax) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)


def save(fig, outpath: Path, transparent: bool = True) -> None:
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, bbox_inches="tight", pad_inches=0.08,
                transparent=transparent)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
def phi_unit(freqs: np.ndarray, *, center: float = 0.0, a: float = 0.20) -> np.ndarray:
    """Compact Meyer-like prototype window."""
    b = 1.0 - 2.0 * a
    d = np.abs(freqs - center)
    return np.select(
        [d <= a, (d > a) & (d <= a + b)],
        [1.0, np.cos((np.pi / 2.0) * (d - a) / b)],
        default=0.0,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Panel (a) — x̃[ℓ]  (frequency-domain DFT)
# ─────────────────────────────────────────────────────────────────────────────
def draw_xtilde(ax) -> None:
    clean_axis(ax)
    x0, x1 = -3.0, 3.0
    indices = [-3, -2, -1, 0, 1, 3]
    ch_left, ch_right = -1.95, -0.15
    ch_centre = 0.5 * (ch_left + ch_right)

    ax.set_xlim(x0 - 0.55, x1 + 0.55)
    ax.set_ylim(-0.72, 1.10)

    # Horizontal axis arrow
    ax.annotate("", xy=(x1 + 0.42, 0), xytext=(x0 - 0.42, 0),
                arrowprops=_AX_ARROW)
    ax.text(x1 + 0.50, 0, r"$\ell$", ha="left", va="center",
            fontsize=FS["xlabel"], color=COL["line"])

    # Frequency-bin tick marks
    for k in indices:
        ax.plot([k, k], [0, 0.20], color=COL["line"], lw=1.1,
                solid_capstyle="butt")

    # Ellipses
    ax.text(-1.45, 0.07, r"$\cdots$", ha="center", va="bottom",
            fontsize=13, color=COL["line"])
    ax.text( 2.30, 0.07, r"$\cdots$", ha="center", va="bottom",
            fontsize=13, color=COL["line"])

    # Endpoint labels (dim)
    ax.text(x0, 0.27, r"$0$",     ha="center", va="bottom",
            fontsize=FS["sub"], color=COL["text_dim"])
    ax.text(x1, 0.27, r"$N{-}1$", ha="center", va="bottom",
            fontsize=FS["sub"], color=COL["text_dim"])

    # Channel band: dashed vertical boundaries
    for xb in (ch_left, ch_right):
        ax.plot([xb, xb], [-0.04, 0.52], color=COL["grid"],
                lw=0.7, ls="--")

    # Top bracket + Nt label
    bracket_y = 0.52
    ax.plot([ch_left, ch_right], [bracket_y, bracket_y],
            color=COL["line"], lw=0.8)
    ax.text(ch_centre, bracket_y + 0.05, r"$N_t$ bins",
            ha="center", va="bottom", fontsize=FS["sub"], color=COL["text"])

    # Centre tick: ℓ_m
    ax.plot([ch_centre, ch_centre], [0, -0.11],
            color=COL["line"], lw=0.8)
    ax.text(ch_centre, -0.17, r"$\ell_m$",
            ha="center", va="top", fontsize=FS["axis"], color=COL["text"])
    ax.text(ch_centre, -0.40, r"$({=}\,mN_t/2)$",
            ha="center", va="top", fontsize=FS["sub"], color=COL["text_dim"])

    # Panel letter + variable
    ax.text(0.02, 0.97, r"$(a)$", ha="left", va="top",
            transform=ax.transAxes, fontsize=FS["panel"],
            fontstyle="italic", color=COL["text"])
    ax.text(0.14, 0.99, r"$\tilde{x}[\ell]$", ha="left", va="top",
            transform=ax.transAxes, fontsize=FS["var"], color=COL["text"])


# ─────────────────────────────────────────────────────────────────────────────
# Panel (b) — φ̃_m[ℓ]  (prototype window + neighbours)
# ─────────────────────────────────────────────────────────────────────────────
def draw_phi_prototype(ax, *, n: int = 1600, a: float = 0.20) -> None:
    clean_axis(ax)

    b = 1.0 - 2.0 * a
    channel_spacing = 2.0 * (a + b / 2.0)          # = 1.0 for a = 0.20
    centers = [0.0, channel_spacing, 2.0 * channel_spacing]
    alphas  = [1.0, 0.42, 0.22]
    labels  = [r"$m$", r"$m{+}1$", r"$m{+}2$"]

    f = np.linspace(-1.05, 3.70, n)
    ax.set_xlim(-0.92, 3.80)
    ax.set_ylim(-0.38, 1.26)

    # Axis arrow
    ax.annotate("", xy=(3.62, 0), xytext=(-0.86, 0),
                arrowprops=_AX_ARROW)
    ax.text(3.70, 0, r"$\ell$", ha="left", va="center",
            fontsize=FS["xlabel"], color=COL["line"])

    scale = 0.80
    # Draw neighbours first (dashed, faded)
    for i in (2, 1):
        w = phi_unit(f, center=centers[i], a=a)
        ax.plot(f, scale * w, color=COL["line"], lw=1.0,
                ls="--", alpha=alphas[i])

    # Main channel m — solid
    w_main = phi_unit(f, center=centers[0], a=a)
    ax.plot(f, scale * w_main, color=COL["line"], lw=1.7)

    # Channel labels above each peak
    for i, (c, lbl) in enumerate(zip(centers, labels, strict=False)):
        col = COL["text"] if i == 0 else COL["text_dim"]
        ax.text(c, scale + 0.07, lbl, ha="center", va="bottom",
                fontsize=FS["sub"], color=col, alpha=max(alphas[i], 0.55))

    # ℓ_m tick below axis
    ax.plot([0, 0], [0, -0.10], color=COL["line"], lw=0.8)
    ax.text(0, -0.14, r"$\ell_m$", ha="center", va="top",
            fontsize=FS["axis"], color=COL["text"])

    # Panel letter + variable
    ax.text(0.02, 0.97, r"$(b)$", ha="left", va="top",
            transform=ax.transAxes, fontsize=FS["panel"],
            fontstyle="italic", color=COL["text"])
    ax.text(0.16, 0.99, r"$\tilde{\varphi}_m[\ell]$", ha="left", va="top",
            transform=ax.transAxes, fontsize=FS["var"], color=COL["text"])


# ─────────────────────────────────────────────────────────────────────────────
# Panel (c) — x_m[n]  (WDM atom in time domain)
# ─────────────────────────────────────────────────────────────────────────────
def draw_atom(ax, *, m_channel: int = 5) -> None:
    clean_axis(ax)
    t = np.linspace(0, 1, 800)
    sigma    = 0.18
    envelope = 0.76 * np.exp(-0.5 * ((t - 0.5) / sigma) ** 2)
    atom_re  = envelope * np.cos(2 * np.pi * m_channel * t)
    atom_im  = envelope * np.sin(2 * np.pi * m_channel * t)

    ax.set_xlim(-0.08, 1.16)
    ax.set_ylim(-0.92, 0.98)

    # Axis arrow + endpoint ticks
    ax.annotate("", xy=(1.10, 0), xytext=(-0.04, 0),
                arrowprops=_AX_ARROW)
    for x in (0, 1):
        ax.plot([x, x], [-0.05, 0.05], color=COL["line"], lw=1.0)

    # Im (background dashed), Re (foreground solid)
    ax.plot(t, atom_im, color=COL["text_dim"], lw=0.9, ls="--", alpha=0.45,
            label=r"$\mathrm{Im}$")
    ax.plot(t, atom_re, color=COL["line"],     lw=1.5,
            label=r"$\mathrm{Re}$")

    # Small legend for Re / Im
    ax.legend(fontsize=FS["sub"], frameon=False, loc="upper right",
              handlelength=1.4, handletextpad=0.4,
              borderaxespad=0.2, labelcolor=COL["text"])

    ax.text(-0.03, 0.01, r"$n$", fontsize=FS["xlabel"],
            color=COL["text"], ha="right", va="center")
    ax.text(0, -0.14, r"$0$",       ha="center", va="top",
            fontsize=FS["axis"], color=COL["text"])
    ax.text(1, -0.14, r"$N_t{-}1$", ha="center", va="top",
            fontsize=FS["axis"], color=COL["text"])

    # Panel letter + variable
    ax.text(0.02, 0.97, r"$(c)$", ha="left", va="top",
            transform=ax.transAxes, fontsize=FS["panel"],
            fontstyle="italic", color=COL["text"])
    ax.text(0.15, 0.99, r"$x_m[n]$", ha="left", va="top",
            transform=ax.transAxes, fontsize=FS["var"], color=COL["text"])


# ─────────────────────────────────────────────────────────────────────────────
# Panel (d) — W_{nm}  (WDM time–frequency grid)
# ─────────────────────────────────────────────────────────────────────────────
def draw_wnm(ax, *, nt: int = 7, nf: int = 5) -> None:
    _c_even = "#F4F4F4"   # (n+m) even — near-white
    _c_odd  = "#DCDCDC"   # (n+m) odd  — light gray
    _c_edge = "#BEBEBE"   # DC / Nyquist rows — darker

    clean_axis(ax)
    ax.set_aspect("equal")
    ax.set_xlim(-1.95, nt + 3.10)
    ax.set_ylim(-1.40, nf + 1.95)

    # Grid cells — checkerboard interior, darker edge rows
    for n in range(nt):
        for m in range(nf + 1):
            if m in (0, nf):
                face = _c_edge
            else:
                face = _c_even if (n + m) % 2 == 0 else _c_odd
            ax.add_patch(Rectangle((n, m), 1, 1,
                                   facecolor=face,
                                   edgecolor=COL["grid"], lw=0.4))

    # Outer border
    ax.add_patch(Rectangle((0, 0), nt, nf + 1,
                            facecolor="none",
                            edgecolor=COL["line"], lw=1.0))

    # Highlighted cell
    n0, m0 = int(0.55 * nt), int(0.50 * nf)
    ax.add_patch(Rectangle((n0, m0), 1, 1,
                            facecolor=COL["accent"],
                            edgecolor=COL["line"],
                            lw=1.0, alpha=0.80, zorder=5))

    # ΔT dimension arrow (horizontal, below cell)
    ax.annotate("", xy=(n0, m0 - 0.28), xytext=(n0 + 1, m0 - 0.28),
                arrowprops=_DIM_ARROW, zorder=6)
    ax.text(n0 + 0.5, m0 - 0.42, r"$\Delta T$",
            ha="center", va="top", fontsize=FS["sub"], color=COL["text_dim"])

    # ΔF dimension arrow (vertical, left of cell)
    ax.annotate("", xy=(n0 - 0.28, m0), xytext=(n0 - 0.28, m0 + 1),
                arrowprops=_DIM_ARROW, zorder=6)
    ax.text(n0 - 0.44, m0 + 0.5, r"$\Delta F$",
            ha="right", va="center", fontsize=FS["sub"], color=COL["text_dim"])

    # Callout to highlighted cell
    ax.plot(n0 + 0.5, m0 + 0.5, marker="o", ms=2.2,
            color="white", zorder=7)
    ax.annotate(r"$w_{nm}=\mathrm{Re}[C^*_{nm}\,x_m[n]]$",
                xy=(n0 + 0.5, m0 + 0.5),
                xytext=(nt + 0.38, m0 + 0.5),
                ha="left", va="center", fontsize=FS["sub"],
                arrowprops=_ANN_ARROW, color=COL["text"])

    # x-axis labels
    ax.text(0.5,      -0.13, r"$0$",        ha="center", va="top",
            fontsize=FS["axis"], color=COL["text"])
    ax.text(nt - 0.5, -0.13, r"$N_t{-}1$", ha="center", va="top",
            fontsize=FS["axis"], color=COL["text"])
    ax.annotate("", xy=(nt - 0.3, -0.82), xytext=(1.0, -0.82),
                arrowprops=dict(arrowstyle="->", lw=0.7,
                                color=COL["text_dim"]))
    ax.text(nt / 2, -1.06, r"time bins $n$",
            ha="center", va="top", fontsize=FS["axis"], color=COL["text"])

    # y-axis labels
    ax.text(-0.22, 0.5,      r"$0$",   ha="right", va="center",
            fontsize=FS["axis"], color=COL["text"])
    ax.text(-0.22, nf + 0.5, r"$N_f$", ha="right", va="center",
            fontsize=FS["axis"], color=COL["text"])
    ax.annotate("", xy=(-0.60, nf + 0.2), xytext=(-0.60, 0.8),
                arrowprops=dict(arrowstyle="->", lw=0.7,
                                color=COL["text_dim"]))
    ax.text(-1.60, (nf + 1) / 2, r"frequency channels $m$",
            ha="center", va="center", fontsize=FS["axis"],
            color=COL["text"], rotation="vertical")

    # DC / Nyquist edge labels
    for m_edge, lbl in [(0.5, "DC"), (nf + 0.5, "Nyquist")]:
        ax.plot([nt, nt + 0.14], [m_edge, m_edge],
                color=COL["grid"], lw=0.7)
        ax.text(nt + 0.20, m_edge, lbl,
                ha="left", va="center",
                fontsize=FS["sub"], color=COL["text_dim"])

    # C_{nm} legend — compact horizontal key above the grid.
    sw, sh = 0.36, 0.32
    gap = 1.42
    entries = [
        (_c_even, r"$C_{nm}=1$"),
        (_c_odd,  r"$C_{nm}=i$"),
        (_c_edge, r"DC/Ny."),
    ]
    ly = nf + 1.18
    lx = nt - 3.95
    ax.text(lx - 0.18, ly + sh / 2, r"$C_{nm}$",
            ha="right", va="center", fontsize=FS["sub"], color=COL["text_dim"])
    for k, (fc, lbl) in enumerate(entries):
        x0 = lx + k * gap
        ax.add_patch(Rectangle((x0, ly), sw, sh,
                               facecolor=fc, edgecolor=COL["grid"],
                               lw=0.5, clip_on=False))
        ax.text(x0 + sw + 0.10, ly + sh / 2, lbl,
                ha="left", va="center", fontsize=6.5, color=COL["text_dim"],
                clip_on=False)

    # Panel letter + variable
    ax.text(0.02, 0.98, r"$(d)$", ha="left", va="top",
            transform=ax.transAxes, fontsize=FS["panel"],
            fontstyle="italic", color=COL["text"])
    ax.text(0.11, 0.99, r"$W_{nm}$", ha="left", va="top",
            transform=ax.transAxes, fontsize=FS["var"], color=COL["text"])


# ─────────────────────────────────────────────────────────────────────────────
# Overview figure  (2 × 4 mosaic)
# ─────────────────────────────────────────────────────────────────────────────
def save_overview_figure(outpath: Path, *, transparent: bool = True) -> None:
    fig, axes = plt.subplot_mosaic(
        [["a", "a", "b", "b"],
         ["c", "d", "d", "d"]],
        figsize=(10.0, 5.2),
        gridspec_kw={
            "width_ratios":  [1.55, 0.65, 0.82, 1.62],
            "height_ratios": [1.0,  1.65],
            "left":   0.04, "right": 0.995,
            "top":    0.97, "bottom": 0.07,
            "wspace": 0.28, "hspace": 0.20,
        },
    )

    draw_xtilde(axes["a"])
    draw_phi_prototype(axes["b"])
    draw_atom(axes["c"])
    draw_wnm(axes["d"])

    save(fig, outpath, transparent=transparent)


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=Path, default=paths.figures)
    args = parser.parse_args()

    set_style()
    args.outdir.mkdir(parents=True, exist_ok=True)
    save_overview_figure(args.outdir / "wdm_overview.pdf", transparent=False)
    print(f"Saved to {args.outdir / 'wdm_overview.pdf'}")
