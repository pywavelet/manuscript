"""Publication-quality schematic of the WDM transform layout."""

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
    "accent2":      "#A65F3A",   # muted rust — second highlighted cell
    "accent_light": "#E8E8E8",   # C_{nm}=i cell fill
}

EXAMPLES = [
    {"n": 2, "m": 2, "color": COL["accent"],  "label": "A"},
    {"n": 5, "m": 4, "color": COL["accent2"], "label": "B"},
]

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


def c_nm(n: int, m: int) -> complex:
    """Wilson phase factor for an interior WDM cell."""
    return 1.0 + 0.0j if (n + m) % 2 == 0 else 0.0 + 1.0j


def c_nm_label(n: int, m: int) -> str:
    return r"1" if (n + m) % 2 == 0 else r"i"


def panel_label(ax, label: str, title: str) -> None:
    ax.text(0.02, 0.94, label, ha="left", va="top",
            transform=ax.transAxes, fontsize=FS["panel"],
            fontstyle="italic", color=COL["text"])
    ax.text(0.15, 0.96, title, ha="left", va="top",
            transform=ax.transAxes, fontsize=FS["var"], color=COL["text"])


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
# New panel (a) — two frequency-domain cells
# ─────────────────────────────────────────────────────────────────────────────
def draw_frequency_examples(ax, *, nf: int = 6, a: float = 0.28) -> None:
    clean_axis(ax)
    x = np.linspace(0.35, nf - 0.35, 2200)
    offsets = [1.18, 0.18]
    amp = 0.34

    ax.set_xlim(0.45, nf - 0.15)
    ax.set_ylim(-0.34, 1.72)
    ax.annotate("", xy=(nf - 0.22, -0.18), xytext=(0.60, -0.18),
                arrowprops=_AX_ARROW)
    ax.text(nf - 0.10, -0.18, r"$\ell$", ha="left", va="center",
            fontsize=FS["xlabel"], color=COL["line"])

    for ex, y0 in zip(EXAMPLES, offsets, strict=False):
        n, m, color = ex["n"], ex["m"], ex["color"]
        window = phi_unit(x, center=m, a=a)
        local_l = x - m
        phase = np.exp(-2j * np.pi * n * local_l / 2.0)
        g = c_nm(n, m) * window * phase

        ax.fill_between(x, y0, y0 + amp * window, color=color, alpha=0.12,
                        lw=0)
        ax.plot(x, y0 + amp * np.real(g), color=color, lw=1.6)
        ax.plot(x, y0 + amp * np.imag(g), color=color, lw=1.0,
                ls="--", alpha=0.72)
        ax.plot([m, m], [y0 - 0.23, y0 + 0.42], color=COL["grid"],
                lw=0.8, ls=":")
        ax.text(min(m + 0.85, nf - 1.10), y0 + 0.38,
                rf"$n={n},\,m={m},\,C_{{nm}}={c_nm_label(n, m)}$",
                ha="left", va="center", fontsize=FS["sub"],
                color=COL["text"])
        ax.text(m, y0 - 0.25, rf"$m={m}$", ha="center", va="top",
                fontsize=FS["sub"], color=COL["text_dim"])

    ax.plot([], [], color=COL["line"], lw=1.4, label=r"$\mathrm{Re}$")
    ax.plot([], [], color=COL["line"], lw=1.0, ls="--", alpha=0.7,
            label=r"$\mathrm{Im}$")
    ax.legend(fontsize=FS["sub"], frameon=False, loc="upper right",
              handlelength=1.5, handletextpad=0.4, borderaxespad=0.2)
    panel_label(ax, r"$(a)$", r"$\tilde{g}_{nm}[\ell]$")


# ─────────────────────────────────────────────────────────────────────────────
# New panel (b) — matching time-domain atoms
# ─────────────────────────────────────────────────────────────────────────────
def draw_time_examples(ax, *, nt: int = 8) -> None:
    clean_axis(ax)
    tau = np.linspace(0, 1, 1200)
    offsets = [1.02, 0.02]
    amp = 0.34

    ax.set_xlim(-0.05, 1.08)
    ax.set_ylim(-0.44, 1.48)
    ax.annotate("", xy=(1.04, -0.25), xytext=(0.00, -0.25),
                arrowprops=_AX_ARROW)
    ax.text(1.07, -0.25, r"$t$", ha="left", va="center",
            fontsize=FS["xlabel"], color=COL["line"])
    ax.text(0.00, -0.36, r"$0$", ha="center", va="top",
            fontsize=FS["sub"], color=COL["text_dim"])
    ax.text(1.00, -0.36, r"$\Delta T$", ha="center", va="top",
            fontsize=FS["sub"], color=COL["text_dim"])

    for ex, y0 in zip(EXAMPLES, offsets, strict=False):
        n, m, color = ex["n"], ex["m"], ex["color"]
        center = (n + 0.5) / nt
        envelope = np.exp(-0.5 * ((tau - center) / 0.17) ** 2)
        carrier = np.exp(2j * np.pi * (2.0 + 0.55 * m) * (tau - center))
        atom = c_nm(n, m) * envelope * carrier

        ax.fill_between(tau, y0, y0 + amp * envelope, color=color,
                        alpha=0.10, lw=0)
        ax.plot(tau, y0 + amp * np.real(atom), color=color, lw=1.6)
        ax.plot(tau, y0 + amp * np.imag(atom), color=color, lw=1.0,
                ls="--", alpha=0.72)
        ax.plot([center, center], [y0 - 0.23, y0 + 0.42],
                color=COL["grid"], lw=0.8, ls=":")
        ax.text(0.98, y0 + 0.38, rf"$n={n},\,m={m}$",
                ha="right", va="center", fontsize=FS["sub"],
                color=COL["text"])
        ax.text(center, y0 - 0.25, rf"$n={n}$", ha="center", va="top",
                fontsize=FS["sub"], color=COL["text_dim"])

    panel_label(ax, r"$(b)$", r"$g_{nm}(t)$")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 1 redesign — grid plus aligned time and frequency projections
# ─────────────────────────────────────────────────────────────────────────────
def _time_atom(
    t: np.ndarray,
    *,
    n: int,
    m: int,
    nt: int,
    nf: int,
    t_obs: float,
    f_max: float,
) -> np.ndarray:
    dt = t_obs / nt
    df = f_max / nf
    tc = (n + 0.5) * dt
    fc = (m + 0.5) * df
    u = (t - tc) / dt
    envelope = np.exp(-0.5 * (u / 1.15) ** 2) * np.sinc(0.85 * u)
    carrier = np.cos(2.0 * np.pi * fc * (t - tc))
    atom = envelope * carrier
    return atom / np.max(np.abs(atom))


def _freq_window(f: np.ndarray, *, m: int, nf: int, f_max: float) -> np.ndarray:
    df = f_max / nf
    fc = (m + 0.5) * df
    profile = phi_unit((f - fc) / df, center=0.0, a=0.28)
    return profile / np.max(profile)


def draw_projection_grid(
    ax_grid,
    ax_time,
    ax_freq,
    *,
    nt: int = 32,
    nf: int = 16,
    t_obs: float = 512.0,
    f_max: float = 0.5,
) -> None:
    for ax in (ax_grid, ax_time, ax_freq):
        ax.set_facecolor("white")

    dt = t_obs / nt
    df = f_max / nf

    # Main WDM grid.
    ax_grid.set_xlim(0, t_obs)
    ax_grid.set_ylim(0, f_max)
    ax_grid.set_xticks([0, 128, 256, 384, 512])
    ax_grid.set_yticks([0.000, 0.125, 0.250, 0.375, 0.500])
    ax_grid.set_xlabel(r"Time $t$ [seconds]", fontsize=FS["xlabel"])
    ax_grid.set_ylabel(r"Frequency $f$ [Hertz]", fontsize=FS["xlabel"])
    ax_grid.tick_params(labelsize=FS["sub"], length=3, width=0.7)

    for n in range(nt):
        for m in range(nf):
            if m in (0, nf - 1):
                face = "#F1F1F1"
            else:
                face = "#FFFFFF" if (n + m) % 2 == 0 else "#FAFAFA"
            ax_grid.add_patch(Rectangle((n * dt, m * df), dt, df,
                                        facecolor=face,
                                        edgecolor=COL["grid"], lw=0.35))

    for ex in EXAMPLES:
        n, m, color = ex["n"], ex["m"], ex["color"]
        ax_grid.add_patch(Rectangle((n * dt, m * df), dt, df,
                                    facecolor=color,
                                    edgecolor=COL["line"], lw=0.9,
                                    alpha=0.92, zorder=5))
        ax_grid.text((n + 0.5) * dt, (m + 0.5) * df, ex["label"],
                     ha="center", va="center", fontsize=FS["axis"],
                     color="white", fontweight="bold", zorder=6)

    ax_grid.text(0.01, 1.02, r"$W_{nm}$", transform=ax_grid.transAxes,
                 ha="left", va="bottom", fontsize=FS["var"],
                 color=COL["text"])

    # Time-domain atoms, sharing the grid's time axis.
    t = np.linspace(0, t_obs, 3600)
    ax_time.set_xlim(0, t_obs)
    ax_time.set_ylim(-1.10, 1.10)
    ax_time.set_yticks([])
    ax_time.tick_params(axis="x", bottom=False, labelbottom=False)
    ax_time.set_ylabel(r"$g_{nm}(t)$", fontsize=FS["xlabel"])
    ax_time.axhline(0, color=COL["line"], lw=0.65)

    for ex in EXAMPLES:
        n, m, color = ex["n"], ex["m"], ex["color"]
        atom = _time_atom(t, n=n, m=m, nt=nt, nf=nf,
                          t_obs=t_obs, f_max=f_max)
        ax_time.plot(t, 0.74 * atom, color=color, lw=1.35)
        ax_time.axvspan(n * dt, (n + 1) * dt, color=color, alpha=0.10, lw=0)
        ax_time.text((n + 0.5) * dt, 0.92,
                     rf"${ex['label']}:\ (n,m)=({n},{m})$",
                     ha="center", va="top", fontsize=FS["sub"],
                     color=color)

    # Frequency-domain magnitudes, sharing the grid's frequency axis.
    f = np.linspace(0, f_max, 1800)
    ax_freq.set_ylim(0, f_max)
    ax_freq.set_xlim(0, 1.10)
    ax_freq.set_xticks([])
    ax_freq.tick_params(axis="y", left=False, labelleft=False)
    ax_freq.set_title(r"$|\tilde{G}_{nm}(f)|$", fontsize=FS["xlabel"], pad=2)
    ax_freq.axvline(0, color=COL["line"], lw=0.65)

    for ex in EXAMPLES:
        m, color = ex["m"], ex["color"]
        profile = _freq_window(f, m=m, nf=nf, f_max=f_max)
        ax_freq.plot(profile, f, color=color, lw=1.35)
        ax_freq.axhspan(m * df, (m + 1) * df, color=color, alpha=0.10, lw=0)



def draw_cnm_key(ax) -> None:
    ax.set_axis_off()
    ax.text(0.02, 0.82, r"$C_{nm}$", ha="left", va="center",
            fontsize=FS["axis"], color=COL["text_dim"],
            transform=ax.transAxes)
    key = [(r"$1$", "#FFFFFF"), (r"$i$", "#E6E6E6"), ("DC/Ny.", "#F1F1F1")]
    for k, (label, face) in enumerate(key):
        y = 0.60 - 0.22 * k
        ax.add_patch(Rectangle((0.03, y), 0.18, 0.12,
                               transform=ax.transAxes,
                               facecolor=face, edgecolor=COL["grid"],
                               lw=0.6))
        ax.text(0.28, y + 0.06, label, ha="left", va="center",
                fontsize=FS["sub"], color=COL["text_dim"],
                transform=ax.transAxes)


def _schematic_atom(u: np.ndarray, *, n: int, m: int, nt: int) -> np.ndarray:
    center = n + 0.5
    rel = u - center
    envelope = np.exp(-0.5 * (rel / 0.42) ** 2) * np.sinc(1.15 * rel)
    carrier = np.cos(2.0 * np.pi * (0.95 + 0.48 * m) * rel)
    atom = envelope * carrier
    return atom / np.max(np.abs(atom))


def draw_gnm_axis(ax, *, nt: int = 8, nf: int = 6, t_obs: float = 1.0) -> None:
    u = np.linspace(-0.1, nt - 0.9, 2400)
    t = (u / nt) * t_obs
    ax.axhline(0, color=COL["line"], lw=0.65)
    for ex in EXAMPLES:
        n, m, color = ex["n"], ex["m"], ex["color"]
        atom = _schematic_atom(u, n=n, m=m, nt=nt)
        ax.plot(t, 0.82 * atom, color=color, lw=1.35)
        ax.axvspan((n / nt) * t_obs, ((n + 1) / nt) * t_obs,
                   color=color, alpha=0.10, lw=0)
        ax.text(((n + 0.5) / nt) * t_obs, 0.96,
                rf"${ex['label']}$", ha="center", va="top",
                fontsize=FS["axis"], color=color, fontweight="bold")

    ax.set_xlim(0, t_obs)
    ax.set_ylim(-1.08, 1.08)
    ax.set_xticks([0, t_obs])
    ax.set_xticklabels([r"$0$", r"$T$"])
    ax.set_yticks([])
    ax.set_xlabel(r"time $t$", fontsize=FS["xlabel"], labelpad=1)
    ax.set_ylabel(r"$g_{nm}(t)$", fontsize=FS["xlabel"])
    ax.tick_params(axis="x", labelsize=FS["sub"], length=3)
    ax.text(0.02, 0.94, r"$(a)$", transform=ax.transAxes,
            ha="left", va="top", fontsize=FS["panel"],
            fontstyle="italic", color=COL["text"])


def draw_gtilde_axis(ax, *, nf: int = 6, f_max: float = 1.0) -> None:
    f = np.linspace(0, f_max, 1800)
    df = f_max / nf
    for ex in EXAMPLES:
        m, color = ex["m"], ex["color"]
        fc = (m + 0.5) * df
        profile = phi_unit((f - fc) / df, center=0.0, a=0.28)
        ax.fill_between(f, 0, profile, color=color, alpha=0.10, lw=0)
        ax.plot(f, profile, color=color, lw=1.35)
        ax.axvspan(m * df, (m + 1) * df, color=color, alpha=0.08, lw=0)
        ax.text(fc, 1.06, rf"${ex['label']}$",
                ha="center", va="bottom", fontsize=FS["sub"],
                color=color, fontweight="bold")

    ax.set_xlim(0, f_max)
    ax.set_ylim(0, 1.18)
    ax.set_xticks([0, f_max])
    ax.set_xticklabels([r"$0$", r"$f_{\max}$"])
    ax.set_yticks([0, 1])
    ax.set_yticklabels([r"$0$", r"$1$"])
    ax.set_xlabel(r"frequency $f$", fontsize=FS["xlabel"], labelpad=1)
    ax.set_ylabel(r"$|\tilde{G}_{nm}(f)|$", fontsize=FS["xlabel"])
    ax.tick_params(labelsize=FS["sub"], length=3)
    ax.text(0.02, 0.94, r"$(b)$", transform=ax.transAxes,
            ha="left", va="top", fontsize=FS["panel"],
            fontstyle="italic", color=COL["text"])


# ─────────────────────────────────────────────────────────────────────────────
# Panel (d) — W_{nm}  (WDM time–frequency grid)
# ─────────────────────────────────────────────────────────────────────────────
def draw_wnm(ax, *, nt: int = 8, nf: int = 6) -> None:
    _c_even = "#F4F4F4"   # (n+m) even — near-white
    _c_odd  = "#DCDCDC"   # (n+m) odd  — light gray
    _c_edge = "#BEBEBE"   # DC / Nyquist rows — darker

    clean_axis(ax)
    ax.set_aspect("auto")
    ax.set_xlim(-1.95, nt + 3.25)
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

    # Highlighted example cells.
    for letter, ex in zip(["A", "B"], EXAMPLES, strict=False):
        n0, m0, color = ex["n"], ex["m"], ex["color"]
        ax.add_patch(Rectangle((n0, m0), 1, 1,
                               facecolor=color, edgecolor=COL["line"],
                               lw=1.0, alpha=0.82, zorder=5))
        ax.text(n0 + 0.5, m0 + 0.5, letter, ha="center", va="center",
                fontsize=FS["axis"], color="white", fontweight="bold",
                zorder=6)

    n0, m0 = EXAMPLES[0]["n"], EXAMPLES[0]["m"]
    ax.annotate("", xy=(n0, m0 - 0.28), xytext=(n0 + 1, m0 - 0.28),
                arrowprops=_DIM_ARROW, zorder=6)
    ax.text(n0 + 0.5, m0 - 0.42, r"$\Delta T$",
            ha="center", va="top", fontsize=FS["sub"], color=COL["text_dim"])
    ax.annotate("", xy=(n0 - 0.28, m0), xytext=(n0 - 0.28, m0 + 1),
                arrowprops=_DIM_ARROW, zorder=6)
    ax.text(n0 - 0.44, m0 + 0.5, r"$\Delta F$",
            ha="right", va="center", fontsize=FS["sub"], color=COL["text_dim"])

    ax.annotate(r"$w_{nm}=\sqrt{2}\,(-1)^{nm}\,\mathrm{Re}[C^*_{nm}\,x_m[n]]$",
                xy=(EXAMPLES[1]["n"] + 0.5, EXAMPLES[1]["m"] + 0.5),
                xytext=(nt + 0.38, EXAMPLES[1]["m"] + 0.25),
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

    panel_label(ax, r"$(c)$", r"$W_{nm}$")


# ─────────────────────────────────────────────────────────────────────────────
# Overview figure
# ─────────────────────────────────────────────────────────────────────────────
def save_overview_figure(outpath: Path, *, transparent: bool = True) -> None:
    fig = plt.figure(figsize=(10.8, 5.25))
    gs = fig.add_gridspec(
        2, 2,
        width_ratios=[1.00, 2.55],
        height_ratios=[1.0, 1.0],
        left=0.06, right=0.985, bottom=0.12, top=0.955,
        wspace=0.18, hspace=0.36,
    )
    ax_time = fig.add_subplot(gs[0, 0])
    ax_freq = fig.add_subplot(gs[1, 0])
    ax_grid = fig.add_subplot(gs[:, 1])

    draw_gnm_axis(ax_time)
    draw_gtilde_axis(ax_freq)
    draw_wnm(ax_grid)

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
