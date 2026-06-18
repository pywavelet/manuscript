"""Shared plotting helpers for the LISA galactic-binary freq-vs-WDM figures.

These read the lightweight snapshots written by the study's
``export_manuscript_data.py`` (``lisa_gb_demo.npz`` and ``lisa_gb_pp.npz`` in
``src/data/``) and render the three manuscript figures with plain NumPy +
Matplotlib + the ``wdm_transform`` library — no galactic-binary waveform stack
is needed at build time.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

POSTERIOR_LABELS = ["log10(f0 / Hz)", "fdot [Hz/s]", "log10(A)", "phi0 [rad]"]
# Rank-based PP plot is unitless, so its legend uses unit-free symbols.
LATEX = {
    "log10(f0 / Hz)": r"$\log_{10} f_0$",
    "fdot [Hz/s]": r"$\dot f$",
    "log10(A)": r"$\log_{10} A$",
    "phi0 [rad]": r"$\phi_0$",
}

A_WDM = 1.0 / 3.0
D_WDM = 1.0
NT_WDM = 64

# These figures are placed single-column (``width=\linewidth``) in a twocolumn
# PRD, so LaTeX downscales them to the ~3.4 in column width. Building the canvas
# near that width with point sizes below keeps text legible (~7-8 pt rendered)
# instead of shrinking a large canvas to illegibility.
COLUMN_WIDTH_IN = 3.4


def _font_rc(base: float, ticks: float, legend: float) -> dict:
    """rcParams overriding all font sizes for a self-contained figure."""
    return {
        "font.size": base,
        "axes.titlesize": base,
        "axes.labelsize": base,
        "xtick.labelsize": ticks,
        "ytick.labelsize": ticks,
        "legend.fontsize": legend,
    }

FREQ_COL, WDM_COL, TRUTH_COL = "tab:blue", "tab:orange", "black"
DATA_COL, SRC_COL, PSD_COL = "#B8B8B8", "#ff7f0e", "#172919"
PRIOR_COL = "0.45"


@dataclass(frozen=True)
class DemoData:
    seed: int
    dt: float
    n_total: int
    f0: float
    snr: float
    freqs: np.ndarray
    data_rfft: np.ndarray
    signal_rfft: np.ndarray
    psd_inst: np.ndarray
    band_kmin: int
    band_kmax: int
    labels: list[str]
    truth_display: np.ndarray
    samples_freq: np.ndarray
    samples_wdm: np.ndarray
    prior_f0_sigma: float
    prior_fdot_sigma: float


@dataclass(frozen=True)
class PPData:
    n_seeds: int
    labels: list[str]
    ranks_freq: np.ndarray
    ranks_wdm: np.ndarray


def load_demo(path: Path) -> DemoData:
    with np.load(path, allow_pickle=False) as d:
        return DemoData(
            seed=int(d["seed"]),
            dt=float(d["dt"]),
            n_total=int(d["n_total"]),
            f0=float(d["f0"]),
            snr=float(d["snr"]),
            freqs=np.asarray(d["freqs"], dtype=float),
            data_rfft=np.asarray(d["data_rfft"], dtype=np.complex128),
            signal_rfft=np.asarray(d["signal_rfft"], dtype=np.complex128),
            psd_inst=np.asarray(d["psd_inst"], dtype=float),
            band_kmin=int(d["band_kmin"]),
            band_kmax=int(d["band_kmax"]),
            labels=[str(s) for s in np.asarray(d["labels"]).tolist()],
            truth_display=np.asarray(d["truth_display"], dtype=float),
            samples_freq=np.asarray(d["samples_freq"], dtype=float),
            samples_wdm=np.asarray(d["samples_wdm"], dtype=float),
            prior_f0_sigma=float(d["prior_f0_sigma"]),
            prior_fdot_sigma=float(d["prior_fdot_sigma"]),
        )


def load_pp(path: Path) -> PPData:
    with np.load(path, allow_pickle=False) as d:
        return PPData(
            n_seeds=int(d["n_seeds"]),
            labels=[str(s) for s in np.asarray(d["labels"]).tolist()],
            ranks_freq=np.asarray(d["ranks_freq"], dtype=float),
            ranks_wdm=np.asarray(d["ranks_wdm"], dtype=float),
        )


def one_sided_density(rfft: np.ndarray, dt: float, n: int) -> np.ndarray:
    psd = (2.0 * dt / n) * np.abs(np.asarray(rfft)) ** 2
    if psd.size:
        psd[0] *= 0.5
    if n % 2 == 0 and psd.size > 1:
        psd[-1] *= 0.5
    return np.maximum(psd, 1e-60)


# ── 1. data overview (frequency-domain + freq/WDM source-band zooms) ─────────────


def plot_data(demo: DemoData, out: Path) -> Path:
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    from matplotlib.ticker import FuncFormatter, NullFormatter

    from wdm_transform import TimeSeries

    n, dt, freqs, f0 = demo.n_total, demo.dt, demo.freqs, demo.f0
    data_A = np.fft.irfft(demo.data_rfft, n=n)
    psd_data = one_sided_density(demo.data_rfft, dt, n)
    psd_src = one_sided_density(demo.signal_rfft, dt, n)
    psd_inst = np.maximum(demo.psd_inst, 1e-60)

    k0, k1 = demo.band_kmin, min(demo.band_kmax, len(freqs) - 1)
    lo, hi = freqs[k0], freqs[k1]
    source_bins = np.flatnonzero(np.abs(demo.signal_rfft) > 0)
    source_lo, source_hi = freqs[source_bins[[0, -1]]]
    source_pad = 0.05 * (source_hi - source_lo)
    zoom_lo = max(lo, source_lo - source_pad)
    zoom_hi = min(hi, source_hi + source_pad)

    rc = _font_rc(base=16, ticks=14, legend=13)
    plt.rcParams.update(rc)

    # Built ~2x the column width (downscaled to ~0.5x in LaTeX) so the point
    # sizes above render at ~7-8 pt; aspect kept close to the original 11:8.
    fig = plt.figure(figsize=(6.8, 6.1), layout="constrained")
    gs = fig.add_gridspec(2, 1, height_ratios=[1.15, 1], hspace=0.18)
    bottom_gs = gs[1].subgridspec(1, 2, width_ratios=[0.8, 1.2], wspace=0.32)

    # (a) frequency domain: data periodogram, instrument PSD, source; band shaded.
    ax = fig.add_subplot(gs[0])
    ax.axvspan(lo, hi, color=SRC_COL, alpha=0.12, lw=0, zorder=0)
    ax.loglog(freqs[1:], np.where(psd_src[1:] > 1e-50, psd_src[1:], np.nan),
              color=SRC_COL, lw=1.6, label="Injected source", zorder=1)
    ax.loglog(freqs[1:], psd_data[1:], color=DATA_COL, lw=1.0, label="A data", zorder=2)
    ax.loglog(freqs[1:], psd_inst[1:], color=PSD_COL, lw=2.0, label="Instrument PSD", zorder=3)
    ax.set_xlim(1e-4, 3e-3)
    ax.set_ylim(1e-44, 1e-36)
    # Compact 1-2-5 ticks, labelled in mHz (matching the zoom panels) so the
    # band spans plain 0.1-3 instead of 2x10^-4-style scientific labels.
    ax.set_xticks([1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 3e-3])
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v * 1e3:g}"))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel("frequency [mHz]")
    ax.set_ylabel(r"$S^A(f)$ [strain$^2$/Hz]")
    ax.grid(True, which="both", alpha=0.18, ls=":")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend([handles[i] for i in (1, 2, 0)], [labels[i] for i in (1, 2, 0)],
              loc="lower left", frameon=False)
    ax.text(0.01, 0.97, "(a)", transform=ax.transAxes, va="top", fontweight="bold")

    # (b) frequency-domain source-band zoom.
    ax2 = fig.add_subplot(bottom_gs[0])
    m = (freqs >= zoom_lo) & (freqs <= zoom_hi)
    ax2.semilogy(freqs[m] * 1e3, np.where(psd_src[m] > 1e-50, psd_src[m], np.nan),
                 color=SRC_COL, lw=1.8, zorder=1)
    ax2.semilogy(freqs[m] * 1e3, psd_data[m], color=DATA_COL, lw=1.0, zorder=2)
    ax2.semilogy(freqs[m] * 1e3, psd_inst[m], color=PSD_COL, lw=1.4, zorder=3)
    ax2.set_xlim(zoom_lo * 1e3, zoom_hi * 1e3)
    # Two ticks near the panel edges (~20%/80%): the default pair sits near the
    # centre and the wide mHz labels collide.
    btk = np.round(np.linspace(zoom_lo * 1e3, zoom_hi * 1e3, 6)[[1, 4]], 3)
    ax2.set_xticks(btk)
    ax2.set_xticklabels([f"{t:g}" for t in btk])
    ax2.set_xlabel("frequency [mHz]")
    ax2.set_ylabel(r"$S^A(f)$ [strain$^2$/Hz]")
    ax2.grid(True, which="both", alpha=0.18, ls=":")
    ax2.tick_params(labelsize=12)  # narrow panel: avoid wide mHz labels colliding
    ax2.text(0.02, 0.96, "(b)", transform=ax2.transAxes, va="top", fontweight="bold")

    # (c) WDM time-frequency source-band zoom, whitened to ~unit noise.
    # (c) and its colorbar share the right half with a tight gap, so the bar
    # sits snug against the heatmap instead of across a wide column gutter.
    right_gs = bottom_gs[1].subgridspec(1, 2, width_ratios=[1, 0.05], wspace=0.06)
    ax3 = fig.add_subplot(right_gs[0])
    cax = fig.add_subplot(right_gs[1])
    wdm = TimeSeries(np.asarray(data_A, float), dt=dt).to_wdm(nt=NT_WDM, a=A_WDM, d=D_WDM)
    fg = np.asarray(wdm.freq_grid, float)
    noise_on_grid = np.interp(fg, np.linspace(0.0, float(wdm.nyquist), psd_inst.size), psd_inst)
    # Per-pixel WDM noise variance S_nm = N S[m N_t/2] / (2 dt) (ms.tex eq. for
    # Sigma_w); the N factor is what makes |w_nm|/sqrt(S_nm) ~ unit noise.
    var_row = n * noise_on_grid / (2.0 * dt)
    whiten = np.sqrt(np.where(np.isfinite(var_row) & (var_row > 0), var_row, np.inf))
    img = ((np.asarray(wdm.coeffs[0], float) / whiten[None, :]) ** 2).T  # w_nm^2 / S_nm
    tg = np.asarray(wdm.time_grid, float) / 86400.0
    # Likelihood term w^2/S_nm (~chi^2_1, mean 1); fixed window so noise sits near
    # 10^0 and the source saturates toward 10^2, with clean decade ticks.
    norm = LogNorm(vmin=0.01, vmax=100.0)
    im = ax3.imshow(np.nan_to_num(img), aspect="auto",
                    extent=[tg[0], tg[-1], fg[0] * 1e3, fg[-1] * 1e3],
                    origin="lower", cmap="magma", norm=norm,
                    interpolation="nearest", rasterized=True)
    df = max(10.0 * (fg[1] - fg[0]), 1.5e-5)
    ax3.set_ylim(max(1e-4, f0 - df) * 1e3, min(3e-3, f0 + df) * 1e3)
    ax3.set_xlim(tg[0], tg[-1])
    ax3.set_xlabel("time [days]")
    ax3.set_ylabel("frequency [mHz]")
    ax3.tick_params(labelsize=12)
    # Black label (matching (a)/(b)); thin white outline keeps it legible over
    # the dark end of the magma map.
    import matplotlib.patheffects as pe
    ax3.text(0.02, 0.96, "(c)", transform=ax3.transAxes, va="top",
             color="black", fontweight="bold",
             path_effects=[pe.withStroke(linewidth=2, foreground="white")])
    cbar = fig.colorbar(im, cax=cax, label=r"$(w_{nm}^A)^2 / S_{nm}^A$")
    cbar.ax.yaxis.set_ticks_position("left")
    cbar.ax.yaxis.set_label_position("left")
    # Decade ticks with 10^x labels (log colorbar).
    cbar.minorticks_off()
    decades = [0.01, 1.0, 100.0]
    cticks = [t for t in decades if norm.vmin <= t <= norm.vmax]
    cbar.set_ticks(cticks)
    cbar.ax.set_yticklabels([rf"$10^{{{int(round(np.log10(t)))}}}$" for t in cticks])
    cbar.ax.tick_params(labelsize=11)  # short bar: keep ticks from colliding

    fig.canvas.draw()
    ax_pos = ax.get_position()
    ax.set_position([ax_pos.x0, ax_pos.y0, cax.get_position().x1 - ax_pos.x0, ax_pos.height])
    fig.set_layout_engine(None)
    fig.savefig(out)
    plt.close(fig)
    return out


# ── 2. comparison corner (freq vs WDM, prior overlay) ────────────────────────────


def _recenter_circular(label: str, fs_col, ws_col, truth):
    """Unwrap a circular (phi) marginal around the combined posterior mean so the
    cluster is contiguous and centered instead of split across the +-pi cut."""
    if "phi" not in label.lower():
        return fs_col, ws_col, truth
    both = np.concatenate([fs_col, ws_col])
    center = float(np.arctan2(np.mean(np.sin(both)), np.mean(np.cos(both))))
    wrap = lambda x: center + ((x - center + np.pi) % (2 * np.pi) - np.pi)
    return wrap(fs_col), wrap(ws_col), float(wrap(np.array([truth]))[0])


def _corner_display(demo: DemoData):
    """Map the stored (log10 f0, fdot, log10 A, phi0) marginals into readable
    plotting units so the corner needs no 1e-15 / 1e-5 axis multipliers:

      f0   -> offset from the injection in nHz (so the panel is centered at 0),
      fdot -> units of 1e-15 Hz/s,  A -> log10 A,  phi0 -> radians.

    Returns the display-domain freq/WDM columns, truths, axis labels, and the
    prior (center, sigma) per parameter in those same display units.
    """
    f0_inj = demo.f0
    cols_f, cols_w, truth, labels, prior = [], [], [], [], []
    for i, lab in enumerate(demo.labels):
        fcol, wcol = demo.samples_freq[:, i].copy(), demo.samples_wdm[:, i].copy()
        t = float(demo.truth_display[i])
        if lab == "log10(f0 / Hz)":
            f, w, tt = (10.0 ** fcol - f0_inj) * 1e9, (10.0 ** wcol - f0_inj) * 1e9, 0.0
            labels.append(r"$f_0 - f_0^{\mathrm{inj}}$ [nHz]")
            prior.append((0.0, demo.prior_f0_sigma * 1e9))  # f0 prior is Gaussian in linear f0
        elif lab == "fdot [Hz/s]":
            f, w, tt = fcol * 1e15, wcol * 1e15, t * 1e15
            labels.append(r"$\dot f\ [10^{-15}\,\mathrm{Hz/s}]$")
            prior.append((tt, demo.prior_fdot_sigma * 1e15))
        elif lab == "log10(A)":
            f, w, tt = fcol, wcol, t
            labels.append(r"$\log_{10} A$")
            # amplitude prior is Rayleigh(3*A) -> log10(A) std ~0.31 (shape-invariant).
            prior.append((t, 0.31))
        else:  # phi0: periodic, recenter the cluster; prior is exactly uniform.
            f, w, tt = _recenter_circular(lab, fcol, wcol, t)
            labels.append(r"$\phi_0$ [rad]")
            prior.append((None, 1.0))
        cols_f.append(f)
        cols_w.append(w)
        truth.append(tt)
    return np.column_stack(cols_f), np.column_stack(cols_w), truth, labels, prior


def plot_corner(demo: DemoData, out: Path) -> Path:
    import corner
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    plt.rcParams.update(_font_rc(base=14, ticks=10, legend=12))

    fs, ws, truth, latex, prior = _corner_display(demo)

    rng = []
    for i, t in enumerate(truth):
        lo = min(fs[:, i].min(), ws[:, i].min(), t)
        hi = max(fs[:, i].max(), ws[:, i].max(), t)
        pad = 0.08 * (hi - lo + 1e-30)
        rng.append((lo - pad, hi + pad))

    # max_n_ticks=3 thins the long axis values so they stop colliding with the
    # axis titles at single-column width; labelpad pushes the axis titles clear
    # of the wide log10(A) tick values.
    ckw = dict(labels=latex, range=rng, plot_datapoints=False, smooth=1.0, bins=30,
               levels=(0.5, 0.9), label_kwargs={"fontsize": 14}, max_n_ticks=3,
               labelpad=0.12)
    fig = corner.corner(fs, truths=truth, truth_color=TRUTH_COL, color=FREQ_COL,
                        hist_kwargs={"density": True, "lw": 1.4, "color": FREQ_COL}, **ckw)
    corner.corner(ws, fig=fig, color=WDM_COL,
                  hist_kwargs={"density": True, "lw": 1.4, "color": WDM_COL}, **ckw)
    # corner sizes the canvas at ~9.7 in for K=4; shrink (axes are placed in
    # figure fractions, so this scales uniformly) so the point sizes above are
    # only mildly downscaled to the column width.
    fig.set_size_inches(6.6, 6.6)
    axes = np.asarray(fig.axes).reshape(len(latex), len(latex))

    # Prior overlay on each 1-D panel: posteriors are far narrower than the prior.
    for i, (c, s) in enumerate(prior):
        ax = axes[i, i]
        x0, x1 = ax.get_xlim()
        xg = np.linspace(x0, x1, 200)
        pdf = np.ones_like(xg) if c is None else np.exp(-0.5 * ((xg - c) / s) ** 2)
        ax.plot(xg, pdf / pdf.max() * ax.get_ylim()[1] * 0.95, color=PRIOR_COL, ls=":", lw=1.4, zorder=0)

    axes[0, -1].legend(
        handles=[Patch(color=FREQ_COL, label=f"Frequency (SNR={demo.snr:.1f})"),
                 Patch(color=WDM_COL, label="WDM"),
                 plt.Line2D([0], [0], color=PRIOR_COL, ls=":", label="Prior"),
                 plt.Line2D([0], [0], color=TRUTH_COL, ls="-", label="Truth")],
        loc="upper right", frameon=False, fontsize=13)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


# ── 3. population PP plot (WDM solid, freq dashed) ────────────────────────────────


def plot_pp(pp: PPData, out: Path) -> Path:
    import matplotlib.pyplot as plt

    from scipy.stats import binom

    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    x = np.linspace(0, 1, 500)
    colors = {lab: f"C{i}" for i, lab in enumerate(pp.labels)}
    n = pp.n_seeds
    # Nested 1/2/3-sigma pointwise confidence bands under the uniform null:
    # the count below each credible level is Binomial(n, x). Draw widest-first
    # so the darker inner bands sit on top.
    for ci, shade in ((0.9973, "0.90"), (0.9545, "0.82"), (0.6827, "0.72")):
        lo = binom.ppf(0.5 - ci / 2, n, x) / n
        hi = binom.ppf(0.5 + ci / 2, n, x) / n
        ax.fill_between(x, lo, hi, color=shade, lw=0, zorder=0)
    ax.plot([0, 1], [0, 1], color="0.4", lw=0.9, ls=":", zorder=1)
    ranks = {"wdm": pp.ranks_wdm, "freq": pp.ranks_freq}
    for j, lab in enumerate(pp.labels):
        # Lower alpha on the solid (WDM) curves so the near-identical dashed
        # (frequency) curves remain visible underneath where they overlap.
        for dom, ls, alpha, lw in (("wdm", "-", 0.6, 1.6), ("freq", "dashed", 0.9, 1.3)):
            r = np.sort(ranks[dom][:, j])
            cdf = np.searchsorted(r, x, side="right") / n
            ax.plot(x, cdf, ls=ls, color=colors[lab], lw=lw, alpha=alpha,
                    label=(LATEX[lab] if dom == "wdm" else None))
    ax.set_xlabel("credible level")
    ax.set_ylabel("fraction of truths below")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    leg1 = ax.legend(loc="upper left", frameon=False, title="parameter")
    ax.add_artist(leg1)
    ax.legend(loc="lower right", frameon=False, handles=[
        plt.Line2D([0], [0], color="0.3", ls="-", label="WDM"),
        plt.Line2D([0], [0], color="0.3", ls="--", label="Frequency")])
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out
