from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import corner
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from matplotlib.ticker import FixedLocator, FuncFormatter, LogLocator, MaxNLocator, NullFormatter, ScalarFormatter
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

from wdm_transform import TimeSeries
from wdm_transform.signal_processing import wdm_noise_variance

c = 299792458.0
L_LISA = 2.5e9
A_WDM = 1.0 / 3.0
D_WDM = 1.0
DEFAULT_NT_WDM = 432
SECONDS_PER_YEAR = 365.25 * 24 * 3600

DATA_COL = "#D8D8D8"
INJECTION_COL = "#ff7f0e"
NOISE_PSD = "#172919"
FULL_PSD = "#172919"
FINSET_SPINE = "#434242"


@dataclass(frozen=True)
class InjectionData:
    t_obs: float
    dt: float
    freqs: np.ndarray
    data_At: np.ndarray
    noise_psd_A: np.ndarray
    source_params: np.ndarray


@dataclass(frozen=True)
class RunPosterior:
    name: str
    path: Path
    samples: np.ndarray
    labels: list[str]
    truth: np.ndarray
    snr: float | None


def load_injection(path: Path) -> InjectionData:
    with np.load(path) as inj:
        return InjectionData(
            t_obs=float(inj["t_obs"]),
            dt=float(inj["dt"]),
            freqs=np.asarray(inj["freqs"], dtype=float),
            data_At=np.asarray(inj["data_At"], dtype=float),
            noise_psd_A=np.asarray(inj["noise_psd_A"], dtype=float),
            source_params=np.atleast_2d(np.asarray(inj["source_params"], dtype=float)),
        )


def setup_plotting() -> tuple[object, object]:
    matplotlib.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "lines.linewidth": 1.6,
            "mathtext.fontset": "dejavusans",
        }
    )
    return matplotlib, plt


def one_sided_periodogram_density(rfft_coeffs: np.ndarray, dt: float, n_samples: int) -> np.ndarray:
    psd = (2.0 * dt / n_samples) * np.abs(np.asarray(rfft_coeffs)) ** 2
    if psd.size:
        psd[0] *= 0.5
    if n_samples % 2 == 0 and psd.size > 1:
        psd[-1] *= 0.5
    return np.maximum(psd, 1e-60)


def place_local_tdi(segment: np.ndarray, kmin: int, n_freqs: int) -> np.ndarray:
    full = np.zeros(n_freqs, dtype=np.complex128)
    seg = np.asarray(segment, dtype=np.complex128).reshape(-1)
    end = min(kmin + seg.size, n_freqs)
    if end > kmin:
        full[kmin:end] = seg[: end - kmin]
    return full


def _ntilda_e(f: np.ndarray, A: float = 3.0, P: float = 15.0, L: float = L_LISA) -> np.ndarray:
    f_safe = np.where(f > 0.0, f, 1.0)
    fstar = 1.0 / (2.0 * np.pi * L / c)
    return (
        0.5 * (2.0 + np.cos(f_safe / fstar)) * (P / L) ** 2 * 1e-24 * (1.0 + (0.002 / f_safe) ** 4)
        + 2.0
        * (1.0 + np.cos(f_safe / fstar) + np.cos(f_safe / fstar) ** 2)
        * (A / L) ** 2
        * 1e-30
        * (1.0 + (0.0004 / f_safe) ** 2)
        * (1.0 + (f_safe / 0.008) ** 4)
        * (1.0 / (2.0 * np.pi * f_safe)) ** 4
    )


def tdi15_factor(f: np.ndarray, L: float = L_LISA) -> np.ndarray:
    fstar = 1.0 / (2.0 * np.pi * L / c)
    return 4.0 * np.sin(f / fstar) * f / fstar


def noise_tdi15_psd(channel: int, f: np.ndarray, L: float = L_LISA) -> np.ndarray:
    f_arr = np.asarray(f, dtype=float)
    out = np.zeros_like(f_arr, dtype=float)
    pos = f_arr > 0.0
    if np.any(pos):
        out[pos] = _ntilda_e(f_arr[pos], L=L) * tdi15_factor(f_arr[pos], L=L)
    return out


def regenerate_source_rfft(injection: InjectionData) -> np.ndarray:
    import jax
    import jax.numpy as jnp
    import lisaorbits
    from jaxgb.jaxgb import JaxGB

    jax.config.update("jax_enable_x64", True)
    params = np.asarray(injection.source_params[0], dtype=float)
    orbit = lisaorbits.EqualArmlengthOrbits()
    jgb = JaxGB(orbit, t_obs=float(injection.t_obs), t0=0.0, n=256)
    params_j = jnp.asarray(params, dtype=jnp.float64)
    a_loc, _, _ = jgb.get_tdi(params_j, tdi_generation=1.5, tdi_combination="AET")
    kmin = int(jnp.asarray(jgb.get_kmin(params_j[None, 0:1])).reshape(-1)[0])
    return place_local_tdi(np.asarray(a_loc), kmin, len(injection.freqs))


def build_wdm_products(
    data_t: np.ndarray,
    source_t: np.ndarray,
    *,
    dt: float,
    noise_psd: np.ndarray,
    nt: int = DEFAULT_NT_WDM,
):
    total_wdm = TimeSeries(np.asarray(data_t, dtype=float), dt=dt).to_wdm(nt=nt, a=A_WDM, d=D_WDM)
    source_wdm = TimeSeries(np.asarray(source_t, dtype=float), dt=dt).to_wdm(nt=nt, a=A_WDM, d=D_WDM)
    noise_row = np.interp(
        np.asarray(total_wdm.freq_grid, dtype=float),
        np.linspace(0.0, total_wdm.nyquist, len(noise_psd)),
        np.asarray(noise_psd, dtype=float),
        left=float(noise_psd[0]),
        right=float(noise_psd[-1]),
    )
    whitening = np.sqrt(wdm_noise_variance(noise_row, nt=total_wdm.nt, dt=dt))
    return total_wdm, source_wdm, whitening


def wdm_image(wdm, whitening: np.ndarray | None = None) -> np.ndarray:
    coeffs = np.asarray(wdm.coeffs[0], dtype=float)
    if whitening is not None:
        coeffs = coeffs / whitening
    return np.abs(coeffs).T


def add_frequency_inset(
    ax,
    *,
    freqs: np.ndarray,
    psd_data: np.ndarray,
    psd_background: np.ndarray,
    psd_source: np.ndarray,
    source_params: np.ndarray,
) -> None:
    f0 = float(source_params[0])
    half_width = max(18.0 * (freqs[1] - freqs[0]), 1.8e-5)
    freq_lo = max(1e-4, f0 - half_width)
    freq_hi = min(3e-3, f0 + half_width)
    mask = (freqs >= freq_lo) & (freqs <= freq_hi)
    if np.count_nonzero(mask) < 8:
        return

    local_data = np.maximum(psd_data[mask], 1e-60)
    local_bg = np.maximum(psd_background[mask], 1e-60)
    local_src = np.maximum(psd_source[mask], 1e-60)
    local_freqs = freqs[mask]

    inset = inset_axes(
        ax,
        width="31%",
        height="48%",
        loc="lower left",
        bbox_to_anchor=(0.055, 0.09, 1.0, 1.0),
        bbox_transform=ax.transAxes,
        borderpad=0.0,
    )
    inset.set_facecolor("white")
    inset.loglog(local_freqs, local_src, color=INJECTION_COL, lw=5.0, alpha=0.28, zorder=-10, solid_capstyle="round")
    inset.loglog(local_freqs, local_data, color=DATA_COL, lw=1.1, alpha=1.0, zorder=10)
    inset.loglog(local_freqs, local_src, color=INJECTION_COL, lw=1.8, alpha=0.95, zorder=4)
    inset.set_xlim(freq_lo, freq_hi)
    inset.set_ylim(
        max(float(np.nanmin(local_bg)) * 0.5, 1e-44),
        max(float(np.nanmax(np.maximum(local_data, local_src))) * 2.0, 1e-38),
    )
    xticks = np.linspace(freq_lo, freq_hi, 5)
    inset.xaxis.set_major_locator(FixedLocator(xticks))
    inset.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x * 1e3:.4f}"))
    inset.xaxis.set_minor_formatter(NullFormatter())
    inset.yaxis.set_major_locator(LogLocator(base=10.0, numticks=4))
    inset.yaxis.set_minor_formatter(NullFormatter())
    inset.tick_params(axis="both", labelsize=7, pad=1, length=3)
    inset.set_xlabel("Frequency (mHz)", fontsize=7, labelpad=1)
    inset.set_ylabel(r"$S_h(f)$", fontsize=7, labelpad=1)
    for spine in inset.spines.values():
        spine.set_linewidth(1.0)
        spine.set_edgecolor(FINSET_SPINE)

    mark_inset(ax, inset, loc1=2, loc2=4, fc="none", ec=FINSET_SPINE, alpha=0.55)


def add_wdm_panel(fig, ax, *, total_wdm, source_wdm, whitening: np.ndarray, source_params: np.ndarray) -> None:
    total_img = wdm_image(total_wdm, whitening)
    source_img = wdm_image(source_wdm, whitening)
    time_grid = np.asarray(total_wdm.time_grid, dtype=float)
    time_years = time_grid / SECONDS_PER_YEAR
    freq_grid = np.asarray(total_wdm.freq_grid, dtype=float)
    extent = [time_years[0], time_years[-1], freq_grid[0], freq_grid[-1]]

    positive = total_img[total_img > 0.0]
    vmin = max(float(np.nanpercentile(positive, 18)), 0.25)
    vmax = max(float(np.nanpercentile(total_img, 99.5)), vmin * 6.0)
    norm = LogNorm(vmin=vmin, vmax=vmax)

    im = ax.imshow(
        total_img,
        aspect="auto",
        extent=extent,
        origin="lower",
        cmap="magma",
        norm=norm,
        interpolation="nearest",
        rasterized=True,
    )
    ax.set_yscale("log")
    ax.set_ylim(1e-4, 3e-3)
    ax.set_ylabel("Frequency (Hz)", fontweight="bold")
    ax.set_xlabel("Time (yr)", fontweight="bold")
    ax.grid(True, which="both", alpha=0.12, linestyle=":")
    ax.text(0.01, 0.98, "(b)", transform=ax.transAxes, ha="left", va="top", fontweight="bold", color="white")
    ax.set_xticks(np.linspace(time_years[0], time_years[-1], 5))

    cbar = fig.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label("WDM amplitude / noise std")

    delta_f = max(8.0 * total_wdm.delta_f, 1.5e-5)
    freq_lo = max(1e-4, source_params[0] - delta_f)
    freq_hi = min(3e-3, source_params[0] + delta_f)
    source_positive = source_img[source_img > 0.0]
    source_vmin = max(float(np.nanpercentile(source_positive, 20)), 0.35)
    source_vmax = max(float(np.nanpercentile(source_img, 99.7)), source_vmin * 5.0)
    source_norm = LogNorm(vmin=source_vmin, vmax=source_vmax)

    inset = inset_axes(
        ax,
        width="31%",
        height="48%",
        loc="lower left",
        bbox_to_anchor=(0.055, 0.09, 1.0, 1.0),
        bbox_transform=ax.transAxes,
        borderpad=0.0,
    )
    inset.imshow(
        source_img,
        aspect="auto",
        extent=extent,
        origin="lower",
        cmap="magma",
        norm=source_norm,
        interpolation="nearest",
        rasterized=True,
    )
    inset.set_yscale("log")
    inset.set_xlim(time_years[0], time_years[-1])
    inset.set_ylim(freq_lo, freq_hi)
    inset.tick_params(axis="both", colors="white", labelsize=7, pad=1)
    year_ticks = np.linspace(time_years[0], time_years[-1], 4)
    inset.set_xticks(year_ticks)
    inset.set_xticklabels([f"{tick:.2f}" for tick in year_ticks], color="white")
    inset.set_xlabel("Time (yr)", color="white", fontsize=7, labelpad=1)
    inset.set_yticks([freq_lo, source_params[0], freq_hi])
    inset.set_yticklabels([f"{freq_lo * 1e3:.3f}", f"{source_params[0] * 1e3:.3f}", f"{freq_hi * 1e3:.3f}"], color="white")
    inset.set_ylabel("mHz", color="white", fontsize=7, labelpad=1)
    for spine in inset.spines.values():
        spine.set_edgecolor("white")
        spine.set_linewidth(1.1)

    ax.axhspan(freq_lo, freq_hi, facecolor="none", edgecolor="white", linewidth=0.9, alpha=0.35)
    mark_inset(ax, inset, loc1=2, loc2=4, fc="none", ec="white", alpha=0.4)


def create_infographic(injection_path: Path, output_path: Path) -> Path:
    _, plt_module = setup_plotting()
    injection = load_injection(injection_path)
    freqs = injection.freqs
    dt = injection.dt
    source_params = np.asarray(injection.source_params[0], dtype=float)
    source_fft = regenerate_source_rfft(injection)
    data_fft = np.fft.rfft(np.asarray(injection.data_At, dtype=float))
    background_fft = data_fft - source_fft
    source_t = np.fft.irfft(source_fft, n=len(injection.data_At))

    psd_data = one_sided_periodogram_density(data_fft, dt, len(injection.data_At))
    psd_background = one_sided_periodogram_density(background_fft, dt, len(injection.data_At))
    psd_source = one_sided_periodogram_density(source_fft, dt, len(injection.data_At))
    total_psd = np.maximum(np.asarray(injection.noise_psd_A, dtype=float), 1e-60)
    inst_psd = np.maximum(noise_tdi15_psd(0, freqs), 1e-60)
    total_wdm, source_wdm, whitening = build_wdm_products(injection.data_At, source_t, dt=dt, noise_psd=total_psd)

    fig = plt_module.figure(figsize=(10, 9.2))
    gs = fig.add_gridspec(2, 1, hspace=0.28, height_ratios=[1.0, 1.0])

    ax1 = fig.add_subplot(gs[0])
    ax1.loglog(freqs[1:], psd_data[1:], color=DATA_COL, alpha=0.9, lw=1.2, label="A data")
    ax1.loglog(freqs[1:], total_psd[1:], color=NOISE_PSD, lw=2.0, label="Total PSD model")
    ax1.loglog(freqs[1:], inst_psd[1:], color=FULL_PSD, lw=1.4, linestyle="--", label="Instrument PSD")
    ax1.loglog([], [], color=INJECTION_COL, lw=1.8, alpha=0.95, label="Injected source")
    ax1.set_xlim(1e-4, 3e-3)
    ax1.set_ylim(1e-44, 1e-36)
    ax1.set_ylabel(r"$S_h(f)$ [strain$^2$/Hz]", fontweight="bold")
    ax1.grid(True, which="both", alpha=0.18, linestyle=":")
    ax1.legend(loc="upper right", frameon=False)
    ax1.set_xlabel("Frequency (Hz)", fontweight="bold")
    ax1.text(0.01, 0.98, "(a)", transform=ax1.transAxes, ha="left", va="top", fontweight="bold")
    add_frequency_inset(ax1, freqs=freqs, psd_data=psd_data, psd_background=psd_background, psd_source=psd_source, source_params=source_params)

    ax2 = fig.add_subplot(gs[1])
    add_wdm_panel(fig, ax2, total_wdm=total_wdm, source_wdm=source_wdm, whitening=whitening, source_params=source_params)

    fig.savefig(output_path, format="pdf")
    plt_module.close(fig)
    return output_path


def wrap_phase(phi: np.ndarray | float) -> np.ndarray | float:
    return (phi + np.pi) % (2.0 * np.pi) - np.pi


def is_phase_parameter(label: str) -> bool:
    lowered = label.lower()
    return "phi" in lowered or "phase" in lowered


def _normalize_phi(samples: np.ndarray, labels: list[str]) -> np.ndarray:
    wrapped = np.asarray(samples, dtype=float).copy()
    for idx, label in enumerate(labels):
        if is_phase_parameter(label):
            wrapped[:, idx] = wrap_phase(wrapped[:, idx])
    return wrapped


def load_run(path: Path, name: str) -> RunPosterior:
    with np.load(path) as data:
        samples = np.asarray(data["samples_source"], dtype=float)
        labels = [str(item) for item in np.asarray(data["labels"]).tolist()]
        truth = np.asarray(data["truth"], dtype=float).reshape(-1)
        snr_arr = np.asarray(data["snr_optimal"], dtype=float).reshape(-1)
    truth_2d = _normalize_phi(truth[None, :], labels)
    return RunPosterior(
        name=name,
        path=path,
        samples=_normalize_phi(samples, labels),
        labels=labels,
        truth=truth_2d.reshape(-1),
        snr=float(snr_arr[0]) if snr_arr.size else None,
    )


def align_common_labels(run_a: RunPosterior, run_b: RunPosterior) -> tuple[RunPosterior, RunPosterior]:
    common = [label for label in run_a.labels if label in run_b.labels]
    idx_a = [run_a.labels.index(label) for label in common]
    idx_b = [run_b.labels.index(label) for label in common]
    return (
        RunPosterior(run_a.name, run_a.path, run_a.samples[:, idx_a], common, run_a.truth[idx_a], run_a.snr),
        RunPosterior(run_b.name, run_b.path, run_b.samples[:, idx_b], common, run_b.truth[idx_b], run_b.snr),
    )


def _format_axis_label(label: str) -> str:
    label_map = {
        "f0": r"$f_0$",
        "fdot": r"$\dot{f}$",
        "A": r"$A$",
        "phi0": r"$\phi_0$",
        "psi": r"$\psi$",
        "iota": r"$\iota$",
        "SNR": r"SNR",
    }
    if label in label_map:
        return label_map[label]
    for key, value in label_map.items():
        if key in label.lower():
            return label.replace(key, value)
    return label


def _inject_unit_prefix(ax, label: str, truth_value: float | None = None) -> tuple[str, float, bool]:
    compact_label = re.sub(r"[\s$\\{}]", "", label)
    if compact_label == "A":
        return r"$A\ [10^{-23}]$", 1.0e23, False
    match = re.search(r"\[([^\]]+)\]", label)
    if not match:
        return label, 1.0, False
    base_unit = match.group(1)
    if any(param in label.lower() for param in ["phi", "psi", "iota", "phase"]) or base_unit.lower() == "rad":
        return label, 1.0, False
    if "snr" in label.lower():
        return label, 1.0, False
    if "fdot" in label.lower() or r"\dot{f}" in label:
        return r"$\dot{f}\ [10^{-18}\ \mathrm{Hz/s}]$", 1.0e18, False
    if ("f0" in label.lower() or "f_0" in label.lower()) and truth_value is not None:
        lim = ax.get_xlim() if hasattr(ax, "get_xlim") else ax.get_ylim()
        delta_scale = max(abs(lim[0] - truth_value), abs(lim[1] - truth_value))
        if delta_scale > 0.0:
            delta_exp = int(np.floor(np.log10(delta_scale)))
            delta_exp = int(3 * np.floor(delta_exp / 3))
            scale_factor = 10.0 ** (-delta_exp)
            unit_label = rf"10^{{{delta_exp}}}"
        else:
            scale_factor = 1.0
            unit_label = "Hz"
        return rf"$\Delta f_0\ [{unit_label}]$ Hz", scale_factor, True
    lim = ax.get_xlim() if hasattr(ax, "get_xlim") else ax.get_ylim()
    data_mag = np.abs(np.mean(lim))
    if data_mag == 0.0:
        return label, 1.0, False
    mag = np.log10(data_mag)
    prefixes = [(12, "T"), (9, "G"), (6, "M"), (3, "k"), (0, ""), (-3, "m"), (-6, "µ"), (-9, "n"), (-12, "p"), (-15, "f")]
    best_exp, best_prefix = min(prefixes, key=lambda item: abs(mag - item[0]))
    if best_exp == 0:
        return label, 1.0, False
    scale_factor = 10.0 ** best_exp
    return label.replace(f"[{base_unit}]", f"[{best_prefix}{base_unit}]"), scale_factor, False


def _make_scalar_formatter() -> ScalarFormatter:
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-2, 2))
    formatter.set_useOffset(True)
    return formatter


def _apply_scalar_tick_format(ax, axis: str = "both") -> None:
    if axis in ("x", "both"):
        ax.xaxis.set_major_formatter(_make_scalar_formatter())
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
    if axis in ("y", "both"):
        ax.yaxis.set_major_formatter(_make_scalar_formatter())
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
        ax.yaxis.set_offset_position("left")


def _apply_delta_tick_scaling(ax, truth_value: float, scale_factor: float, axis: str = "both") -> None:
    def delta_formatter(val, _pos):
        scaled_delta = (val - truth_value) * scale_factor
        if abs(scaled_delta) < 1e-25:
            return "0"
        if abs(scaled_delta) <= 1e-1:
            return f"{scaled_delta:.4e}"
        return f"{scaled_delta:.6g}"

    if axis in ("x", "both"):
        ax.xaxis.set_major_formatter(FuncFormatter(delta_formatter))
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
    if axis in ("y", "both"):
        ax.yaxis.set_major_formatter(FuncFormatter(delta_formatter))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4))


def _apply_tick_scaling(ax, scale_factor: float, axis: str = "both") -> None:
    if scale_factor == 1.0:
        _apply_scalar_tick_format(ax, axis=axis)
        return

    def formatter_fn(val, _pos):
        scaled = val * scale_factor
        if abs(scaled) < 1e-30:
            return "0"
        if abs(scaled) <= 1e-2:
            return f"{scaled:.3e}"
        return f"{scaled:.5g}"

    if axis in ("x", "both"):
        ax.xaxis.offsetText.set_visible(False)
        ax.xaxis.set_major_formatter(FuncFormatter(formatter_fn))
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
    if axis in ("y", "both"):
        ax.yaxis.offsetText.set_visible(False)
        ax.yaxis.set_major_formatter(FuncFormatter(formatter_fn))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4))


def _reposition_offset_text(ax, row: int, col: int, ndim: int) -> None:
    x_offset = ax.xaxis.get_offset_text()
    y_offset = ax.yaxis.get_offset_text()
    x_offset.set_size(8)
    y_offset.set_size(8)
    if row == ndim - 1:
        x_offset.set_horizontalalignment("right")
        x_offset.set_verticalalignment("top")
        x_offset.set_x(1.0)
        x_offset.set_y(-0.08)
    if col == 0:
        y_offset.set_horizontalalignment("left")
        y_offset.set_verticalalignment("bottom")
        y_offset.set_x(0.0)
        y_offset.set_y(1.02)


def _configure_axes_formatting(fig, ndim: int, truth_values: np.ndarray | None = None) -> None:
    for i, ax in enumerate(fig.axes):
        row = i // ndim
        col = i % ndim
        try:
            ax.ticklabel_format(style="sci", axis="both", scilimits=(-2, 2), useMathText=True)
        except (AttributeError, ValueError):
            pass
        if col == 0:
            y_label = ax.get_ylabel()
            if y_label:
                truth_val = truth_values[row] if truth_values is not None else None
                new_label, scale, is_delta = _inject_unit_prefix(ax, y_label, truth_val)
                ax.set_ylabel(new_label)
                if is_delta and truth_val is not None:
                    _apply_delta_tick_scaling(ax, truth_val, scale, axis="y")
                else:
                    _apply_tick_scaling(ax, scale, axis="y")
        else:
            ax.set_yticklabels([])
        if row == ndim - 1:
            x_label = ax.get_xlabel()
            if x_label:
                truth_val = truth_values[col] if truth_values is not None else None
                new_label, scale, is_delta = _inject_unit_prefix(ax, x_label, truth_val)
                ax.set_xlabel(new_label)
                if is_delta and truth_val is not None:
                    _apply_delta_tick_scaling(ax, truth_val, scale, axis="x")
                else:
                    _apply_tick_scaling(ax, scale, axis="x")
        else:
            ax.set_xticklabels([])
        ax.tick_params(axis="both", labelsize=8, pad=3)
        _reposition_offset_text(ax, row, col, ndim)


def plot_corner(run_a: RunPosterior, run_b: RunPosterior, output_path: Path) -> Path:
    keep = [idx for idx, label in enumerate(run_a.labels) if "snr" not in label.lower()]
    run_a = RunPosterior(run_a.name, run_a.path, run_a.samples[:, keep], [run_a.labels[idx] for idx in keep], run_a.truth[keep], run_a.snr)
    run_b = RunPosterior(run_b.name, run_b.path, run_b.samples[:, keep], [run_b.labels[idx] for idx in keep], run_b.truth[keep], run_b.snr)

    clean_labels = [label.replace("source 1 ", "") for label in run_a.labels]
    latex_labels = [_format_axis_label(lbl) for lbl in clean_labels]

    fig = corner.corner(
        run_a.samples,
        labels=latex_labels,
        truths=run_a.truth,
        truth_color="black",
        color="tab:blue",
        alpha=0.5,
        plot_datapoints=False,
        smooth=1.0,
    )
    corner.corner(
        run_b.samples,
        fig=fig,
        labels=latex_labels,
        truths=None,
        color="tab:orange",
        alpha=0.5,
        plot_datapoints=False,
        smooth=1.0,
    )
    from matplotlib.patches import Patch

    axes = np.asarray(fig.axes).reshape((len(run_a.labels), len(run_a.labels)))
    legend_ax = axes[0, -1]
    _configure_axes_formatting(fig, len(run_a.labels), run_a.truth)
    legend_ax.legend(
        handles=[
            Patch(facecolor="tab:blue", alpha=0.5, label=f"{run_a.name} ($\\rho={run_a.snr:.2f}$)"),
            Patch(facecolor="tab:orange", alpha=0.5, label=f"{run_b.name} ($\\rho={run_b.snr:.2f}$)"),
            plt.Line2D([0], [0], color="black", ls="-", lw=1.5, label="Truth"),
        ],
        loc="upper left",
        fontsize=14,
        frameon=False,
        fancybox=False,
        framealpha=0.0,
    )
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path
