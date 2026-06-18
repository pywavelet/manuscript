import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import paths


# -----------------------
# Load data
# -----------------------
# Prefer the manuscript-local benchmark CSVs (src/data) so the build is
# self-contained and reproducible; fall back to the pinned release copies.
BASE_URL = "https://raw.githubusercontent.com/pywavelet/wdm_transform/v0.5.0/docs/_static"


def _load_benchmark(name: str) -> pd.DataFrame:
    local = paths.data / name
    if local.exists():
        return pd.read_csv(local)
    return pd.read_csv(f"{BASE_URL}/{name}")


data = _load_benchmark("benchmark_data.csv")
fft_data = _load_benchmark("benchmark_fft_data.csv")

data["scalar_s"] = data["scalar_ms"] / 1000.0
fft_data["scalar_s"] = fft_data["scalar_ms"] / 1000.0
data["speedup_batch_vs_serial"] = data["serial_ms"] / data["batch_ms"]

def make_label(row):
    if row["library"] == "numpy":
        return "NumPy"
    elif row["library"] == "jax" and row["device"] == "CPU":
        return "JAX [CPU]"
    elif row["library"] == "jax" and row["device"] == "GPU":
        return "JAX [GPU]"
    else:
        return f"{row['library']} [{row['device']}]"

data["label"] = data.apply(make_label, axis=1)
fft_data["label"] = fft_data.apply(make_label, axis=1)

order = ["NumPy", "JAX [CPU]", "JAX [GPU]"]


# -----------------------
# Paper-style plotting
# -----------------------
plt.rcParams.update({
    "font.size": 9,
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "legend.fontsize": 8,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "lines.linewidth": 1.4,
    "axes.linewidth": 0.8,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "savefig.dpi": 300,
})
colors = {
    "NumPy": "#4C72B0",      
    "JAX [CPU]": "#DD8452",  
    "JAX [GPU]": "#994B4B",  
}

fig, (ax, ax_sp) = plt.subplots(
    2,
    1,
    figsize=(3.5, 3.7),
    sharex=True,
    gridspec_kw={"height_ratios": [3.0, 1.0], "hspace": 0.08},
)

# -----------------------
# Top panel: single-transform (scalar) runtime
# -----------------------
for label in order:
    df = data[data["label"] == label].sort_values("N")
    if df.empty:
        continue

    ax.loglog(
        df["N"],
        df["scalar_s"],
        marker="o",
        ms=3.2,
        lw=1.6,
        label=label,
        color=colors.get(label, "black"),
    )

# FFT overlay (same color per backend, dashed)
for label in order:
    df = fft_data[fft_data["label"] == label].sort_values("N")
    if df.empty:
        continue
    ax.loglog(
        df["N"],
        df["scalar_s"],
        marker="s",
        ms=2.8,
        lw=1.2,
        ls="--",
        color=colors.get(label, "black"),
        alpha=0.85,
    )

# Legend handles for solid=WDM vs dashed=FFT
from matplotlib.lines import Line2D
style_handles = [
    Line2D([0], [0], color="0.25", lw=1.6, ls="-", label="WDM"),
    Line2D([0], [0], color="0.25", lw=1.2, ls="--", label="FFT"),
]

# N log2 N reference (the dominant length-N FFT term at fixed nt), anchored to
# the final NumPy CPU point.
ref_df = data[data["label"] == "NumPy"].sort_values("N")
N_ref = ref_df["N"].to_numpy()
t_ref = ref_df["scalar_s"].to_numpy()
ref = N_ref * np.log2(N_ref)
ref = ref / ref[-1] * t_ref[-1]

ax.loglog(
    N_ref,
    ref,
    color='black',
    ls="-",
    lw=1.0,
    alpha=0.75,
    label=r"$N\log_2 N$",
)

ax.set_ylabel(r"runtime [s]")
ax.grid(True, which="major", ls=":", lw=0.5, alpha=0.45)
ax.grid(False, which="minor")
# Single combined legend above the axes to avoid overlapping the curves
backend_handles, backend_labels = ax.get_legend_handles_labels()
ax.legend(
    backend_handles + style_handles,
    backend_labels + [h.get_label() for h in style_handles],
    frameon=False,
    loc="lower center",
    bbox_to_anchor=(0.5, 1.0),
    ncol=3,
    columnspacing=1.2,
    handlelength=1.8,
    borderaxespad=0.2,
)

# -----------------------
# Bottom panel: batching speedup
# -----------------------
for label in order:
    df = data[data["label"] == label].sort_values("N")
    if df.empty:
        continue

    ax_sp.semilogx(
        df["N"],
        df["speedup_batch_vs_serial"],
        marker="o",
        ms=3.0,
        lw=1.6,
        color=colors.get(label, "black"),
    )

# ax_sp.axhline(1.0, ls="--", lw=0.9, alpha=0.7)

ax_sp.set_xlabel(r"input size $N$")
ax_sp.set_ylabel("speed-up")
ax_sp.grid(True, which="major", ls=":", lw=0.5, alpha=0.45)
ax_sp.grid(False, which="minor")
ax_sp.set_xlim(N_ref.min(), N_ref.max())
fig.align_ylabels()

fig.savefig(paths.figures / "runtimes.pdf", dpi=300, bbox_inches="tight")

plt.close(fig)
