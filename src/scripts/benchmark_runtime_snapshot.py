import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import paths


# -----------------------
# Load data
# -----------------------
data = pd.read_csv(
    "https://raw.githubusercontent.com/pywavelet/wdm_transform/refs/heads/main/docs/_static/benchmark_data.csv"
)

data["label"] = (
    data["library"].str.lower().map({"numpy": "NumPy", "jax": "JAX"})
    + " "
    + data["device"]
)

data["batch_s"] = data["batch_ms"] / 1000.0
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
# Top panel: batched runtime
# -----------------------
for label in order:
    df = data[data["label"] == label].sort_values("N")
    if df.empty:
        continue

    ax.loglog(
        df["N"],
        df["batch_s"],
        marker="o",
        ms=3.2,
        lw=1.6,
        label=label,
        color=colors.get(label, "black"),
    )

# N log N reference, anchored to final NumPy CPU point
ref_df = data[data["label"] == "NumPy"].sort_values("N")
N_ref = ref_df["N"].to_numpy()
t_ref = ref_df["batch_s"].to_numpy()
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
ax.legend(frameon=False, loc="upper left", handlelength=2.2)

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
ax_sp.set_ylabel(r"$t_{\rm serial}/t_{\rm batch}$")
ax_sp.grid(True, which="major", ls=":", lw=0.5, alpha=0.45)
ax_sp.grid(False, which="minor")
ax_sp.set_xlim(N_ref.min(), N_ref.max())
fig.align_ylabels()

fig.savefig(paths.figures / "runtimes.pdf", dpi=300, bbox_inches="tight")

plt.close(fig)