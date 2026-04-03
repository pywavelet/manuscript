from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np

import paths


def _load_results() -> dict:
    with (paths.static / "benchmark_results.json").open("r", encoding="utf-8") as stream:
        return json.load(stream)


def _series(results: dict, section: str, backend: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    records = results[section][backend]
    n_values = np.array(sorted(int(key) for key in records), dtype=float)
    means = np.array([records[str(int(n))]["mean_seconds"] for n in n_values], dtype=float)
    stds = np.array([records[str(int(n))]["std_seconds"] for n in n_values], dtype=float)
    return n_values, means * 1e3, stds * 1e3


def _nlogn_fit(n_values: np.ndarray, means_ms: np.ndarray) -> np.ndarray:
    basis = n_values * np.log2(n_values)
    coefficient = float(np.dot(basis, means_ms) / np.dot(basis, basis))
    return coefficient * basis


def main() -> None:
    results = _load_results()
    paths.figures.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )

    fig, ax = plt.subplots(figsize=(7.0, 4.4), constrained_layout=True)

    colors = {"numpy": "#c44e52", "jax": "#4c72b0"}
    labels = {"numpy": "NumPy", "jax": "JAX"}
    endpoint_text = {}

    for backend in results["forward"]:
        n_values, means_ms, stds_ms = _series(results, "forward", backend)
        ax.plot(
            n_values,
            means_ms,
            marker="o",
            linewidth=2.2,
            markersize=4.5,
            color=colors.get(backend, "0.3"),
            label=labels.get(backend, backend),
        )
        ax.plot(
            n_values,
            _nlogn_fit(n_values, means_ms),
            linewidth=1.5,
            linestyle="--",
            color=colors.get(backend, "0.3"),
            alpha=0.85,
            label="_nolegend_",
        )
        ax.fill_between(
            n_values,
            np.maximum(means_ms - stds_ms, 1e-6),
            means_ms + stds_ms,
            color=colors.get(backend, "0.3"),
            alpha=0.14,
        )
        endpoint_text[backend] = (n_values[-1], means_ms[-1])

    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("data length $N$")
    ax.set_ylabel("forward runtime [ms]")
    ax.grid(True, which="major", alpha=0.18, linewidth=0.8)
    ax.grid(True, which="minor", alpha=0.08, linewidth=0.5)
    ax.legend(frameon=False, loc="upper left")
    ax.set_xlim(1800, 1.3e6)

    x_numpy, y_numpy = endpoint_text["numpy"]
    x_jax, y_jax = endpoint_text["jax"]
    ax.annotate(
        f"{y_jax:.1f} ms",
        xy=(x_jax, y_jax),
        xytext=(-36, -4),
        textcoords="offset points",
        color=colors["jax"],
        fontsize=10,
        ha="right",
    )
    ax.annotate(
        f"{y_numpy:.1f} ms",
        xy=(x_numpy, y_numpy),
        xytext=(-36, 8),
        textcoords="offset points",
        color=colors["numpy"],
        fontsize=10,
        ha="right",
    )

    speedup = y_numpy / y_jax
    ax.text(
        0.98,
        0.06,
        f"largest-$N$ speedup: {speedup:.1f}$\\times$",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=10,
    )
    ax.text(
        0.02,
        0.06,
        r"dashed: least-squares $c\,N\log_2 N$ fit",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=9.5,
        color="0.35",
    )

    fig.savefig(paths.figures / "runtimes.pdf", bbox_inches="tight")


if __name__ == "__main__":
    main()
