from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np

import paths


def _load_results() -> dict:
    with (paths.static / "benchmark_results.json").open("r", encoding="utf-8") as stream:
        return json.load(stream)


def _series(results: dict, backend: str) -> tuple[np.ndarray, np.ndarray]:
    records = results["error"][backend]
    n_values = np.array(sorted(int(key) for key in records), dtype=float)
    errors = np.array([records[str(int(n))]["max_abs_error"] for n in n_values], dtype=float)
    return n_values, errors


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

    for backend in results["error"]:
        n_values, errors = _series(results, backend)
        ax.plot(
            n_values,
            errors,
            marker="o",
            linewidth=2.0,
            markersize=4.0,
            color=colors.get(backend, "0.3"),
            label=labels.get(backend, backend),
        )

    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("data length $N$")
    ax.set_ylabel("max absolute round-trip error")
    ax.grid(True, which="major", alpha=0.18, linewidth=0.8)
    ax.grid(True, which="minor", alpha=0.08, linewidth=0.5)
    ax.legend(frameon=False, loc="upper left")
    ax.set_xlim(1800, 1.3e6)
    ax.axhspan(1e-16, 1e-10, color="0.75", alpha=0.15)
    ax.text(
        0.98,
        0.08,
        "roundoff-scale regime",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=10,
        color="0.35",
    )

    fig.savefig(paths.figures / "mse.pdf", bbox_inches="tight")


if __name__ == "__main__":
    main()
