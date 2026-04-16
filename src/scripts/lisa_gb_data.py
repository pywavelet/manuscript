from __future__ import annotations

import paths

from lisa_gb_support import align_common_labels, create_infographic, load_run, plot_corner


def main() -> None:
    paths.figures.mkdir(parents=True, exist_ok=True)
    create_infographic(
        injection_path=paths.data / "injection.npz",
        output_path=paths.figures / "lisa_gb_demo_data.pdf",
    )
    run_a = load_run(paths.data / "wdm_posteriors.npz", "WDM")
    run_b = load_run(paths.data / "freq_posteriors.npz", "Frequency")
    run_a, run_b = align_common_labels(run_a, run_b)
    plot_corner(run_a, run_b, paths.figures / "lisa_gb_demo_corner.pdf")


if __name__ == "__main__":
    main()
