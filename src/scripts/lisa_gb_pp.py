from __future__ import annotations

import paths

from lisa_gb_support import load_pp, plot_pp


def main() -> None:
    paths.figures.mkdir(parents=True, exist_ok=True)
    pp = load_pp(paths.data / "lisa_gb_pp.npz")
    plot_pp(pp, paths.figures / "lisa_gb_demo_pp.pdf")


if __name__ == "__main__":
    main()
