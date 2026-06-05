from __future__ import annotations

import paths

from lisa_gb_support import load_demo, plot_corner


def main() -> None:
    paths.figures.mkdir(parents=True, exist_ok=True)
    demo = load_demo(paths.data / "lisa_gb_demo.npz")
    plot_corner(demo, paths.figures / "lisa_gb_demo_corner.pdf")


if __name__ == "__main__":
    main()
