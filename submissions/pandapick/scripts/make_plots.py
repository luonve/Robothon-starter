"""Generate figures from real rollouts -> results/*.png for the README."""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pandapick.pipeline import run_episode
from pandapick.model import SORT_BINS, STACK_PAD

RES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
TEAL, AMBER = "#2dd4bf", "#f59e0b"


def main(seeds=6):
    pp = [run_episode(s, "pick_place", 3, log=False)[0] for s in range(seeds)]
    so = [run_episode(s, "sort", 3, log=False)[0] for s in range(seeds)]

    # 1) per-episode success (place vs sort)
    fig, ax = plt.subplots(figsize=(7, 3.4), dpi=130)
    x = np.arange(seeds)
    ax.bar(x - 0.2, [r["success"] for r in pp], 0.4, label="pick & place", color=TEAL)
    ax.bar(x + 0.2, [r["success"] for r in so], 0.4, label="colour sort", color=AMBER)
    ax.axhline(3, ls="--", color="#888", lw=1, label="all 3 cubes")
    ax.set_xlabel("seed"); ax.set_ylabel("cubes succeeded / 3"); ax.set_ylim(0, 3.4)
    ax.set_title("PandaPick — per-episode success across two jobs")
    ax.legend(fontsize=8, loc="lower right"); ax.grid(alpha=0.3, axis="y"); fig.tight_layout()
    p1 = os.path.join(RES, "accuracy_plot.png"); fig.savefig(p1); plt.close(fig); print("wrote", p1)

    # 2) workspace map: cube starts (feeders) + destinations
    fig, ax = plt.subplots(figsize=(6.4, 5.0), dpi=130)
    for s in range(seeds):
        _, _, c = run_episode(s, "sort", 3, log=False) if False else (None, None, None)
    # plot feeder samples + bins
    from pandapick.model import sample_scene
    for s in range(seeds):
        sc = sample_scene(s, "sort", 3)
        for f in sc["feeders"]:
            ax.scatter(*f["xy"], c={"R": "#dc5a46", "G": "#5ac878", "B": "#5a82e6"}[f["color"]],
                       s=60, edgecolors="k", linewidths=0.4)
    for col, (bx, by) in SORT_BINS.items():
        ax.scatter(bx, by, marker="s", s=260, c={"R": "#dc5a46", "G": "#5ac878", "B": "#5a82e6"}[col],
                   edgecolors="k", label=f"bin {col}")
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)"); ax.set_aspect("equal")
    ax.set_title("Workspace — randomized cubes (top) sorted to colour bins (bottom)")
    ax.legend(fontsize=8, loc="upper right"); ax.grid(alpha=0.3); fig.tight_layout()
    p2 = os.path.join(RES, "coverage_plot.png"); fig.savefig(p2); plt.close(fig); print("wrote", p2)


if __name__ == "__main__":
    main()
