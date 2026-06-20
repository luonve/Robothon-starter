"""Ve bieu do tu ket qua benchmark that -> results/*.png cho README."""
import os, sys, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pandapick.pipeline import run_episode
from pandapick.model import BIN_POS

RES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")


def main(n=16):
    rows = [run_episode(s, log=False)[0] for s in range(n)]
    xy = np.array([r["scene"]["cube_xy"] for r in rows])
    errs = np.array([r["place_err_m"] * 1000 for r in rows])
    okp = np.array([r["place_ok"] for r in rows])

    # 1) workspace coverage (vi tri cube + bin), mau theo success
    fig, ax = plt.subplots(figsize=(6.4, 5.2), dpi=130)
    ax.scatter(xy[okp, 0], xy[okp, 1], c="#5ac882", s=90, label="place OK", edgecolors="k", linewidths=0.5)
    if (~okp).any():
        ax.scatter(xy[~okp, 0], xy[~okp, 1], c="#eb785a", s=90, label="place fail", edgecolors="k")
    ax.scatter([BIN_POS[0]], [BIN_POS[1]], marker="s", c="#7aa6e0", s=200, label="bin", edgecolors="k")
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)"); ax.set_aspect("equal")
    ax.set_title(f"PandaPick — randomized cube positions ({int(okp.sum())}/{n} placed)")
    ax.legend(fontsize=8, loc="upper right"); ax.grid(alpha=0.3); fig.tight_layout()
    p1 = os.path.join(RES, "coverage_plot.png"); fig.savefig(p1); plt.close(fig); print("wrote", p1)

    # 2) place accuracy
    fig, ax = plt.subplots(figsize=(6.4, 3.2), dpi=130)
    ax.bar(range(n), errs, color=["#5ac882" if o else "#eb785a" for o in okp])
    ax.axhline(errs.mean(), ls="--", color="#444", label=f"mean {errs.mean():.1f} mm")
    ax.set_xlabel("episode"); ax.set_ylabel("place error (mm)")
    ax.set_title("Placement accuracy per episode"); ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y"); fig.tight_layout()
    p2 = os.path.join(RES, "accuracy_plot.png"); fig.savefig(p2); plt.close(fig); print("wrote", p2)


if __name__ == "__main__":
    main()
