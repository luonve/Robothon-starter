"""Run a multi-task benchmark (pick-and-place + colour sorting) and collect a demo dataset.

Outputs results/benchmark.json, results/benchmark.csv and results/demo_dataset.npz.
Every figure is measured from the MuJoCo rollout; nothing is hard-coded.
"""
from __future__ import annotations
import json
import os
import numpy as np

from .pipeline import run_episode

RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
TASKS = ["pick_place", "sort"]


def run_all(n_episodes: int = 8, n_cubes: int = 3, save_dataset: bool = True, verbose: bool = True):
    rows = []
    ds = {"qpos": [], "qvel": [], "ee_pos": [], "grip": [], "cube_pos": [],
          "action_qtarget": [], "phase": [], "episode": [], "task": []}
    ep = 0
    for seed in range(n_episodes):
        for task in TASKS:
            res, recs, _ = run_episode(seed, task=task, n=n_cubes, log=save_dataset)
            res["episode"] = ep
            rows.append(res)
            for r in recs:
                ds["qpos"].append(r["qpos"]); ds["qvel"].append(r["qvel"])
                ds["ee_pos"].append(r["ee_pos"]); ds["grip"].append(r["grip"])
                ds["cube_pos"].append(r["cube_pos"]); ds["action_qtarget"].append(r["action_qtarget"])
                ds["phase"].append(r["phase"]); ds["episode"].append(ep); ds["task"].append(task)
            if verbose:
                print(f"  ep {ep:02d} [{task:10s}] picked={res['picked']}/{n_cubes} success={res['success']}/{n_cubes}")
            ep += 1

    summary = summarize(rows, ds, n_cubes)
    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, "benchmark.json"), "w", encoding="utf-8") as fp:
        json.dump({"summary": summary, "episodes": rows}, fp, indent=2, ensure_ascii=False)
    _write_csv(rows)
    if save_dataset:
        _save_dataset(ds)
    return summary, rows


def summarize(rows, ds, n_cubes):
    pp = [r for r in rows if r["task"] == "pick_place"]
    so = [r for r in rows if r["task"] == "sort"]
    tot_cubes = lambda rs: max(1, len(rs) * n_cubes)
    errs = [r["mean_place_err_m"] for r in pp if r.get("mean_place_err_m") is not None]
    return {
        "n_episodes": len(rows),
        "tasks": TASKS,
        "pick_success_rate": round(sum(r["picked"] for r in rows) / max(1, len(rows) * n_cubes), 3),
        "place_success_rate": round(sum(r["success"] for r in pp) / tot_cubes(pp), 3),
        "sort_accuracy": round(sum(r["success"] for r in so) / tot_cubes(so), 3),
        "mean_place_err_mm": round(float(np.mean(errs)) * 1000, 1) if errs else None,
        "dataset_steps": len(ds["qpos"]),
        "obs_dim": 14,    # qpos7 + ee3 + grip1 + cube3
        "action_dim": 7,
    }


def _save_dataset(ds):
    arrs = {k: np.array(v) for k, v in ds.items() if k not in ("phase", "task")}
    arrs["phase"] = np.array(ds["phase"]); arrs["task"] = np.array(ds["task"])
    np.savez_compressed(os.path.join(RESULTS, "demo_dataset.npz"), **arrs)


def _write_csv(rows):
    import csv
    keys = ["episode", "seed", "task", "picked", "success"]
    with open(os.path.join(RESULTS, "benchmark.csv"), "w", newline="", encoding="utf-8") as fp:
        w = csv.DictWriter(fp, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in keys})
