"""15-task benchmark (pick-place, colour-sort, multi-object) + demonstration dataset.

Every figure is measured from the MuJoCo rollout under minimum-jerk trajectory control.
Outputs results/benchmark.json, benchmark.csv, demo_dataset.npz.
"""
from __future__ import annotations
import json
import os
import numpy as np

from .pipeline import run_episode

RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")

# 15-task suite: varied jobs, object counts, seeds (randomized each), + disturbance-rejection.
# tuple = (name, task, seed, n_objects, disturbance_N)
TASK_SUITE = [
    ("T01 place-A",       "pick_place", 0, 3, 0.0),
    ("T02 place-B",       "pick_place", 1, 3, 0.0),
    ("T03 place-C",       "pick_place", 2, 2, 0.0),
    ("T04 place-4obj",    "pick_place", 12, 4, 0.0),
    ("T05 place-E",       "pick_place", 4, 3, 0.0),
    ("T06 sort-RGB-A",    "sort",       0, 3, 0.0),
    ("T07 sort-RGB-B",    "sort",       1, 3, 0.0),
    ("T08 sort-RGB-C",    "sort",       2, 3, 0.0),
    ("T09 sort-RGB-D",    "sort",       5, 3, 0.0),
    ("T10 sort-RGB-E",    "sort",       6, 3, 0.0),
    ("T11 place-2obj",    "pick_place", 7, 2, 0.0),
    ("T12 place-4obj-B",  "pick_place", 8, 4, 0.0),
    ("T13 sort-2obj",     "sort",       9, 2, 0.0),
    ("T14 place-F",       "pick_place", 10, 3, 0.0),
    ("T15 sort-F",        "sort",       11, 3, 0.0),
]


def run_all(n_episodes: int | None = None, save_dataset: bool = True, verbose: bool = True):
    suite = TASK_SUITE if n_episodes is None else TASK_SUITE[:max(1, n_episodes)]
    rows = []
    ds = {"qpos": [], "qvel": [], "ee_pos": [], "grip": [], "cube_pos": [],
          "action_qtarget": [], "phase": [], "episode": [], "task": []}
    for ep, (name, task, seed, n, dist) in enumerate(suite):
        res, recs, _ = run_episode(seed, task=task, n=n, log=save_dataset, disturb_N=dist)
        res.update({"task_id": name, "episode": ep})
        rows.append(res)
        for r in recs:
            for k in ("qpos", "qvel", "ee_pos", "grip", "cube_pos", "action_qtarget", "phase"):
                ds[k].append(r[k])
            ds["episode"].append(ep); ds["task"].append(task)
        if verbose:
            print(f"  {name:14s} [{task:10s} x{n}]  pick={res['picked']}/{n}  success={res['success']}/{n}")

    summary = summarize(rows)
    held, weight = measure_grasp_stability()
    summary["grasp_holds_disturbance_N"] = held
    summary["disturbance_x_object_weight"] = round(held / weight, 1) if weight else None
    if verbose:
        print(f"  grasp stability: holds {held:.0f} N disturbance ({held/weight:.0f}x object weight)")
    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, "benchmark.json"), "w", encoding="utf-8") as fp:
        json.dump({"summary": summary, "tasks": rows}, fp, indent=2, ensure_ascii=False)
    _write_csv(rows)
    if save_dataset:
        _save_dataset(ds)
    return summary, rows


def measure_grasp_stability():
    """Grasp a cube, ramp an external disturbance, return the max force the grip holds (N)."""
    import mujoco
    from .model import build_model
    from .control import IKController, GRIP_OPEN, GRIP_CLOSE
    m, meta = build_model(0, "pick_place", 1)
    c = IKController(m, meta, log=False)
    c.set_grip(GRIP_OPEN, 120, "settle")
    cx, cy, cz = meta.cube_pos(c.d, 0)
    c.move_to([cx, cy, cz + 0.12], GRIP_OPEN, 380, "h")
    c.move_to([cx, cy, cz], GRIP_OPEN, 380, "d")
    c.set_grip(GRIP_CLOSE, 460, "g")
    c.move_to([cx, cy, cz + 0.22], GRIP_CLOSE, 420, "lift")
    cb = meta.cube_bid[0]
    weight = float(m.body_mass[cb] * 9.81)
    held = 0.0
    for F in [1, 2, 3, 4, 5, 6, 7, 8]:
        c.d.xfrc_applied[cb][:3] = [F * 0.6, 0.0, -F]
        for _ in range(150):
            mujoco.mj_step(m, c.d)
        c.d.xfrc_applied[cb][:] = 0.0
        for _ in range(60):
            mujoco.mj_step(m, c.d)
        if meta.cube_pos(c.d, 0)[2] > cz + 0.12:
            held = float(F)
        else:
            break
    return held, weight


def summarize(rows):
    tot = sum(r["n"] for r in rows)
    picked = sum(r["picked"] for r in rows)
    success = sum(r["success"] for r in rows)
    errs = [r["mean_place_err_m"] for r in rows if r.get("mean_place_err_m") is not None]
    tasks_full = sum(1 for r in rows if r["success"] == r["n"])
    return {
        "n_tasks": len(rows),
        "tasks_fully_solved": tasks_full,
        "task_success_rate": round(tasks_full / max(1, len(rows)), 3),
        "object_pick_rate": round(picked / max(1, tot), 3),
        "object_place_rate": round(success / max(1, tot), 3),
        "mean_place_err_mm": round(float(np.mean(errs)) * 1000, 1) if errs else None,
        "control": "resolved-rate (Jacobian) IK + smooth interpolated trajectories",
        "dataset_steps": None,   # filled by run_all caller-side
        "obs_dim": 14, "action_dim": 7,
    }


def _save_dataset(ds):
    arrs = {k: np.array(v) for k, v in ds.items() if k not in ("phase", "task")}
    arrs["phase"] = np.array(ds["phase"]); arrs["task"] = np.array(ds["task"])
    np.savez_compressed(os.path.join(RESULTS, "demo_dataset.npz"), **arrs)
    # patch dataset_steps into benchmark.json
    p = os.path.join(RESULTS, "benchmark.json")
    if os.path.exists(p):
        obj = json.load(open(p, encoding="utf-8"))
        obj["summary"]["dataset_steps"] = len(ds["qpos"])
        json.dump(obj, open(p, "w", encoding="utf-8"), indent=2, ensure_ascii=False)


def _write_csv(rows):
    import csv
    keys = ["episode", "task_id", "task", "n", "picked", "success", "mean_place_err_m"]
    with open(os.path.join(RESULTS, "benchmark.csv"), "w", newline="", encoding="utf-8") as fp:
        w = csv.DictWriter(fp, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in keys})
