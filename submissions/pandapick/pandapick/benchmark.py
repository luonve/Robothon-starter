"""Chay N episode pick-and-place co domain randomization -> gom DATASET demo + metrics.

Xuat: results/benchmark.json (metrics) + results/demo_dataset.npz (obs/action cho imitation
learning) + results/benchmark.csv. Tat ca so do tu mo phong, khong hard-code.
"""
from __future__ import annotations
import json
import os
import numpy as np

from .pipeline import run_episode
from .model import sample_scene

RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")


def run_all(n_episodes: int = 12, save_dataset: bool = True, verbose: bool = True):
    rows = []
    ds = {"qpos": [], "qvel": [], "ee_pos": [], "grip": [], "cube_pos": [],
          "action_qtarget": [], "phase": [], "episode": []}
    for ep in range(n_episodes):
        res, recs, _ = run_episode(ep, log=save_dataset)
        rows.append(res)
        for r in recs:
            ds["qpos"].append(r["qpos"]); ds["qvel"].append(r["qvel"])
            ds["ee_pos"].append(r["ee_pos"]); ds["grip"].append(r["grip"])
            ds["cube_pos"].append(r["cube_pos"]); ds["action_qtarget"].append(r["action_qtarget"])
            ds["phase"].append(r["phase"]); ds["episode"].append(ep)
        if verbose:
            print(f"  ep {ep:02d}: pick={res['pick_ok']} place={res['place_ok']} "
                  f"err={res['place_err_m']*1000:.0f}mm")

    summary = summarize(rows, ds)
    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, "benchmark.json"), "w", encoding="utf-8") as fp:
        json.dump({"summary": summary, "episodes": rows}, fp, indent=2, ensure_ascii=False)
    _write_csv(rows)
    if save_dataset:
        _save_dataset(ds)
    return summary, rows


def summarize(rows, ds):
    n = len(rows)
    picks = sum(r["pick_ok"] for r in rows)
    places = sum(r["place_ok"] for r in rows)
    errs = [r["place_err_m"] for r in rows if r["place_ok"]]
    # phu vung lam viec (dien tich bao boi cac vi tri cube)
    xy = np.array([r["scene"]["cube_xy"] for r in rows])
    cov = float((xy[:, 0].max() - xy[:, 0].min()) * (xy[:, 1].max() - xy[:, 1].min())) if n > 1 else 0.0
    return {
        "n_episodes": n,
        "pick_success_rate": round(picks / max(1, n), 3),
        "place_success_rate": round(places / max(1, n), 3),
        "mean_place_err_mm": round(float(np.mean(errs)) * 1000, 1) if errs else None,
        "max_place_err_mm": round(float(np.max(errs)) * 1000, 1) if errs else None,
        "dataset_steps": len(ds["qpos"]),
        "workspace_coverage_m2": round(cov, 4),
        "obs_dim": 13,   # qpos7 + ee3 + grip1 + cube? (obs = qpos+ee+grip+cube_pos)
        "action_dim": 7,
    }


def _save_dataset(ds):
    arrs = {k: np.array(v) for k, v in ds.items() if k != "phase"}
    arrs["phase"] = np.array(ds["phase"])
    np.savez_compressed(os.path.join(RESULTS, "demo_dataset.npz"), **arrs)


def _write_csv(rows):
    import csv
    keys = ["seed", "pick_ok", "place_ok", "place_err_m", "n_records"]
    with open(os.path.join(RESULTS, "benchmark.csv"), "w", newline="", encoding="utf-8") as fp:
        w = csv.DictWriter(fp, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in keys})
