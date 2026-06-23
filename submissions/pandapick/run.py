#!/usr/bin/env python
"""PandaPick — entry point.

  python run.py                # run benchmark + generate the demonstration dataset
  python run.py --episodes 20  # number of seeds
  python run.py --quick        # quick smoke run
  python run.py --demo         # render the demo video
  python run.py --ablation     # closed-loop vs open-loop grasp force control (measured)
  python run.py --audit        # re-runnable honesty audit of the closed-loop claims
"""
from __future__ import annotations
import argparse
import time


def main():
    from pandapick.benchmark import TASK_SUITE
    n_full = len(TASK_SUITE)
    ap = argparse.ArgumentParser(description="PandaPick closed-loop force-regulated manipulation cell")
    ap.add_argument("--tasks", type=int, default=None, help=f"limit to first N tasks (default: full {n_full})")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--ablation", action="store_true", help="closed-loop vs open-loop grasp force control")
    ap.add_argument("--fragile", action="store_true", help="fragile force-budget: closed INTACT vs open CRACKED")
    ap.add_argument("--audit", action="store_true", help="re-runnable honesty audit of the closed-loop claims")
    args = ap.parse_args()

    if args.demo:
        from pandapick.record_demo import record
        print(f"[OK] Video: {record()}")
        return

    if args.audit:
        import audit
        audit.main()
        return

    if args.ablation:
        import json
        from pandapick.benchmark import run_ablation
        print("[START] closed-loop vs open-loop grasp force control (identical seeds)")
        print(json.dumps(run_ablation(), indent=2))
        return

    if args.fragile:
        import json
        from pandapick.benchmark import run_fragile_budget
        print("[START] fragile force-budget: closed gentle (INTACT) vs open binary slam (CRACKED), identical seeds")
        print(json.dumps(run_fragile_budget(), indent=2))
        return

    from pandapick.benchmark import run_all
    n = 3 if args.quick else args.tasks
    print(f"[START] PandaPick — {n_full if n is None else n}-task benchmark")
    t0 = time.time()
    summary, rows = run_all(n_episodes=n)
    print("\n========== SUMMARY ==========")
    for k, v in summary.items():
        print(f"  {k:26s}: {v}")
    print(f"  elapsed: {time.time()-t0:.1f}s")
    print("  output: results/benchmark.json + demo_dataset.npz")


if __name__ == "__main__":
    main()
