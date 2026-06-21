#!/usr/bin/env python
"""PandaPick — entry point.

  python run.py                # run benchmark + generate the demonstration dataset
  python run.py --episodes 20  # number of seeds
  python run.py --quick        # quick smoke run
  python run.py --demo         # render the demo video
"""
from __future__ import annotations
import argparse
import time


def main():
    ap = argparse.ArgumentParser(description="PandaPick pick-and-place data-collection")
    ap.add_argument("--tasks", type=int, default=None, help="limit to first N tasks (default: full 15)")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--demo", action="store_true")
    args = ap.parse_args()

    if args.demo:
        from pandapick.record_demo import record
        print(f"[OK] Video: {record()}")
        return

    from pandapick.benchmark import run_all
    n = 3 if args.quick else args.tasks
    print(f"[START] PandaPick — {'15' if n is None else n}-task benchmark")
    t0 = time.time()
    summary, rows = run_all(n_episodes=n)
    print("\n========== SUMMARY ==========")
    for k, v in summary.items():
        print(f"  {k:26s}: {v}")
    print(f"  elapsed: {time.time()-t0:.1f}s")
    print("  output: results/benchmark.json + demo_dataset.npz")


if __name__ == "__main__":
    main()
