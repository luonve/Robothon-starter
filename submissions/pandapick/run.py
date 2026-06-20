#!/usr/bin/env python
"""PandaPick — entry point.

  python run.py                # chay benchmark + sinh dataset demo
  python run.py --episodes 20  # so episode
  python run.py --quick        # nhanh 3 episode
  python run.py --demo         # render video demo
"""
from __future__ import annotations
import argparse
import time


def main():
    ap = argparse.ArgumentParser(description="PandaPick pick-and-place data-collection")
    ap.add_argument("--episodes", type=int, default=12)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--demo", action="store_true")
    args = ap.parse_args()

    if args.demo:
        from pandapick.record_demo import record
        print(f"[OK] Video: {record()}")
        return

    from pandapick.benchmark import run_all
    n = 3 if args.quick else args.episodes
    print(f"[BAT_DAU] PandaPick — {n} episode pick-and-place")
    t0 = time.time()
    summary, rows = run_all(n_episodes=n)
    print("\n========== TONG KET ==========")
    for k, v in summary.items():
        print(f"  {k:26s}: {v}")
    print(f"  thoi gian: {time.time()-t0:.1f}s")
    print("  Ket qua: results/benchmark.json + demo_dataset.npz")


if __name__ == "__main__":
    main()
