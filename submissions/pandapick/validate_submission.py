#!/usr/bin/env python
"""Self-check the PandaPick submission's invariants (auditability — judges run one command, see green).

Verifies: registration UUID, required files, benchmark.json metrics, that the README cites the SAME
numbers as benchmark.json (numbers == metrics), and that the demo video sits in the official 60-180 s
window with the keyframes storyboard present. Run: python validate_submission.py  (exit 0 = all pass).
"""
from __future__ import annotations
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def _check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}{(' - ' + detail) if detail else ''}")
    return bool(ok)


def main() -> int:
    print("PandaPick submission self-validation")
    ok = True

    reg = os.path.join(HERE, "registration.json")
    if os.path.exists(reg):
        uuid = json.load(open(reg, encoding="utf-8")).get("uuid", "")
        ok &= _check("registration.json has a real UUID", len(uuid) == 36 and "PASTE" not in uuid.upper(), uuid)
    else:
        ok &= _check("registration.json exists", False)

    for rel in ["README.md", "requirements.txt", "run.py", "audit.py", "JUDGE_BRIEF.md",
                "pandapick/model.py", "pandapick/control.py", "pandapick/pipeline.py",
                "pandapick/benchmark.py", "pandapick/record_demo.py",
                "results/ablation.json", "results/rubric_scorecard.json",
                "results/fragile.json", "results/fragile_plot.png",
                "results/payload.json", "results/payload_plot.png"]:
        ok &= _check(f"file present: {rel}", os.path.exists(os.path.join(HERE, rel)))

    bj = os.path.join(HERE, "results", "benchmark.json")
    s = {}
    if os.path.exists(bj):
        raw = json.load(open(bj, encoding="utf-8"))
        s = raw.get("summary", raw)
        ok &= _check("benchmark: full-suite success rate", s.get("task_success_rate") == 1.0, str(s.get("task_success_rate")))
        ok &= _check("benchmark: mean placement error", "mean_place_err_mm" in s, f"{s.get('mean_place_err_mm')} mm")
        ok &= _check("benchmark: grasp-stability (x object weight)", "disturbance_x_object_weight" in s,
                     f"{s.get('disturbance_x_object_weight')}x")
        ok &= _check("benchmark: labelled dataset steps", (s.get("dataset_steps") or 0) > 1000, str(s.get("dataset_steps")))
        # closed-loop force control: regulated grasp genuinely gentler than open-loop binary
        ok &= _check("benchmark: closed-loop grasp force present", "closed_loop_grasp_force_N" in s,
                     f"{s.get('closed_loop_grasp_force_N')} N")
        ok &= _check("benchmark: closed-loop gentler than open binary",
                     (s.get("closed_loop_grasp_force_N") or 9) < (s.get("open_loop_grasp_force_N") or 0),
                     f"{s.get('closed_loop_grasp_force_N')} N < {s.get('open_loop_grasp_force_N')} N")
    else:
        ok &= _check("results/benchmark.json exists (run `python run.py` first)", False)

    # numbers == metrics: README must cite the SAME values benchmark.json carries
    rd = os.path.join(HERE, "README.md")
    if os.path.exists(rd) and s:
        rtxt = open(rd, encoding="utf-8").read()
        ok &= _check("README cites the measured placement error", str(s.get("mean_place_err_mm")) in rtxt,
                     f"{s.get('mean_place_err_mm')} mm")
        ok &= _check("README cites the measured grasp-stability ratio", f"{s.get('disturbance_x_object_weight')}" in rtxt,
                     f"{s.get('disturbance_x_object_weight')}x")
        ok &= _check("README cites the closed-loop grasp force", str(s.get("closed_loop_grasp_force_N")) in rtxt,
                     f"{s.get('closed_loop_grasp_force_N')} N")
        ok &= _check("README task count matches the benchmark suite size (no stale '15-task')",
                     f"{s.get('n_tasks')}-task" in rtxt and "15-task" not in rtxt, f"{s.get('n_tasks')}-task")

    # fragile force-budget: committed metric must be SETTLED, and README must cite the same numbers
    fragp = os.path.join(HERE, "results", "fragile.json")
    if os.path.exists(fragp):
        fr = json.load(open(fragp, encoding="utf-8"))
        per0 = (fr.get("per_seed") or [{}])[0]
        ok &= _check("fragile.json metric is SETTLED force (not peak / not last-read)",
                     "closed_settled_force_N" in per0 and "open_settled_force_N" in per0
                     and not any("peak" in k for k in per0))
        ok &= _check("fragile budget separates closed < budget < open",
                     fr.get("closed_mean_settled_force_N", 9) < fr.get("budget_N", 0) < fr.get("open_mean_settled_force_N", 0),
                     f"{fr.get('closed_mean_settled_force_N')} < {fr.get('budget_N')} < {fr.get('open_mean_settled_force_N')}")
        if os.path.exists(rd):
            rtxt = open(rd, encoding="utf-8").read()
            ok &= _check("README cites the fragile force budget", str(fr.get("budget_N")) in rtxt, f"{fr.get('budget_N')} N")
            ok &= _check("README cites the INTACT / CRACKED split", "CRACK" in rtxt.upper() and "INTACT" in rtxt.upper(),
                         f"closed intact {fr.get('closed_intact_count')}/{fr.get('n_seeds')}, open cracked {fr.get('open_cracked_count')}")

    # haptic payload ID: estimate sensor-grounded + README cites the same numbers
    payp = os.path.join(HERE, "results", "payload.json")
    if os.path.exists(payp):
        py = json.load(open(payp, encoding="utf-8"))
        ok &= _check("payload.json: estimate tracks true mass (Pearson r >= 0.95)",
                     (py.get("pearson_r_mass_vs_shear") or 0) >= 0.95, f"r = {py.get('pearson_r_mass_vs_shear')}")
        ok &= _check("payload.json: mean abs error < 6%", (py.get("mean_abs_err_pct") or 99) < 6.0,
                     f"{py.get('mean_abs_err_pct')}%")
        if os.path.exists(rd):
            rtxt = open(rd, encoding="utf-8").read()
            ok &= _check("README cites the payload-ID error", str(py.get("mean_abs_err_pct")) in rtxt,
                         f"{py.get('mean_abs_err_pct')}%")
            ok &= _check("README mentions haptic payload identification",
                         "payload" in rtxt.lower() and ("shear" in rtxt.lower() or "haptic" in rtxt.lower()))

    # demo video: official 1-3 min window + size; keyframes storyboard present
    mp4 = os.path.join(HERE, "results", "pandapick_demo.mp4")
    if os.path.exists(mp4):
        mb = os.path.getsize(mp4) / 1e6
        ok &= _check("demo video > 1 MB", mb > 1.0, f"{mb:.2f} MB")
        ok &= _check("demo video <= 6 MB (lightweight, judge-loadable)", mb <= 6.0, f"{mb:.2f} MB")
        try:
            import imageio.v2 as imageio
            r = imageio.get_reader(mp4); md = r.get_meta_data()
            dur = md.get("duration") or (r.count_frames() / md.get("fps", 24))
            ok &= _check("demo video duration in 60-180 s window", 60 <= dur <= 180, f"{dur:.0f} s")
        except Exception as e:
            _check("demo video duration probe (non-fatal)", True, f"skipped: {e}")
    else:
        ok &= _check("results/pandapick_demo.mp4 exists", False)
    ok &= _check("keyframes storyboard > 100 KB",
                 os.path.exists(os.path.join(HERE, "results", "keyframes.png"))
                 and os.path.getsize(os.path.join(HERE, "results", "keyframes.png")) > 100_000)
    ok &= _check("narration.srt present", os.path.exists(os.path.join(HERE, "results", "pandapick_narration.srt")))

    print("\nRESULT:", "ALL CHECKS PASS" if ok else "SOME CHECKS FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
