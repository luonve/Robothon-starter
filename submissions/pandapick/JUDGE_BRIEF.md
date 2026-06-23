# PandaPick — Judge Brief (60-second review path)

**One-line:** a Franka Panda that autonomously solves a 15-task pick-place / colour-sort suite at
**100%** and logs every step as a labelled imitation dataset — manipulation reframed as
**demonstration-data generation**.

## Inspect first (everything is one command: `python run.py`)

1. **`results/pandapick_demo.mp4`** (~4 MB, ~76 s) + `results/pandapick_narration.srt` + `results/keyframes.png` (8-shot storyboard) — cinematic, with a live HUD; every on-screen number is read from the sim.
2. **`results/benchmark.json`** — every headline number, measured (no hand-written values).
3. **`python validate_submission.py`** — re-checks the numbers cross-file and the video gate; prints `ALL CHECKS PASS`.

## Headline numbers (all measured from MuJoCo rollouts)

| Metric                  | Value                                                                                      |
| ----------------------- | ------------------------------------------------------------------------------------------ |
| 15-task suite success   | **15 / 15 = 100%**                                                                         |
| Mean placement error    | **13.3 mm**                                                                                |
| Grasp stability         | holds a **5 N** disturbance = **19.9× the object weight**                                  |
| Ablation (design value) | interpolated trajectory **100%** vs hard joint slew **53.3%** place success → **+46.7 pp** |
| Labelled dataset        | **139,960** state-action steps (obs 14-D, action 7-D)                                      |
| Compute                 | CPU only, no GPU, one command                                                              |

## Rubric → evidence

| Axis         | Evidence                                                                                                         |
| ------------ | ---------------------------------------------------------------------------------------------------------------- |
| Runnability  | `run.py` (one CPU command) + `validate_submission.py`                                                            |
| MuJoCo depth | `model.py` — `MjSpec`, `mj_jacSite` Jacobian IK, free-body contact, external-force disturbance, offscreen render |
| Task design  | `benchmark.py` 15-task suite (pick-place, colour-sort, multi-object), 100% solved                                |
| Control      | `control.py` resolved-rate (Jacobian) IK + interpolated trajectories (ablation: +46.7 pp)                        |
| Dexterity    | grasp → transport → place of randomized objects; grasp holds 19.9× object weight                                 |
| Engineering  | small typed modules, pinned deps, vendored model untouched, self-validator                                       |
| Presentation | `pandapick_demo.mp4` (cinematic, ~4 MB) + `keyframes.png` + `narration.srt` + plots                              |
| Innovation   | manipulation as a **labelled demonstration-data pipeline**                                                       |

## Integrity (what is deterministic vs measured)

- The expert is a **scripted resolved-rate IK controller** — it is the _labeller / demonstration generator_, by design (not an RL policy; we do not claim one).
- Cubes are genuine **freejoint** bodies — **no teleport, no weld shortcut**. Every HUD/README number is read live from `mj_forward` / `qpos` / `mj_jacSite` / `mj_contactForce`.
- `validate_submission.py` enforces that the README cites the same values as `benchmark.json`.
