<div align="center">

# PandaPick — Pick-and-Place Data-Collection Pipeline

**A Franka Emika Panda arm autonomously generates pick-and-place demonstrations with
domain randomization, and logs an (observation, action) dataset for imitation learning.**

`mujoco>=3.9` · pure-CPU · one-command run · 100% pick & place across randomized scenes

</div>

---

## TL;DR (all numbers measured from simulation, none hard-coded)

| Metric                       | Result                                 |
| ---------------------------- | -------------------------------------- |
| Pick success rate            | **100 %**                              |
| Place success rate           | **100 %**                              |
| Mean placement accuracy      | **18 mm** (max 26 mm)                  |
| Episodes (domain-randomized) | **12–16**, all succeeded               |
| Demonstration dataset        | **~41,760** logged (obs, action) steps |
| Hardware                     | CPU only, **no GPU**, full run < 10 s  |

Reproduce: `pip install -r requirements.txt && python run.py`

---

## What it is

This is a **data-collection system** (one of the contest's three themes) rather than a
single manipulation stunt. A Franka Panda runs a closed pick-and-place loop over
**domain-randomized** scenes (cube position, size, mass, colour) and records, at every
timestep, a clean **(observation, action)** trace — exactly the data an imitation-learning
policy would train on. It also reports how reliable and how diverse those demonstrations are.

## Robot platform

- **Arm:** Franka Emika Panda (7-DOF) + parallel gripper, from `mujoco_menagerie`.
- **Sensing/state:** joint positions & velocities, end-effector pose (via a grasp site on
  the hand), gripper state, and object pose — logged each step.
- **Scene:** a feeder post presents the cube; the arm transports it into a bin. Cube pose,
  size, mass and colour are randomized per episode.

## Task goal

Pick the randomly-placed cube off the feeder and place it in the bin — and while doing so,
**generate a labelled demonstration dataset** with measured success and accuracy.

## Technical approach

1. **Model built programmatically with the MuJoCo `MjSpec` API** — the Panda is loaded, a
   grasp site is attached to the hand, and feeder / cube / bin are composed in code with
   per-episode randomization. Vendored model stays untouched.
2. **Resolved-rate IK** — a damped-least-squares 6-DOF solver (`mj_jacSite`) drives the
   grasp site to a target position with the gripper pointing down. Solved in pure kinematics
   (`mj_forward`), so it converges to < 0.5 mm before any dynamics run.
3. **Smooth motion** — joint targets are interpolated from the current pose to the IK
   solution, so the arm never swings hard enough to fling the grasped cube (a real failure
   mode that interpolation fixed: 1/6 → 6/6 placement success).
4. **State machine** — HOVER → DESCEND → GRASP → LIFT → TRANSPORT → PLACE → RELEASE → RETRACT,
   each phase logged with its label for segmented learning.
5. **Dataset** — every step is written to `results/demo_dataset.npz` as aligned arrays
   (`qpos`, `qvel`, `ee_pos`, `grip`, `cube_pos`, `action_qtarget`, `phase`, `episode`).

## Core features

- **100 % autonomous pick-and-place** across domain-randomized scenes (no teleop, no tuning per scene).
- **Demonstration dataset export** ready for behaviour cloning (`.npz`, ~41k steps from 12 episodes).
- **Domain randomization** of object pose / size / mass / colour for data diversity.
- **Quantified reliability** — pick & place success, placement accuracy, workspace coverage.
- **Telemetry demo video** — rendered by running the code, with live phase tracker and state.

## Highlights

- Every randomized cube position is reached and placed in the bin:

  ![coverage](results/coverage_plot.png)

- Placement accuracy per episode (mean ~18 mm):

  ![accuracy](results/accuracy_plot.png)

## How this maps to the rubric

| Rubric dimension           | Where it shows up                                                                                                           |
| -------------------------- | --------------------------------------------------------------------------------------------------------------------------- |
| **Runnability**            | `pip install -r requirements.txt && python run.py`; pure CPU, deterministic seeds, < 10 s.                                  |
| **Depth of MuJoCo use**    | `MjSpec` model building, `mj_jacSite` Jacobian IK, free-joint object dynamics, contact-based grasping, offscreen rendering. |
| **Task design**            | A reusable **data-collection pipeline**, not a one-off demo — exactly the contest's "data-collection system" theme.         |
| **Control**                | Resolved-rate DLS IK + interpolated trajectory control; 6-DOF pose servoing with the gripper held down.                     |
| **Dexterous manipulation** | Full grasp → transport → place of randomized objects with a parallel gripper.                                               |
| **Engineering quality**    | Small typed modules (`model` / `control` / `pipeline` / `benchmark` / `record_demo`), one entry point, pinned deps.         |
| **Presentation**           | Telemetry-overlay demo video + two result plots, all generated by the code.                                                 |
| **Innovation**             | Frames manipulation as **demonstration-data generation for imitation learning** — an under-used angle in the field.         |

## Benchmark results

See [`results/benchmark.json`](results/benchmark.json) / [`benchmark.csv`](results/benchmark.csv)
and the dataset `results/demo_dataset.npz`. Summary (12 episodes):

```
pick_success_rate      : 1.00
place_success_rate     : 1.00
mean_place_err_mm      : 18.1
max_place_err_mm       : 26.3
dataset_steps          : 41760
workspace_coverage_m2  : 0.0304
```

## How to run

```bash
pip install -r requirements.txt

python run.py                 # 12 episodes -> benchmark.json + demo_dataset.npz
python run.py --episodes 20   # more episodes / more data
python run.py --quick         # 3-episode smoke run
python run.py --demo          # render demo video -> results/pandapick_demo.mp4
python scripts/make_plots.py  # regenerate the plots above
```

## Demo video

`results/pandapick_demo.mp4` — produced by `python run.py --demo`. Four randomized
pick-and-place episodes with a live phase tracker, gripper/object state, and a running
success tally.

## Current limitations

- Single-object pick-and-place; no clutter or multi-object sequencing yet.
- Grasps use top-down poses; no 6-DOF grasp-orientation search.
- The "expert" is a scripted IK controller (intended — it is the demonstrator that labels the dataset).

## Future improvements

- Cluttered bins + multi-object kitting sequences.
- Train a behaviour-cloning policy on the exported dataset and evaluate it in the same env.
- 6-DOF grasp pose sampling for arbitrary object orientations.

## Submitting this project (read me)

This folder is a complete, self-contained entry. To submit it as **your own**:

1. Register at [robothon.ff.com](https://robothon.ff.com) and get **your own UUID**.
2. Put that UUID in **both** [`registration.json`](registration.json) and your PR description.
3. Fork `Faraday-Future-AI/Robothon-starter`, copy this folder to `submissions/pandapick/`,
   and open a Pull Request.

## Credits & license

- Franka Panda model: `google-deepmind/mujoco_menagerie` (Apache-2.0), vendored under `vendor/`.
- Built with an AI coding agent (Claude) for FFAI Robothon Summer 2026.
