# MuJoCo Robotics Simulation Hackathon

Build a robot simulation task, interactive system, or data-collection environment with [Google DeepMind MuJoCo](https://github.com/google-deepmind/mujoco).

This repository is the official Hackathon starter repo. Submit your work by opening a **Pull Request**. Submissions are reviewed based on the PR code, models, run instructions, demo video, and registration UUID.

## How to Participate

1. Register on the official Hackathon platform and obtain your **registration UUID** (token).
2. Fork this repository.
3. Build your project on a feature branch.
4. Add your submission under `submissions/<your-project-name>/`.
5. Open a Pull Request to this repository.
6. Fill in your registration UUID in the required locations (see below).

## Registration UUID (Required)

Participants must obtain a **registration UUID** from the official Hackathon platform before submitting. This UUID proves that you registered for the event.

### Where to put your UUID

You must include the same UUID in **both** places below:

#### 1. Submission folder — `registration.json`

Create this file inside your submission directory:

```
submissions/<your-project-name>/registration.json
```

Example:

```json
{
  "uuid": "00000000-0000-0000-0000-000000000000",
  "participant_name": "Your Name or Team Name",
  "project_name": "Your Project Name"
}
```

Replace the example UUID with the one issued by the Hackathon platform. Do not share or reuse another participant's UUID.

#### 2. Pull Request description

When you open your PR, the PR template will ask for your registration UUID. Paste the same UUID there.

If the template is not shown automatically, add this line at the top of your PR description:

```markdown
Registration UUID: 00000000-0000-0000-0000-000000000000
```

Submissions without a valid UUID in both places may be rejected during review.

Use `submissions/SUBMISSION_TEMPLATE/` as a starting point for your folder structure.

## Core Requirements

- Use MuJoCo as the primary physics simulation engine
- You may use any robot platform: arms, mobile robots, quadrupeds, humanoids, grippers, UGVs, dexterous hands, etc.
- You may use open-source robot models or custom MJCF models
- Build a runnable simulation task, interactive system, or data-collection environment
- Submit via Pull Request
- Include a demo video or video link in the PR

## Recommended Directions

- **Advanced teleoperation**: keyboard, gamepad, VR, Web UI, motion capture
- **Long-horizon tasks**: navigation, grasping, carrying, assembly, door opening, tidying, cleaning
- **Data collection**: auto-generated trajectories, states, actions, images, depth, sensor streams, labels
- **Dexterous manipulation**: multi-finger grasping, in-hand rotation, tool use, button presses, bottle opening
- **Real-world scenarios**: K12 education, campus security, home service, warehouse logistics, industrial inspection
- **Open exploration**: any creative MuJoCo robotics simulation project

## Encouraged Robot Platforms

Open-source robot models are welcome, for example:

- Unitree Go1 / Go2 / G1
- Boston Dynamics Spot
- Franka Emika Panda
- Shadow Hand
- LEAP Hand
- Robotiq Gripper
- Other MuJoCo / MJCF open models

Reference libraries:

- [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie)
- [MuJoCo Model Gallery](https://mujoco.readthedocs.io/en/latest/models.html)

## PR Submission Checklist

Each PR should include:

- Project source code
- MuJoCo scene files / robot models / related assets
- Run instructions: dependencies, install steps, launch commands, controls
- Demo video or video link
- `registration.json` with your platform-issued UUID
- A short project summary covering:
  - Project name
  - Robot platform used
  - Task goal
  - Technical approach
  - Core features
  - Highlights
  - Current limitations
  - Future improvements

## Demo Video Requirements

The video must be produced by running your submitted code and should show:

- Simulation startup
- Robot platform and task scene
- Task execution
- Teleoperation, autonomous control, or data-collection logic
- Final result or task state

Recommended length: 1–3 minutes.

## Judging Criteria

- **Reproducibility**: does the code run cleanly and is it easy to reproduce?
- **MuJoCo depth**: MJCF, physics, collision, joints, sensors, actuators
- **Task design**: clarity, challenge, real-world relevance
- **Control capability**: teleop, autonomy, policy control, planning, or data collection
- **Dexterity**: multi-finger coordination and fine manipulation (if applicable)
- **Engineering quality**: code structure, docs, configuration, asset management
- **Presentation**: demo video clarity and persuasiveness
- **Innovation**: novelty in scene, robot, task, or application

## Example Topics

- Campus security patrol with Boston Dynamics Spot
- Rough-terrain inspection with Unitree Go1 / Go2
- K12 lab organization with Franka Panda
- Fine grasping and in-hand rotation with Shadow Hand / LEAP Hand
- Web / gamepad / VR teleoperation stack
- Auto-generated grasp trajectory dataset
- Home-service long-horizon tasks: open door, pick, place, clean desktop

## Goal

Deliver a runnable, demonstrable, reproducible MuJoCo robotics simulation project via Pull Request.

---

## About This Repository

This is the official Hackathon starter repository. It includes bundled robot assets and example scripts you can fork and extend.

### Quick Start

```bash
python3 -m pip install -r requirements.txt
```

Run examples:

```bash
python examples/run_ff_master_demo.py
python examples/run_zsl1_demo.py
```

Open MuJoCo Viewer:

```bash
python -m mujoco.viewer
```

### Bundled Assets

| Path | Description |
|------|-------------|
| `assets/ff-master/` | FF Master humanoid (ultra / hand / fist variants) |
| `assets/zsl-1/` | ZSL-1 robot URDF / MuJoCo model |
| `examples/` | Example run scripts |
| `model_catalog.json` | Reference list of recommended open-source robot models |
| `submissions/SUBMISSION_TEMPLATE/` | Submission folder template with UUID placeholder |
