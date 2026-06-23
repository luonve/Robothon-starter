"""Scene builder (MjSpec): Franka Panda + N cubes on feeder posts + colour bins / tote.

Supports two jobs: pick_place and sort (route by colour). Each cube sits on a slim
feeder post so the long gripper fingers can straddle the post and close on the cube body.
The IK solver parks every cube far away while solving (multi-object kinematics).
"""
from __future__ import annotations
import os
import numpy as np
import mujoco

_HERE = os.path.dirname(os.path.abspath(__file__))
_PANDA_XML = os.path.normpath(os.path.join(_HERE, "..", "vendor", "mujoco_menagerie",
                                            "franka_emika_panda", "scene.xml"))

HOME = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])
GRASP_SITE_OFFSET = 0.0584
FEEDER_TOP_Z = 0.10
HALF = 0.022                              # cube half-extent (fixed)

# 3 part colours for the sort job (R/G/B) + matching bins
COLORS = {"R": (0.85, 0.30, 0.25, 1), "G": (0.30, 0.65, 0.40, 1), "B": (0.30, 0.45, 0.85, 1)}
SORT_BINS = {"R": (0.40, -0.26), "G": (0.50, -0.30), "B": (0.60, -0.26)}
STACK_PAD = (0.50, -0.10)


def _rng(seed):
    return np.random.default_rng(seed)


def sample_scene(seed: int, task: str = "pick_place", n: int = 3) -> dict:
    """Sample a randomized scene from the seed and job type."""
    r = _rng(seed)
    feeders = []
    xs = np.linspace(0.44, 0.60, n)
    for i in range(n):
        fx = float(xs[i] + r.uniform(-0.01, 0.01))
        fy = float(r.uniform(0.08, 0.16))
        col = list(COLORS)[i % 3] if task == "sort" else "R"
        feeders.append({"xy": (fx, fy), "color": col})
    return {"task": task, "n": n, "feeders": feeders, "mass": float(r.uniform(0.025, 0.04))}


def build_spec(scene: dict) -> "mujoco.MjSpec":
    spec = mujoco.MjSpec.from_file(_PANDA_XML)
    spec.visual.global_.offwidth = 1920
    spec.visual.global_.offheight = 1080

    hsite = spec.body("hand").add_site()
    hsite.name = "grasp"; hsite.pos = [0, 0, GRASP_SITE_OFFSET]
    hsite.size = [0.005, 0.005, 0.005]; hsite.rgba = [0, 1, 0, 0]
    wb = spec.worldbody

    for i, f in enumerate(scene["feeders"]):
        fx, fy = f["xy"]
        fp = wb.add_geom(); fp.name = f"feeder{i}"; fp.type = mujoco.mjtGeom.mjGEOM_BOX
        fp.size = [0.009, 0.009, FEEDER_TOP_Z / 2]; fp.pos = [fx, fy, FEEDER_TOP_Z / 2]
        fp.rgba = [0.30, 0.32, 0.36, 1]
        cube = wb.add_body(); cube.name = f"cube{i}"; cube.pos = [fx, fy, FEEDER_TOP_Z + HALF + 0.002]
        cj = cube.add_joint(); cj.name = f"cube{i}_free"; cj.type = mujoco.mjtJoint.mjJNT_FREE
        g = cube.add_geom(); g.name = f"cube{i}_g"; g.type = mujoco.mjtGeom.mjGEOM_BOX
        g.size = [HALF, HALF, HALF]; g.rgba = list(COLORS[f["color"]]); g.mass = scene["mass"]
        g.condim = 4; g.friction = [2.0, 0.05, 0.001]

    # destination: colour bins (sort) or a single tote/pad
    if scene["task"] == "sort":
        for col, (bx, by) in SORT_BINS.items():
            _add_bin(wb, f"bin_{col}", bx, by, COLORS[col])
    else:
        # single tote for pick_place at the pad location
        bx, by = STACK_PAD
        pad = wb.add_geom(); pad.name = "stack_pad"; pad.type = mujoco.mjtGeom.mjGEOM_BOX
        pad.size = [0.04, 0.04, 0.004]; pad.pos = [bx, by, 0.004]; pad.rgba = [0.4, 0.4, 0.45, 1]
        if scene["task"] in ("pick_place", "fragile"):   # fragile dung cung tote/bin nhu pick_place
            _add_bin(wb, "bin_R", bx, by, (0.26, 0.45, 0.38, 1))
    return spec


def _add_bin(wb, name, bx, by, rgba):
    fl = wb.add_geom(); fl.name = name + "_floor"; fl.type = mujoco.mjtGeom.mjGEOM_BOX
    fl.size = [0.045, 0.045, 0.004]; fl.pos = [bx, by, 0.004]; fl.rgba = list(rgba)
    for dx, dy, sx, sy in [(0.045, 0, 0.004, 0.045), (-0.045, 0, 0.004, 0.045),
                           (0, 0.045, 0.045, 0.004), (0, -0.045, 0.045, 0.004)]:
        w = wb.add_geom(); w.type = mujoco.mjtGeom.mjGEOM_BOX
        w.size = [sx, sy, 0.018]; w.pos = [bx + dx, by + dy, 0.018]; w.rgba = list(rgba)


def build_model(seed: int, task: str = "pick_place", n: int = 3):
    scene = sample_scene(seed, task, n)
    spec = build_spec(scene)
    model = spec.compile()
    return model, Meta(model, scene)


class Meta:
    def __init__(self, m, scene):
        self.m = m
        self.scene = scene
        self.n = scene["n"]
        self.grasp_site = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "grasp")
        self.grip_act = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, "actuator8")
        self.arm_range = m.jnt_range[:7].copy()
        self.cube_jadr = [m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"cube{i}_free")]
                          for i in range(self.n)]
        self.cube_bid = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"cube{i}") for i in range(self.n)]
        self.colors = [f["color"] for f in scene["feeders"]]
        # Fingertip geoms cho doc luc tiep xuc (mj_contactForce): cac pad la geom KHONG TEN
        # (dinh nghia qua class) -> PHAI loc theo body id cua left_finger/right_finger, KHONG tra ten.
        lf = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "left_finger")
        rf = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "right_finger")
        self.finger_geoms = frozenset(i for i in range(m.ngeom) if m.geom_bodyid[i] in (lf, rf))
        self.cube_geom = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, f"cube{i}_g") for i in range(self.n)]

    def cube_pos(self, d, i):
        a = self.cube_jadr[i]
        return d.qpos[a:a + 3].copy()

    def ee_pos(self, d):
        return d.site_xpos[self.grasp_site].copy()
