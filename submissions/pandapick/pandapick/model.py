"""Build scene MuJoCo bang MjSpec: Franka Panda + feeder post (trinh cube) + bin.

Cube nam tren feeder post manh (ngon gripper straddle qua post -> kep than cube,
fingertip khong cham san). Domain randomization: vi tri cube, kich thuoc, khoi luong, mau.
"""
from __future__ import annotations
import os
import numpy as np
import mujoco

_HERE = os.path.dirname(os.path.abspath(__file__))
_PANDA_XML = os.path.normpath(os.path.join(_HERE, "..", "vendor", "mujoco_menagerie",
                                            "franka_emika_panda", "scene.xml"))

# Tu the home cua arm (gripper huong xuong)
HOME = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])
GRASP_SITE_OFFSET = 0.0584         # tam kep cach body hand doc local-z
FEEDER_TOP_Z = 0.10                # do cao mat feeder (cube dat tren day)
BIN_POS = (0.45, -0.28, 0.0)       # vi tri thung dat


def _rng(seed):
    return np.random.default_rng(seed)


def sample_scene(seed: int) -> dict:
    """Sinh tham so scene ngau nhien (deterministic theo seed)."""
    r = _rng(seed)
    # vi tri cube trong vung arm voi toi (gripper huong xuong)
    cx = float(r.uniform(0.42, 0.58))
    cy = float(r.uniform(-0.12, 0.16))
    half = float(r.uniform(0.018, 0.024))       # nua canh cube
    mass = float(r.uniform(0.02, 0.05))
    rgba = tuple(r.uniform(0.25, 0.9, size=3)) + (1.0,)
    return {"cube_xy": (cx, cy), "cube_half": half, "cube_mass": mass, "cube_rgba": rgba}


def build_spec(scene: dict) -> "mujoco.MjSpec":
    cx, cy = scene["cube_xy"]
    half = scene["cube_half"]
    spec = mujoco.MjSpec.from_file(_PANDA_XML)
    spec.visual.global_.offwidth = 1920
    spec.visual.global_.offheight = 1080

    # site tam kep gan vao body hand
    hsite = spec.body("hand").add_site()
    hsite.name = "grasp"
    hsite.pos = [0, 0, GRASP_SITE_OFFSET]
    hsite.size = [0.005, 0.005, 0.005]
    hsite.rgba = [0, 1, 0, 0]

    wb = spec.worldbody

    # feeder post manh trinh cube
    fz = FEEDER_TOP_Z / 2
    feeder = wb.add_geom()
    feeder.name = "feeder"; feeder.type = mujoco.mjtGeom.mjGEOM_BOX
    feeder.size = [0.009, 0.009, fz]; feeder.pos = [cx, cy, fz]
    feeder.rgba = [0.30, 0.32, 0.36, 1]

    # cube tren feeder
    cube = wb.add_body()
    cube.name = "cube"; cube.pos = [cx, cy, FEEDER_TOP_Z + half + 0.002]
    cube.add_joint().type = mujoco.mjtJoint.mjJNT_FREE
    g = cube.add_geom()
    g.name = "cube_g"; g.type = mujoco.mjtGeom.mjGEOM_BOX
    g.size = [half, half, half]; g.rgba = list(scene["cube_rgba"])
    g.mass = scene["cube_mass"]; g.condim = 4; g.friction = [2.0, 0.05, 0.001]

    # bin (thung dat) — 4 thanh + day
    bx, by, _ = BIN_POS
    floor = wb.add_geom(); floor.name = "bin_floor"; floor.type = mujoco.mjtGeom.mjGEOM_BOX
    floor.size = [0.06, 0.06, 0.004]; floor.pos = [bx, by, 0.004]; floor.rgba = [0.26, 0.45, 0.38, 1]
    for dx, dy, sx, sy in [(0.06, 0, 0.004, 0.06), (-0.06, 0, 0.004, 0.06),
                           (0, 0.06, 0.06, 0.004), (0, -0.06, 0.06, 0.004)]:
        wgt = wb.add_geom(); wgt.type = mujoco.mjtGeom.mjGEOM_BOX
        wgt.size = [sx, sy, 0.02]; wgt.pos = [bx + dx, by + dy, 0.02]; wgt.rgba = [0.28, 0.48, 0.40, 1]

    return spec


def build_model(seed: int):
    scene = sample_scene(seed)
    spec = build_spec(scene)
    model = spec.compile()
    return model, Meta(model, scene)


class Meta:
    def __init__(self, m, scene):
        self.m = m
        self.scene = scene
        self.grasp_site = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "grasp")
        self.grip_act = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, "actuator8")
        self.cube_jadr = m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "cube_free")]
        self.cube_bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "cube")
        self.arm_range = m.jnt_range[:7].copy()
        self.bin_pos = np.array(BIN_POS)

    def cube_pos(self, d):
        return d.qpos[self.cube_jadr:self.cube_jadr + 3].copy()

    def ee_pos(self, d):
        return d.site_xpos[self.grasp_site].copy()
