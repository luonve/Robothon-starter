"""1 episode pick-and-place + thu thap du lieu demo.

State machine: HOVER -> DESCEND -> GRASP -> LIFT -> TRANSPORT -> PLACE -> RELEASE -> RETRACT.
Tra ket qua (pick/place success, do chinh xac, cycle time) + ban ghi (obs, action).
"""
from __future__ import annotations
import numpy as np

from .model import build_model, FEEDER_TOP_Z
from .control import IKController, GRIP_OPEN, GRIP_CLOSE


def run_episode(seed: int, log: bool = False, renderer=None, cam=None, fast: bool = False):
    model, meta = build_model(seed)
    c = IKController(model, meta, log=log)
    if renderer is not None:
        c._renderer = renderer; c._cam = cam
    s = 1 if fast else 1
    # de cube settle tren feeder
    c.set_grip(GRIP_OPEN, steps=120, phase="settle")

    cube0 = meta.cube_pos(c.d)
    cx, cy, cz = cube0
    half = meta.scene["cube_half"]
    grasp_z = cz                                  # tam cube
    bin_xy = meta.bin_pos[:2]

    res = {"seed": seed, "scene": {k: (list(v) if isinstance(v, (tuple, list, np.ndarray)) else v)
                                   for k, v in meta.scene.items()}}

    # 1. HOVER tren cube
    c.move_to([cx, cy, cz + 0.12], GRIP_OPEN, steps=420, phase="hover")
    # 2. DESCEND toi tam cube
    c.move_to([cx, cy, grasp_z], GRIP_OPEN, steps=420, phase="descend")
    z_before = meta.cube_pos(c.d)[2]
    # 3. GRASP (dong kep)
    c.set_grip(GRIP_CLOSE, steps=500, phase="grasp")
    # 4. LIFT
    c.move_to([cx, cy, cz + 0.22], GRIP_CLOSE, steps=450, phase="lift")
    z_lift = meta.cube_pos(c.d)[2]
    pick_ok = bool(z_lift > cz + 0.12)
    # 5. TRANSPORT toi tren bin
    c.move_to([bin_xy[0], bin_xy[1], cz + 0.24], GRIP_CLOSE, steps=550, phase="transport")
    # 6. PLACE (ha xuong sat day bin -> tha gan, khong nay ra)
    c.move_to([bin_xy[0], bin_xy[1], 0.055 + half], GRIP_CLOSE, steps=420, phase="place")
    # 7. RELEASE
    c.set_grip(GRIP_OPEN, steps=300, phase="release")
    # 8. RETRACT
    c.move_to([bin_xy[0], bin_xy[1], cz + 0.24], GRIP_OPEN, steps=300, phase="retract")

    cube_f = meta.cube_pos(c.d)
    place_err = float(np.linalg.norm(cube_f[:2] - bin_xy))
    place_ok = bool(pick_ok and place_err < 0.06 and cube_f[2] < 0.10)
    res.update({
        "pick_ok": pick_ok,
        "place_ok": place_ok,
        "place_err_m": round(place_err, 4),
        "cube_final": cube_f.round(4).tolist(),
        "n_records": len(c.records),
    })
    return res, c.records, c
