"""One episode per job type, plus demonstration logging.

- pick_place: lift each cube and drop it in the tote.
- sort:       lift each cube and route it to the matching colour bin (R/G/B).
Per-cube state machine: hover -> descend -> grasp -> lift -> transport -> place -> release -> retract.
"""
from __future__ import annotations
import numpy as np

from .model import build_model, HALF, FEEDER_TOP_Z, SORT_BINS, STACK_PAD
from .control import IKController, GRIP_OPEN, GRIP_CLOSE


def _pick_place_one(c, meta, i, dest_xy, dest_z, precise=False, disturb_N=0.0, mode="closed", firm=True):
    """mode='closed' -> grasp DIEU KHIEN LUC (grasp_to_force regulate ve setpoint); mode='open' ->
    baseline binary close (ablation chat luong dieu khien luc).
    firm=True -> sau regulate bop chac (GRIP_CLOSE) cho carry vung; firm=False -> GIU GRIP NHE (gia tri
    regulate) suot carry -> dung cho task 'fragile' (luc settled thuc su nhe toi luc dat, ko bi firm-up undo)."""
    c.active_cube = i
    cube = meta.cube_pos(c.d, i)
    cx, cy, cz = cube
    dx, dy = dest_xy
    place_steps = 720 if precise else 440      # stack can settle sat IK -> chinh xac hon
    rel_h = 0.004 if precise else 0.012         # release low so the cube barely shifts on drop
    c.move_to([cx, cy, cz + 0.12], GRIP_OPEN, 380, "hover")
    c.move_to([cx, cy, cz], GRIP_OPEN, 380, "descend")
    if mode == "closed":
        _, hold = c.grasp_to_force(i, phase="grasp", firm=firm)   # CLOSED-LOOP: dong den luc muc tieu (firm tuy task)
    else:
        c.set_grip(GRIP_CLOSE, 460, "grasp"); hold = GRIP_CLOSE   # OPEN-LOOP baseline: binary slam
    # FREEZE ctrl kep o gia tri hoi tu suot lift/transport (chong vong lap chong lai spike quan tinh)
    c.move_to([cx, cy, cz + 0.24], hold, 420, "lift")
    picked = meta.cube_pos(c.d, i)[2] > cz + 0.12
    # disturbance rejection during transport
    if disturb_N:
        c.d.xfrc_applied[meta.cube_bid[i]][:3] = [disturb_N * 0.6, 0.0, -disturb_N]
    c.move_to([dx, dy, dest_z + 0.14], hold, 520, "transport")
    c.d.xfrc_applied[meta.cube_bid[i]][:] = 0.0
    c.move_to([dx, dy, dest_z + rel_h], hold, place_steps, "place")
    c.set_grip(GRIP_OPEN, 280, "release")
    c.move_to([dx, dy, dest_z + 0.16], GRIP_OPEN, 260, "retract")
    return picked


def run_episode(seed, task="pick_place", n=3, log=False, renderer=None, cam=None, disturb_N=0.0,
                mode="closed"):
    model, meta = build_model(seed, task, n)
    c = IKController(model, meta, log=log)
    if renderer is not None:
        c._renderer = renderer; c._cam = cam
    c.set_grip(GRIP_OPEN, 120, "settle")

    res = {"seed": seed, "task": task, "n": n, "disturb_N": disturb_N, "mode": mode}
    picks, placeds, errs = 0, 0, []

    if task == "stack":
        bx, by = STACK_PAD
        for k in range(n):
            dest_z = (2 * k + 1) * HALF              # each cube goes to an increasing height
            ok = _pick_place_one(c, meta, k, (bx, by), dest_z, precise=True, mode=mode)
            picks += int(ok)
        # tower check: every cube lifted + near the stack axis
        heights = sorted(meta.cube_pos(c.d, i)[2] for i in range(n))
        xy_ok = all(np.linalg.norm(meta.cube_pos(c.d, i)[:2] - np.array([bx, by])) < HALF * 1.4 for i in range(n))
        tower_h = heights[-1]
        stacked = sum(1 for h in heights if h > HALF * 1.2)   # count cubes lifted off the table
        placeds = stacked if xy_ok else 0
        res.update({"tower_height_m": round(float(tower_h), 4),
                    "cubes_stacked": int(placeds), "xy_aligned": bool(xy_ok)})
    elif task == "sort":
        for i in range(n):
            col = meta.colors[i]
            bx, by = SORT_BINS[col]
            ok = _pick_place_one(c, meta, i, (bx, by), HALF + 0.02, mode=mode)
            picks += int(ok)
            cf = meta.cube_pos(c.d, i)
            in_bin = np.linalg.norm(cf[:2] - np.array([bx, by])) < 0.05 and cf[2] < 0.12
            placeds += int(ok and in_bin)
        res.update({"sorted_correct": int(placeds)})
    elif task == "fragile":
        # FRAGILE: pick-place voi GRIP NHE suot carry (firm=False) — luc settled (~1.15N) giu nguyen
        # toi luc dat, ko bi firm-up bop chac undo. Day la job that chay duoc o che do nhe (place 100%);
        # tuong phan crush-vs-save (open-loop slam lam vo) nam o ablation run_fragile_budget (fragile.json).
        bx, by = STACK_PAD
        for i in range(n):
            ok = _pick_place_one(c, meta, i, (bx, by), HALF + 0.02, disturb_N=disturb_N, mode=mode, firm=False)
            picks += int(ok)
            cf = meta.cube_pos(c.d, i)
            in_bin = np.linalg.norm(cf[:2] - np.array([bx, by])) < 0.06 and cf[2] < 0.14
            placeds += int(ok and in_bin)
            errs.append(float(np.linalg.norm(cf[:2] - np.array([bx, by]))))
        res.update({"placed": int(placeds), "mean_place_err_m": round(float(np.mean(errs)), 4) if errs else None,
                    "gentle_carry": True})
    else:   # pick_place
        bx, by = STACK_PAD
        for i in range(n):
            ok = _pick_place_one(c, meta, i, (bx, by), HALF + 0.02, disturb_N=disturb_N, mode=mode)
            picks += int(ok)
            cf = meta.cube_pos(c.d, i)
            in_bin = np.linalg.norm(cf[:2] - np.array([bx, by])) < 0.06 and cf[2] < 0.14
            placeds += int(ok and in_bin)
            errs.append(float(np.linalg.norm(cf[:2] - np.array([bx, by]))))
        res.update({"placed": int(placeds), "mean_place_err_m": round(float(np.mean(errs)), 4) if errs else None})

    res.update({"picked": picks, "success": placeds, "n_records": len(c.records)})
    return res, c.records, c
