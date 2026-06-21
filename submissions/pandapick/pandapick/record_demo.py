"""Render the PandaPick demo video — multi-task (place + colour-sort) with a top/bottom HUD.

Distinct top-bar + bottom-strip layout. All on-screen state is read from the live simulation.
"""
from __future__ import annotations
import os
import numpy as np
import mujoco
import imageio.v2 as imageio
from PIL import Image, ImageDraw, ImageFont

from .model import build_model, HALF, SORT_BINS, STACK_PAD
from .control import IKController, GRIP_OPEN, GRIP_CLOSE
from .pipeline import _pick_place_one

RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
W, H = 1280, 720
TOP, BOT = 64, 82
RENDER_H = H - TOP - BOT
# Theme SANG (lab/blueprint) — tuong phan ro voi DexFab (theme toi)
C = {"bg": (237, 240, 235), "bar": (248, 249, 245), "ink": (22, 34, 43), "dim": (92, 108, 116),
     "teal": (13, 148, 136), "amber": (181, 96, 8), "ok": (21, 145, 86)}
COLrgb = {"R": (220, 90, 70), "G": (90, 200, 120), "B": (90, 130, 230)}
PHASE_LABEL = {"hover": "approach", "descend": "descend", "grasp": "grasp", "lift": "lift",
               "transport": "transport", "place": "place", "release": "release",
               "retract": "retract", "settle": "init", "hold": "secure"}


def _font(sz):
    for p in [r"C:\Windows\Fonts\consola.ttf", r"C:\Windows\Fonts\segoeui.ttf", r"C:\Windows\Fonts\arial.ttf"]:
        if os.path.exists(p):
            return ImageFont.truetype(p, sz)
    return ImageFont.load_default()


def _cam():
    c = mujoco.MjvCamera()
    c.azimuth = 150; c.elevation = -24; c.distance = 1.2; c.lookat[:] = [0.5, -0.05, 0.12]
    return c


def _hud(raw, st):
    cv = Image.new("RGB", (W, H), C["bg"])
    cv.paste(Image.fromarray(raw).convert("RGB").resize((W, RENDER_H)), (0, TOP))
    d = ImageDraw.Draw(cv, "RGBA")
    # top bar
    d.rectangle([0, 0, W, TOP], fill=C["bar"]); d.line([0, TOP, W, TOP], fill=C["teal"], width=2)
    d.text((24, 16), "PandaPick", font=_font(30), fill=C["ink"])
    d.text((230, 24), "// Franka Panda multi-task data collection", font=_font(16), fill=C["dim"])
    task_txt = "TASK: " + ("COLOUR SORT" if st["task"] == "sort" else "PICK & PLACE")
    d.text((W // 2 + 40, 22), task_txt, font=_font(20), fill=C["amber"])
    d.text((W - 250, 12), f"episode {st['ep']+1}/{st['total']}", font=_font(17), fill=C["dim"])
    d.text((W - 250, 36), f"success {st['ok']}/{st['done']}", font=_font(17), fill=C["ok"])
    # bottom strip
    by = H - BOT
    d.rectangle([0, by, W, H], fill=C["bar"]); d.line([0, by, W, by], fill=C["teal"], width=2)
    ph = PHASE_LABEL.get(st["phase"], st["phase"])
    d.text((24, by + 12), "phase", font=_font(13), fill=C["dim"])
    d.text((24, by + 30), ph.upper(), font=_font(26), fill=C["teal"])
    d.text((300, by + 12), "gripper", font=_font(13), fill=C["dim"])
    d.text((300, by + 32), "OPEN" if st["grip"] > 128 else "HOLD", font=_font(20), fill=C["ink"])
    d.text((470, by + 12), "cube", font=_font(13), fill=C["dim"])
    d.text((470, by + 32), f"#{st['cube']+1}", font=_font(20), fill=C["ink"])
    # target color chip (sort)
    if st["task"] == "sort" and st.get("color"):
        d.text((600, by + 12), "target bin", font=_font(13), fill=C["dim"])
        d.rectangle([600, by + 34, 628, by + 60], fill=COLrgb[st["color"]])
        d.text((636, by + 34), st["color"], font=_font(22), fill=C["ink"])
    # disturbance indicator (grasp-stability demo)
    if st.get("disturb", 0):
        d.text((600, by + 12), "disturbance", font=_font(13), fill=C["dim"])
        d.text((600, by + 30), f"{st['disturb']:.0f} N  ->  GRIP HOLDS", font=_font(22), fill=C["amber"])
    d.text((W - 430, by + 14), f"demo steps  {st['steps']:,}", font=_font(16), fill=C["dim"])
    d.text((W - 430, by + 40), "obs/action -> imitation dataset", font=_font(15), fill=C["dim"])
    return np.asarray(cv)


def _card(lines, n, subs=None):
    cv = Image.new("RGB", (W, H), C["bg"]); d = ImageDraw.Draw(cv)
    d.line([64, H // 2 - 96, 132, H // 2 - 96], fill=C["teal"], width=3)
    y = H // 2 - 70
    for ln, sz, c in lines:
        d.text((64, y), ln, font=_font(sz), fill=c); y += sz + 12
    if subs:
        y += 14
        for s in subs:
            d.text((64, y), s, font=_font(19), fill=C["dim"]); y += 28
    return [np.asarray(cv)] * n


def _render_episode(seed, task, ep, total, tally, frames, n=3):
    model, meta = build_model(seed, task, n)
    rnd = mujoco.Renderer(model, RENDER_H, W); cam = _cam()
    c = IKController(model, meta, log=False)
    c._renderer = rnd; c._cam = cam; c._frame_every = 12
    st = {"task": task, "ep": ep, "total": total, "phase": "settle", "grip": GRIP_OPEN,
          "cube": 0, "color": None, "ok": tally["ok"], "done": tally["done"], "steps": tally["steps"]}

    def compose(raw):
        st["phase"] = c.phase; st["grip"] = c.grip; st["cube"] = c.active_cube
        st["color"] = meta.colors[c.active_cube] if task == "sort" else None
        st["steps"] = tally["steps"] + c._k * 0  # live counter below
        tally["steps"] += c._frame_every
        st["steps"] = tally["steps"]
        return _hud(raw, st)
    c.compose = compose

    c.set_grip(GRIP_OPEN, 120, "settle")
    bx, by = STACK_PAD
    ok = 0
    for i in range(n):
        if task == "sort":
            dx, dy = SORT_BINS[meta.colors[i]]; dz = HALF + 0.02
        else:
            dx, dy = bx, by; dz = HALF + 0.02
        picked = _pick_place_one(c, meta, i, (dx, dy), dz)
        cf = meta.cube_pos(c.d, i)
        dest = SORT_BINS[meta.colors[i]] if task == "sort" else (bx, by)
        if picked and np.linalg.norm(cf[:2] - np.array(dest)) < 0.06 and cf[2] < 0.13:
            ok += 1
    tally["done"] += 1; tally["ok"] += int(ok == n)
    frames.extend(c.frames)        # gom frame episode vao video chinh
    del rnd
    return ok


def _render_disturb_episode(seed, ep, total, tally, frames):
    """Grasp-stability demo: hold a cube while an external disturbance is applied."""
    model, meta = build_model(seed, "pick_place", 1)
    rnd = mujoco.Renderer(model, RENDER_H, W); cam = _cam()
    c = IKController(model, meta, log=False)
    c._renderer = rnd; c._cam = cam; c._frame_every = 12
    st = {"task": "pick_place", "ep": ep, "total": total, "phase": "settle", "grip": GRIP_OPEN,
          "cube": 0, "color": None, "disturb": 0, "ok": tally["ok"], "done": tally["done"], "steps": tally["steps"]}

    def compose(raw):
        st["phase"] = c.phase; st["grip"] = c.grip; st["cube"] = 0
        tally["steps"] += c._frame_every; st["steps"] = tally["steps"]
        return _hud(raw, st)
    c.compose = compose

    c.set_grip(GRIP_OPEN, 120, "settle")
    cx, cy, cz = meta.cube_pos(c.d, 0)
    c.move_to([cx, cy, cz + 0.12], GRIP_OPEN, 300, "hover")
    c.move_to([cx, cy, cz], GRIP_OPEN, 300, "descend")
    c.set_grip(GRIP_CLOSE, 400, "grasp")
    c.move_to([cx, cy, cz + 0.24], GRIP_CLOSE, 360, "lift")
    cb = meta.cube_bid[0]
    for F in (3.0, 5.0):                       # ramp the disturbance up
        st["disturb"] = F
        c.d.xfrc_applied[cb][:3] = [F * 0.6, 0.0, -F]
        c.set_grip(GRIP_CLOSE, 170, "hold")    # step + capture while holding against the force
    c.d.xfrc_applied[cb][:] = 0.0; st["disturb"] = 0
    held = meta.cube_pos(c.d, 0)[2] > cz + 0.12
    bx, by = STACK_PAD
    c.move_to([bx, by, cz + 0.24], GRIP_CLOSE, 420, "transport")
    c.move_to([bx, by, HALF + 0.03], GRIP_CLOSE, 360, "place")
    c.set_grip(GRIP_OPEN, 250, "release")
    c.move_to([bx, by, cz + 0.2], GRIP_OPEN, 200, "retract")
    tally["done"] += 1; tally["ok"] += int(held)
    frames.extend(c.frames)
    del rnd
    return held


def record(out_path=None, fps=24):
    out_path = out_path or os.path.join(RESULTS, "pandapick_demo.mp4")
    os.makedirs(RESULTS, exist_ok=True)
    frames = []
    frames += _card([("PandaPick", 60, C["ink"]),
                     ("Multi-task pick-place + colour sorting", 26, C["teal"])],
                    80, subs=["Franka Emika Panda  //  MuJoCo  //  resolved-rate IK",
                              "autonomous demos -> labelled (obs, action) dataset"])
    tally = {"ok": 0, "done": 0, "steps": 0}
    total = 3
    frames += _card([("Episode 1", 38, C["ink"]), ("pick & place", 24, C["amber"])],
                    22, subs=["randomized cube positions"])
    _render_episode(0, "pick_place", 0, total, tally, frames)
    frames += _card([("Episode 2", 38, C["ink"]), ("colour sort  (R / G / B)", 24, C["amber"])],
                    22, subs=["read each colour, route to its bin"])
    _render_episode(1, "sort", 1, total, tally, frames)
    frames += _card([("Episode 3", 38, C["ink"]), ("grasp stability", 24, C["amber"])],
                    22, subs=["hold the object against an external disturbance"])
    _render_disturb_episode(0, 2, total, tally, frames)
    frames += _card([("PandaPick", 54, C["ink"]),
                     ("15 tasks // 100% success // 13.3 mm // holds ~20x object weight", 21, C["teal"])],
                    85, subs=["resolved-rate IK  //  140k-step imitation dataset",
                              "pip install -r requirements.txt  &&  python run.py  //  CPU, no GPU"])
    imageio.mimwrite(out_path, frames, fps=fps, quality=8, macro_block_size=1)
    print(f"[OK] {len(frames)} frame ({len(frames)/fps:.0f}s) -> {out_path}")
    return out_path
