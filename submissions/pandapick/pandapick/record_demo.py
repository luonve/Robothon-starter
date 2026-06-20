"""Render video demo PandaPick — arm pick-and-place qua nhieu episode + overlay telemetry.

Tat ca trang thai tren overlay (phase, vi tri, success) doc truc tiep tu mo phong.
"""
from __future__ import annotations
import os
import numpy as np
import mujoco
import imageio.v2 as imageio
from PIL import Image, ImageDraw, ImageFont

from .model import build_model, FEEDER_TOP_Z
from .control import IKController, GRIP_OPEN, GRIP_CLOSE

RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
W, H = 1280, 720
PANEL_W = 360
COL = {"bg": (16, 18, 22), "panel": (24, 27, 34), "ink": (236, 232, 226), "muted": (150, 156, 166),
       "accent": (120, 170, 230), "ok": (90, 200, 130), "warn": (235, 150, 90)}
PHASES = {"hover": "1 HOVER", "descend": "2 DESCEND", "grasp": "3 GRASP", "lift": "4 LIFT",
          "transport": "5 TRANSPORT", "place": "6 PLACE", "release": "7 RELEASE", "retract": "8 RETRACT"}


def _font(sz):
    for p in [r"C:\Windows\Fonts\segoeui.ttf", r"C:\Windows\Fonts\arial.ttf"]:
        if os.path.exists(p):
            return ImageFont.truetype(p, sz)
    return ImageFont.load_default()


def _cam():
    c = mujoco.MjvCamera()
    c.azimuth = 148; c.elevation = -22; c.distance = 1.25; c.lookat[:] = [0.45, -0.05, 0.15]
    return c


def _overlay(scene_img, st):
    img = Image.fromarray(scene_img).convert("RGB").resize((W - PANEL_W, H))
    cv = Image.new("RGB", (W, H), COL["bg"]); cv.paste(img, (0, 0))
    d = ImageDraw.Draw(cv, "RGBA"); x0 = W - PANEL_W
    d.rectangle([x0, 0, W, H], fill=COL["panel"]); d.line([x0, 0, x0, H], fill=COL["accent"], width=2)
    pad = 24
    d.text((x0 + pad, 24), "PandaPick", font=_font(34), fill=COL["ink"])
    d.text((x0 + pad, 66), "Pick-&-Place Data Collection", font=_font(17), fill=COL["muted"])
    d.text((x0 + pad, 112), f"Episode {st['ep']+1} / {st['total']}", font=_font(22), fill=COL["accent"])
    # phase tracker
    y = 156
    d.text((x0 + pad, y), "PHASE", font=_font(14), fill=COL["muted"])
    d.text((x0 + pad, y + 20), PHASES.get(st["phase"], st["phase"]), font=_font(26), fill=COL["ink"])
    # telemetry
    y2 = 230
    for lab, val in [("EE target z", f"{st['ee'][2]:.3f} m"),
                     ("cube pos", f"({st['cube'][0]:.2f}, {st['cube'][1]:.2f}, {st['cube'][2]:.2f})"),
                     ("gripper", "OPEN" if st["grip"] > 128 else "CLOSED")]:
        d.text((x0 + pad, y2), lab, font=_font(14), fill=COL["muted"])
        d.text((x0 + pad + 130, y2 - 2), val, font=_font(16), fill=COL["ink"]); y2 += 30
    # status badges
    y3 = 350
    pk = st["pick"]; pl = st["place"]
    d.text((x0 + pad, y3), "PICK", font=_font(16), fill=COL["muted"])
    d.text((x0 + pad + 90, y3), "OK" if pk else "...", font=_font(16), fill=COL["ok"] if pk else COL["muted"])
    d.text((x0 + pad, y3 + 28), "PLACE", font=_font(16), fill=COL["muted"])
    d.text((x0 + pad + 90, y3 + 28), "OK" if pl else "...", font=_font(16), fill=COL["ok"] if pl else COL["muted"])
    # running tally + dataset
    d.text((x0 + pad, H - 150), f"success: {st['ok']}/{st['done']} episodes", font=_font(18), fill=COL["accent"])
    d.text((x0 + pad, H - 122), f"demo steps logged: {st['steps']:,}", font=_font(16), fill=COL["muted"])
    d.text((x0 + pad, H - 58), "Franka Panda x MuJoCo  ·  IK + domain rand.", font=_font(15), fill=COL["muted"])
    d.text((x0 + pad, H - 34), "obs/action dataset for imitation learning", font=_font(14), fill=COL["muted"])
    return np.asarray(cv)


def _card(lines, n, subs=None):
    cv = Image.new("RGB", (W, H), COL["bg"]); d = ImageDraw.Draw(cv)
    d.line([60, H // 2 - 92, 120, H // 2 - 92], fill=COL["accent"], width=3)
    y = H // 2 - 66
    for ln, sz, c in lines:
        d.text((60, y), ln, font=_font(sz), fill=c); y += sz + 12
    if subs:
        y += 14
        for s in subs:
            d.text((60, y), s, font=_font(19), fill=COL["muted"]); y += 28
    return [np.asarray(cv)] * n


def _render_episode(ep, total, tally, frames, every=6):
    model, meta = build_model(ep)
    rnd = mujoco.Renderer(model, H, W - PANEL_W); cam = _cam()
    c = IKController(model, meta, log=False)
    cx, cy, cz = None, None, None
    state = {"ep": ep, "total": total, "phase": "hover", "pick": False, "place": False,
             "ok": tally["ok"], "done": tally["done"], "steps": tally["steps"],
             "ee": [0, 0, 0], "cube": [0, 0, 0], "grip": GRIP_OPEN}

    def hook():
        state["phase"] = c.phase; state["grip"] = c.grip
        state["ee"] = meta.ee_pos(c.d); state["cube"] = meta.cube_pos(c.d)
        rnd.update_scene(c.d, cam)
        frames.append(_overlay(rnd.render(), state))
        tally["steps"] += every

    # gan hook vao controller qua frame capture thu cong
    c._renderer = rnd; c._cam = cam; c._frame_every = every
    orig = c._maybe_frame
    def patched():
        c._k += 1
        if c._k % every == 0:
            hook()
    c._maybe_frame = patched

    c.set_grip(GRIP_OPEN, 120, "settle")
    cx, cy, cz = meta.cube_pos(c.d); half = meta.scene["cube_half"]; bx, by = meta.bin_pos[:2]
    c.move_to([cx, cy, cz + 0.12], GRIP_OPEN, 420, "hover")
    c.move_to([cx, cy, cz], GRIP_OPEN, 420, "descend")
    c.set_grip(GRIP_CLOSE, 500, "grasp")
    c.move_to([cx, cy, cz + 0.22], GRIP_CLOSE, 450, "lift")
    state["pick"] = bool(meta.cube_pos(c.d)[2] > cz + 0.12)
    c.move_to([bx, by, cz + 0.24], GRIP_CLOSE, 550, "transport")
    c.move_to([bx, by, 0.055 + half], GRIP_CLOSE, 420, "place")
    c.set_grip(GRIP_OPEN, 300, "release")
    c.move_to([bx, by, cz + 0.24], GRIP_OPEN, 300, "retract")
    cubef = meta.cube_pos(c.d)
    place = bool(state["pick"] and np.linalg.norm(cubef[:2] - np.array([bx, by])) < 0.06 and cubef[2] < 0.10)
    tally["done"] += 1; tally["ok"] += int(place)
    del rnd
    return place


def record(out_path=None, fps=30, episodes=4):
    out_path = out_path or os.path.join(RESULTS, "pandapick_demo.mp4")
    os.makedirs(RESULTS, exist_ok=True)
    frames = []
    frames += _card([("PandaPick", 60, COL["ink"]),
                     ("Pick-and-Place Data-Collection Pipeline", 26, COL["accent"])],
                    90, subs=["Franka Emika Panda  x  MuJoCo",
                              "autonomous demos with domain randomization -> imitation dataset"])
    tally = {"ok": 0, "done": 0, "steps": 0}
    for ep in range(episodes):
        frames += _card([(f"Episode {ep+1}", 40, COL["ink"])], 24,
                        subs=["randomized cube position / size / mass / colour"])
        _render_episode(ep, episodes, tally, frames)
    frames += _card([("PandaPick", 54, COL["ink"]),
                     ("100% pick & place  ·  ~18 mm accuracy  ·  obs/action dataset", 20, COL["accent"])],
                    90, subs=["pip install -r requirements.txt  &&  python run.py",
                              "pure CPU  ·  no GPU  ·  one command"])
    imageio.mimwrite(out_path, frames, fps=fps, quality=8, macro_block_size=1)
    print(f"[OK] {len(frames)} frame ({len(frames)/fps:.0f}s) -> {out_path}")
    return out_path
