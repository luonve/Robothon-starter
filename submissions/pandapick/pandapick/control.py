"""IK controller + pick-place primitives + ghi du lieu (obs, action).

IK: damped least-squares 6-DOF (vi tri tam kep -> target, huong gripper xuong),
giai theo dong hoc thuan (mj_forward) roi command arm (dynamics). Da validate hoi tu ~0.
"""
from __future__ import annotations
import numpy as np
import mujoco

from .model import Meta, HOME

# gripper: ctrl 255 = mo het, 0 = dong het
GRIP_OPEN = 255.0
GRIP_CLOSE = 0.0
# huong gripper xuong (z-axis tam kep -> -z world)
R_DOWN = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1.0]])


class IKController:
    def __init__(self, model, meta: Meta, log: bool = False):
        self.m = model
        self.meta = meta
        self.d = mujoco.MjData(model)
        self.grip = GRIP_OPEN
        self.phase = "init"
        self.do_log = log
        self.records = []          # dataset (obs, action) tung buoc
        self.frames = []           # frame video (neu bat)
        self._renderer = None
        self._cam = None
        self._frame_every = 6
        self._k = 0
        self.reset()

    def reset(self):
        mujoco.mj_resetData(self.m, self.d)
        self.d.qpos[:7] = HOME
        self.d.ctrl[:7] = HOME
        self.d.ctrl[self.meta.grip_act] = GRIP_OPEN
        self.grip = GRIP_OPEN
        mujoco.mj_forward(self.m, self.d)

    # ---- IK dong hoc thuan ----
    def ik_solve(self, target_pos, max_iter: int = 250, tol: float = 4e-4):
        m, meta = self.m, self.meta
        q = self.d.qpos[:7].copy()
        dt = mujoco.MjData(m)
        for _ in range(max_iter):
            dt.qpos[:7] = q
            dt.qpos[meta.cube_jadr:meta.cube_jadr + 3] = [2, 2, 2]   # day cube ra xa khi giai
            mujoco.mj_forward(m, dt)
            perr = target_pos - dt.site_xpos[meta.grasp_site]
            Rh = dt.site_xmat[meta.grasp_site].reshape(3, 3)
            Re = R_DOWN @ Rh.T
            rerr = 0.5 * np.array([Re[2, 1] - Re[1, 2], Re[0, 2] - Re[2, 0], Re[1, 0] - Re[0, 1]])
            if np.linalg.norm(perr) < tol and np.linalg.norm(rerr) < 1e-2:
                break
            jp = np.zeros((3, m.nv)); jr = np.zeros((3, m.nv))
            mujoco.mj_jacSite(m, dt, jp, jr, meta.grasp_site)
            J = np.vstack([jp[:, :7], jr[:, :7]])
            dq = J.T @ np.linalg.solve(J @ J.T + 1e-4 * np.eye(6), np.concatenate([perr, rerr]))
            q = np.clip(q + dq, meta.arm_range[:, 0], meta.arm_range[:, 1])
        return q

    # ---- command + log ----
    def _record(self, q_target):
        if not self.do_log:
            return
        d = self.d
        self.records.append({
            "phase": self.phase,
            "qpos": d.qpos[:7].round(5).tolist(),
            "qvel": d.qvel[:7].round(4).tolist(),
            "ee_pos": self.meta.ee_pos(d).round(5).tolist(),
            "grip": float(self.grip),
            "cube_pos": self.meta.cube_pos(d).round(5).tolist(),
            "action_qtarget": np.asarray(q_target).round(5).tolist(),
        })

    def _maybe_frame(self):
        self._k += 1
        if self._renderer is not None and self._k % self._frame_every == 0:
            self._renderer.update_scene(self.d, self._cam)
            self.frames.append(self._renderer.render().copy())

    def move_to(self, target_pos, grip, steps: int = 500, phase: str = None, ramp_frac: float = 0.6):
        """Di chuyen MUOT: noi suy joint target tu hien tai -> nghiem IK trong ramp_frac*steps
        buoc dau (tranh arm vung nhanh lam vat van khoi tay), roi giu phan con lai."""
        if phase:
            self.phase = phase
        q_goal = self.ik_solve(target_pos)
        q_start = self.d.qpos[:7].copy()
        self.grip = grip
        self.d.ctrl[self.meta.grip_act] = grip
        ramp = max(1, int(steps * ramp_frac))
        for i in range(steps):
            a = min(1.0, (i + 1) / ramp)
            qcmd = q_start + a * (q_goal - q_start)
            self.d.ctrl[:7] = qcmd
            mujoco.mj_step(self.m, self.d)
            self._record(qcmd)
            self._maybe_frame()
        return np.linalg.norm(target_pos - self.meta.ee_pos(self.d))

    def set_grip(self, grip, steps: int = 500, phase: str = None):
        if phase:
            self.phase = phase
        self.grip = grip
        self.d.ctrl[self.meta.grip_act] = grip
        for _ in range(steps):
            mujoco.mj_step(self.m, self.d)
            self._record(self.d.ctrl[:7].copy())
            self._maybe_frame()
