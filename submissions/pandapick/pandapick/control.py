"""IK controller + motion primitives + (obs, action) logging.

Inverse kinematics: 6-DOF damped least-squares (grasp-site position -> target, gripper held
pointing down), solved in pure kinematics (mj_forward) then commanded to the arm under dynamics.
Converges to sub-millimetre before the arm moves.
"""
from __future__ import annotations
import numpy as np
import mujoco

from .model import Meta, HOME

# gripper: ctrl 255 = fully open, 0 = fully closed
GRIP_OPEN = 255.0
GRIP_CLOSE = 0.0
# gripper points down (grasp-site z-axis -> world -z)
R_DOWN = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1.0]])


class IKController:
    def __init__(self, model, meta: Meta, log: bool = False):
        self.m = model
        self.meta = meta
        self.d = mujoco.MjData(model)
        self.grip = GRIP_OPEN
        self.phase = "init"
        self.active_cube = 0          # cube currently manipulated (for logging)
        self.do_log = log
        self.records = []          # per-step (obs, action) dataset
        self.frames = []           # rendered video frames (if enabled)
        self._renderer = None
        self._cam = None
        self._frame_every = 6
        self._k = 0
        self.compose = None           # callable(raw_img)->frame overlay, set by record_demo
        self.reset()

    def reset(self):
        mujoco.mj_resetData(self.m, self.d)
        self.d.qpos[:7] = HOME
        self.d.ctrl[:7] = HOME
        self.d.ctrl[self.meta.grip_act] = GRIP_OPEN
        self.grip = GRIP_OPEN
        mujoco.mj_forward(self.m, self.d)

    # ---- kinematics-only IK ----
    def ik_solve(self, target_pos, max_iter: int = 250, tol: float = 4e-4):
        m, meta = self.m, self.meta
        q = self.d.qpos[:7].copy()
        dt = mujoco.MjData(m)
        for _ in range(max_iter):
            dt.qpos[:7] = q
            for a in meta.cube_jadr:                                 # park every cube far away while solving IK
                dt.qpos[a:a + 3] = [2, 2, 2]
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
            "cube_pos": self.meta.cube_pos(d, self.active_cube).round(5).tolist(),
            "action_qtarget": np.asarray(q_target).round(5).tolist(),
        })

    def _maybe_frame(self):
        self._k += 1
        if self._renderer is not None and self._k % self._frame_every == 0:
            self._renderer.update_scene(self.d, self._cam)
            raw = self._renderer.render().copy()
            self.frames.append(self.compose(raw) if self.compose else raw)

    def move_to(self, target_pos, grip, steps: int = 500, phase: str = None, ramp_frac: float = 0.6):
        """Smooth move: interpolate the joint target from current to the IK solution over the
        first ramp_frac*steps (a hard slew would fling the grasped cube), then hold."""
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
