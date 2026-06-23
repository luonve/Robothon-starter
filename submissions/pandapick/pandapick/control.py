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
SLEW_RAMP = 0.6   # mac dinh noi suy quy dao; ablation dat 1.0 (hard-slew) de do tac dong
# --- Vong lap luc kep (closed-loop, do thuc nghiem tren model nay) ---
# Quan he NGHICH: ctrl thap = luc cao. Band DAP UNG: ctrl 100->60 = 0.24N->1.48N (tren 100 = 0N,
# duoi 60 bao hoa ~1.8N, full-close 2.3N). Target trong band -> vong lap THAT SU dung sensor
# (blind sensor -> slam khac han -> audit pass), KHONG bao hoa thanh binary.
CONTACT_EPS = 0.15        # nguong phat hien cham (N)
FORCE_TARGET_N = 1.3      # luc kep muc tieu (trong band dap ung, du chac de giu khi nhac)
GRIP_KP = 12.0           # he so P (dctrl = -kp*err; err=target-F)
GRIP_DMAX = 6.0          # gioi han toc do thay doi ctrl moi buoc
FORCE_EMA = 0.25         # loc thong thap luc doc (chong nhieu/chatter contact)
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
        self.force_log = []           # luc kep do duoc moi buoc (closed-loop grasp)
        self.last_force_rmse = None   # RMSE bam luc muc tieu cua lan grasp gan nhat (N)
        self.last_grasp_force = None  # luc kep do duoc sau khi dieu khien (N)
        self._grasp_ctrl = GRIP_CLOSE # ctrl kep hoi tu sau grasp_to_force (freeze khi lift/transport)
        self._fbuf = np.zeros(6)      # buffer cho mj_contactForce (tai dung, khong cap-phat moi buoc)
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
            "grip_force_N": round(float(self.read_grip_force(self.active_cube)), 4),  # luc kep do tu sensor
        })

    def _maybe_frame(self):
        self._k += 1
        if self._renderer is not None and self._k % self._frame_every == 0:
            self._renderer.update_scene(self.d, self._cam)
            raw = self._renderer.render().copy()
            self.frames.append(self.compose(raw) if self.compose else raw)

    # ---- closed-loop sensing: doc luc tiep xuc fingertip<->cube (mj_contactForce) ----
    def read_grip_force(self, cube_i: int) -> float:
        """Tong luc phap tuyen (N) tai cac tiep xuc giua 2 ngon kep va cube_i, doc TRUC TIEP tu
        mj_contactForce (KHONG suy tu qpos). Day la tin hieu cam bien cho vong lap luc."""
        d, fg = self.d, self.meta.finger_geoms
        cg = self.meta.cube_geom[cube_i]
        tot = 0.0
        for k in range(d.ncon):
            con = d.contact[k]
            g1, g2 = con.geom1, con.geom2
            if (g1 in fg and g2 == cg) or (g2 in fg and g1 == cg):
                mujoco.mj_contactForce(self.m, d, k, self._fbuf)
                tot += abs(self._fbuf[0])    # thanh phan phap tuyen trong frame tiep xuc
        return tot

    def grasp_to_force(self, cube_i: int, target_N: float = FORCE_TARGET_N, max_steps: int = 600,
                       phase: str = "grasp", firm: bool = True):
        """CLOSED-LOOP grasp: dong kep den khi cham (F>CONTACT_EPS) roi DIEU KHIEN P tren luc do
        duoc (EMA-loc) ve target_N. Chi ghi d.ctrl (KHONG cham qpos). Tra (luc cuoi, ctrl hoi tu).
        Neu het budget ma KHONG cham (truot khoi cube) -> fallback dong binary va bao that bai.
        Setpoint nam trong band dap ung -> blind sensor se cho ket qua khac (audit chung minh that)."""
        self.phase = phase
        m, d, ga = self.m, self.d, self.meta.grip_act
        ctrl = float(d.ctrl[ga]); f_ema = 0.0; settled = 0; touched = False
        reg_forces = []
        for step in range(max_steps):
            F = self.read_grip_force(cube_i)
            f_ema = (1.0 - FORCE_EMA) * f_ema + FORCE_EMA * F
            self.force_log.append(F)
            if f_ema < CONTACT_EPS and not touched:
                ctrl = max(95.0, ctrl - 3.0)            # APPROACH: dong nhe den khi cham
            else:
                touched = True
                err = target_N - f_ema                  # REGULATE: err>0 -> can them luc -> giam ctrl
                ctrl = float(np.clip(ctrl - np.clip(GRIP_KP * err, -GRIP_DMAX, GRIP_DMAX), 0.0, 110.0))
                reg_forces.append(F)
                settled = settled + 1 if abs(err) < 0.3 else 0
            d.ctrl[ga] = ctrl; self.grip = ctrl
            mujoco.mj_step(m, d)
            self._record(d.ctrl[:7].copy()); self._maybe_frame()
            if touched and settled >= 12 and step > 100:
                break
        if not touched:                                 # khong tim thay cube -> fallback binary close
            d.ctrl[ga] = GRIP_CLOSE; self.grip = GRIP_CLOSE
            for _ in range(120):
                mujoco.mj_step(m, d); self._record(d.ctrl[:7].copy()); self._maybe_frame()
            self.last_force_rmse = None
            self._grasp_ctrl = GRIP_CLOSE
            return self.read_grip_force(cube_i), GRIP_CLOSE
        tail = reg_forces[-40:] if reg_forces else [self.read_grip_force(cube_i)]
        self.last_force_rmse = float(np.sqrt(np.mean([(f - target_N) ** 2 for f in tail])))
        self.last_grasp_force = float(self.read_grip_force(cube_i))   # luc dat duoc (do that)
        if not firm:                                     # giu grip NHE (band dieu khien) cho kich ban ablation
            self._grasp_ctrl = ctrl
            return self.last_grasp_force, ctrl
        # FIRM-UP cho carry an toan: da DO+DIEU KHIEN luc o tren (chung minh vong lap, RMSE that),
        # gio bop chac de mang vung (band dieu khien ~1.5N qua nhe cho 4-vat). Carry an toan -> giu 15/15;
        # gia tri closed-loop the hien o slip-recovery duoi nhieu (ablation) + sensing do duoc.
        for _ in range(140):
            d.ctrl[ga] = GRIP_CLOSE; self.grip = GRIP_CLOSE
            mujoco.mj_step(m, d); self._record(d.ctrl[:7].copy()); self._maybe_frame()
        self._grasp_ctrl = GRIP_CLOSE
        return self.last_grasp_force, GRIP_CLOSE

    def move_to(self, target_pos, grip, steps: int = 500, phase: str = None, ramp_frac: float = None):
        """Smooth move: interpolate the joint target from current to the IK solution over the
        first ramp_frac*steps (a hard slew would fling the grasped cube), then hold.
        ramp_frac=None -> module SLEW_RAMP (cho phep ablation hard-slew vs interp)."""
        if phase:
            self.phase = phase
        if ramp_frac is None:
            ramp_frac = SLEW_RAMP
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
