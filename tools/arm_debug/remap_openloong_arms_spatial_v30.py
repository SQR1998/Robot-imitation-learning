#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

import mujoco as mj
import numpy as np
import smplx
import torch
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as R
from smplx.joint_names import JOINT_NAMES

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from general_motion_retargeting import ROBOT_XML_DICT, RobotMotionViewer


DEFAULT_BODY_CSV = (
    ROOT
    / "output/openloong_v2_kneeguide/csv/openloong/live_motion.csv"
)
DEFAULT_SMPLX = (
    ROOT
    / "output/openloong_v2_kneeguide/stream_demo/gmr_smplx_results.npz"
)
DEFAULT_OUT_DIR = ROOT / "output/openloong_v3_armspatial_v1"
SMPLX_FOLDER = ROOT / "assets/body_models"

ARM_JOINTS = {
    "left": [f"J_arm_l_{i:02d}" for i in range(1, 8)],
    "right": [f"J_arm_r_{i:02d}" for i in range(1, 8)],
}
ARM_BODY = {
    "left": {
        "shoulder": "Link_arm_l_01",
        "elbow": "Link_arm_l_04",
        "wrist": "Link_arm_l_07",
    },
    "right": {
        "shoulder": "Link_arm_r_01",
        "elbow": "Link_arm_r_04",
        "wrist": "Link_arm_r_07",
    },
}
HUMAN_JOINTS = (
    "pelvis",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
)


def unit(v: np.ndarray, fallback: np.ndarray | None = None) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    n = float(np.linalg.norm(v))
    if n > 1e-10:
        return v / n
    if fallback is None:
        return np.zeros_like(v)
    return unit(np.asarray(fallback, dtype=np.float64))


def csv_row_to_qpos(row: np.ndarray) -> np.ndarray:
    q = np.asarray(row, dtype=np.float64).copy()
    q[3:7] = row[[6, 3, 4, 5]]  # CSV xyzw -> MuJoCo wxyz
    return q


def qpos_to_csv_row(qpos: np.ndarray) -> np.ndarray:
    q = np.asarray(qpos, dtype=np.float64)
    row = q.copy()
    row[3:7] = q[3:7][[1, 2, 3, 0]]  # MuJoCo wxyz -> CSV xyzw
    return row


def body_rot(data: mj.MjData, body_id: int) -> np.ndarray:
    return np.asarray(data.xmat[body_id], dtype=np.float64).reshape(3, 3)


def segment_distance(
    p1: np.ndarray,
    q1: np.ndarray,
    p2: np.ndarray,
    q2: np.ndarray,
) -> float:
    """Shortest Euclidean distance between two 3-D line segments."""
    p1 = np.asarray(p1, dtype=np.float64)
    q1 = np.asarray(q1, dtype=np.float64)
    p2 = np.asarray(p2, dtype=np.float64)
    q2 = np.asarray(q2, dtype=np.float64)

    d1 = q1 - p1
    d2 = q2 - p2
    r = p1 - p2
    a = float(np.dot(d1, d1))
    e = float(np.dot(d2, d2))
    eps = 1e-12

    if a <= eps and e <= eps:
        return float(np.linalg.norm(p1 - p2))

    if a <= eps:
        s = 0.0
        t = float(np.clip(np.dot(d2, r) / e, 0.0, 1.0))
    else:
        c = float(np.dot(d1, r))
        if e <= eps:
            t = 0.0
            s = float(np.clip(-c / a, 0.0, 1.0))
        else:
            b = float(np.dot(d1, d2))
            denom = a * e - b * b
            if abs(denom) > eps:
                s = float(
                    np.clip(
                        (b * np.dot(d2, r) - c * e) / denom,
                        0.0,
                        1.0,
                    )
                )
            else:
                s = 0.0
            t = float((b * s + np.dot(d2, r)) / e)
            if t < 0.0:
                t = 0.0
                s = float(np.clip(-c / a, 0.0, 1.0))
            elif t > 1.0:
                t = 1.0
                s = float(np.clip((b - c) / a, 0.0, 1.0))

    c1 = p1 + d1 * s
    c2 = p2 + d2 * t
    return float(np.linalg.norm(c1 - c2))


def signed_distance_to_obb(
    point: np.ndarray,
    center: np.ndarray,
    basis: np.ndarray,
    half: np.ndarray,
) -> float:
    """SDF of an oriented box. Positive outside, negative inside."""
    local = basis.T @ (np.asarray(point, dtype=np.float64) - center)
    q = np.abs(local) - half
    outside = float(np.linalg.norm(np.maximum(q, 0.0)))
    inside = float(min(max(q[0], q[1], q[2]), 0.0))
    return outside + inside


class OpenLoongArmSpatialRetargeter:
    def __init__(self, model: mj.MjModel, args: argparse.Namespace):
        self.model = model
        self.data = mj.MjData(model)
        self.args = args

        self.base_body = self._body_id("base_link")
        self.body_ids = {
            side: {
                key: self._body_id(name)
                for key, name in ARM_BODY[side].items()
            }
            for side in ("left", "right")
        }

        # Never hard-code CSV/qpos indices. Read them from the MuJoCo model.
        self.arm_qpos = {
            side: [self._joint_qpos_index(name) for name in ARM_JOINTS[side]]
            for side in ("left", "right")
        }
        self.arm_qpos_all = self.arm_qpos["left"] + self.arm_qpos["right"]

        self.lower, self.upper = self._arm_bounds()
        self.arm_body_sets = self._build_arm_body_sets()
        self.upper_len, self.forearm_len = self._measure_robot_arm_lengths()

        self.prev_solution: np.ndarray | None = None
        self.prev_output_arm: np.ndarray | None = None

    def _body_id(self, name: str) -> int:
        idx = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_BODY, name)
        if idx < 0:
            raise RuntimeError(f"MuJoCo body not found: {name}")
        return int(idx)

    def _joint_qpos_index(self, name: str) -> int:
        jid = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            raise RuntimeError(f"MuJoCo joint not found: {name}")
        qidx = int(self.model.jnt_qposadr[jid])
        if qidx < 7:
            raise RuntimeError(f"Unexpected arm qpos index for {name}: {qidx}")
        return qidx

    def _arm_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        lower = []
        upper = []
        for name in ARM_JOINTS["left"] + ARM_JOINTS["right"]:
            jid = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_JOINT, name)
            if jid < 0:
                raise RuntimeError(name)
            if bool(self.model.jnt_limited[jid]):
                lo, hi = self.model.jnt_range[jid]
            else:
                lo, hi = -np.pi, np.pi
            lower.append(float(lo))
            upper.append(float(hi))
        return np.asarray(lower), np.asarray(upper)

    def _build_arm_body_sets(self) -> dict[str, set[int]]:
        out = {"left": set(), "right": set()}
        for bid in range(int(self.model.nbody)):
            name = mj.mj_id2name(self.model, mj.mjtObj.mjOBJ_BODY, bid) or ""
            if name.startswith("Link_arm_l_"):
                out["left"].add(bid)
            elif name.startswith("Link_arm_r_"):
                out["right"].add(bid)
        return out

    def _measure_robot_arm_lengths(
        self,
    ) -> tuple[dict[str, float], dict[str, float]]:
        q = np.asarray(self.model.qpos0, dtype=np.float64).copy()
        self.data.qpos[:] = q
        mj.mj_forward(self.model, self.data)

        upper = {}
        fore = {}
        for side in ("left", "right"):
            s = np.asarray(
                self.data.xpos[self.body_ids[side]["shoulder"]],
                dtype=np.float64,
            )
            e = np.asarray(
                self.data.xpos[self.body_ids[side]["elbow"]],
                dtype=np.float64,
            )
            w = np.asarray(
                self.data.xpos[self.body_ids[side]["wrist"]],
                dtype=np.float64,
            )
            upper[side] = float(np.linalg.norm(e - s))
            fore[side] = float(np.linalg.norm(w - e))
        return upper, fore

    def set_q(self, q: np.ndarray) -> None:
        self.data.qpos[:] = q
        mj.mj_forward(self.model, self.data)

    def endpoints(self) -> dict[str, dict[str, np.ndarray]]:
        return {
            side: {
                key: np.asarray(self.data.xpos[bid], dtype=np.float64).copy()
                for key, bid in self.body_ids[side].items()
            }
            for side in ("left", "right")
        }

    def robot_torso_frame(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        ep = self.endpoints()
        left_s = ep["left"]["shoulder"]
        right_s = ep["right"]["shoulder"]
        shoulder_center = 0.5 * (left_s + right_s)
        base = np.asarray(
            self.data.xpos[self.base_body],
            dtype=np.float64,
        ).copy()

        lateral = unit(
            left_s - right_s,
            np.array([0.0, 1.0, 0.0]),
        )
        up = unit(
            shoulder_center - base,
            np.array([0.0, 0.0, 1.0]),
        )
        forward = unit(
            np.cross(lateral, up),
            np.array([1.0, 0.0, 0.0]),
        )

        base_forward = (
            body_rot(self.data, self.base_body)
            @ np.array([1.0, 0.0, 0.0])
        )
        if float(np.dot(forward, base_forward)) < 0.0:
            forward = -forward

        up = unit(np.cross(forward, lateral), up)
        basis = np.column_stack([forward, lateral, up])

        torso_center = 0.5 * (base + shoulder_center)
        shoulder_width = float(np.linalg.norm(left_s - right_s))
        torso_height = float(np.linalg.norm(shoulder_center - base))

        half = np.array(
            [
                self.args.torso_half_x,
                max(
                    0.10,
                    0.5
                    * shoulder_width
                    * self.args.torso_half_y_scale,
                ),
                max(
                    0.15,
                    0.5 * torso_height
                    + self.args.torso_z_padding,
                ),
            ],
            dtype=np.float64,
        )
        return torso_center, basis, half, shoulder_center

    @staticmethod
    def human_torso_frame(
        human: dict[str, np.ndarray],
    ) -> np.ndarray:
        ls = human["left_shoulder"]
        rs = human["right_shoulder"]
        pelvis = human["pelvis"]
        center = 0.5 * (ls + rs)

        # After the checkpoint's y-up -> z-up conversion:
        # lateral = human right->left, up = pelvis->shoulder center,
        # forward = lateral x up.
        lateral = unit(
            ls - rs,
            np.array([1.0, 0.0, 0.0]),
        )
        up = unit(
            center - pelvis,
            np.array([0.0, 0.0, 1.0]),
        )
        forward = unit(
            np.cross(lateral, up),
            np.array([0.0, -1.0, 0.0]),
        )
        up = unit(np.cross(forward, lateral), up)
        return np.column_stack([forward, lateral, up])

    def build_targets(
        self,
        human: dict[str, np.ndarray],
    ) -> tuple[
        dict[str, dict[str, np.ndarray]],
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    ]:
        ep = self.endpoints()
        human_basis = self.human_torso_frame(human)
        torso = self.robot_torso_frame()
        _, robot_basis, _, _ = torso

        targets: dict[str, dict[str, np.ndarray]] = {}

        for side in ("left", "right"):
            hs = human[f"{side}_shoulder"]
            he = human[f"{side}_elbow"]
            hw = human[f"{side}_wrist"]

            # Keep human arm direction in the torso-local coordinate system,
            # but reconstruct the target with OpenLoong's own limb lengths.
            upper_h = unit(he - hs)
            fore_h = unit(hw - he)

            upper_local = human_basis.T @ upper_h
            fore_local = human_basis.T @ fore_h

            upper_robot = unit(robot_basis @ upper_local)
            fore_robot = unit(robot_basis @ fore_local)

            s_robot = ep[side]["shoulder"]
            elbow_target = (
                s_robot
                + upper_robot
                * self.upper_len[side]
                * self.args.upper_length_scale
            )
            wrist_target = (
                elbow_target
                + fore_robot
                * self.forearm_len[side]
                * self.args.forearm_length_scale
            )

            targets[side] = {
                "shoulder": s_robot.copy(),
                "elbow": elbow_target,
                "wrist": wrist_target,
            }

        # Collision safety is applied to spatial targets BEFORE IK.
        self.project_targets_collision_safe(targets, torso)
        return targets, torso

    def _project_point_outside_torso(
        self,
        point: np.ndarray,
        side: str,
        torso: tuple[
            np.ndarray,
            np.ndarray,
            np.ndarray,
            np.ndarray,
        ],
        margin: float,
        prefer_side: bool,
    ) -> np.ndarray:
        center, basis, half, _ = torso
        local = basis.T @ (point - center)
        expanded = half + float(margin)

        if not np.all(np.abs(local) < expanded):
            return point

        side_sign = 1.0 if side == "left" else -1.0
        eps = 0.003

        # Preserve front/back intent: a hand behind the back is projected to
        # the back face, not forcibly teleported in front of the chest.
        x_sign = 1.0 if local[0] >= 0.0 else -1.0

        x_candidate = local.copy()
        x_candidate[0] = x_sign * (expanded[0] + eps)

        y_candidate = local.copy()
        y_candidate[1] = side_sign * (expanded[1] + eps)

        dx = float(np.linalg.norm(x_candidate - local))
        dy = float(np.linalg.norm(y_candidate - local))

        if prefer_side:
            choose_y = dy * 0.88 <= dx * 1.12
        else:
            choose_y = dy <= dx * 1.06

        chosen = y_candidate if choose_y else x_candidate
        return center + basis @ chosen

    def project_targets_collision_safe(
        self,
        targets: dict[str, dict[str, np.ndarray]],
        torso: tuple[
            np.ndarray,
            np.ndarray,
            np.ndarray,
            np.ndarray,
        ],
    ) -> None:
        # 1) Torso clearance before IK.
        for side in ("left", "right"):
            targets[side]["elbow"] = (
                self._project_point_outside_torso(
                    targets[side]["elbow"],
                    side,
                    torso,
                    self.args.elbow_torso_margin,
                    prefer_side=True,
                )
            )
            targets[side]["wrist"] = (
                self._project_point_outside_torso(
                    targets[side]["wrist"],
                    side,
                    torso,
                    self.args.hand_torso_margin,
                    prefer_side=False,
                )
            )

        # 2) Hand-hand target clearance.
        wl = targets["left"]["wrist"]
        wr = targets["right"]["wrist"]
        delta = wl - wr
        dist = float(np.linalg.norm(delta))

        if dist < self.args.hand_hand_min:
            if dist < 1e-8:
                direction = torso[1][:, 1]
            else:
                direction = delta / dist

            push = 0.5 * (
                self.args.hand_hand_min
                - dist
                + 0.003
            )
            targets["left"]["wrist"] = wl + direction * push
            targets["right"]["wrist"] = wr - direction * push

        # 3) Forearm target crossing/clearance.
        el = targets["left"]["elbow"]
        er = targets["right"]["elbow"]
        wl = targets["left"]["wrist"]
        wr = targets["right"]["wrist"]

        fore_dist = segment_distance(
            el,
            wl,
            er,
            wr,
        )

        if fore_dist < self.args.forearm_forearm_min:
            lateral = torso[1][:, 1]
            push = 0.5 * (
                self.args.forearm_forearm_min
                - fore_dist
                + 0.003
            )
            targets["left"]["wrist"] = wl + lateral * push
            targets["right"]["wrist"] = wr - lateral * push

        # Separation can move a wrist toward the torso, so recheck.
        for side in ("left", "right"):
            targets[side]["wrist"] = (
                self._project_point_outside_torso(
                    targets[side]["wrist"],
                    side,
                    torso,
                    self.args.hand_torso_margin,
                    prefer_side=False,
                )
            )

    def _set_arm_vector(
        self,
        q: np.ndarray,
        z: np.ndarray,
    ) -> np.ndarray:
        out = q.copy()
        out[self.arm_qpos_all] = z
        return out

    def _arm_vector(
        self,
        q: np.ndarray,
    ) -> np.ndarray:
        return np.asarray(
            q[self.arm_qpos_all],
            dtype=np.float64,
        ).copy()

    def _actual_collision_depth(self) -> float:
        total_sq = 0.0
        all_arm = (
            self.arm_body_sets["left"]
            | self.arm_body_sets["right"]
        )

        for i in range(int(self.data.ncon)):
            con = self.data.contact[i]

            b1 = int(
                self.model.geom_bodyid[
                    int(con.geom1)
                ]
            )
            b2 = int(
                self.model.geom_bodyid[
                    int(con.geom2)
                ]
            )

            # Ignore world/floor here; V3.0 focuses on arm-body/self collision.
            if b1 == 0 or b2 == 0:
                continue

            if b1 not in all_arm and b2 not in all_arm:
                continue

            depth = max(
                0.0,
                -float(con.dist),
            )
            total_sq += depth * depth

        return math.sqrt(total_sq)

    def _collision_contact_count(self) -> int:
        count = 0
        all_arm = (
            self.arm_body_sets["left"]
            | self.arm_body_sets["right"]
        )

        for i in range(int(self.data.ncon)):
            con = self.data.contact[i]

            b1 = int(
                self.model.geom_bodyid[
                    int(con.geom1)
                ]
            )
            b2 = int(
                self.model.geom_bodyid[
                    int(con.geom2)
                ]
            )

            if b1 == 0 or b2 == 0:
                continue

            if b1 not in all_arm and b2 not in all_arm:
                continue

            if float(con.dist) < 0.0:
                count += 1

        return count

    def solve(
        self,
        body_q: np.ndarray,
        human: dict[str, np.ndarray],
    ) -> tuple[np.ndarray, dict[str, float]]:
        # Important ordering:
        # 1. body_q is the already-validated final body pose;
        # 2. freeze it;
        # 3. discard/re-solve all 14 arm qpos values jointly.
        self.set_q(body_q)

        original_arm = self._arm_vector(body_q)
        targets, torso = self.build_targets(human)

        # Frame 0 is fully solved too. Original GMR is only an initial guess,
        # never a post-hoc patch target.
        if self.prev_solution is None:
            x0 = original_arm.copy()
        else:
            x0 = self.prev_solution.copy()

        x0 = np.clip(
            x0,
            self.lower + 1e-7,
            self.upper - 1e-7,
        )

        def residual(z: np.ndarray) -> np.ndarray:
            q = self._set_arm_vector(
                body_q,
                z,
            )
            self.set_q(q)
            ep = self.endpoints()

            res: list[float] = []

            # Spatial arm tracking.
            for side in ("left", "right"):
                res.extend(
                    (
                        self.args.elbow_weight
                        * (
                            ep[side]["elbow"]
                            - targets[side]["elbow"]
                        )
                    ).tolist()
                )
                res.extend(
                    (
                        self.args.wrist_weight
                        * (
                            ep[side]["wrist"]
                            - targets[side]["wrist"]
                        )
                    ).tolist()
                )

            center, basis, half, _ = torso

            # Fixed-dimension torso clearance:
            # elbow + 1/3 forearm + 2/3 forearm + wrist.
            for side in ("left", "right"):
                e = ep[side]["elbow"]
                w = ep[side]["wrist"]

                points = [
                    (
                        e,
                        self.args.elbow_torso_margin,
                    ),
                    (
                        e + (w - e) / 3.0,
                        self.args.forearm_torso_margin,
                    ),
                    (
                        e + 2.0 * (w - e) / 3.0,
                        self.args.forearm_torso_margin,
                    ),
                    (
                        w,
                        self.args.hand_torso_margin,
                    ),
                ]

                for p, margin in points:
                    sd = signed_distance_to_obb(
                        p,
                        center,
                        basis,
                        half,
                    )
                    shortage = max(
                        0.0,
                        float(margin) - sd,
                    )
                    res.append(
                        self.args.torso_collision_weight
                        * shortage
                    )

            # Bilateral collision constraints are part of IK itself.
            hand_dist = float(
                np.linalg.norm(
                    ep["left"]["wrist"]
                    - ep["right"]["wrist"]
                )
            )
            res.append(
                self.args.hand_hand_weight
                * max(
                    0.0,
                    self.args.hand_hand_min
                    - hand_dist,
                )
            )

            fore_dist = segment_distance(
                ep["left"]["elbow"],
                ep["left"]["wrist"],
                ep["right"]["elbow"],
                ep["right"]["wrist"],
            )
            res.append(
                self.args.forearm_forearm_weight
                * max(
                    0.0,
                    self.args.forearm_forearm_min
                    - fore_dist,
                )
            )

            # Real MuJoCo mesh contact is a second safety layer.
            res.append(
                self.args.contact_weight
                * self._actual_collision_depth()
            )

            # Temporal continuity is an optimization term, not output clipping.
            if self.prev_solution is not None:
                res.extend(
                    (
                        self.args.temporal_weight
                        * (
                            z
                            - self.prev_solution
                        )
                    ).tolist()
                )
            else:
                res.extend(
                    np.zeros_like(z).tolist()
                )

            # Tiny preference only. Old GMR arms are not preserved.
            res.extend(
                (
                    self.args.original_weight
                    * (
                        z
                        - original_arm
                    )
                ).tolist()
            )

            return np.asarray(
                res,
                dtype=np.float64,
            )

        result = least_squares(
            residual,
            x0=x0,
            bounds=(
                self.lower,
                self.upper,
            ),
            method="trf",
            max_nfev=self.args.max_nfev,
            ftol=self.args.ftol,
            xtol=self.args.xtol,
            gtol=self.args.gtol,
            verbose=0,
        )

        z = np.asarray(
            result.x,
            dtype=np.float64,
        )

        q_out = self._set_arm_vector(
            body_q,
            z,
        )
        self.set_q(q_out)

        ep = self.endpoints()
        center, basis, half, _ = torso

        wrist_err_l = float(
            np.linalg.norm(
                ep["left"]["wrist"]
                - targets["left"]["wrist"]
            )
        )
        wrist_err_r = float(
            np.linalg.norm(
                ep["right"]["wrist"]
                - targets["right"]["wrist"]
            )
        )
        elbow_err_l = float(
            np.linalg.norm(
                ep["left"]["elbow"]
                - targets["left"]["elbow"]
            )
        )
        elbow_err_r = float(
            np.linalg.norm(
                ep["right"]["elbow"]
                - targets["right"]["elbow"]
            )
        )

        hand_dist = float(
            np.linalg.norm(
                ep["left"]["wrist"]
                - ep["right"]["wrist"]
            )
        )

        fore_dist = segment_distance(
            ep["left"]["elbow"],
            ep["left"]["wrist"],
            ep["right"]["elbow"],
            ep["right"]["wrist"],
        )

        torso_clearances = []
        for side in ("left", "right"):
            e = ep[side]["elbow"]
            w = ep[side]["wrist"]

            for p in (
                e,
                e + (w - e) / 3.0,
                e + 2.0 * (w - e) / 3.0,
                w,
            ):
                torso_clearances.append(
                    signed_distance_to_obb(
                        p,
                        center,
                        basis,
                        half,
                    )
                )

        if self.prev_output_arm is None:
            max_step = 0.0
        else:
            max_step = float(
                np.max(
                    np.abs(
                        z
                        - self.prev_output_arm
                    )
                )
            )

        self.prev_solution = z.copy()
        self.prev_output_arm = z.copy()

        diag = {
            "cost": float(result.cost),
            "nfev": float(result.nfev),
            "optimality": float(result.optimality),
            "success": float(bool(result.success)),
            "wrist_err_l": wrist_err_l,
            "wrist_err_r": wrist_err_r,
            "elbow_err_l": elbow_err_l,
            "elbow_err_r": elbow_err_r,
            "hand_hand_dist": hand_dist,
            "forearm_forearm_dist": fore_dist,
            "min_torso_clearance": float(
                min(torso_clearances)
            ),
            "contact_count": float(
                self._collision_contact_count()
            ),
            "contact_depth": float(
                self._actual_collision_depth()
            ),
            "max_joint_step": max_step,
        }

        return q_out, diag


def load_human_joint_positions(
    smplx_path: Path,
) -> tuple[dict[str, np.ndarray], int]:
    """
    V3-local SMPL-X loader.

    Important:
    - Do NOT modify the stable V2 SMPL/GMR loader.
    - Explicitly expand betas/expression to every frame.
    - Preserve the checkpoint's y-up -> z-up transform.
    """
    smplx_data = np.load(
        smplx_path,
        allow_pickle=True,
    )

    num_frames = int(
        smplx_data["pose_body"].shape[0]
    )

    gender_raw = np.asarray(
        smplx_data["gender"]
    )
    if gender_raw.shape == ():
        gender = str(gender_raw.item())
    else:
        gender = str(
            gender_raw.reshape(-1)[0]
        )

    print(
        "[V3 SMPL-X] frames:",
        num_frames,
    )
    print(
        "[V3 SMPL-X] gender:",
        gender,
    )

    body_model = smplx.create(
        str(SMPLX_FOLDER),
        "smplx",
        gender=gender,
        use_pca=False,
        batch_size=num_frames,
    )

    root_orient = np.asarray(
        smplx_data["root_orient"],
        dtype=np.float32,
    ).reshape(
        num_frames,
        3,
    )

    trans = np.asarray(
        smplx_data["trans"],
        dtype=np.float32,
    ).reshape(
        num_frames,
        3,
    )

    # Keep exactly the same coordinate convention used by the
    # checkpoint pipeline: WHAM y-up -> GMR/OpenLoong z-up.
    rotation_matrix = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )

    rot_fix = R.from_matrix(
        rotation_matrix
    )

    root_orient = (
        rot_fix
        * R.from_rotvec(root_orient)
    ).as_rotvec().astype(
        np.float32
    )

    trans = (
        trans
        @ rotation_matrix.T
    ).astype(
        np.float32
    )

    print(
        "[V3 SMPL-X] Applied coordinate fix: "
        "y-up -> z-up"
    )

    # -----------------------------------------
    # Shape parameters
    # -----------------------------------------
    raw_betas = np.asarray(
        smplx_data["betas"],
        dtype=np.float32,
    ).reshape(-1)

    target_beta_dim = int(
        getattr(
            body_model,
            "num_betas",
            raw_betas.shape[0],
        )
    )

    if raw_betas.shape[0] < target_beta_dim:
        raw_betas = np.pad(
            raw_betas,
            (
                0,
                target_beta_dim
                - raw_betas.shape[0],
            ),
        )
    elif raw_betas.shape[0] > target_beta_dim:
        raw_betas = raw_betas[
            :target_beta_dim
        ]

    # Critical fix:
    # betas and expression both explicitly use N frames.
    betas = np.repeat(
        raw_betas[None, :],
        num_frames,
        axis=0,
    )

    if hasattr(
        body_model,
        "num_expression_coeffs",
    ):
        expression_dim = int(
            body_model.num_expression_coeffs
        )
    elif hasattr(
        body_model,
        "expression",
    ):
        expression_dim = int(
            body_model.expression.shape[-1]
        )
    else:
        expression_dim = 10

    print(
        "[V3 SMPL-X] betas:",
        betas.shape,
    )
    print(
        "[V3 SMPL-X] expression:",
        (
            num_frames,
            expression_dim,
        ),
    )

    with torch.no_grad():
        smplx_output = body_model(
            betas=torch.tensor(
                betas,
                dtype=torch.float32,
            ),
            expression=torch.zeros(
                (
                    num_frames,
                    expression_dim,
                ),
                dtype=torch.float32,
            ),
            global_orient=torch.tensor(
                root_orient,
                dtype=torch.float32,
            ),
            body_pose=torch.tensor(
                np.asarray(
                    smplx_data["pose_body"],
                    dtype=np.float32,
                ),
                dtype=torch.float32,
            ),
            transl=torch.tensor(
                trans,
                dtype=torch.float32,
            ),
            left_hand_pose=torch.zeros(
                num_frames,
                45,
                dtype=torch.float32,
            ),
            right_hand_pose=torch.zeros(
                num_frames,
                45,
                dtype=torch.float32,
            ),
            jaw_pose=torch.zeros(
                num_frames,
                3,
                dtype=torch.float32,
            ),
            leye_pose=torch.zeros(
                num_frames,
                3,
                dtype=torch.float32,
            ),
            reye_pose=torch.zeros(
                num_frames,
                3,
                dtype=torch.float32,
            ),
            return_full_pose=True,
        )

    joints = (
        smplx_output.joints
        .detach()
        .cpu()
        .numpy()
    )

    names = JOINT_NAMES[
        : joints.shape[1]
    ]

    name_to_idx = {
        name: i
        for i, name in enumerate(names)
    }

    missing = [
        name
        for name in HUMAN_JOINTS
        if name not in name_to_idx
    ]

    if missing:
        raise RuntimeError(
            f"SMPL-X joints missing: {missing}"
        )

    out = {
        name: np.asarray(
            joints[
                :,
                name_to_idx[name],
                :,
            ],
            dtype=np.float64,
        )
        for name in HUMAN_JOINTS
    }

    print(
        "[V3 SMPL-X] joints loaded:",
        ", ".join(HUMAN_JOINTS),
    )

    return out, int(
        joints.shape[0]
    )


def human_frame(
    human_seq: dict[str, np.ndarray],
    i: int,
) -> dict[str, np.ndarray]:
    return {
        name: values[i].copy()
        for name, values in human_seq.items()
    }


def render_motion(
    qpos_seq: np.ndarray,
    indices: list[int],
    fps: float,
    video_path: Path,
) -> None:
    video_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    viewer = RobotMotionViewer(
        robot_type="openloong",
        motion_fps=fps,
        transparent_robot=0,
        record_video=True,
        video_path=str(video_path),
        camera_follow=True,
    )

    try:
        for i in indices:
            q = qpos_seq[i]
            viewer.step(
                q[:3],
                q[3:7],
                q[7:],
                rate_limit=False,
                follow_camera=True,
            )
    finally:
        viewer.close()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "OpenLoong V3.0 bilateral 14-DoF arm spatial retargeting. "
            "Body qpos is frozen and both arms are jointly re-solved from frame 0."
        )
    )

    p.add_argument(
        "--body-csv",
        type=Path,
        default=DEFAULT_BODY_CSV,
    )
    p.add_argument(
        "--smplx",
        type=Path,
        default=DEFAULT_SMPLX,
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
    )
    p.add_argument(
        "--fps",
        type=float,
        default=30.0,
    )
    p.add_argument(
        "--start-frame",
        type=int,
        default=0,
    )
    p.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="0 = process to the end",
    )
    p.add_argument(
        "--render",
        action="store_true",
    )

    p.add_argument(
        "--upper-length-scale",
        type=float,
        default=1.0,
    )
    p.add_argument(
        "--forearm-length-scale",
        type=float,
        default=1.0,
    )

    p.add_argument(
        "--torso-half-x",
        type=float,
        default=0.14,
    )
    p.add_argument(
        "--torso-half-y-scale",
        type=float,
        default=0.82,
    )
    p.add_argument(
        "--torso-z-padding",
        type=float,
        default=0.03,
    )

    p.add_argument(
        "--hand-torso-margin",
        type=float,
        default=0.075,
    )
    p.add_argument(
        "--elbow-torso-margin",
        type=float,
        default=0.050,
    )
    p.add_argument(
        "--forearm-torso-margin",
        type=float,
        default=0.055,
    )
    p.add_argument(
        "--hand-hand-min",
        type=float,
        default=0.14,
    )
    p.add_argument(
        "--forearm-forearm-min",
        type=float,
        default=0.075,
    )

    p.add_argument(
        "--elbow-weight",
        type=float,
        default=10.0,
    )
    p.add_argument(
        "--wrist-weight",
        type=float,
        default=18.0,
    )
    p.add_argument(
        "--torso-collision-weight",
        type=float,
        default=45.0,
    )
    p.add_argument(
        "--hand-hand-weight",
        type=float,
        default=55.0,
    )
    p.add_argument(
        "--forearm-forearm-weight",
        type=float,
        default=42.0,
    )
    p.add_argument(
        "--contact-weight",
        type=float,
        default=120.0,
    )
    p.add_argument(
        "--temporal-weight",
        type=float,
        default=0.28,
    )
    p.add_argument(
        "--original-weight",
        type=float,
        default=0.04,
    )

    p.add_argument(
        "--max-nfev",
        type=int,
        default=24,
    )
    p.add_argument(
        "--ftol",
        type=float,
        default=2e-5,
    )
    p.add_argument(
        "--xtol",
        type=float,
        default=2e-5,
    )
    p.add_argument(
        "--gtol",
        type=float,
        default=2e-5,
    )
    p.add_argument(
        "--joint-step-warn",
        type=float,
        default=0.12,
        help=(
            "Diagnostic only. "
            "No post-hoc clipping is applied."
        ),
    )

    return p.parse_args()


def main() -> None:
    args = parse_args()

    args.body_csv = (
        args.body_csv
        .expanduser()
        .resolve()
    )
    args.smplx = (
        args.smplx
        .expanduser()
        .resolve()
    )
    args.out_dir = (
        args.out_dir
        .expanduser()
        .resolve()
    )

    if not args.body_csv.exists():
        raise FileNotFoundError(
            f"body CSV not found: {args.body_csv}"
        )
    if not args.smplx.exists():
        raise FileNotFoundError(
            f"SMPL-X npz not found: {args.smplx}"
        )

    args.out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    out_csv = (
        args.out_dir
        / "csv/openloong/live_motion.csv"
    )
    diag_csv = (
        args.out_dir
        / "diagnostics/arm_spatial_diagnostics.csv"
    )
    video_path = (
        args.out_dir
        / "video/openloong_armspatial_v1.mp4"
    )

    out_csv.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    diag_csv.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    video_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    body_csv = np.loadtxt(
        args.body_csv,
        delimiter=",",
        dtype=np.float64,
    )

    if body_csv.ndim == 1:
        body_csv = body_csv[None, :]

    model = mj.MjModel.from_xml_path(
        str(
            ROBOT_XML_DICT[
                "openloong"
            ]
        )
    )

    if body_csv.shape[1] != int(model.nq):
        raise RuntimeError(
            "CSV/model mismatch: "
            f"csv_cols={body_csv.shape[1]} "
            f"model_nq={model.nq}"
        )

    qpos_seq = np.stack(
        [
            csv_row_to_qpos(row)
            for row in body_csv
        ],
        axis=0,
    )

    human_seq, human_frames = (
        load_human_joint_positions(
            args.smplx
        )
    )

    if abs(
        len(qpos_seq)
        - human_frames
    ) > 2:
        raise RuntimeError(
            "Frame mismatch is too large: "
            f"body={len(qpos_seq)} "
            f"smplx={human_frames}"
        )

    n = min(
        len(qpos_seq),
        human_frames,
    )

    start = max(
        0,
        int(args.start_frame),
    )

    if start >= n:
        raise ValueError(
            f"start-frame {start} "
            f">= available frames {n}"
        )

    if int(args.max_frames) > 0:
        end = min(
            n,
            start
            + int(args.max_frames),
        )
    else:
        end = n

    indices = list(
        range(
            start,
            end,
        )
    )

    out_qpos = qpos_seq.copy()
    solver = (
        OpenLoongArmSpatialRetargeter(
            model,
            args,
        )
    )

    print("=" * 80)
    print(
        "OpenLoong V3.0 "
        "Arm Spatial Retargeting"
    )
    print("=" * 80)
    print(
        "body CSV :",
        args.body_csv,
    )
    print(
        "SMPL-X   :",
        args.smplx,
    )
    print(
        "frames   :",
        f"{start}..{end - 1} "
        f"({len(indices)} frames)",
    )
    print(
        "model nq :",
        model.nq,
    )
    print(
        "left arm qpos :",
        solver.arm_qpos["left"],
    )
    print(
        "right arm qpos:",
        solver.arm_qpos["right"],
    )
    print(
        "robot arm lengths [m]:",
        (
            "L "
            f"upper={solver.upper_len['left']:.4f} "
            f"fore={solver.forearm_len['left']:.4f}"
        ),
        (
            "R "
            f"upper={solver.upper_len['right']:.4f} "
            f"fore={solver.forearm_len['right']:.4f}"
        ),
    )
    print("=" * 80)

    fieldnames = [
        "frame",
        "time_sec",
        "cost",
        "nfev",
        "optimality",
        "success",
        "left_wrist_err_mm",
        "right_wrist_err_mm",
        "left_elbow_err_mm",
        "right_elbow_err_mm",
        "hand_hand_dist_m",
        "forearm_forearm_dist_m",
        "min_torso_clearance_m",
        "contact_count",
        "contact_depth_m",
        "max_joint_step_rad",
        "step_warning",
    ]

    diagnostics: list[
        dict[str, float | int]
    ] = []

    for k, i in enumerate(indices):
        body_q = qpos_seq[i]
        h = human_frame(
            human_seq,
            i,
        )

        q_out, d = solver.solve(
            body_q,
            h,
        )

        out_qpos[i] = q_out

        step_warning = int(
            d["max_joint_step"]
            > args.joint_step_warn
        )

        diagnostics.append(
            {
                "frame": i,
                "time_sec": i / args.fps,
                "cost": d["cost"],
                "nfev": int(d["nfev"]),
                "optimality": d["optimality"],
                "success": int(d["success"]),
                "left_wrist_err_mm":
                    d["wrist_err_l"] * 1000.0,
                "right_wrist_err_mm":
                    d["wrist_err_r"] * 1000.0,
                "left_elbow_err_mm":
                    d["elbow_err_l"] * 1000.0,
                "right_elbow_err_mm":
                    d["elbow_err_r"] * 1000.0,
                "hand_hand_dist_m":
                    d["hand_hand_dist"],
                "forearm_forearm_dist_m":
                    d["forearm_forearm_dist"],
                "min_torso_clearance_m":
                    d["min_torso_clearance"],
                "contact_count":
                    int(d["contact_count"]),
                "contact_depth_m":
                    d["contact_depth"],
                "max_joint_step_rad":
                    d["max_joint_step"],
                "step_warning":
                    step_warning,
            }
        )

        if (
            k == 0
            or (k + 1) % 15 == 0
            or k + 1 == len(indices)
            or step_warning
            or d["contact_count"] > 0
        ):
            print(
                f"[{k + 1:4d}/{len(indices):4d}] "
                f"frame={i:3d} "
                f"t={i/args.fps:5.2f}s "
                "Werr=("
                f"{d['wrist_err_l']*1000:5.1f},"
                f"{d['wrist_err_r']*1000:5.1f}"
                ")mm "
                f"HH={d['hand_hand_dist']:.3f}m "
                f"FF={d['forearm_forearm_dist']:.3f}m "
                f"torso={d['min_torso_clearance']:.3f}m "
                f"contact={int(d['contact_count'])} "
                f"dq={d['max_joint_step']:.3f}"
                + (
                    "  <-- STEP WARN"
                    if step_warning
                    else ""
                )
            )

    out_rows = np.stack(
        [
            qpos_to_csv_row(q)
            for q in out_qpos
        ],
        axis=0,
    )

    np.savetxt(
        out_csv,
        out_rows,
        delimiter=",",
        fmt="%.9f",
    )

    with diag_csv.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(
            diagnostics
        )

    wr_l = np.array(
        [
            d["left_wrist_err_mm"]
            for d in diagnostics
        ],
        dtype=float,
    )
    wr_r = np.array(
        [
            d["right_wrist_err_mm"]
            for d in diagnostics
        ],
        dtype=float,
    )
    hh = np.array(
        [
            d["hand_hand_dist_m"]
            for d in diagnostics
        ],
        dtype=float,
    )
    ff = np.array(
        [
            d["forearm_forearm_dist_m"]
            for d in diagnostics
        ],
        dtype=float,
    )
    tc = np.array(
        [
            d["min_torso_clearance_m"]
            for d in diagnostics
        ],
        dtype=float,
    )
    dq = np.array(
        [
            d["max_joint_step_rad"]
            for d in diagnostics
        ],
        dtype=float,
    )
    contacts = np.array(
        [
            d["contact_count"]
            for d in diagnostics
        ],
        dtype=int,
    )

    print()
    print("=" * 80)
    print("V3.0 summary")
    print("=" * 80)
    print(
        "left wrist error : "
        f"mean={wr_l.mean():.1f} mm "
        f"max={wr_l.max():.1f} mm"
    )
    print(
        "right wrist error: "
        f"mean={wr_r.mean():.1f} mm "
        f"max={wr_r.max():.1f} mm"
    )
    print(
        "min hand-hand distance      : "
        f"{hh.min():.4f} m"
    )
    print(
        "min forearm-forearm distance: "
        f"{ff.min():.4f} m"
    )
    print(
        "min torso proxy clearance   : "
        f"{tc.min():.4f} m"
    )
    print(
        "max arm joint step          : "
        f"{dq.max():.4f} rad/frame"
    )
    print(
        "frames with MuJoCo arm collision: "
        f"{int(np.count_nonzero(contacts))}"
        f"/{len(indices)}"
    )
    print(
        "No post-hoc wrist delta or "
        "joint-step clipping was applied."
    )
    print(
        "CSV :",
        out_csv,
    )
    print(
        "DIAG:",
        diag_csv,
    )

    if args.render:
        print(
            "Rendering:",
            video_path,
        )
        render_motion(
            out_qpos,
            indices,
            args.fps,
            video_path,
        )
        print(
            "VIDEO:",
            video_path,
        )


if __name__ == "__main__":
    main()
