#!/usr/bin/env python3

from __future__ import annotations

import argparse
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np


ROOT = Path(__file__).resolve().parents[2]

DEFAULT_URDF = (
    ROOT
    / "dataset/openloong_master1/robot/"
    "LingLong2.0_20260616/LingLong2.0.urdf"
)

DEFAULT_CSV = (
    ROOT
    / "output/linglong20_arm_v13_boundary_collision_safe/"
    "linglong20_arm_v13_boundary_collision_safe_qpos37.csv"
)


ARM_EXPECTED = [
    ("left_shoulder_pitch_joint", 23),
    ("left_shoulder_roll_joint", 24),
    ("left_shoulder_yaw_joint", 25),
    ("left_elbow_joint", 26),
    ("left_wrist_roll_joint", 27),
    ("left_wrist_pitch_joint", 28),
    ("left_wrist_yaw_joint", 29),

    ("right_shoulder_pitch_joint", 30),
    ("right_shoulder_roll_joint", 31),
    ("right_shoulder_yaw_joint", 32),
    ("right_elbow_joint", 33),
    ("right_wrist_roll_joint", 34),
    ("right_wrist_pitch_joint", 35),
    ("right_wrist_yaw_joint", 36),
]


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--csv",
        type=Path,
        default=DEFAULT_CSV,
    )

    p.add_argument(
        "--urdf",
        type=Path,
        default=DEFAULT_URDF,
    )

    p.add_argument(
        "--fps",
        type=float,
        default=30.0,
    )

    p.add_argument(
        "--speed",
        type=float,
        default=0.5,
        help="1.0=normal speed, 0.5=half speed",
    )

    p.add_argument(
        "--start",
        type=int,
        default=0,
    )

    p.add_argument(
        "--end",
        type=int,
        default=-1,
    )

    p.add_argument(
        "--loop",
        action="store_true",
    )

    p.add_argument(
        "--no-follow",
        action="store_true",
    )

    return p.parse_args()


def build_model(urdf: Path):
    print("Loading URDF:")
    print(" ", urdf)

    spec = mujoco.MjSpec.from_file(
        str(urdf)
    )

    # LingLong2.0 URDF mesh filenames already contain "meshes/...".
    # The URDF also sets compiler meshdir="meshes/", which would make
    # MuJoCo look for "meshes/meshes/...".
    #
    # Point meshdir at the URDF directory itself:
    #   URDF_DIR + meshes/foo.STL
    #
    # Do NOT modify the original contest URDF.
    print(
        "compiler meshdir before:",
        spec.compiler.meshdir,
    )

    spec.compiler.meshdir = str(
        urdf.resolve().parent
    )

    print(
        "compiler meshdir after :",
        spec.compiler.meshdir,
    )

    base = spec.body("base_link")

    if base is None:
        raise RuntimeError(
            "base_link not found in LingLong2.0 URDF"
        )

    # URDF root is fixed by default.
    # Our trajectory is:
    # root xyz(3) + root quat(4) + 30 joints
    #
    # Therefore add the same floating root used by qpos37.
    base.add_freejoint(
        name="viewer_root_free_joint"
    )

    # Visual ground reference.
    ground = spec.worldbody.add_geom()
    ground.name = "viewer_ground"
    ground.type = mujoco.mjtGeom.mjGEOM_PLANE
    ground.size = [5.0, 5.0, 0.1]
    ground.pos = [0.0, 0.0, -0.025]
    ground.rgba = [0.35, 0.35, 0.35, 1.0]
    ground.contype = 0
    ground.conaffinity = 0

    model = spec.compile()

    return model


def verify_model(model):
    print()
    print("=" * 80)
    print("MODEL CHECK")
    print("=" * 80)

    print("nq =", model.nq)
    print("nv =", model.nv)
    print("njnt =", model.njnt)

    if model.nq != 37:
        raise RuntimeError(
            f"Expected nq=37, got nq={model.nq}. "
            "Stop playback to avoid wrong joint mapping."
        )

    print()
    print("Arm qpos mapping:")

    bad = []

    for name, expected_qpos in ARM_EXPECTED:
        jid = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_JOINT,
            name,
        )

        if jid < 0:
            raise RuntimeError(
                f"Joint not found: {name}"
            )

        qadr = int(
            model.jnt_qposadr[jid]
        )

        print(
            f"{name:32s} "
            f"qpos={qadr:2d} "
            f"expected={expected_qpos:2d}"
        )

        if qadr != expected_qpos:
            bad.append(
                (name, qadr, expected_qpos)
            )

    if bad:
        print()
        print("BAD MAPPING:")
        for item in bad:
            print(item)

        raise RuntimeError(
            "LingLong qpos mapping does not match "
            "the V1.3 qpos37 layout."
        )

    print()
    print("QPOS37 MAPPING: PASS")


def load_motion(path: Path):
    q = np.loadtxt(
        path,
        delimiter=",",
    )

    if q.ndim == 1:
        q = q[None, :]

    if q.shape[1] != 37:
        raise RuntimeError(
            f"Expected CSV shape Nx37, got {q.shape}"
        )

    if not np.isfinite(q).all():
        raise RuntimeError(
            "CSV contains non-finite values"
        )

    # Normalize floating-base quaternion.
    norms = np.linalg.norm(
        q[:, 3:7],
        axis=1,
        keepdims=True,
    )

    if np.any(norms < 1e-8):
        raise RuntimeError(
            "Invalid root quaternion"
        )

    q[:, 3:7] /= norms

    return q


def main():
    args = parse_args()

    model = build_model(
        args.urdf
    )

    verify_model(model)

    qseq = load_motion(
        args.csv
    )

    print()
    print("=" * 80)
    print("MOTION")
    print("=" * 80)

    print("CSV:")
    print(" ", args.csv)

    print("shape:", qseq.shape)

    start = max(
        0,
        int(args.start),
    )

    if args.end < 0:
        end = len(qseq) - 1
    else:
        end = min(
            int(args.end),
            len(qseq) - 1,
        )

    if start > end:
        raise ValueError(
            f"start={start} > end={end}"
        )

    fps = float(args.fps)
    speed = float(args.speed)

    if fps <= 0:
        raise ValueError(
            "fps must be > 0"
        )

    if speed <= 0:
        raise ValueError(
            "speed must be > 0"
        )

    dt = 1.0 / (
        fps * speed
    )

    print(
        f"frames: {start}..{end}"
    )

    print(
        f"fps: {fps:.2f}"
    )

    print(
        f"speed: {speed:.2f}x"
    )

    print()
    print(
        "Close the MuJoCo window to exit."
    )

    print(
        "Mouse: rotate / zoom camera normally."
    )

    data = mujoco.MjData(
        model
    )

    # Initial pose.
    data.qpos[:] = qseq[start]

    mujoco.mj_forward(
        model,
        data,
    )

    with mujoco.viewer.launch_passive(
        model,
        data,
    ) as viewer:

        # Initial camera.
        viewer.cam.distance = 3.0
        viewer.cam.azimuth = 145.0
        viewer.cam.elevation = -12.0

        while viewer.is_running():

            for frame in range(
                start,
                end + 1,
            ):
                if not viewer.is_running():
                    break

                t0 = time.perf_counter()

                data.qpos[:] = (
                    qseq[frame]
                )

                data.qvel[:] = 0.0

                mujoco.mj_forward(
                    model,
                    data,
                )

                if not args.no_follow:
                    viewer.cam.lookat[:] = [
                        float(
                            data.qpos[0]
                        ),
                        float(
                            data.qpos[1]
                        ),
                        float(
                            data.qpos[2]
                        ) + 0.55,
                    ]

                viewer.sync()

                if (
                    frame == start
                    or frame == end
                    or frame % 30 == 0
                    or frame in (
                        341,
                        422,
                        425,
                        448,
                    )
                ):
                    print(
                        f"\rframe "
                        f"{frame:03d}/{len(qseq)-1} "
                        f"time={frame/fps:6.2f}s",
                        end="",
                        flush=True,
                    )

                elapsed = (
                    time.perf_counter()
                    - t0
                )

                remain = dt - elapsed

                if remain > 0:
                    time.sleep(remain)

            print()

            if not args.loop:
                # Hold final frame until viewer closes.
                while viewer.is_running():
                    viewer.sync()
                    time.sleep(0.03)

                break


if __name__ == "__main__":
    main()
