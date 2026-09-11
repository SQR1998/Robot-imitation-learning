#!/usr/bin/env python3

from pathlib import Path
import sys

import mujoco
import numpy as np


ROOT = Path(__file__).resolve().parents[2]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from general_motion_retargeting import (
    GeneralMotionRetargeting,
)
from general_motion_retargeting.params import (
    IK_CONFIG_DICT,
)
from general_motion_retargeting.utils.smpl import (
    load_smplx_file,
    get_smplx_data_offline_fast,
)


XML = (
    ROOT
    / "dataset/openloong_master1/robot"
    / "LingLong2.0_20260616"
    / "LingLong2.0_floating.xml"
)

IK_CONFIG = (
    ROOT
    / "general_motion_retargeting/ik_configs"
    / "smplx_to_linglong20.json"
)

SMPLX_FILE = (
    ROOT
    / "output/openloong_v2_kneeguide"
    / "stream_demo"
    / "gmr_smplx_results.npz"
)

BODY_MODEL_DIR = (
    ROOT
    / "assets/body_models"
)

OUT_DIR = (
    ROOT
    / "output/linglong20_body_v1_smoke"
)

OUT_CSV = (
    OUT_DIR
    / "linglong20_body_v1_qpos37.csv"
)

NUM_FRAMES = 120


def main():
    for p in (
        XML,
        IK_CONFIG,
        SMPLX_FILE,
        BODY_MODEL_DIR,
    ):
        if not p.exists():
            raise FileNotFoundError(p)

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 90)
    print("LINGLONG 2.0 BODY V1 SMOKE TEST")
    print("=" * 90)

    print("robot :", XML)
    print("config:", IK_CONFIG)
    print("smplx :", SMPLX_FILE)

    # Temporary registration.
    # Do not modify params.py until this V1 is validated.
    IK_CONFIG_DICT["smplx"]["linglong2"] = (
        IK_CONFIG
    )

    (
        smplx_data,
        body_model,
        smplx_output,
        human_height,
    ) = load_smplx_file(
        str(SMPLX_FILE),
        str(BODY_MODEL_DIR),
        coord_fix="auto",
    )

    print()
    print(
        "estimated human height:",
        float(human_height),
    )

    (
        human_frames,
        aligned_fps,
    ) = get_smplx_data_offline_fast(
        smplx_data,
        body_model,
        smplx_output,
        tgt_fps=30,
    )

    print("human frames:", len(human_frames))
    print("aligned fps :", aligned_fps)

    n = min(
        NUM_FRAMES,
        len(human_frames),
    )

    gmr = GeneralMotionRetargeting(
        src_human="smplx",
        tgt_robot="linglong2",
        robot_path=str(XML),
        actual_human_height=float(human_height),
        verbose=False,
    )

    qpos = []
    errors = []

    for i in range(n):
        q, err = gmr.retarget(
            human_frames[i],
            offset_to_ground=False,
        )

        q = np.asarray(
            q,
            dtype=np.float64,
        )

        if q.shape != (37,):
            raise RuntimeError(
                f"frame {i}: "
                f"qpos shape={q.shape}"
            )

        if not np.all(np.isfinite(q)):
            raise RuntimeError(
                f"frame {i}: non-finite qpos"
            )

        qpos.append(q)
        errors.append(err)

        if (
            i == 0
            or (i + 1) % 30 == 0
            or i == n - 1
        ):
            print(
                f"frame {i:03d} | "
                f"root_z={q[2]:.4f} | "
                f"err1={err[0]:.4f} | "
                f"err2={err[1]:.4f}"
            )

    qpos = np.stack(
        qpos,
        axis=0,
    )

    np.savetxt(
        OUT_CSV,
        qpos,
        delimiter=",",
        fmt="%.9f",
    )

    model = mujoco.MjModel.from_xml_path(
        str(XML)
    )

    quat_norm = np.linalg.norm(
        qpos[:, 3:7],
        axis=1,
    )

    max_step = float(
        np.max(
            np.abs(
                np.diff(
                    qpos[:, 7:],
                    axis=0,
                )
            )
        )
    )

    violations = []

    for jid in range(1, model.njnt):
        if not bool(
            model.jnt_limited[jid]
        ):
            continue

        qadr = int(
            model.jnt_qposadr[jid]
        )

        lo, hi = model.jnt_range[jid]

        vals = qpos[:, qadr]

        below = float(
            np.min(vals - lo)
        )

        above = float(
            np.max(vals - hi)
        )

        if below < -1e-6 or above > 1e-6:
            name = mujoco.mj_id2name(
                model,
                mujoco.mjtObj.mjOBJ_JOINT,
                jid,
            )

            violations.append(
                (
                    name,
                    below,
                    above,
                )
            )

    print()
    print("=" * 90)
    print("RESULT")
    print("=" * 90)

    print("shape            :", qpos.shape)
    print(
        "finite           :",
        bool(np.all(np.isfinite(qpos))),
    )
    print(
        "quat norm min/max:",
        float(quat_norm.min()),
        float(quat_norm.max()),
    )
    print(
        "root z min/max   :",
        float(qpos[:, 2].min()),
        float(qpos[:, 2].max()),
    )
    print(
        "max joint step   :",
        max_step,
        "rad/frame",
    )

    print(
        "joint violations :",
        len(violations),
    )

    for item in violations:
        print("  ", item)

    print()
    print("saved:", OUT_CSV)

    assert qpos.shape == (n, 37)
    assert np.all(np.isfinite(qpos))
    assert np.max(
        np.abs(quat_norm - 1.0)
    ) < 1e-5

    print()
    print(
        "LINGLONG20 BODY V1 NUMERIC SMOKE: PASS"
    )


if __name__ == "__main__":
    main()
