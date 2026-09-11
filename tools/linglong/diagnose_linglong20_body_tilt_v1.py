from pathlib import Path
import argparse
import numpy as np
import mujoco


def quat_wxyz_to_matrix(q):
    q = np.asarray(q, dtype=np.float64)
    q = q / max(np.linalg.norm(q), 1e-12)

    mat = np.zeros(9, dtype=np.float64)
    mujoco.mju_quat2Mat(mat, q)
    return mat.reshape(3, 3)


def roll_pitch_yaw_deg_from_matrix(R):
    # ZYX convention:
    # R = Rz(yaw) @ Ry(pitch) @ Rx(roll)

    pitch = np.arctan2(
        -R[2, 0],
        np.sqrt(
            R[0, 0] ** 2
            + R[1, 0] ** 2
        ),
    )

    roll = np.arctan2(
        R[2, 1],
        R[2, 2],
    )

    yaw = np.arctan2(
        R[1, 0],
        R[0, 0],
    )

    return np.degrees(
        [roll, pitch, yaw]
    )


def tilt_deg_from_up(R):
    # Local +Z in world coordinates.
    up = R[:, 2]

    c = float(
        np.clip(
            up[2] / max(
                np.linalg.norm(up),
                1e-12,
            ),
            -1.0,
            1.0,
        )
    )

    return float(
        np.degrees(
            np.arccos(c)
        )
    )


def body_id(model, name):
    idx = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_BODY,
        name,
    )

    return idx


def build_model(urdf):
    spec = mujoco.MjSpec.from_file(
        str(urdf)
    )

    # LingLong URDF mesh filenames already contain meshes/.
    spec.compiler.meshdir = str(
        urdf.resolve().parent
    )

    base = spec.body("base_link")

    base.add_freejoint(
        name="diagnostic_root_free_joint"
    )

    model = spec.compile()
    data = mujoco.MjData(model)

    return model, data


def print_joint_map(model):
    print()
    print("=" * 100)
    print("JOINT / QPOS MAP")
    print("=" * 100)

    for jid in range(model.njnt):

        name = mujoco.mj_id2name(
            model,
            mujoco.mjtObj.mjOBJ_JOINT,
            jid,
        )

        qadr = int(
            model.jnt_qposadr[jid]
        )

        dofadr = int(
            model.jnt_dofadr[jid]
        )

        jtype = int(
            model.jnt_type[jid]
        )

        print(
            f"{jid:2d} "
            f"qpos={qadr:2d} "
            f"dof={dofadr:2d} "
            f"type={jtype} "
            f"{name}"
        )


def print_candidate_joints(model):
    print()
    print("=" * 100)
    print("TORSO / WAIST JOINT CANDIDATES")
    print("=" * 100)

    keywords = (
        "waist",
        "torso",
        "lumbar",
        "hip",
        "pelvis",
    )

    for jid in range(model.njnt):

        name = mujoco.mj_id2name(
            model,
            mujoco.mjtObj.mjOBJ_JOINT,
            jid,
        )

        if not name:
            continue

        if any(
            key in name.lower()
            for key in keywords
        ):
            qadr = int(
                model.jnt_qposadr[jid]
            )

            rng = model.jnt_range[jid]

            print(
                f"{name:32s} "
                f"qpos={qadr:2d} "
                f"range=["
                f"{np.degrees(rng[0]):7.2f}, "
                f"{np.degrees(rng[1]):7.2f}] deg"
            )


def stats(name, x):
    x = np.asarray(
        x,
        dtype=np.float64,
    )

    print(
        f"{name:28s} "
        f"mean={np.mean(x):8.3f} "
        f"median={np.median(x):8.3f} "
        f"min={np.min(x):8.3f} "
        f"max={np.max(x):8.3f} "
        f"std={np.std(x):8.3f}"
    )


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--urdf",
        default=(
            "dataset/openloong_master1/"
            "robot/LingLong2.0_20260616/"
            "LingLong2.0.urdf"
        ),
    )

    parser.add_argument(
        "--csv",
        default=(
            "output/"
            "linglong20_arm_v15_strong_extension/"
            "linglong20_arm_v15_strong_extension_qpos37.csv"
        ),
    )

    args = parser.parse_args()

    urdf = Path(args.urdf).resolve()
    csv = Path(args.csv).resolve()

    print("URDF:")
    print(" ", urdf)

    print("CSV:")
    print(" ", csv)

    model, data = build_model(
        urdf
    )

    print()
    print("=" * 100)
    print("MODEL")
    print("=" * 100)

    print("nq =", model.nq)
    print("nv =", model.nv)
    print("njnt =", model.njnt)

    if model.nq != 37:
        raise RuntimeError(
            f"Expected nq=37, got {model.nq}"
        )

    qpos = np.loadtxt(
        csv,
        delimiter=",",
    )

    if (
        qpos.ndim != 2
        or qpos.shape[1] != 37
    ):
        raise RuntimeError(
            f"Unexpected CSV shape: "
            f"{qpos.shape}"
        )

    print("qpos shape =", qpos.shape)

    print_joint_map(model)
    print_candidate_joints(model)

    # ---------------------------------------------------------
    # Find important bodies.
    # ---------------------------------------------------------

    body_candidates = [
        "base_link",
        "waist_yaw_link",
        "waist_roll_link",
        "waist_pitch_link",
    ]

    body_ids = {}

    print()
    print("=" * 100)
    print("BODY CANDIDATES")
    print("=" * 100)

    for name in body_candidates:

        idx = body_id(
            model,
            name,
        )

        if idx >= 0:
            body_ids[name] = idx
            print(
                f"{name:24s} id={idx}"
            )
        else:
            print(
                f"{name:24s} NOT FOUND"
            )

    # ---------------------------------------------------------
    # Per-frame orientation.
    # ---------------------------------------------------------

    root_roll = []
    root_pitch = []
    root_yaw = []
    root_tilt = []

    body_rpy = {
        name: []
        for name in body_ids
    }

    body_tilt = {
        name: []
        for name in body_ids
    }

    for i in range(len(qpos)):

        q = qpos[i].copy()

        # Normalize root quaternion.
        q[3:7] /= max(
            np.linalg.norm(
                q[3:7]
            ),
            1e-12,
        )

        root_R = quat_wxyz_to_matrix(
            q[3:7]
        )

        rpy = (
            roll_pitch_yaw_deg_from_matrix(
                root_R
            )
        )

        root_roll.append(
            rpy[0]
        )
        root_pitch.append(
            rpy[1]
        )
        root_yaw.append(
            rpy[2]
        )
        root_tilt.append(
            tilt_deg_from_up(
                root_R
            )
        )

        data.qpos[:] = q

        mujoco.mj_forward(
            model,
            data,
        )

        for name, bid in body_ids.items():

            R = (
                data.xmat[bid]
                .reshape(3, 3)
                .copy()
            )

            body_rpy[name].append(
                roll_pitch_yaw_deg_from_matrix(
                    R
                )
            )

            body_tilt[name].append(
                tilt_deg_from_up(
                    R
                )
            )

    print()
    print("=" * 100)
    print("ROOT ORIENTATION STATISTICS")
    print("=" * 100)

    stats(
        "root roll [deg]",
        root_roll,
    )

    stats(
        "root pitch [deg]",
        root_pitch,
    )

    stats(
        "root yaw [deg]",
        root_yaw,
    )

    stats(
        "root total tilt [deg]",
        root_tilt,
    )

    print()
    print("=" * 100)
    print("BODY WORLD-ORIENTATION STATISTICS")
    print("=" * 100)

    for name in body_ids:

        arr = np.asarray(
            body_rpy[name]
        )

        print()
        print(name)

        stats(
            "  world roll [deg]",
            arr[:, 0],
        )

        stats(
            "  world pitch [deg]",
            arr[:, 1],
        )

        stats(
            "  world yaw [deg]",
            arr[:, 2],
        )

        stats(
            "  world tilt [deg]",
            body_tilt[name],
        )

    # ---------------------------------------------------------
    # Directly print scalar waist joint trajectories.
    # ---------------------------------------------------------

    print()
    print("=" * 100)
    print("WAIST JOINT TRAJECTORIES")
    print("=" * 100)

    for jid in range(model.njnt):

        name = mujoco.mj_id2name(
            model,
            mujoco.mjtObj.mjOBJ_JOINT,
            jid,
        )

        if not name:
            continue

        if "waist" not in name.lower():
            continue

        if int(
            model.jnt_type[jid]
        ) == int(
            mujoco.mjtJoint.mjJNT_FREE
        ):
            continue

        qadr = int(
            model.jnt_qposadr[jid]
        )

        vals_deg = np.degrees(
            qpos[:, qadr]
        )

        stats(
            f"{name} qpos={qadr}",
            vals_deg,
        )

    # ---------------------------------------------------------
    # Representative frames.
    # ---------------------------------------------------------

    sample_frames = [
        0,
        50,
        100,
        150,
        200,
        250,
        300,
        350,
        400,
        450,
        500,
        550,
        600,
        650,
        660,
    ]

    print()
    print("=" * 100)
    print("REPRESENTATIVE FRAMES")
    print("=" * 100)

    for i in sample_frames:

        if i >= len(qpos):
            continue

        print(
            f"frame {i:3d} | "
            f"root roll={root_roll[i]:7.2f} "
            f"pitch={root_pitch[i]:7.2f} "
            f"yaw={root_yaw[i]:7.2f} "
            f"tilt={root_tilt[i]:7.2f}"
        )

        for name in body_ids:

            arr = np.asarray(
                body_rpy[name]
            )

            print(
                f"          {name:18s} "
                f"roll={arr[i,0]:7.2f} "
                f"pitch={arr[i,1]:7.2f} "
                f"tilt={body_tilt[name][i]:7.2f}"
            )

    # ---------------------------------------------------------
    # Identify strongest tilt frames.
    # ---------------------------------------------------------

    print()
    print("=" * 100)
    print("TOP ROOT-TILT FRAMES")
    print("=" * 100)

    order = np.argsort(
        np.asarray(root_tilt)
    )[::-1]

    for idx in order[:20]:

        print(
            f"frame={idx:3d} "
            f"tilt={root_tilt[idx]:7.2f} "
            f"roll={root_roll[idx]:7.2f} "
            f"pitch={root_pitch[idx]:7.2f}"
        )

    if "waist_pitch_link" in body_ids:

        torso_tilt = np.asarray(
            body_tilt[
                "waist_pitch_link"
            ]
        )

        print()
        print("=" * 100)
        print("TOP WAIST_PITCH_LINK TILT FRAMES")
        print("=" * 100)

        order = np.argsort(
            torso_tilt
        )[::-1]

        arr = np.asarray(
            body_rpy[
                "waist_pitch_link"
            ]
        )

        for idx in order[:20]:

            print(
                f"frame={idx:3d} "
                f"tilt={torso_tilt[idx]:7.2f} "
                f"roll={arr[idx,0]:7.2f} "
                f"pitch={arr[idx,1]:7.2f}"
            )

    print()
    print("BODY TILT DIAGNOSTIC: PASS")


if __name__ == "__main__":
    main()
