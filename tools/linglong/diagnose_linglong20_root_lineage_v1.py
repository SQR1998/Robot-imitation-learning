from pathlib import Path
import numpy as np
import mujoco


ROOT = Path(".").resolve()

REF = (
    ROOT
    / "output"
    / "linglong20_arm_v15_strong_extension"
    / "linglong20_arm_v15_strong_extension_qpos37.csv"
)


def load_csv(path):
    try:
        q = np.loadtxt(
            path,
            delimiter=",",
        )
    except Exception:
        return None

    if (
        q.ndim != 2
        or q.shape[1] != 37
    ):
        return None

    return q


def quat_to_R(q):
    q = np.asarray(
        q,
        dtype=np.float64,
    )

    q = q / max(
        np.linalg.norm(q),
        1e-12,
    )

    m = np.zeros(
        9,
        dtype=np.float64,
    )

    mujoco.mju_quat2Mat(
        m,
        q,
    )

    return m.reshape(3, 3)


def rpy_deg(R):

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
        [
            roll,
            pitch,
            yaw,
        ]
    )


def root_stats(q):

    rolls = []
    pitches = []
    tilts = []

    for row in q:

        R = quat_to_R(
            row[3:7]
        )

        rpy = rpy_deg(R)

        rolls.append(
            rpy[0]
        )

        pitches.append(
            rpy[1]
        )

        up = R[:, 2]

        c = float(
            np.clip(
                up[2]
                / max(
                    np.linalg.norm(up),
                    1e-12,
                ),
                -1.0,
                1.0,
            )
        )

        tilts.append(
            np.degrees(
                np.arccos(c)
            )
        )

    return (
        np.asarray(rolls),
        np.asarray(pitches),
        np.asarray(tilts),
    )


ref = load_csv(REF)

if ref is None:
    raise RuntimeError(
        f"Cannot load reference: {REF}"
    )


files = sorted(
    ROOT.glob(
        "output/**/*.csv"
    )
)


print("=" * 150)
print(
    "LINGLONG ROOT ORIENTATION LINEAGE"
)
print("=" * 150)

print(
    f"{'mean tilt':>10s} "
    f"{'max tilt':>10s} "
    f"{'mean roll':>10s} "
    f"{'mean pitch':>11s} "
    f"{'root diff':>12s}  "
    f"path"
)

print("-" * 150)


count = 0

for path in files:

    q = load_csv(path)

    if q is None:
        continue

    if len(q) != len(ref):
        continue

    roll, pitch, tilt = (
        root_stats(q)
    )

    root_diff = float(
        np.max(
            np.abs(
                q[:, 0:7]
                - ref[:, 0:7]
            )
        )
    )

    print(
        f"{np.mean(tilt):10.3f} "
        f"{np.max(tilt):10.3f} "
        f"{np.mean(roll):10.3f} "
        f"{np.mean(pitch):11.3f} "
        f"{root_diff:12.6g}  "
        f"{path.relative_to(ROOT)}"
    )

    count += 1


print()
print(
    "valid qpos37 files:",
    count,
)

print()
print(
    "Interpretation:"
)

print(
    "root diff == 0  -> exact same root trajectory as V1.5"
)

print(
    "Find the EARLIEST stage that already has mean tilt ~9.55 deg."
)

print()
print(
    "ROOT LINEAGE DIAGNOSTIC: PASS"
)
