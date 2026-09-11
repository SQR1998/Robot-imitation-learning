#!/usr/bin/env python3

from pathlib import Path

import mujoco
import numpy as np


ROOT = Path(__file__).resolve().parents[2]

URDF = (
    ROOT
    / "dataset/openloong_master1/robot"
    / "LingLong2.0_20260616"
    / "LingLong2.0.urdf"
)

CSV = (
    ROOT
    / "output/linglong20_arm_v15_strong_extension"
    / "linglong20_arm_v15_strong_extension_qpos37.csv"
)


def build_model():
    spec = mujoco.MjSpec.from_file(str(URDF))
    spec.compiler.meshdir = str(URDF.parent)

    base = spec.body("base_link")
    base.add_freejoint(name="ground_diag_root")

    floor = spec.worldbody.add_geom()
    floor.name = "ground_diag_plane"
    floor.type = mujoco.mjtGeom.mjGEOM_PLANE
    floor.size = [5.0, 5.0, 0.1]
    floor.pos = [0.0, 0.0, 0.0]

    model = spec.compile()
    data = mujoco.MjData(model)

    return model, data


def geom_name(model, gid):
    return mujoco.mj_id2name(
        model,
        mujoco.mjtObj.mjOBJ_GEOM,
        gid,
    )


def body_name(model, bid):
    return mujoco.mj_id2name(
        model,
        mujoco.mjtObj.mjOBJ_BODY,
        bid,
    )


model, data = build_model()

q = np.loadtxt(
    CSV,
    delimiter=",",
)

assert q.shape == (661, 37)


plane_gid = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_GEOM,
    "ground_diag_plane",
)


left_geoms = []
right_geoms = []


print("=" * 90)
print("FOOT / ANKLE GEOMS")
print("=" * 90)

for gid in range(model.ngeom):

    if gid == plane_gid:
        continue

    bid = int(
        model.geom_bodyid[gid]
    )

    bname = body_name(
        model,
        bid,
    ) or ""

    lname = bname.lower()

    if (
        "ankle" not in lname
        and "foot" not in lname
        and "toe" not in lname
    ):
        continue

    print(
        f"geom={gid:3d} "
        f"body={bname:32s} "
        f"geom_name={geom_name(model, gid)}"
    )

    if "left" in lname:
        left_geoms.append(gid)

    if "right" in lname:
        right_geoms.append(gid)


print()
print("left geoms :", left_geoms)
print("right geoms:", right_geoms)

if not left_geoms or not right_geoms:
    raise RuntimeError(
        "没有正确找到左右脚 geom"
    )


def signed_distance_to_ground(gid):

    fromto = np.zeros(
        6,
        dtype=np.float64,
    )

    return float(
        mujoco.mj_geomDistance(
            model,
            data,
            int(gid),
            int(plane_gid),
            2.0,
            fromto,
        )
    )


left_z = []
right_z = []
lowest = []


for i in range(len(q)):

    data.qpos[:] = q[i]

    mujoco.mj_forward(
        model,
        data,
    )

    ld = min(
        signed_distance_to_ground(g)
        for g in left_geoms
    )

    rd = min(
        signed_distance_to_ground(g)
        for g in right_geoms
    )

    left_z.append(ld)
    right_z.append(rd)
    lowest.append(
        min(ld, rd)
    )


left_z = np.asarray(left_z)
right_z = np.asarray(right_z)
lowest = np.asarray(lowest)


def stats(name, x):

    print(
        f"{name:16s}"
        f" min={x.min(): .6f}"
        f" p01={np.percentile(x,1): .6f}"
        f" p05={np.percentile(x,5): .6f}"
        f" median={np.median(x): .6f}"
        f" max={x.max(): .6f}"
    )


print()
print("=" * 90)
print("GROUND DISTANCE")
print("=" * 90)

stats(
    "left",
    left_z,
)

stats(
    "right",
    right_z,
)

stats(
    "lowest",
    lowest,
)


print()
print(
    "penetration frames:",
    int(
        np.sum(
            lowest < 0.0
        )
    ),
    "/",
    len(lowest),
)


worst = np.argsort(
    lowest
)[:30]


print()
print("=" * 90)
print("WORST FRAMES")
print("=" * 90)

for i in worst:

    print(
        f"frame={i:3d} "
        f"left={left_z[i]*1000:8.2f} mm "
        f"right={right_z[i]*1000:8.2f} mm "
        f"lowest={lowest[i]*1000:8.2f} mm"
    )


np.savez(
    "/tmp/linglong20_v15_ground_diag.npz",
    left=left_z,
    right=right_z,
    lowest=lowest,
)

print()
print(
    "saved diagnostic: "
    "/tmp/linglong20_v15_ground_diag.npz"
)

print(
    "GROUND DIAGNOSTIC: PASS"
)
