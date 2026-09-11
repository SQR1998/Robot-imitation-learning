#!/usr/bin/env python3

from pathlib import Path
from collections import Counter, defaultdict
import sys

import mujoco
import numpy as np


ROOT = Path(__file__).resolve().parents[2]

CSV = (
    ROOT
    / "output/linglong20_arm_v1"
    / "linglong20_arm_v1_qpos37.csv"
)

XML = (
    ROOT
    / "dataset/openloong_master1/robot"
    / "LingLong2.0_20260616"
    / "LingLong2.0_floating.xml"
)


def name_of(model, objtype, idx):

    if idx < 0:
        return "UNKNOWN"

    name = mujoco.mj_id2name(
        model,
        objtype,
        int(idx),
    )

    return (
        name
        if name is not None
        else f"unnamed_{idx}"
    )


def geom_info(model, geom_id):

    body_id = int(
        model.geom_bodyid[geom_id]
    )

    geom_name = name_of(
        model,
        mujoco.mjtObj.mjOBJ_GEOM,
        geom_id,
    )

    body_name = name_of(
        model,
        mujoco.mjtObj.mjOBJ_BODY,
        body_id,
    )

    return (
        geom_name,
        body_name,
    )


def arm_side(body_name):

    name = body_name.lower()

    arm_words = (
        "shoulder",
        "elbow",
        "wrist",
    )

    if not any(
        word in name
        for word in arm_words
    ):
        return None

    if name.startswith("left_"):
        return "left"

    if name.startswith("right_"):
        return "right"

    return None


def target_region(body_name):

    name = body_name.lower()

    # Torso / trunk.
    if (
        name == "base_link"
        or "waist" in name
        or "torso" in name
        or "chest" in name
    ):
        return "torso"

    # Upper-leg chain.
    #
    # hip_pitch / hip_roll / hip_yaw bodies form the
    # robot thigh-side kinematic chain.
    if (
        "hip_pitch" in name
        or "hip_roll" in name
        or "hip_yaw" in name
    ):
        return "thigh"

    return None


def arm_part(body_name):

    name = body_name.lower()

    if "shoulder" in name:
        return "upper_arm"

    if "elbow" in name:
        return "forearm"

    if "wrist" in name:
        return "wrist"

    return "arm"


def longest_run(frames):

    if not frames:
        return (
            None,
            None,
            0,
        )

    values = sorted(
        set(frames)
    )

    best_start = values[0]
    best_end = values[0]

    start = values[0]
    prev = values[0]

    for f in values[1:]:

        if f == prev + 1:
            prev = f
            continue

        if (
            prev - start
            > best_end - best_start
        ):
            best_start = start
            best_end = prev

        start = f
        prev = f

    if (
        prev - start
        > best_end - best_start
    ):
        best_start = start
        best_end = prev

    return (
        best_start,
        best_end,
        best_end
        - best_start
        + 1,
    )


def main():

    qpos = np.loadtxt(
        CSV,
        delimiter=",",
        dtype=np.float64,
    )

    assert qpos.shape == (
        661,
        37,
    )


    model = mujoco.MjModel.from_xml_path(
        str(XML)
    )

    data = mujoco.MjData(
        model
    )


    pair_counter = Counter()

    class_counter = Counter()

    pair_frames = defaultdict(
        list
    )

    class_frames = defaultdict(
        list
    )

    class_min_dist = {}

    frame_events = defaultdict(
        list
    )


    print("=" * 110)
    print(
        "LINGLONG ARM V1 SELF-COLLISION DIAGNOSIS"
    )
    print("=" * 110)

    print(
        "frames:",
        qpos.shape[0],
    )

    print()


    for frame_idx, q in enumerate(qpos):

        data.qpos[:] = q

        mujoco.mj_forward(
            model,
            data,
        )


        for ci in range(
            data.ncon
        ):

            contact = data.contact[ci]

            g1 = int(
                contact.geom1
            )

            g2 = int(
                contact.geom2
            )


            geom1, body1 = geom_info(
                model,
                g1,
            )

            geom2, body2 = geom_info(
                model,
                g2,
            )


            side1 = arm_side(
                body1
            )

            side2 = arm_side(
                body2
            )

            region1 = target_region(
                body1
            )

            region2 = target_region(
                body2
            )


            side = None
            arm_body = None
            arm_geom = None

            region = None
            target_body = None
            target_geom = None


            if (
                side1 is not None
                and region2 is not None
            ):
                side = side1
                arm_body = body1
                arm_geom = geom1

                region = region2
                target_body = body2
                target_geom = geom2


            elif (
                side2 is not None
                and region1 is not None
            ):
                side = side2
                arm_body = body2
                arm_geom = geom2

                region = region1
                target_body = body1
                target_geom = geom1


            else:
                continue


            part = arm_part(
                arm_body
            )

            dist = float(
                contact.dist
            )


            class_key = (
                side,
                part,
                region,
            )

            pair_key = (
                arm_body,
                target_body,
            )


            class_counter[
                class_key
            ] += 1

            pair_counter[
                pair_key
            ] += 1


            class_frames[
                class_key
            ].append(
                frame_idx
            )

            pair_frames[
                pair_key
            ].append(
                frame_idx
            )


            if (
                class_key
                not in class_min_dist
                or dist
                < class_min_dist[
                    class_key
                ]
            ):
                class_min_dist[
                    class_key
                ] = dist


            frame_events[
                frame_idx
            ].append(
                (
                    side,
                    part,
                    region,
                    arm_body,
                    target_body,
                    dist,
                )
            )


    print("=" * 110)
    print(
        "1. COLLISION CLASS SUMMARY"
    )
    print("=" * 110)


    if not class_counter:

        print(
            "No arm-vs-torso/thigh contacts detected."
        )

    else:

        for key, count in (
            class_counter
            .most_common()
        ):

            side, part, region = key

            frames = class_frames[
                key
            ]

            unique_frames = sorted(
                set(frames)
            )

            (
                run_start,
                run_end,
                run_len,
            ) = longest_run(
                unique_frames
            )


            print(
                f"{side:5s} "
                f"{part:10s} -> "
                f"{region:6s} | "
                f"contacts={count:4d} | "
                f"frames={len(unique_frames):4d} | "
                f"first={unique_frames[0]:3d} "
                f"last={unique_frames[-1]:3d} | "
                f"longest={run_start:3d}-{run_end:3d} "
                f"({run_len:3d}) | "
                f"min_dist="
                f"{class_min_dist[key]*1000:8.3f} mm"
            )


    print()
    print("=" * 110)
    print(
        "2. BODY PAIRS"
    )
    print("=" * 110)


    for pair, count in (
        pair_counter
        .most_common(30)
    ):

        frames = sorted(
            set(
                pair_frames[
                    pair
                ]
            )
        )

        print(
            f"{pair[0]:32s} <-> "
            f"{pair[1]:32s} | "
            f"contacts={count:4d} | "
            f"frames={len(frames):4d} | "
            f"{frames[0]:3d}..{frames[-1]:3d}"
        )


    print()
    print("=" * 110)
    print(
        "3. FRAMES WITH MOST ARM COLLISIONS"
    )
    print("=" * 110)


    ranked_frames = sorted(
        frame_events.items(),
        key=lambda x: len(x[1]),
        reverse=True,
    )


    for (
        frame,
        events,
    ) in ranked_frames[:25]:

        print()
        print(
            f"frame {frame:03d}: "
            f"{len(events)} contacts"
        )

        for event in events[:12]:

            (
                side,
                part,
                region,
                arm_body,
                target_body,
                dist,
            ) = event

            print(
                f"    "
                f"{side:5s} "
                f"{part:10s} -> "
                f"{region:6s} | "
                f"{arm_body} <-> "
                f"{target_body} | "
                f"dist={dist*1000:8.3f} mm"
            )


    print()
    print("=" * 110)
    print(
        "4. TOTAL COLLISION FRAMES"
    )
    print("=" * 110)


    collision_frames = sorted(
        frame_events.keys()
    )


    print(
        "collision frame count:",
        len(
            collision_frames
        ),
        "/",
        len(qpos),
    )


    if collision_frames:

        print(
            "first collision frame:",
            collision_frames[0],
        )

        print(
            "last collision frame:",
            collision_frames[-1],
        )


    print()
    print(
        "LINGLONG ARM V1 "
        "SELF-COLLISION DIAGNOSIS: PASS"
    )


if __name__ == "__main__":
    main()
