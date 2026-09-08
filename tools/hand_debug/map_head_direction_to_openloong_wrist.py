from pathlib import Path

import mujoco as mj
import numpy as np
from scipy.optimize import least_squares

from general_motion_retargeting import (
    ROBOT_XML_DICT,
    RobotMotionViewer,
)


# ============================================================
# 路径
# ============================================================

SRC_CSV = Path(
    "output/openloong_v2_kneeguide/"
    "csv/openloong/live_motion.csv"
)

OUT_DIR = Path(
    "output/openloong_v2_handmap_poc"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

OUT_CSV = (
    OUT_DIR
    / "openloong_hand_headalign_15_18.csv"
)

OUT_VIDEO = (
    OUT_DIR
    / "openloong_hand_headalign_15_18.mp4"
)


# ============================================================
# 时间
# ============================================================

FPS = 30.0

# 真正需要正确的区间
FULL_START = 15.0
FULL_END = 18.0

# 为避免15.0秒突然跳变，
# 前后各留0.4秒平滑进入/退出
BLEND_START = 14.6
BLEND_END = 18.4


# ============================================================
# CSV / qpos 索引
#
# CSV列位置与qpos位置一致，
# 唯一区别只是 root quaternion 保存为 xyzw。
# ============================================================

R_ARM_06 = 14
R_ARM_07 = 15

L_ARM_06 = 21
L_ARM_07 = 22


# ============================================================
# 参数
# ============================================================

# 优化中：
# 手指方向是主要目标
DIR_WEIGHT = 5.0

# 尽量保持Link_arm_07关节原点位置
POS_WEIGHT = 16.0

# 不希望腕关节无意义地大幅偏离原动作
REG_WEIGHT = 0.10


# 优化结果再做5帧中心平滑
SMOOTH_RADIUS = 2

# 最终“修正量”每帧最大变化
# 5度/frame @30fps = 150度/s
MAX_DELTA_STEP = np.deg2rad(5.0)


# ============================================================
# 基础函数
# ============================================================

def smoothstep(x):
    x = float(
        np.clip(x, 0.0, 1.0)
    )

    return (
        x * x * (3.0 - 2.0 * x)
    )


def blend_strength(t):
    """
    14.6 -> 15.0 : 0 -> 1
    15.0 -> 18.0 : 1
    18.0 -> 18.4 : 1 -> 0
    """

    if t < BLEND_START:
        return 0.0

    if t < FULL_START:
        return smoothstep(
            (t - BLEND_START)
            / (FULL_START - BLEND_START)
        )

    if t <= FULL_END:
        return 1.0

    if t <= BLEND_END:
        return smoothstep(
            (BLEND_END - t)
            / (BLEND_END - FULL_END)
        )

    return 0.0


def csv_row_to_qpos(row):
    """
    CSV:
        root quaternion = xyzw

    MuJoCo:
        root quaternion = wxyz
    """

    q = row.copy()

    q[3:7] = row[
        [6, 3, 4, 5]
    ]

    return q


def unit(v):
    v = np.asarray(
        v,
        dtype=np.float64,
    )

    n = float(
        np.linalg.norm(v)
    )

    if n < 1e-12:
        return np.zeros_like(v)

    return v / n


def body_rot(data, body_id):
    return np.asarray(
        data.xmat[body_id],
        dtype=np.float64,
    ).reshape(3, 3)


def angle_deg(a, b):
    a = unit(a)
    b = unit(b)

    d = float(
        np.clip(
            np.dot(a, b),
            -1.0,
            1.0,
        )
    )

    return float(
        np.rad2deg(
            np.arccos(d)
        )
    )


def centered_smooth(x, radius):
    out = x.copy()

    for i in range(len(x)):

        lo = max(
            0,
            i - radius,
        )

        hi = min(
            len(x),
            i + radius + 1,
        )

        out[i] = np.mean(
            x[lo:hi]
        )

    return out


def limit_delta_velocity(delta, max_step):
    """
    限制“我们额外加上的腕部修正量”。

    原来的GMR动作本身不受影响。
    """

    out = delta.copy()

    # forward
    for i in range(
        1,
        len(out),
    ):

        change = (
            out[i] - out[i - 1]
        )

        change = np.clip(
            change,
            -max_step,
            max_step,
        )

        out[i] = (
            out[i - 1]
            + change
        )

    # backward
    for i in range(
        len(out) - 2,
        -1,
        -1,
    ):

        change = (
            out[i] - out[i + 1]
        )

        change = np.clip(
            change,
            -max_step,
            max_step,
        )

        out[i] = (
            out[i + 1]
            + change
        )

    return out


# ============================================================
# 数据
# ============================================================

x = np.loadtxt(
    SRC_CSV,
    delimiter=",",
)

assert x.shape == (
    661,
    38,
), x.shape

assert np.isfinite(x).all()

y = x.copy()


# ============================================================
# MuJoCo
# ============================================================

model = mj.MjModel.from_xml_path(
    str(
        ROBOT_XML_DICT[
            "openloong"
        ]
    )
)

data = mj.MjData(model)


HEAD_ID = mj.mj_name2id(
    model,
    mj.mjtObj.mjOBJ_BODY,
    "Link_head_pitch",
)

R_HAND_ID = mj.mj_name2id(
    model,
    mj.mjtObj.mjOBJ_BODY,
    "Link_arm_r_07",
)

L_HAND_ID = mj.mj_name2id(
    model,
    mj.mjtObj.mjOBJ_BODY,
    "Link_arm_l_07",
)


assert HEAD_ID >= 0
assert R_HAND_ID >= 0
assert L_HAND_ID >= 0


# OpenLoong末端手部在自身坐标系中的长轴。
#
# 右手模型从J_arm_r_07继续沿-Y方向延伸，
# 左手模型镜像，沿+Y方向延伸。
R_FINGER_LOCAL = np.array(
    [0.0, -1.0, 0.0]
)

L_FINGER_LOCAL = np.array(
    [0.0, +1.0, 0.0]
)


# 头部正前方
HEAD_FORWARD_LOCAL = np.array(
    [1.0, 0.0, 0.0]
)


# 机械限位
R06_BOUNDS = (
    -1.8326,
    +1.8326,
)

R07_BOUNDS = (
    -1.0472,
    +1.0472,
)

L06_BOUNDS = (
    -1.8326,
    +1.8326,
)

L07_BOUNDS = (
    -1.0472,
    +1.0472,
)


# ============================================================
# 单手优化
# ============================================================

def solve_hand(
    original_q,
    side,
    x0,
):

    if side == "right":

        q06_idx = R_ARM_06
        q07_idx = R_ARM_07

        hand_id = R_HAND_ID

        finger_local = (
            R_FINGER_LOCAL
        )

        lower = np.array([
            R06_BOUNDS[0],
            R07_BOUNDS[0],
        ])

        upper = np.array([
            R06_BOUNDS[1],
            R07_BOUNDS[1],
        ])

    else:

        q06_idx = L_ARM_06
        q07_idx = L_ARM_07

        hand_id = L_HAND_ID

        finger_local = (
            L_FINGER_LOCAL
        )

        lower = np.array([
            L06_BOUNDS[0],
            L07_BOUNDS[0],
        ])

        upper = np.array([
            L06_BOUNDS[1],
            L07_BOUNDS[1],
        ])


    # --------------------------------------------------------
    # 原始姿态
    # --------------------------------------------------------

    data.qpos[:] = original_q

    mj.mj_forward(
        model,
        data,
    )


    original_hand_pos = np.asarray(
        data.xpos[hand_id],
        dtype=np.float64,
    ).copy()


    head_forward = (
        body_rot(
            data,
            HEAD_ID,
        )
        @ HEAD_FORWARD_LOCAL
    )

    head_forward = unit(
        head_forward
    )


    original_joint = np.array([
        original_q[q06_idx],
        original_q[q07_idx],
    ])


    # --------------------------------------------------------
    # residual
    # --------------------------------------------------------

    def residual(z):

        q = original_q.copy()

        q[q06_idx] = z[0]
        q[q07_idx] = z[1]

        data.qpos[:] = q

        mj.mj_forward(
            model,
            data,
        )


        hand_rot = body_rot(
            data,
            hand_id,
        )

        finger_world = (
            hand_rot
            @ finger_local
        )

        finger_world = unit(
            finger_world
        )


        # 指尖方向应该和头部方向一致
        dir_residual = (
            DIR_WEIGHT
            * (
                finger_world
                - head_forward
            )
        )


        # 尽量别让手腕位置乱跑
        pos_residual = (
            POS_WEIGHT
            * (
                np.asarray(
                    data.xpos[
                        hand_id
                    ]
                )
                - original_hand_pos
            )
        )


        # 尽可能小改动
        regularization = (
            REG_WEIGHT
            * (
                z
                - original_joint
            )
        )


        return np.concatenate([
            dir_residual,
            pos_residual,
            regularization,
        ])


    x0 = np.clip(
        np.asarray(
            x0,
            dtype=np.float64,
        ),
        lower,
        upper,
    )


    result = least_squares(
        residual,
        x0=x0,
        bounds=(
            lower,
            upper,
        ),
        method="trf",
        max_nfev=30,
        ftol=1e-7,
        xtol=1e-7,
        gtol=1e-7,
    )


    return result.x


# ============================================================
# 逐帧求解
# ============================================================

start_frame = int(
    np.floor(
        BLEND_START * FPS
    )
)

end_frame = int(
    np.ceil(
        BLEND_END * FPS
    )
)


print("=" * 72)
print("OpenLoong V2.4 hand mapping POC")
print("=" * 72)
print("source:", SRC_CSV)
print(
    "optimization frames:",
    start_frame,
    "~",
    end_frame,
)
print(
    "full alignment:",
    FULL_START,
    "~",
    FULL_END,
    "sec",
)
print("=" * 72)


r_target = x[
    :,
    [
        R_ARM_06,
        R_ARM_07,
    ]
].copy()

l_target = x[
    :,
    [
        L_ARM_06,
        L_ARM_07,
    ]
].copy()


prev_r = r_target[
    start_frame
].copy()

prev_l = l_target[
    start_frame
].copy()


for frame_id in range(
    start_frame,
    end_frame + 1,
):

    q = csv_row_to_qpos(
        x[frame_id]
    )


    r_sol = solve_hand(
        q,
        "right",
        prev_r,
    )

    l_sol = solve_hand(
        q,
        "left",
        prev_l,
    )


    r_target[
        frame_id
    ] = r_sol

    l_target[
        frame_id
    ] = l_sol


    prev_r = r_sol
    prev_l = l_sol


    if (
        frame_id == start_frame
        or frame_id % 15 == 0
        or frame_id == end_frame
    ):

        print(
            f"solve "
            f"{frame_id}/{end_frame} "
            f"t={frame_id/FPS:.2f}s"
        )


# ============================================================
# 对目标腕关节做中心平滑
# ============================================================

for col in range(2):

    segment = r_target[
        start_frame:
        end_frame + 1,
        col,
    ]

    r_target[
        start_frame:
        end_frame + 1,
        col,
    ] = centered_smooth(
        segment,
        SMOOTH_RADIUS,
    )


    segment = l_target[
        start_frame:
        end_frame + 1,
        col,
    ]

    l_target[
        start_frame:
        end_frame + 1,
        col,
    ] = centered_smooth(
        segment,
        SMOOTH_RADIUS,
    )


# ============================================================
# 转为“相对原动作的修正量”
# ============================================================

r_original = x[
    :,
    [
        R_ARM_06,
        R_ARM_07,
    ]
]

l_original = x[
    :,
    [
        L_ARM_06,
        L_ARM_07,
    ]
]


r_delta = (
    r_target
    - r_original
)

l_delta = (
    l_target
    - l_original
)


# 只在实验窗口内保留修正
for i in range(len(x)):

    s = blend_strength(
        i / FPS
    )

    r_delta[i] *= s
    l_delta[i] *= s


# ============================================================
# 防止机器人腕部突变
# ============================================================

for col in range(2):

    r_delta[
        :,
        col,
    ] = limit_delta_velocity(
        r_delta[:, col],
        MAX_DELTA_STEP,
    )

    l_delta[
        :,
        col,
    ] = limit_delta_velocity(
        l_delta[:, col],
        MAX_DELTA_STEP,
    )


# ============================================================
# 写回
# ============================================================

y[
    :,
    R_ARM_06
] = (
    r_original[:, 0]
    + r_delta[:, 0]
)

y[
    :,
    R_ARM_07
] = (
    r_original[:, 1]
    + r_delta[:, 1]
)

y[
    :,
    L_ARM_06
] = (
    l_original[:, 0]
    + l_delta[:, 0]
)

y[
    :,
    L_ARM_07
] = (
    l_original[:, 1]
    + l_delta[:, 1]
)


# 再做一次机械限位保险
y[:, R_ARM_06] = np.clip(
    y[:, R_ARM_06],
    *R06_BOUNDS,
)

y[:, R_ARM_07] = np.clip(
    y[:, R_ARM_07],
    *R07_BOUNDS,
)

y[:, L_ARM_06] = np.clip(
    y[:, L_ARM_06],
    *L06_BOUNDS,
)

y[:, L_ARM_07] = np.clip(
    y[:, L_ARM_07],
    *L07_BOUNDS,
)


np.savetxt(
    OUT_CSV,
    y,
    delimiter=",",
    fmt="%.9f",
)


# ============================================================
# 数值验证
# ============================================================

def evaluate(csv_data):

    right_errors = []
    left_errors = []

    for frame_id in range(
        int(FULL_START * FPS),
        int(FULL_END * FPS) + 1,
    ):

        q = csv_row_to_qpos(
            csv_data[
                frame_id
            ]
        )

        data.qpos[:] = q

        mj.mj_forward(
            model,
            data,
        )


        head_forward = (
            body_rot(
                data,
                HEAD_ID,
            )
            @ HEAD_FORWARD_LOCAL
        )


        right_dir = (
            body_rot(
                data,
                R_HAND_ID,
            )
            @ R_FINGER_LOCAL
        )

        left_dir = (
            body_rot(
                data,
                L_HAND_ID,
            )
            @ L_FINGER_LOCAL
        )


        right_errors.append(
            angle_deg(
                right_dir,
                head_forward,
            )
        )

        left_errors.append(
            angle_deg(
                left_dir,
                head_forward,
            )
        )


    return (
        np.asarray(
            right_errors
        ),
        np.asarray(
            left_errors
        ),
    )


before_r, before_l = evaluate(
    x
)

after_r, after_l = evaluate(
    y
)


print()
print("=" * 72)
print("15~18s hand / head direction angle")
print("=" * 72)

print(
    "RIGHT before:"
)

print(
    f"  mean={before_r.mean():.2f} deg "
    f"max={before_r.max():.2f}"
)

print(
    "RIGHT after:"
)

print(
    f"  mean={after_r.mean():.2f} deg "
    f"max={after_r.max():.2f}"
)

print()

print(
    "LEFT before:"
)

print(
    f"  mean={before_l.mean():.2f} deg "
    f"max={before_l.max():.2f}"
)

print(
    "LEFT after:"
)

print(
    f"  mean={after_l.mean():.2f} deg "
    f"max={after_l.max():.2f}"
)


print()
print("maximum absolute modification:")

print(
    "R06:",
    f"{np.rad2deg(np.max(np.abs(r_delta[:,0]))):.2f} deg"
)

print(
    "R07:",
    f"{np.rad2deg(np.max(np.abs(r_delta[:,1]))):.2f} deg"
)

print(
    "L06:",
    f"{np.rad2deg(np.max(np.abs(l_delta[:,0]))):.2f} deg"
)

print(
    "L07:",
    f"{np.rad2deg(np.max(np.abs(l_delta[:,1]))):.2f} deg"
)


print()
print(
    "CSV:",
    OUT_CSV,
)


# ============================================================
# 离线渲染
# ============================================================

print()
print("=" * 72)
print("Rendering 661-frame MuJoCo video")
print("=" * 72)


viewer = RobotMotionViewer(
    robot_type="openloong",

    camera_follow=True,
    motion_fps=30,

    record_video=True,
    video_path=str(
        OUT_VIDEO
    ),

    video_width=608,
    video_height=1072,

    window_width=600,
    window_height=900,

    camera_lookat_height_offset=0.45,
    camera_elevation=-28.0,
    camera_distance_scale=1.70,
)


try:

    for i in range(
        len(y)
    ):

        root_pos = y[
            i,
            :3
        ]


        # CSV xyzw -> MuJoCo wxyz
        q_xyzw = y[
            i,
            3:7
        ]

        q_wxyz = q_xyzw[
            [3, 0, 1, 2]
        ]


        viewer.step(
            root_pos=root_pos,

            root_rot=q_wxyz,

            dof_pos=y[
                i,
                7:
            ],

            rate_limit=False,

            follow_camera=True,
        )


        if (
            i == 0
            or (i + 1) % 100 == 0
            or i == len(y) - 1
        ):

            print(
                f"rendered "
                f"{i+1}/{len(y)}"
            )


finally:

    viewer.close()


print()
print("=" * 72)
print("DONE")
print("CSV  :", OUT_CSV)
print("VIDEO:", OUT_VIDEO)
print("=" * 72)
