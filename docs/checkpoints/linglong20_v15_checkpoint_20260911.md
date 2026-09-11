# LingLong2.0 Retarget Checkpoint - V1.5

Date: 2026-09-11

## Branch

Target branch:
`openloong-linglong20-retarget`

## Current version

Main retarget script:

`tools/linglong/retarget_linglong20_arm_v15_strong_extension.py`

Viewer:

`tools/linglong/view_linglong20_qpos37.py`

Generated trajectory:

`output/linglong20_arm_v15_strong_extension/linglong20_arm_v15_strong_extension_qpos37.csv`

## Motion format

- frames: 661
- qpos dimension: 37
- MuJoCo nq: 37
- MuJoCo nv: 36
- robot movable joints: 30
- root: xyz(3) + quaternion(4)
- arm mapping verified

Arm qpos:

- left shoulder pitch: 23
- left shoulder roll: 24
- left shoulder yaw: 25
- left elbow: 26
- left wrist roll: 27
- left wrist pitch: 28
- left wrist yaw: 29

- right shoulder pitch: 30
- right shoulder roll: 31
- right shoulder yaw: 32
- right elbow: 33
- right wrist roll: 34
- right wrist pitch: 35
- right wrist yaw: 36

## V1.5 arm changes

V1.5 adds:

1. stronger human arm extension detection
2. model-specific elbow straight calibration
3. direct elbow-joint extension residual
4. geometric straight-arm residual
5. reduced V1 elbow-posture resistance during extension
6. reduced elbow continuity/acceleration resistance during extension
7. previous collision-aware IK retained
8. previous temporal anti-flip / boundary bridge retained

## Straight elbow calibration

LEFT:
- local arm index: 3
- straight elbow joint: approximately 90 deg
- geometric bend: approximately 0.01 deg

RIGHT:
- local arm index: 3
- straight elbow joint: approximately 90 deg
- geometric bend: approximately 0.00 deg

## Current validation

Non-arm preservation:

`qpos[0:23] max diff = 0.0`

LEFT:
- penetration frames: 1
- penetration/contact frame: 422
- actual contact frames: 1
- minimum signed distance: about -12.70 mm

RIGHT:
- penetration frames: 0
- actual contact frames: 0
- minimum signed distance: about +10.29 mm

Temporal:

- max arm step: about 8.02 deg
- frame: 619
- joint: right_wrist_roll_joint

Output:

- shape: (661, 37)
- finite: True

## Visual status

Current visual judgement:

- overall motion direction is basically correct
- arm retargeting is substantially improved
- elbow extension still requires further refinement
- wrist behavior still requires final inspection/refinement
- robot body appears continuously tilted during playback

## Important next issue

The body tilt is NOT introduced by the current arm retarget stage because:

`qpos[0:23] max diff = 0.0`

The next debugging stage should therefore inspect the upstream body trajectory, especially:

1. root quaternion qpos[3:7]
2. waist_yaw_joint
3. waist_pitch_joint
4. head/torso visual orientation
5. source body CSV vs LingLong2.0 model frame convention
6. whether the tilt exists in the source trajectory before arm retargeting
7. whether the viewer/model coordinate convention creates an apparent tilt

Do not modify V1.5 until this checkpoint has been preserved.

