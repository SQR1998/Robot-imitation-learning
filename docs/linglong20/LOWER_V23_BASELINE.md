# LingLong 2.0 Lower-Body Retarget V2.3 Baseline

Date: 2026-09-11

## Status

Lower-body retarget V2.3 is visually accepted as the first usable
LingLong 2.0 lower-body baseline.

Internal motion format:

- 661 frames
- 37-D qpos
- 30 FPS output
- root quaternion: WXYZ

## Mapping design

### Hip / knee chain

The old V1 absolute-point mapping is no longer used for the lower body.

V2 uses:

- LingLong's own fixed hip anchors
- LingLong thigh length: 0.3823 m
- LingLong shank length: 0.4200 m
- human hip->knee direction
- human knee->ankle direction

The lateral component is morphology-scaled with:

- lateral gain = 0.57

The value 0.57 was calibrated from the source SMPL-X geometry and
LingLong leg morphology rather than manually selected.

### Leg twist

V2.1 adds knee-plane / leg-twist mapping.

This substantially reduced the abnormal hip-yaw solution seen in V2.0.

### Ankle

V2.3 maps:

SMPL-X foot local +Y
    -> human sole/up normal
    -> LingLong foot +Z
    -> ankle_pitch + ankle_roll

The previous V1 ankle mapping is discarded.

The ankle does not attempt to reproduce full foot yaw because LingLong
has no ankle-yaw joint.

## Verified visual behavior

Accepted:

- no obvious knee valgus / inward collapse
- no knee-to-knee collision
- no knee interpenetration
- no obvious leg reversal
- ankle pitch visually reasonable
- ankle roll visually reasonable
- no persistent ankle-limit saturation
- no obvious ankle flipping
- overall lower-body motion visually acceptable

## V2.3 numerical ankle result

Left ankle pitch:
-38.78 deg .. +4.08 deg

Left ankle roll:
-11.20 deg .. +12.18 deg

Right ankle pitch:
-30.43 deg .. +11.24 deg

Right ankle roll:
-5.86 deg .. +8.97 deg

All ankle limit-hit counts:
0

Sole-normal fit is effectively exact for the 2-DoF ankle target.

## Remaining optimization

Do not add collision correction unless a remaining collision persists
after mapping optimization.

Next priorities:

1. diagnose toe-yaw / hip-yaw relationship
2. improve lower-body temporal continuity where necessary
3. verify root / foot ground behavior
4. only then freeze final lower-body mapping
5. upper-body retargeting comes afterwards

The LingLong 2.0 model has no hand/finger DoFs. Human hand information,
if reused later, will only provide wrist orientation guidance.
