#!/usr/bin/env python3

from pathlib import Path
import hashlib

ROOT = Path(__file__).resolve().parents[2]

robot_dir = (
    ROOT
    / "dataset/openloong_master1/robot"
    / "LingLong2.0_20260616"
)

src = robot_dir / "LingLong2.0.urdf"
dst = robot_dir / "LingLong2.0_mujoco.urdf"

EXPECTED_SHA256 = (
    "eb2a6341eb9c511b0d378d06fb320a0da30eed82dd2948c787f56562a6f890f8"
)

if not src.exists():
    raise FileNotFoundError(src)

digest = hashlib.sha256(src.read_bytes()).hexdigest()

print("source :", src)
print("sha256 :", digest)

if digest != EXPECTED_SHA256:
    raise RuntimeError(
        "LingLong2.0.urdf SHA256 mismatch. "
        "Refusing to modify an unknown model."
    )

text = src.read_text(encoding="utf-8")

old = 'meshdir="meshes/"'
new = 'meshdir="."'

if old not in text:
    raise RuntimeError(
        f'Expected {old!r} not found in source URDF'
    )

text = text.replace(old, new)

dst.write_text(text, encoding="utf-8")

print("created:", dst)
print()
print("MuJoCo compatibility change:")
print('  meshdir="meshes/"')
print("       ->")
print('  meshdir="."')
