"""Builds the MuJoCo XML for a 3D two-leg creature.

A Morphology3D holds the body values: the limb lengths, the torso size and the
density. build_mjcf() turns one of these into a model file. random_morphology3d()
gives a new body for each episode.

Axes: +z is up, +x is the walk direction, +y is left.

The creature cannot collide with itself. The floor uses contype 1 and
conaffinity 1. Each part of the creature uses contype 2 and conaffinity 1. Two
parts touch only if one contype matches the other conaffinity, so two creature
parts never touch, and a foot still touches the floor.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Ranges for a random body, in metres and kg/m3. They are centred on the fixed
# body, so the fixed body scales to all zeros and sits in the middle. They are
# narrow on purpose. Widen them after the policy handles this width.
TORSO_HEIGHT_RANGE = (0.36, 0.52)   # z, the height of the tall box
TORSO_WIDTH_RANGE = (0.24, 0.36)    # y
TORSO_DEPTH = 0.26                  # x
THIGH_LENGTH_RANGE = (0.45, 0.65)
SHIN_LENGTH_RANGE = (0.45, 0.65)
DENSITY_RANGE = (700.0, 1100.0)

LIMB_RADIUS = 0.06
# The torso is light, so its mass does not raise the centre of mass. The legs use
# the body density, which keeps the mass low where it helps balance.
BODY_DENSITY = 400.0
HIP_SEPARATION = 0.28   # 0.28 beat 0.40 in a direct test: farther, and it fell later
FOOT_LENGTH = 0.22                  # x
FOOT_WIDTH = 0.16                   # y (wider foot = more lateral stability)
FOOT_HEIGHT = 0.05                  # z
FOOT_FORWARD = 0.06                 # forward offset of foot center from ankle (toe bias)

# Motor strength. The control value is -1 to +1, so gear is the largest torque.
HIP_PITCH_GEAR = 80.0
HIP_ROLL_GEAR = 50.0
KNEE_GEAR = 60.0
ANKLE_GEAR = 30.0   # capable enough to balance (v9's gear-15 ankle was too weak and it fell)

N_MORPHOLOGY_OBS = 7

# Four joints per leg. The ankle holds the foot flat and gives push-off. Without
# it the foot tilts with the shin and catches on the ground.
# This order sets the order of the action vector. Do not change it.
JOINT_NAMES = [
    "left_hip_pitch", "left_hip_roll", "left_knee", "left_ankle",
    "right_hip_pitch", "right_hip_roll", "right_knee", "right_ankle",
]
ACTUATOR_NAMES = list(JOINT_NAMES)
FOOT_TOUCH_SENSORS = ["left_foot_touch", "right_foot_touch"]


@dataclass(frozen=True)
class LegSpec3D:
    thigh_length: float
    shin_length: float


@dataclass(frozen=True)
class Morphology3D:
    torso_height: float
    torso_width: float
    density: float
    left: LegSpec3D
    right: LegSpec3D

    def standing_height(self) -> float:
        """Torso-center z at which the longer leg's foot rests on the floor."""
        longest = max(
            self.left.thigh_length + self.left.shin_length,
            self.right.thigh_length + self.right.shin_length,
        )
        return self.torso_height / 2.0 + longest + FOOT_HEIGHT

    def as_observation(self) -> np.ndarray:
        def norm(x: float, lo: float, hi: float) -> float:
            return 2.0 * (x - lo) / (hi - lo) - 1.0

        return np.array(
            [
                norm(self.torso_height, *TORSO_HEIGHT_RANGE),
                norm(self.torso_width, *TORSO_WIDTH_RANGE),
                norm(self.density, *DENSITY_RANGE),
                norm(self.left.thigh_length, *THIGH_LENGTH_RANGE),
                norm(self.left.shin_length, *SHIN_LENGTH_RANGE),
                norm(self.right.thigh_length, *THIGH_LENGTH_RANGE),
                # The right shin was missing here. Each leg is sampled on its own, so
                # this value changed and the policy could not read it. It is last in
                # the list, so an older network can grow by one input column.
                norm(self.right.shin_length, *SHIN_LENGTH_RANGE),
            ],
            dtype=np.float32,
        )


def random_morphology3d(rng: np.random.Generator) -> Morphology3D:
    def leg() -> LegSpec3D:
        return LegSpec3D(
            thigh_length=float(rng.uniform(*THIGH_LENGTH_RANGE)),
            shin_length=float(rng.uniform(*SHIN_LENGTH_RANGE)),
        )

    return Morphology3D(
        torso_height=float(rng.uniform(*TORSO_HEIGHT_RANGE)),
        torso_width=float(rng.uniform(*TORSO_WIDTH_RANGE)),
        density=float(rng.uniform(*DENSITY_RANGE)),
        left=leg(),
        right=leg(),
    )


def fixed_morphology3d() -> Morphology3D:
    # Long legs. The creature must bend its knees to stand at a normal height.
    # Short legs reach the ground straight, and the knees then stay straight.
    leg = LegSpec3D(thigh_length=0.55, shin_length=0.55)
    return Morphology3D(
        torso_height=0.44,   # vertical height (tall rectangle)
        torso_width=0.30,    # wide, so a bent-knee gait is less easy to tip over
        density=900.0,
        left=leg,
        right=leg,
    )


def _leg_xml(side: str, sign: float, spec: LegSpec3D, torso_height: float, density: float) -> str:
    """side: 'left'/'right'; sign: +1 for left (+y), -1 for right (-y)."""
    hip_y = sign * HIP_SEPARATION / 2.0
    hip_z = -torso_height / 2.0  # legs hang from the bottom of the tall body
    thigh_fromto = f"0 0 0 0 0 {-spec.thigh_length}"
    shin_fromto = f"0 0 0 0 0 {-spec.shin_length}"
    foot_half = f"{FOOT_LENGTH / 2.0} {FOOT_WIDTH / 2.0} {FOOT_HEIGHT / 2.0}"
    foot_pos = f"{FOOT_FORWARD} 0 {-FOOT_HEIGHT / 2.0}"  # in the foot body frame, below the ankle

    return f"""
      <body name="{side}_thigh" pos="0 {hip_y} {hip_z}">
        <joint name="{side}_hip_pitch" type="hinge" axis="0 1 0" range="-1.4 1.4"/>
        <joint name="{side}_hip_roll" type="hinge" axis="1 0 0" range="-0.5 0.5"/>
        <geom name="{side}_thigh" type="capsule" fromto="{thigh_fromto}" size="{LIMB_RADIUS}" density="{density}"/>
        <body name="{side}_shin" pos="0 0 {-spec.thigh_length}">
          <joint name="{side}_knee" type="hinge" axis="0 1 0" range="-2.4 0.0"/>
          <geom name="{side}_shin" type="capsule" fromto="{shin_fromto}" size="{LIMB_RADIUS}" density="{density}"/>
          <body name="{side}_foot" pos="0 0 {-spec.shin_length}">
            <joint name="{side}_ankle" type="hinge" axis="0 1 0" range="-0.6 0.4"/>
            <geom name="{side}_foot" type="box" pos="{foot_pos}" size="{foot_half}" density="{density}"/>
            <site name="{side}_foot_site" type="box" pos="{foot_pos}" size="{foot_half}"/>
          </body>
        </body>
      </body>"""


def build_mjcf(m: Morphology3D) -> str:
    h = m.standing_height() + 0.02
    # A plain tall box: x is depth, y is width, z is height.
    torso_half = f"{TORSO_DEPTH / 2.0} {m.torso_width / 2.0} {m.torso_height / 2.0}"

    left = _leg_xml("left", +1.0, m.left, m.torso_height, m.density)
    right = _leg_xml("right", -1.0, m.right, m.torso_height, m.density)

    return f"""<mujoco model="biped">
  <compiler angle="radian" inertiafromgeom="true"/>
  <option timestep="0.005" gravity="0 0 -9.81" iterations="20"/>
  <default>
    <geom contype="2" conaffinity="1" friction="1.0 0.1 0.1"/>
    <motor ctrllimited="true" ctrlrange="-1 1"/>
  </default>
  <worldbody>
    <geom name="floor" type="plane" size="50 50 0.1" contype="1" conaffinity="1" friction="1.0 0.1 0.1"/>
    <body name="torso" pos="0 0 {h}">
      <freejoint name="root"/>
      <geom name="torso" type="box" size="{torso_half}" density="{BODY_DENSITY}"/>
{left}
{right}
    </body>
  </worldbody>
  <actuator>
    <motor name="left_hip_pitch" joint="left_hip_pitch" gear="{HIP_PITCH_GEAR}"/>
    <motor name="left_hip_roll" joint="left_hip_roll" gear="{HIP_ROLL_GEAR}"/>
    <motor name="left_knee" joint="left_knee" gear="{KNEE_GEAR}"/>
    <motor name="left_ankle" joint="left_ankle" gear="{ANKLE_GEAR}"/>
    <motor name="right_hip_pitch" joint="right_hip_pitch" gear="{HIP_PITCH_GEAR}"/>
    <motor name="right_hip_roll" joint="right_hip_roll" gear="{HIP_ROLL_GEAR}"/>
    <motor name="right_knee" joint="right_knee" gear="{KNEE_GEAR}"/>
    <motor name="right_ankle" joint="right_ankle" gear="{ANKLE_GEAR}"/>
  </actuator>
  <sensor>
    <touch name="left_foot_touch" site="left_foot_site"/>
    <touch name="right_foot_touch" site="right_foot_site"/>
  </sensor>
</mujoco>"""
