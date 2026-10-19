"""Procedural MJCF (MuJoCo XML) generation for a 3D biped.

This is the 3D successor to walker/creature.py. Where the 2D version built a
pymunk body imperatively, here we emit an MJCF string that MuJoCo compiles into
a model. The morphology-lottery idea is unchanged: a Morphology3D describes the
body (limb lengths, torso size, density), and random_morphology3d() samples a new
one per episode.

Why 3D is harder than the 2D version
------------------------------------
A 2D biped only has to balance in the sagittal (fore-aft) plane. A 3D biped can
also fall sideways, so each hip gets TWO actuated axes: a pitch joint (swing the
leg forward/back, the primary walking motion) and a roll joint (move the leg
sideways, needed for lateral balance). Knees are a single pitch axis. Feet are
rigid boxes (no ankle joint) that give a base of support in both x and y.

Coordinate convention (MuJoCo): +z up, +x forward (walking direction), +y left.

Collision filtering
-------------------
We disable all self-collision among the creature's own geoms (matching the 2D
project's intent) while keeping foot/ground contact, via contype/conaffinity
bitmasks: floor is (contype=1, conaffinity=1); every creature geom is
(contype=2, conaffinity=1). Two geoms collide iff (contype_a & conaffinity_b) or
(contype_b & conaffinity_a) is nonzero -- creature-creature evaluates to 0 (no
self collision), creature-floor evaluates to 1 (contact kept).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# --- randomization ranges (meters / kg per m^3) ---
# RE-CENTRED 2026-08-23 on the known-good body (legs 0.55/0.55, torso 0.44 x 0.30, density
# 900) so fixed_morphology3d() sits at EXACTLY the centre of every range, i.e. normalises to
# the all-zeros morphology vector. Previously the trained body fell OUTSIDE these ranges
# (legs 0.55 vs a (0.32,0.50) range -> normalised 1.56, torso_width 0.30 -> 1.80), so turning
# on --randomize would have handed the policy systematically shorter, narrower bodies than
# the one it mastered, with morphology inputs far outside the network's trained range.
# These are deliberately narrow (~+-18%) as a curriculum starting point -- widen once the
# policy copes.
# Simple body: a plain VERTICALLY TALL rectangular box (no cockpit/head details).
# torso_height is its vertical (z) height, torso_width its lateral (y) span, and
# TORSO_DEPTH its fore-aft (x) depth.
TORSO_HEIGHT_RANGE = (0.36, 0.52)   # z extent (vertical height of the tall body)
TORSO_WIDTH_RANGE = (0.24, 0.36)    # y extent (lateral)
TORSO_DEPTH = 0.26                  # x extent (fore-aft) -- wider from the side view
THIGH_LENGTH_RANGE = (0.45, 0.65)
SHIN_LENGTH_RANGE = (0.45, 0.65)
DENSITY_RANGE = (700.0, 1100.0)     # kg/m^3 (true volumetric density now, 3D)

LIMB_RADIUS = 0.06
# The body is a hollow shell -- low density so its mass up high doesn't raise the
# center of mass (which is what makes a tall body hard to balance). The legs keep
# the morphology density, so the mass stays low where it helps balance.
BODY_DENSITY = 400.0
HIP_SEPARATION = 0.28               # controlled A/B at BOTHKNEE_WEIGHT=1.6 (v3 vs v4, only this differing):
                                    # 0.28 beat 0.40 on distance (9.34 vs 5.72 m), endurance (211 vs 146 steps)
                                    # and knee symmetry (0.80 vs 0.69). An earlier note claiming 0.28 failed was
                                    # wrong -- it compared runs differing in TWO variables and measured v3 on the
                                    # 0.40 body it never trained on. This is walk3d_bothknee_v3's geometry.
FOOT_LENGTH = 0.22                  # x
FOOT_WIDTH = 0.16                   # y (wider foot = more lateral stability)
FOOT_HEIGHT = 0.05                  # z
FOOT_FORWARD = 0.06                 # forward offset of foot center from ankle (toe bias)

# actuator strengths (max torque = gear, since ctrl is clamped to [-1, 1])
HIP_PITCH_GEAR = 80.0
HIP_ROLL_GEAR = 50.0
KNEE_GEAR = 60.0
ANKLE_GEAR = 30.0   # capable enough to balance (v9's gear-15 ankle was too weak and it fell)

N_MORPHOLOGY_OBS = 7

# Per leg now: hip pitch + hip roll + knee + ANKLE (pitch). The ankle lets the
# foot pivot to stay flat on the ground instead of tilting rigidly with the
# shin (which causes the foot to slap/catch -> the hobbling gait), and gives a
# push-off. Order matters: it fixes the action-vector and qpos layout.
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
                # right shin was MISSING here: random_morphology3d samples each leg
                # independently, so this dimension varied invisibly and the policy could not
                # condition on it -- fatal for the mismatched-leg case that makes the
                # morphology lottery interesting. Appended LAST so the warm-start can extend
                # the network's input layer by one column at the end.
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
    # Longer legs (0.55 each vs 0.40) so the creature MUST bend its knees to stand
    # at a normal height -> genuinely knee-bent gait (stiff-legged no longer reaches
    # a comfortable height). This is the geometric fix for the stiff knees.
    leg = LegSpec3D(thigh_length=0.55, shin_length=0.55)
    return Morphology3D(
        torso_height=0.44,   # vertical height (tall rectangle)
        torso_width=0.30,    # lateral span -- wider so it's less tippy for bent-knee walking
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
    # Plain tall rectangular body: x = fore-aft depth, y = lateral width, z = vertical height.
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
