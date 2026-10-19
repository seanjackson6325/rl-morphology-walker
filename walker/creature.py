"""Builds a pymunk biped from a Morphology spec: torso + two legs (thigh+shin),
each leg actuated at the hip and knee by a motor constraint.

Joint actuation model
----------------------
Each actuated joint is a `pymunk.SimpleMotor`, which drives the *relative
angular velocity* between two bodies, capped by a maximum torque
(`max_force`). This mirrors how a real servo/DC motor behaves: it chases a
target speed but can only supply so much torque while doing it. The RL
action for a joint sets the target speed; physics (gravity, contact forces,
the other joints) determines whether the motor can actually reach it.

Self-collision
---------------
A creature's own limbs must not collide with each other (a thigh swinging
into the torso shouldn't bounce off it) but every creature must still collide
with the ground. pymunk handles this via `ShapeFilter` groups: shapes that
share a nonzero group id never collide with each other, regardless of
anything else. Every shape belonging to one creature gets the same group id;
the ground keeps the default group (0), so it still collides with everyone.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import pymunk

from walker.morphology import LegSpec, Morphology

HIP_TORQUE = 45.0          # N*m, max torque the hip motor can supply
KNEE_TORQUE = 35.0         # N*m, max torque the knee motor can supply
HIP_ANGLE_LIMIT = 1.2      # rad, relative angle limit (~69 deg each way)
KNEE_ANGLE_LIMIT = 2.2     # rad
LIMB_FRICTION = 1.0
GROUND_FRICTION = 1.0

# Feet: a horizontal segment rigidly attached to the bottom of each shin, so
# the lower leg is L-shaped. Without this the leg ends in a knife-edge (the
# bottom of a thin vertical box) and the creature is an inverted pendulum on a
# point -- it has NO base of support and is mathematically required to topple.
# The foot extends mostly forward (toe) with a small heel, giving a footprint
# the creature can balance and push off from.
FOOT_FORWARD = 0.16        # m, toe extent in +x (walking direction)
FOOT_BACK = 0.05           # m, heel extent in -x
FOOT_HEIGHT = 0.05         # m, foot thickness

GROUND_COLLISION_TYPE = 0
FOOT_COLLISION_TYPE = 1

_next_group_id = 1


def _new_group_id() -> int:
    global _next_group_id
    gid = _next_group_id
    _next_group_id += 1
    return gid


@dataclass
class Leg:
    thigh: pymunk.Body
    shin: pymunk.Body
    thigh_shape: pymunk.Shape
    shin_shape: pymunk.Shape
    foot_shape: pymunk.Shape
    hip_motor: pymunk.SimpleMotor
    knee_motor: pymunk.SimpleMotor


@dataclass
class Creature:
    morphology: Morphology
    torso: pymunk.Body
    torso_shape: pymunk.Shape
    left: Leg
    right: Leg
    standing_height: float
    bodies: list = field(default_factory=list)
    shapes: list = field(default_factory=list)

    @property
    def motors(self) -> list[pymunk.SimpleMotor]:
        """Order: [hip_L, knee_L, hip_R, knee_R] -- this fixes the action vector layout."""
        return [self.left.hip_motor, self.left.knee_motor, self.right.hip_motor, self.right.knee_motor]

    def foot_contact(self, leg: Leg, ground_shape: pymunk.Shape) -> bool:
        touching = False

        def _check(arbiter: pymunk.Arbiter) -> None:
            nonlocal touching
            if ground_shape in arbiter.shapes:
                touching = True

        leg.shin.each_arbiter(_check)
        return touching


def _build_leg(
    space: pymunk.Space,
    torso: pymunk.Body,
    spec: LegSpec,
    hip_anchor_local: tuple[float, float],
    hip_world: tuple[float, float],
    density: float,
    group_id: int,
) -> Leg:
    # Thigh: top pinned to the hip, hangs straight down initially.
    thigh_mass = density * spec.thigh_width * spec.thigh_length
    thigh_moment = pymunk.moment_for_box(thigh_mass, (spec.thigh_width, spec.thigh_length))
    thigh = pymunk.Body(thigh_mass, thigh_moment)
    thigh.position = (hip_world[0], hip_world[1] - spec.thigh_length / 2.0)
    thigh_shape = pymunk.Poly.create_box(thigh, (spec.thigh_width, spec.thigh_length))
    thigh_shape.friction = LIMB_FRICTION
    thigh_shape.filter = pymunk.ShapeFilter(group=group_id)
    space.add(thigh, thigh_shape)

    hip_joint = pymunk.PivotJoint(torso, thigh, hip_anchor_local, (0, spec.thigh_length / 2.0))
    hip_limit = pymunk.RotaryLimitJoint(torso, thigh, -HIP_ANGLE_LIMIT, HIP_ANGLE_LIMIT)
    hip_motor = pymunk.SimpleMotor(torso, thigh, 0.0)
    hip_motor.max_force = HIP_TORQUE
    space.add(hip_joint, hip_limit, hip_motor)

    # Shin: top pinned to the knee (bottom of thigh), hangs straight down initially.
    # The foot mass is folded into the shin body so the existing explicit
    # mass/moment stays a reasonable approximation for the L-shaped lower leg.
    knee_world = (thigh.position.x, thigh.position.y - spec.thigh_length / 2.0)
    foot_area = (FOOT_FORWARD + FOOT_BACK) * FOOT_HEIGHT
    shin_mass = density * (spec.shin_width * spec.shin_length + foot_area)
    shin_moment = pymunk.moment_for_box(shin_mass, (spec.shin_width, spec.shin_length))
    shin = pymunk.Body(shin_mass, shin_moment)
    shin.position = (knee_world[0], knee_world[1] - spec.shin_length / 2.0)
    shin_shape = pymunk.Poly.create_box(shin, (spec.shin_width, spec.shin_length))
    shin_shape.friction = LIMB_FRICTION
    shin_shape.filter = pymunk.ShapeFilter(group=group_id)
    space.add(shin, shin_shape)

    # Foot: an L-shape attached at the bottom of the shin, in the shin's local
    # frame. This is the actual ground-contact shape (the base of support).
    shin_bottom = -spec.shin_length / 2.0
    foot_verts = [
        (-FOOT_BACK, shin_bottom),
        (FOOT_FORWARD, shin_bottom),
        (FOOT_FORWARD, shin_bottom - FOOT_HEIGHT),
        (-FOOT_BACK, shin_bottom - FOOT_HEIGHT),
    ]
    foot_shape = pymunk.Poly(shin, foot_verts)
    foot_shape.friction = LIMB_FRICTION
    foot_shape.filter = pymunk.ShapeFilter(group=group_id)
    foot_shape.collision_type = FOOT_COLLISION_TYPE
    space.add(foot_shape)

    knee_joint = pymunk.PivotJoint(thigh, shin, (0, -spec.thigh_length / 2.0), (0, spec.shin_length / 2.0))
    knee_limit = pymunk.RotaryLimitJoint(thigh, shin, -KNEE_ANGLE_LIMIT, KNEE_ANGLE_LIMIT)
    knee_motor = pymunk.SimpleMotor(thigh, shin, 0.0)
    knee_motor.max_force = KNEE_TORQUE
    space.add(knee_joint, knee_limit, knee_motor)

    return Leg(
        thigh=thigh,
        shin=shin,
        thigh_shape=thigh_shape,
        shin_shape=shin_shape,
        foot_shape=foot_shape,
        hip_motor=hip_motor,
        knee_motor=knee_motor,
    )


def build_creature(space: pymunk.Space, morphology: Morphology, origin_x: float, ground_y: float) -> Creature:
    """Adds a fresh creature to `space`, standing upright at x=origin_x with
    its longer leg's foot resting at ground_y, and returns a handle to it."""

    m = morphology
    group_id = _new_group_id()

    max_leg_len = max(
        m.left.thigh_length + m.left.shin_length,
        m.right.thigh_length + m.right.shin_length,
    )
    # +FOOT_HEIGHT: the foot sits below the bottom of the shin, so the torso
    # must spawn that much higher for the feet to rest on the ground.
    torso_y = ground_y + max_leg_len + FOOT_HEIGHT + m.torso_height / 2.0

    torso_mass = m.torso_density * m.torso_width * m.torso_height
    torso_moment = pymunk.moment_for_box(torso_mass, (m.torso_width, m.torso_height))
    torso = pymunk.Body(torso_mass, torso_moment)
    torso.position = (origin_x, torso_y)
    torso_shape = pymunk.Poly.create_box(torso, (m.torso_width, m.torso_height))
    torso_shape.friction = LIMB_FRICTION
    torso_shape.filter = pymunk.ShapeFilter(group=group_id)
    space.add(torso, torso_shape)

    hip_y_local = -m.torso_height / 2.0
    left_hip_local = (-m.torso_width * 0.25, hip_y_local)
    right_hip_local = (m.torso_width * 0.25, hip_y_local)
    left_hip_world = (origin_x + left_hip_local[0], torso_y + hip_y_local)
    right_hip_world = (origin_x + right_hip_local[0], torso_y + hip_y_local)

    left = _build_leg(space, torso, m.left, left_hip_local, left_hip_world, m.leg_density, group_id)
    right = _build_leg(space, torso, m.right, right_hip_local, right_hip_world, m.leg_density, group_id)

    creature = Creature(
        morphology=m,
        torso=torso,
        torso_shape=torso_shape,
        left=left,
        right=right,
        standing_height=torso_y,
    )
    creature.bodies = [torso, left.thigh, left.shin, right.thigh, right.shin]
    creature.shapes = [
        torso_shape,
        left.thigh_shape,
        left.shin_shape,
        left.foot_shape,
        right.thigh_shape,
        right.shin_shape,
        right.foot_shape,
    ]
    return creature
