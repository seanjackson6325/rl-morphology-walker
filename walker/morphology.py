"""Procedural biped morphology generation.

This is the "domain randomization" piece: instead of one fixed body, every
episode can sample a new set of physical parameters from these ranges. The
policy never sees these ranges directly, but it does see the *resulting*
parameters (normalized) as part of its observation, via `as_observation()`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

TORSO_WIDTH_RANGE = (0.35, 0.55)        # m
TORSO_HEIGHT_RANGE = (0.20, 0.32)       # m
LIMB_LENGTH_RANGE = (0.28, 0.50)        # m, applies to thigh and shin independently
LIMB_WIDTH = 0.08                       # m, fixed for simplicity
# "Areal" density (kg/m^2, since this is 2D). Kept low enough that the joint
# motors (HIP_TORQUE/KNEE_TORQUE in creature.py) can comfortably lift and swing
# a leg: at the old 300-900 range a single ~19 kg segment needed ~37 N*m just
# to hold horizontal, exceeding the motor's torque. the creature literally
# could not pick up its own leg.
DENSITY_RANGE = (80.0, 160.0)           # kg/m^2

N_MORPHOLOGY_OBS = 8


@dataclass(frozen=True)
class LegSpec:
    thigh_length: float
    shin_length: float
    thigh_width: float = LIMB_WIDTH
    shin_width: float = LIMB_WIDTH


@dataclass(frozen=True)
class Morphology:
    torso_width: float
    torso_height: float
    torso_density: float
    leg_density: float
    left: LegSpec
    right: LegSpec

    def as_observation(self) -> np.ndarray:
        """Body parameters the policy is allowed to see, rescaled to ~[-1, 1]."""

        def norm(x: float, lo: float, hi: float) -> float:
            return 2.0 * (x - lo) / (hi - lo) - 1.0

        return np.array(
            [
                norm(self.torso_width, *TORSO_WIDTH_RANGE),
                norm(self.torso_height, *TORSO_HEIGHT_RANGE),
                norm(self.torso_density, *DENSITY_RANGE),
                norm(self.leg_density, *DENSITY_RANGE),
                norm(self.left.thigh_length, *LIMB_LENGTH_RANGE),
                norm(self.left.shin_length, *LIMB_LENGTH_RANGE),
                norm(self.right.thigh_length, *LIMB_LENGTH_RANGE),
                norm(self.right.shin_length, *LIMB_LENGTH_RANGE),
            ],
            dtype=np.float32,
        )


def random_morphology(rng: np.random.Generator) -> Morphology:
    def random_leg() -> LegSpec:
        return LegSpec(
            thigh_length=float(rng.uniform(*LIMB_LENGTH_RANGE)),
            shin_length=float(rng.uniform(*LIMB_LENGTH_RANGE)),
        )

    return Morphology(
        torso_width=float(rng.uniform(*TORSO_WIDTH_RANGE)),
        torso_height=float(rng.uniform(*TORSO_HEIGHT_RANGE)),
        torso_density=float(rng.uniform(*DENSITY_RANGE)),
        leg_density=float(rng.uniform(*DENSITY_RANGE)),
        left=random_leg(),
        right=random_leg(),
    )


def fixed_morphology() -> Morphology:
    """One deterministic body. Used for the physics sanity check and the
    fixed-body baseline training run, before domain randomization is turned on.
    """
    leg = LegSpec(thigh_length=0.40, shin_length=0.40)
    return Morphology(
        torso_width=0.45,
        torso_height=0.25,
        torso_density=120.0,
        leg_density=120.0,
        left=leg,
        right=leg,
    )
