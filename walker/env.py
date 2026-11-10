"""Gymnasium environment for the 2D walker.

This file turns the pymunk rig in walker.creature into a learning problem.

The observation holds the torso pose and speed, the joint angles and speeds, the
foot contacts, and the body values. The body values are necessary. Without them
the same observation can need two different actions on two different bodies, and
the problem has no correct answer.

The reward pays for forward speed and subtracts a small cost for control effort
and for a large torso angle. An episode stops when the creature falls.
"""

from __future__ import annotations

from typing import Optional

import gymnasium as gym
import numpy as np
import pygame
import pymunk
from gymnasium import spaces

from walker.creature import GROUND_COLLISION_TYPE, GROUND_FRICTION, build_creature
from walker.morphology import N_MORPHOLOGY_OBS, fixed_morphology, random_morphology

PHYSICS_DT = 1.0 / 240.0
SUBSTEPS_PER_STEP = 4  # control runs at 60 Hz; physics is solved at 240 Hz for stability
MAX_MOTOR_RATE = 4.0  # rad/s

# Clean, standard Walker2d-style reward. Most of the earlier thrashing on these
# weights was futile because the body had no feet and couldn't walk at all; with
# a real base of support a conventional reward is enough.
FORWARD_WEIGHT = 1.0
ENERGY_WEIGHT = 0.001
ANGLE_WEIGHT = 0.05
ALIVE_BONUS = 0.05  # small per-step "healthy" reward; now safe since standing is achievable
FALL_PENALTY = 5.0
FALL_HEIGHT_FRACTION = 0.6  # episode ends if torso drops below this fraction of its settled height
MAX_TILT = 0.7  # rad (~40 deg); episode ends if torso tips past this

N_TORSO_OBS = 5
N_JOINT_OBS = 8
N_CONTACT_OBS = 2
N_OBS = N_TORSO_OBS + N_JOINT_OBS + N_CONTACT_OBS + N_MORPHOLOGY_OBS  # 23

PPM = 300.0
SCREEN_W, SCREEN_H = 1200, 600
GROUND_SCREEN_Y = SCREEN_H - 80


class CreatureWalkerEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 60}

    def __init__(
        self,
        randomize: bool = True,
        render_mode: Optional[str] = None,
        max_episode_steps: int = 1000,
    ):
        super().__init__()
        self.randomize = randomize
        self.render_mode = render_mode
        self.max_episode_steps = max_episode_steps

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(N_OBS,), dtype=np.float32)

        self.space: Optional[pymunk.Space] = None
        self.ground: Optional[pymunk.Shape] = None
        self.creature = None
        self._standing_height = 1.0
        self._steps = 0

        self._screen = None
        self._clock = None

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)

        self.space = pymunk.Space()
        self.space.gravity = (0.0, -9.81)

        self.ground = pymunk.Segment(self.space.static_body, (-200, 0), (200, 0), 0.05)
        self.ground.friction = GROUND_FRICTION
        self.ground.collision_type = GROUND_COLLISION_TYPE
        self.space.add(self.ground)

        morphology = random_morphology(self.np_random) if self.randomize else fixed_morphology()
        self.creature = build_creature(self.space, morphology, origin_x=0.0, ground_y=0.0)

        # Let the body settle under gravity into a natural standing pose before
        # the episode officially starts. the spawn pose is a kinematic
        # idealization (all joints at 0), not necessarily a physical equilibrium.
        for _ in range(60):
            self.space.step(PHYSICS_DT)
        self._standing_height = self.creature.torso.position.y

        self._steps = 0
        obs = self._get_obs()
        return obs, {}

    def step(self, action: np.ndarray):
        action = np.clip(action, -1.0, 1.0)
        for motor, a in zip(self.creature.motors, action):
            motor.rate = float(a) * MAX_MOTOR_RATE

        for _ in range(SUBSTEPS_PER_STEP):
            self.space.step(PHYSICS_DT)
        self._steps += 1

        obs = self._get_obs()
        reward, terminated = self._compute_reward(action)
        truncated = self._steps >= self.max_episode_steps

        if self.render_mode == "human":
            self.render()

        return obs, reward, terminated, truncated, {}

    def _get_obs(self) -> np.ndarray:
        torso = self.creature.torso
        torso_obs = np.array(
            [
                torso.angle,
                torso.angular_velocity,
                torso.position.y - self._standing_height,
                torso.velocity.x,
                torso.velocity.y,
            ],
            dtype=np.float32,
        )

        left, right = self.creature.left, self.creature.right
        joint_obs = np.array(
            [
                left.thigh.angle - torso.angle,
                left.thigh.angular_velocity - torso.angular_velocity,
                left.shin.angle - left.thigh.angle,
                left.shin.angular_velocity - left.thigh.angular_velocity,
                right.thigh.angle - torso.angle,
                right.thigh.angular_velocity - torso.angular_velocity,
                right.shin.angle - right.thigh.angle,
                right.shin.angular_velocity - right.thigh.angular_velocity,
            ],
            dtype=np.float32,
        )

        contact_obs = np.array(
            [
                float(self.creature.foot_contact(left, self.ground)),
                float(self.creature.foot_contact(right, self.ground)),
            ],
            dtype=np.float32,
        )

        morphology_obs = self.creature.morphology.as_observation()

        return np.concatenate([torso_obs, joint_obs, contact_obs, morphology_obs])

    def _compute_reward(self, action: np.ndarray) -> tuple[float, bool]:
        torso = self.creature.torso

        forward_term = FORWARD_WEIGHT * torso.velocity.x
        energy_term = -ENERGY_WEIGHT * float(np.sum(np.square(action)))
        angle_term = -ANGLE_WEIGHT * (torso.angle**2)
        reward = forward_term + energy_term + angle_term + ALIVE_BONUS

        fallen = (
            torso.position.y < FALL_HEIGHT_FRACTION * self._standing_height
            or abs(torso.angle) > MAX_TILT
        )
        if fallen:
            reward -= FALL_PENALTY
        return reward, fallen

    def _to_screen(self, point, camera_x: float) -> tuple[int, int]:
        sx = SCREEN_W / 2 + (point[0] - camera_x) * PPM
        sy = GROUND_SCREEN_Y - point[1] * PPM
        return int(sx), int(sy)

    def render(self):
        if self._screen is None:
            pygame.init()
            self._screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
            pygame.display.set_caption("Creature Walker")
            self._clock = pygame.time.Clock()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                return

        camera_x = self.creature.torso.position.x
        self._screen.fill((30, 30, 40))
        pygame.draw.line(self._screen, (90, 70, 50), (0, GROUND_SCREEN_Y), (SCREEN_W, GROUND_SCREEN_Y), 4)

        for shape in self.creature.shapes:
            verts = [shape.body.local_to_world(v) for v in shape.get_vertices()]
            points = [self._to_screen((v.x, v.y), camera_x) for v in verts]
            pygame.draw.polygon(self._screen, (120, 180, 220), points)
            pygame.draw.polygon(self._screen, (20, 20, 20), points, 2)

        pygame.display.flip()
        self._clock.tick(self.metadata["render_fps"])

    def close(self):
        if self._screen is not None:
            pygame.quit()
            self._screen = None
