"""Gymnasium environment for the 3D walker.

The 3D version of walker/env.py. The physics engine is MuJoCo. The training code
is the same for both versions.

Each hip has two joints. The pitch joint swings the leg. The roll joint moves the
leg sideways. A 2D creature cannot fall sideways, so it does not need the roll
joint.

The body values are always in the observation, also for a fixed body. The
observation size is therefore constant, and one network shape serves both
experiments.
"""

from __future__ import annotations

from typing import Optional

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

from walker.mjcf import (
    FOOT_HEIGHT,
    FOOT_TOUCH_SENSORS,
    N_MORPHOLOGY_OBS,
    build_mjcf,
    fixed_morphology3d,
    random_morphology3d,
)

FRAME_SKIP = 5          # physics steps per control step -> 40 Hz control

# Start pose, in radians. A start from straight legs makes a step hard to find.
# A small bend puts the legs where a step is an easy change. The signs are for
# the backward knee: the knee angle is negative when the knee bends.
HIP_PITCH_CROUCH = 0.5
KNEE_CROUCH = -1.0
ANKLE_CROUCH = 0.4      # hold the foot flat, or it catches on the first step

FORWARD_WEIGHT = 2.5
ALIVE_BONUS = 0.15
CTRL_COST = 0.001
LATERAL_WEIGHT = 0.3       # sideways speed
LATERAL_POS_WEIGHT = 0.5   # sideways drift
HEADING_WEIGHT = 0.6       # face +x
UPRIGHT_WEIGHT = 0.9
SMOOTH_WEIGHT = 0.01       # large changes between two actions

# Foot clearance and the flight penalty were for the forward knee. The backward
# knee bends without help and learns better without them, so both are off.
CLEARANCE_WEIGHT = 0.0
CLEARANCE_CAP = 0.12       # m
CLEARANCE_SPEED_CAP = 1.0  # m/s
FLIGHT_PENALTY = 1.0       # keep one foot down, or it bounds and falls

KNEE_WEIGHT = 0.5          # bend the swing knee
KNEE_CAP = 1.2             # rad

# Pay for the WEAKER of the two knees, not for the sum. A sum lets one knee
# collect half the term, and the creature then drags the other leg straight.
# The term uses knee movement per step, not knee angle: a deep fixed bend would
# also satisfy an angle. The speed gate below stops a creature that moves its
# knees in one place.
# The weight was 2.5 at first. The creature then took very short fast steps,
# because short steps give high knee movement. 1.6 keeps both knees and gives
# the forward term priority again.
BOTHKNEE_WEIGHT = 1.6
BOTHKNEE_CAP = 0.05        # rad per step where the term stops to grow
BOTHKNEE_ALPHA = 1.0 / 20  # average over about one step cycle

FLATFOOT_WEIGHT = 0.0      # off: the ankle range already stops a tiptoe gait

# Height reward. Without it the creature walks on its knees, because the upright
# term reads the torso angle only, not the torso height. The target is a peak,
# not a ramp: too tall also loses reward. A ramp let it stand on straight legs.
STAND_WEIGHT = 0.8
STAND_TARGET_FRAC = 0.80
STAND_TOL = 0.20

FALL_HEIGHT_FRACTION = 0.6   # stop below this part of the standing height
MIN_UPRIGHT = 0.6            # stop above a tilt of about 53 degrees

N_JOINTS = 8   # hip pitch, hip roll, knee and ankle, for each leg
N_OBS = 1 + 4 + N_JOINTS + 6 + N_JOINTS + 2 + N_MORPHOLOGY_OBS


class CreatureWalker3DEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 40}

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

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(N_JOINTS,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(N_OBS,), dtype=np.float32)

        self.model = None
        self.data = None
        self.morphology = None
        self._standing_height = 1.0
        self._torso_id = -1
        self._sensor_adr = []
        self._steps = 0
        self._viewer = None

    def _build(self):
        self.morphology = random_morphology3d(self.np_random) if self.randomize else fixed_morphology3d()
        xml = build_mjcf(self.morphology)
        self.model = mujoco.MjModel.from_xml_string(xml)
        self.data = mujoco.MjData(self.model)
        self._standing_height = self.morphology.standing_height()
        self._torso_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "torso")
        self._sensor_adr = [
            self.model.sensor_adr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, name)]
            for name in FOOT_TOUCH_SENSORS
        ]
        self._foot_geom_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name)
            for name in ("left_foot", "right_foot")
        ]
        self._knee_adr = [
            self.model.jnt_qposadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)]
            for name in ("left_knee", "right_knee")
        ]
        self._foot_body_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
            for name in ("left_foot", "right_foot")
        ]

    def _set_crouch_pose(self):
        """Set the start pose, then move the torso down until the lower foot touches
        the ground. This works for any body."""
        # qpos holds the free joint in 0 to 6, then the 8 hinges in order:
        # left hip pitch 7, hip roll 8, knee 9, ankle 10; right 11 to 14.
        qpos = self.data.qpos
        qpos[7] = HIP_PITCH_CROUCH    # left hip pitch
        qpos[9] = KNEE_CROUCH         # left knee
        qpos[10] = ANKLE_CROUCH       # left ankle (foot flat)
        qpos[11] = HIP_PITCH_CROUCH   # right hip pitch
        qpos[13] = KNEE_CROUCH        # right knee
        qpos[14] = ANKLE_CROUCH       # right ankle (foot flat)
        mujoco.mj_forward(self.model, self.data)
        lowest = min(
            float(self.data.geom_xpos[g][2]) - FOOT_HEIGHT / 2.0 for g in self._foot_geom_ids
        )
        qpos[2] += 0.02 - lowest     # lower (or raise) torso so the lower foot is 2cm above ground
        mujoco.mj_forward(self.model, self.data)

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)

        # Build a new model only for a new body. A fixed body keeps its model and
        # data. This is faster, and it keeps an open viewer valid: the viewer binds
        # to one model object, and a new object each episode freezes the window.
        if self.model is None or self.randomize:
            self._build()
            if self.render_mode == "human" and self._viewer is not None:
                self._viewer.close()  # force re-bind to the new model/data
                self._viewer = None
        else:
            mujoco.mj_resetData(self.model, self.data)

        # Start in a crouched ready stance (much easier to learn stepping from).
        self._set_crouch_pose()
        self._prev_action = np.zeros(N_JOINTS, dtype=np.float64)
        # Knee movement, averaged per leg. It starts at the cap and falls to the true
        # value in about 1 second. A start at zero would remove reward from the first
        # 40 steps of each episode, and tie this term to the episode length.
        self._prev_knee = np.array([self.data.qpos[k] for k in self._knee_adr], dtype=np.float64)
        self._knee_ema = np.full(2, BOTHKNEE_CAP, dtype=np.float64)

        # A small random change to the start pose.
        self.data.qpos[:] += self.np_random.uniform(-0.01, 0.01, size=self.model.nq)
        self.data.qvel[:] += self.np_random.uniform(-0.01, 0.01, size=self.model.nv)
        mujoco.mj_forward(self.model, self.data)

        self._steps = 0
        return self._get_obs(), {}

    def step(self, action: np.ndarray):
        action = np.clip(action, -1.0, 1.0).astype(np.float64)
        self.data.ctrl[:] = action

        x_before = self.data.xpos[self._torso_id][0]
        y_before = self.data.xpos[self._torso_id][1]
        for _ in range(FRAME_SKIP):
            mujoco.mj_step(self.model, self.data)
        x_after = self.data.xpos[self._torso_id][0]
        y_after = self.data.xpos[self._torso_id][1]

        dt = FRAME_SKIP * self.model.opt.timestep
        vx = (x_after - x_before) / dt
        vy = (y_after - y_before) / dt

        R = self.data.xmat[self._torso_id].reshape(3, 3)
        upright = float(R[2, 2])
        heading = float(R[0, 0])   # torso forward-axis x-component; 1.0 = facing +x (straight ahead)
        torso_z = float(self.data.xpos[self._torso_id][2])
        torso_y = float(self.data.xpos[self._torso_id][1])

        forward_term = FORWARD_WEIGHT * vx
        lateral_term = -LATERAL_WEIGHT * abs(vy)
        lateral_pos_term = -LATERAL_POS_WEIGHT * abs(torso_y)   # penalize how far it has veered off the line
        heading_term = HEADING_WEIGHT * heading                 # reward facing straight ahead
        ctrl_term = -CTRL_COST * float(np.sum(np.square(action)))
        upright_term = UPRIGHT_WEIGHT * upright

        # Pay for a lifted foot only when the other foot is down. Pay nothing when
        # both feet are off the ground, or the creature hops.
        left_contact = self.data.sensordata[self._sensor_adr[0]] > 1.0
        right_contact = self.data.sensordata[self._sensor_adr[1]] > 1.0

        def _clear(gid):
            return min(max(float(self.data.geom_xpos[gid][2]) - FOOT_HEIGHT / 2.0, 0.0), CLEARANCE_CAP)

        clearance = 0.0
        if left_contact and not right_contact:      # right leg swinging
            clearance = _clear(self._foot_geom_ids[1])
        elif right_contact and not left_contact:     # left leg swinging
            clearance = _clear(self._foot_geom_ids[0])
        # A lifted foot pays only while the creature moves forward.
        clearance_term = CLEARANCE_WEIGHT * clearance * float(np.clip(vx, 0.0, CLEARANCE_SPEED_CAP))
        flight_term = -FLIGHT_PENALTY if (not left_contact and not right_contact) else 0.0

        # Bend the knee of the swing leg. A negative angle is a bent knee.
        knee_bend = 0.0
        if left_contact and not right_contact:       # right leg swinging
            knee_bend = max(0.0, -float(self.data.qpos[self._knee_adr[1]]))
        elif right_contact and not left_contact:      # left leg swinging
            knee_bend = max(0.0, -float(self.data.qpos[self._knee_adr[0]]))
        # Again, only while the creature moves forward.
        knee_term = KNEE_WEIGHT * min(knee_bend, KNEE_CAP) * float(np.clip(vx, 0.0, 1.0))

        # Pay for the knee that moves less. A single reading of the two knee angles
        # would be wrong: in a correct gait one knee is straight while the other bends,
        # so the smaller of the two is near zero. The average over one step cycle
        # answers the useful question instead: does this knee work at all?
        knee_now = np.array([self.data.qpos[k] for k in self._knee_adr], dtype=np.float64)
        self._knee_ema = (1.0 - BOTHKNEE_ALPHA) * self._knee_ema + BOTHKNEE_ALPHA * np.abs(knee_now - self._prev_knee)
        self._prev_knee = knee_now
        bothknee = float(np.min(np.clip(self._knee_ema, 0.0, BOTHKNEE_CAP))) / BOTHKNEE_CAP
        # The speed gate stops a creature that moves its knees in one place.
        bothknee_term = BOTHKNEE_WEIGHT * bothknee * float(np.clip(vx, 0.0, 1.0))

        # Pay a foot on the ground for a flat angle. This stops a tiptoe gait.
        flatfoot = 0.0
        if left_contact:
            flatfoot += float(self.data.xmat[self._foot_body_ids[0]].reshape(3, 3)[2, 2])
        if right_contact:
            flatfoot += float(self.data.xmat[self._foot_body_ids[1]].reshape(3, 3)[2, 2])
        flatfoot_term = FLATFOOT_WEIGHT * flatfoot

        height_frac = torso_z / self._standing_height
        stand = 1.0 - abs(height_frac - STAND_TARGET_FRAC) / STAND_TOL  # peaked at the target height
        stand_term = STAND_WEIGHT * float(np.clip(stand, 0.0, 1.0))

        smooth_term = -SMOOTH_WEIGHT * float(np.sum(np.square(action - self._prev_action)))
        self._prev_action = action

        reward = (
            forward_term + lateral_term + lateral_pos_term + heading_term + ctrl_term + upright_term
            + clearance_term + knee_term + bothknee_term + flatfoot_term + stand_term + flight_term + smooth_term + ALIVE_BONUS
        )

        self._steps += 1
        fallen = torso_z < FALL_HEIGHT_FRACTION * self._standing_height or upright < MIN_UPRIGHT
        terminated = bool(fallen)
        truncated = self._steps >= self.max_episode_steps

        if self.render_mode == "human":
            self.render()

        return self._get_obs(), reward, terminated, truncated, {}

    def _get_obs(self) -> np.ndarray:
        qpos = self.data.qpos
        qvel = self.data.qvel
        height = np.array([qpos[2]], dtype=np.float32)
        quat = qpos[3:7].astype(np.float32)
        joint_angles = qpos[7 : 7 + N_JOINTS].astype(np.float32)
        torso_vel = qvel[0:6].astype(np.float32)
        joint_vels = qvel[6 : 6 + N_JOINTS].astype(np.float32)
        contacts = np.array(
            [1.0 if self.data.sensordata[adr] > 1.0 else 0.0 for adr in self._sensor_adr],
            dtype=np.float32,
        )
        morphology = self.morphology.as_observation()
        return np.concatenate([height, quat, joint_angles, torso_vel, joint_vels, contacts, morphology])

    def render(self):
        import time

        if self._viewer is None:
            import mujoco.viewer as mj_viewer

            self._viewer = mj_viewer.launch_passive(self.model, self.data)
            with self._viewer.lock():
                self._viewer.cam.distance = 7.0   # zoom out so the whole body + stride is visible
                self._viewer.cam.azimuth = 90      # side-on view of the sagittal (forward) gait
                self._viewer.cam.elevation = -12

        # Follow the torso, or the creature walks out of the frame.
        with self._viewer.lock():
            self._viewer.cam.lookat[0] = float(self.data.xpos[self._torso_id][0])
            self._viewer.cam.lookat[1] = float(self.data.xpos[self._torso_id][1])
            self._viewer.cam.lookat[2] = 0.8

        self._viewer.sync()
        # Run at about real time. This runs for playback only, never in training.
        time.sleep(FRAME_SKIP * self.model.opt.timestep)

    def close(self):
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None
