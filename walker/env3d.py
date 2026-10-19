"""Gymnasium environment for the 3D MuJoCo biped walker.

The 3D successor to walker/env.py. The PPO/training code is unchanged -- this
just swaps the physics backend (MuJoCo instead of pymunk) and grows the
observation/action spaces.

What's new vs. the 2D env, and why
----------------------------------
* Action is 6-D now (hip pitch + hip roll + knee, per leg) instead of 4-D --
  the hip ROLL axes are what let the policy balance laterally, the degree of
  freedom 2D didn't have.
* The reward adds a LATERAL drift penalty (|v_y|) and an uprightness term.
  In 2D the creature was confined to the sagittal plane so it could only fall
  forward/back; in 3D it can also fall sideways or veer off course, so we must
  actively reward going straight and staying upright.
* The morphology vector is ALWAYS in the observation (even for a fixed body),
  so the observation size is constant (31) and one policy architecture serves
  both the fixed-body and randomized-body experiments.
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

FRAME_SKIP = 5          # physics steps per control step; control dt = 5 * 0.005 = 0.025 s (40 Hz)

# Crouched "ready stance" start pose (radians). Standing from locked-straight
# legs makes a stepping gait hard to discover (the only way to start moving is an
# awkward break from full extension), which is why the straight-leg start
# plateaued. A slight forward hip pitch + backward knee flex lowers the center of
# mass and puts the legs where alternating steps are an easy local move -- a
# standard trick for bipedal-locomotion RL. hip_pitch<0 tilts the thigh forward;
# knee>0 flexes the shin backward; ~knee = 2*|hip_pitch| keeps the feet under the
# torso so the start is balanced. NOTE: signs are for the BACKWARD-bending
# (avian / AT-ST chicken-walker) knee, range [-2.4, 0]: knee flexes negative,
# thigh tilts back, shin swings forward to keep the foot under the torso.
HIP_PITCH_CROUCH = 0.5
KNEE_CROUCH = -1.0   # deeper bend to start near the bent-knee target height (long legs)
ANKLE_CROUCH = 0.4   # tilt the foot flat to the ground at the start (bent knee tilts the shin, so
                     # ankle=0 leaves the foot toe-up and it stubs/stumbles on the first step)

FORWARD_WEIGHT = 2.5    # strong: bentleg_v1 stood still in a squat (0.1m); make forward progress clearly worth it
ALIVE_BONUS = 0.15
CTRL_COST = 0.001
LATERAL_WEIGHT = 0.3    # penalize sideways velocity
LATERAL_POS_WEIGHT = 0.5  # penalize sideways POSITION drift (|y|)
HEADING_WEIGHT = 0.6    # reward facing +x; moderate (the 1.2 in v18 helped yaw but contributed to instability)
UPRIGHT_WEIGHT = 0.9    # strong: the tall body was reclining back to walk on stiff legs + ankles
SMOOTH_WEIGHT = 0.01    # penalize abrupt action changes (anti-hobble); low so it doesn't suppress knee-stepping

# Foot-clearance reward: pays the creature for lifting a swing foot off the
# ground, which forces knee-bending steps instead of a stiff-legged waddle.
# CRUCIAL: only paid during SINGLE SUPPORT (the other foot planted). An earlier
# version rewarded every airborne foot independently, so a flight phase (both
# feet up) earned DOUBLE clearance -- which is exactly what produced a peg-leg
# hop (one locked straight support leg, hop, other leg lifted). Single-support
# clearance + a flight penalty (below) instead reward true alternating steps.
# NOTE: the foot-clearance + flight-penalty machinery below was all developed to
# coax the HUMAN (forward-bending) knee out of degenerate gaits. The avian
# (backward-bending, AT-ST) knee bends its knees naturally and learned best with
# a SIMPLE reward (the original v1), so for the chicken walker these are disabled
# (weights 0) -- they over-constrained it and made it fall fast. Restore them for
# any future human-knee / morphology-lottery work.
CLEARANCE_WEIGHT = 0.0  # a capable ankle can satisfy foot-lift without the knee; use KNEE_WEIGHT (below) instead
KNEE_WEIGHT = 0.5       # reward bending the swing knee (weak lever on its own; real fix for bent knees is leg geometry)
KNEE_CAP = 1.2          # rad, cap on rewarded knee flexion
# Weakest-link BOTH-knees reward, replacing the old SUMMED bent-knee term. The sum let one
# working knee collect half the term, so locking the other knee cost only ~5% of step reward
# -- cheap against the stability and zero holding torque of a rigid strut leg, so the policy
# rationally dragged one straight leg (measured: left knee 4.3 deg/step, right 0.46 deg/step).
# min() over the legs makes a locked knee score ZERO instead of half.
# Calibration: per-leg EMA of |dknee| measured 0.069 (working) vs 0.011 (locked) rad/step, so
# CAP=0.05 saturates on a real gait (no payoff for jitter) while the limp scores 22% of max.
# What this buys, measured on the limping policy: the old depth-based sum paid it 0.14/step of
# a 0.36 max, leaving only 0.22/step of headroom for fixing the locked knee. The min form pays
# 0.22 of a 1.00 max, leaving 0.78/step -- a 3.5x stronger gradient aimed at exactly the defect.
# NOT an asymmetry penalty |usage_L - usage_R|: that is minimized perfectly by TWO locked knees.
# Rate (|dknee|), not flexion depth: depth is farmable by a permanent deep bend (the v5
# "walking on its knees" failure); cyclic rate is not.
# 1.0 was measured too weak (bothknee_v1): at 1.0 the policy correctly computed that a fast
# LIMP beats a symmetric walk -- forward alone pays 4.23/step at 1.69 m/s, so locking a knee
# and keeping the speed won by ~0.58/step. At 2.5 a symmetric ~1.1 m/s walk wins by ~0.77/step
# instead. Safe direction to push BECAUSE of the clip(vx,0,1) gate below: under 1.0 m/s the
# term cuts itself off, so unlike lowering FORWARD_WEIGHT this cannot produce a squatter
# (FORWARD_WEIGHT 1.0 is what gave the bentleg_v1/v2 squats at 0.1 m).
# 2.5 fixed the knee symmetry (ratio 0.10 -> 0.98) but overshot: at 2.5 this term was 35% of all
# reward (2.07 of 5.84/step), so the policy bought KNEE MOTION at the expense of TRAVEL -- stride
# collapsed 0.64 -> 0.22 m and cadence doubled to 4.6 steps/s (mincing). Note the knees were NOT
# low-amplitude: 50 deg per cycle, nearly the limper's 62 deg. The term pays for knee motion whether
# or not it produces a stride. 1.6 sits above the ~1.4 break-even where a symmetric walk still beats
# a fast limp, while returning priority to forward travel.
BOTHKNEE_WEIGHT = 1.6
BOTHKNEE_CAP = 0.05        # rad/step of knee motion at which the term saturates
BOTHKNEE_ALPHA = 1.0 / 20  # EMA time constant ~0.5 s ~ one gait cycle
FLATFOOT_WEIGHT = 0.0   # disabled: the ankle range limit (mjcf, +0.4 max) mechanically prevents tiptoe now;
                        # the reward version (0.4) over-constrained balance in v14
CLEARANCE_CAP = 0.12    # m, target swing-foot lift height
CLEARANCE_SPEED_CAP = 1.0  # m/s; clearance reward is scaled by clip(vx,0,this) so it only pays while moving
FLIGHT_PENALTY = 1.0    # re-enabled: keeps >=1 foot down so it WALKS instead of bounding (bounding on the tall body fell in ~3s)

# Stand-tall reward: without it, the foot-clearance reward + crouch start get
# satisfied by a deep permanent knee-bend (walking on its knees), since nothing
# else rewards height (the upright term only cares about torso ORIENTATION, not
# how low it is). This pays full reward once the torso is near full standing
# height and ramps to zero in a deep crouch, while still allowing the natural
# slight knee bend of real walking (full reward kicks in at 0.85 of standing
# height, not 1.0, so it isn't forced to lock its legs straight).
STAND_WEIGHT = 0.8
# PEAKED height target: full reward at STAND_TARGET_FRAC of the (straight-leg)
# standing height, falling to zero STAND_TOL away in EITHER direction. Penalizing
# "too tall" (not just "too low") is what forces the long legs to bend -- a plain
# ramp let it stand tall on straight legs and collect full reward.
STAND_TARGET_FRAC = 0.80   # raised from 0.72: a walkable knee bend (~55 deg) instead of a too-deep 86 deg squat
STAND_TOL = 0.20

FALL_HEIGHT_FRACTION = 0.6   # terminate if torso drops below this fraction of standing height (a deep kneel no longer counts as "alive")
MIN_UPRIGHT = 0.6            # terminate if torso z-axis . world-up drops below this (~53 deg tilt) -- forbids the recline gait

# observation layout (see _get_obs): height(1) + quat(4) + joint angles(8)
# + torso lin/ang vel(6) + joint vels(6) + foot contacts(2) + morphology(6)
N_JOINTS = 8  # hip pitch + hip roll + knee + ankle, per leg
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
        """Put the hinges in the crouched ready stance, then drop the torso so the
        lower foot rests just on the ground (works for any morphology)."""
        # qpos layout: [0:7] free joint, then 8 hinges in declaration order:
        # left:  hip_pitch(7) hip_roll(8) knee(9) ankle(10)
        # right: hip_pitch(11) hip_roll(12) knee(13) ankle(14)
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

        # Rebuild the model only when we actually need a new body: the first
        # reset, or every reset under domain randomization. For a fixed body we
        # reuse the same MjModel/MjData (just reset state in place) -- this is
        # cheaper, and crucially keeps any attached viewer valid across episodes
        # (the viewer binds to specific model/data objects; replacing them every
        # episode is what made earlier playback freeze on episode 0).
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
        # Per-leg knee-motion EMA for the weakest-link both-knees reward. Optimistic init
        # (starts AT the cap, decays to the truth in ~1 s) keeps the term neutral at episode
        # start; a zero init would dock the first ~40 steps of every episode and couple this
        # term to episode length. Safe to pay early: ending an episode forfeits ~5.7/step.
        self._prev_knee = np.array([self.data.qpos[k] for k in self._knee_adr], dtype=np.float64)
        self._knee_ema = np.full(2, BOTHKNEE_CAP, dtype=np.float64)

        # Small randomization of the start pose for robustness (standard practice).
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

        # Foot contacts. Reward swing-foot clearance ONLY in single support (one
        # foot planted); penalize flight (both feet off) so it walks, not hops.
        left_contact = self.data.sensordata[self._sensor_adr[0]] > 1.0
        right_contact = self.data.sensordata[self._sensor_adr[1]] > 1.0

        def _clear(gid):
            return min(max(float(self.data.geom_xpos[gid][2]) - FOOT_HEIGHT / 2.0, 0.0), CLEARANCE_CAP)

        clearance = 0.0
        if left_contact and not right_contact:      # right leg swinging
            clearance = _clear(self._foot_geom_ids[1])
        elif right_contact and not left_contact:     # left leg swinging
            clearance = _clear(self._foot_geom_ids[0])
        # Gate clearance by forward speed so lifting a foot only pays while
        # actually traveling -- otherwise the policy farms it by marching in
        # place (v7's failure: ~0.9m in 25s while waving one leg).
        clearance_term = CLEARANCE_WEIGHT * clearance * float(np.clip(vx, 0.0, CLEARANCE_SPEED_CAP))
        flight_term = -FLIGHT_PENALTY if (not left_contact and not right_contact) else 0.0

        # Reward bending the SWING leg's knee (avian knee: more negative = more bent),
        # so it steps with its knees instead of stiff legs + ankle.
        knee_bend = 0.0
        if left_contact and not right_contact:       # right leg swinging
            knee_bend = max(0.0, -float(self.data.qpos[self._knee_adr[1]]))
        elif right_contact and not left_contact:      # left leg swinging
            knee_bend = max(0.0, -float(self.data.qpos[self._knee_adr[0]]))
        # Gate by forward speed so bending a knee only pays WHILE moving forward
        # -- otherwise the policy farms it by stepping in place (v10: ~0.26 m/s).
        knee_term = KNEE_WEIGHT * min(knee_bend, KNEE_CAP) * float(np.clip(vx, 0.0, 1.0))

        # Continuous bent-knee posture reward on BOTH knees (avian: bent = negative).
        # Instantaneous min(flexion_L, flexion_R) would be WRONG -- in a healthy gait the
        # stance knee is extended while the swing knee is flexed, so the instantaneous min is
        # near zero for CORRECT walking. The EMA over ~one gait cycle asks the right question:
        # "is this knee being used cyclically?" -- then pay for the WORSE leg.
        knee_now = np.array([self.data.qpos[k] for k in self._knee_adr], dtype=np.float64)
        self._knee_ema = (1.0 - BOTHKNEE_ALPHA) * self._knee_ema + BOTHKNEE_ALPHA * np.abs(knee_now - self._prev_knee)
        self._prev_knee = knee_now
        bothknee = float(np.min(np.clip(self._knee_ema, 0.0, BOTHKNEE_CAP))) / BOTHKNEE_CAP
        # vx-gated like the other gait terms, or the policy farms knee motion standing still
        # (the v7/v10 march-in-place failure).
        bothknee_term = BOTHKNEE_WEIGHT * bothknee * float(np.clip(vx, 0.0, 1.0))

        # Flat-foot reward: reward a PLANTED foot for being flat (its up-axis
        # vertical, xmat[2,2]~1). Discourages the tiptoe/toe-point gait.
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

        # Pan the camera to follow the torso so the walker stays centered as it
        # moves forward instead of striding out of frame.
        with self._viewer.lock():
            self._viewer.cam.lookat[0] = float(self.data.xpos[self._torso_id][0])
            self._viewer.cam.lookat[1] = float(self.data.xpos[self._torso_id][1])
            self._viewer.cam.lookat[2] = 0.8

        self._viewer.sync()
        # Pace playback to roughly real time so the gait is watchable (render is
        # only ever called for render_mode="human", never during training).
        time.sleep(FRAME_SKIP * self.model.opt.timestep)

    def close(self):
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None
