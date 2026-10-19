"""Extend the fixed-body walker's network to the 36-D morphology observation.

Why surgery instead of retraining: adding the missing right-shin input changes the
observation from 35-D to 36-D, so PPO.load() cannot resume the 40M-step walker directly.
Rather than pay for that gait again, we widen the first layer by one column.

Two things have to be handled, not just one:

1. NEW INPUT (right shin, appended last). Initialise its column to ZERO so it contributes
   nothing at first; the policy learns to use it during the randomised run.

2. RE-CENTRED RANGES -- the subtle one. The old policy trained on a FIXED body, so its
   morphology inputs were the constant [0.5, 1.8, 0.0, 1.5556, 1.5556, 1.5556] and the
   network effectively baked that constant into its behaviour. Under the new ranges the same
   body normalises to all zeros, so simply copying the weights would silently change what the
   network computes. We therefore fold the old constant contribution into the bias:

       y = W_other x_other + W_morph m_old + b        (old)
       y = W_other x_other + 0 . m_new     + b'       (new, with b' = b + W_morph m_old)

   which is exactly equivalent on the fixed body, and morphology-blind elsewhere -- a clean
   starting point that cannot silently inherit a wrong morphology response.
"""

from __future__ import annotations

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from walker.env3d import CreatureWalker3DEnv, N_OBS

SRC = "models/BEST_walk3d_bothknee_v5_33M.zip"
DST = "models/warmstart_morph36"
N_OBS_OLD = 35
N_NON_MORPH = 29                      # height(1)+quat(4)+joints(8)+torso_vel(6)+joint_vels(8)+contacts(2)
M_OLD = np.array([0.5, 1.8, 0.0, 1.5555556, 1.5555556, 1.5555556], dtype=np.float64)

old = PPO.load(SRC)
assert old.observation_space.shape == (N_OBS_OLD,), old.observation_space.shape
assert N_OBS == 36, f"expected 36-D observation, got {N_OBS}"

env = DummyVecEnv([lambda: CreatureWalker3DEnv(randomize=True, render_mode=None, max_episode_steps=1000)])
new = PPO("MlpPolicy", env, learning_rate=3e-4, n_steps=2048, batch_size=256, n_epochs=10,
          gamma=0.99, gae_lambda=0.95, clip_range=0.2, ent_coef=0.0, verbose=0)

sd_old, sd_new = old.policy.state_dict(), new.policy.state_dict()
out = {}
for key, val in sd_old.items():
    if key.endswith(".0.weight") and val.shape[1] == N_OBS_OLD:        # the input layers
        W = val.detach().cpu().numpy()
        W2 = np.zeros((W.shape[0], N_OBS), dtype=W.dtype)
        W2[:, :N_NON_MORPH] = W[:, :N_NON_MORPH]                       # keep proprioception
        # morphology columns left at ZERO (blind at init, learned during training)
        out[key] = torch.as_tensor(W2)
        bkey = key.replace(".weight", ".bias")
        b = sd_old[bkey].detach().cpu().numpy()
        fold = W[:, N_NON_MORPH:N_OBS_OLD] @ M_OLD                     # old constant -> bias
        out[bkey] = torch.as_tensor((b + fold).astype(b.dtype))
    elif key.endswith(".0.bias") and sd_old[key.replace(".bias", ".weight")].shape[1] == N_OBS_OLD:
        continue                                                       # handled above
    else:
        out[key] = val.clone()

missing = set(sd_new) - set(out)
assert not missing, f"unmapped params: {missing}"
new.policy.load_state_dict(out)
new.save(DST)
print(f"saved {DST}.zip  ({N_OBS_OLD}-D -> {N_OBS}-D observation)")

# --- verify the surgery is behaviour-preserving on the FIXED body ---
fixed = CreatureWalker3DEnv(randomize=False, render_mode=None, max_episode_steps=1000)
obs_new, _ = fixed.reset()
obs_old = np.concatenate([obs_new[:N_NON_MORPH], M_OLD.astype(np.float32)])
a_old, _ = old.predict(obs_old, deterministic=True)
a_new, _ = new.predict(obs_new, deterministic=True)
print(f"max action difference on an identical fixed-body state: {np.abs(a_old - a_new).max():.2e}")
print("(should be ~1e-7; anything larger means the fold is wrong)")
fixed.close()
