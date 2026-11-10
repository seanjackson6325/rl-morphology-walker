"""Grows the trained network to the 36-value observation.

The right shin was missing from the observation. Adding it takes the observation
from 35 values to 36, and PPO.load() then refuses the old file. This script widens
the first layer instead, so the 40 million training steps are not lost.

Two changes are necessary, not one:

1. The new input column starts at zero. It has no effect at first, and the policy
   learns to use it during training.

2. The ranges moved. The old policy trained on a fixed body, so its body values
   were the constant [0.5, 1.8, 0, 1.556, 1.556, 1.556]. The same body now scales
   to all zeros. A copy of the weights would therefore change the output. The old
   constant goes into the layer bias instead:

       y = W_other x + W_body m_old + b      becomes
       y = W_other x + 0            + b2     with b2 = b + W_body m_old

   The result is the same output on the fixed body, and no body response at first.
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

# Check that the new network gives the same actions on the fixed body.
fixed = CreatureWalker3DEnv(randomize=False, render_mode=None, max_episode_steps=1000)
obs_new, _ = fixed.reset()
obs_old = np.concatenate([obs_new[:N_NON_MORPH], M_OLD.astype(np.float32)])
a_old, _ = old.predict(obs_old, deterministic=True)
a_new, _ = new.predict(obs_new, deterministic=True)
print(f"largest action difference on the same state: {np.abs(a_old - a_new).max():.2e}")
print("A value near 1e-7 is correct. A larger value means the bias is wrong.")
fixed.close()
