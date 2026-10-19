"""Gait evaluation for the 3D walker. One script, all the metrics that mattered.

    python evaluate.py models/BEST_walk3d_bothknee_v5_33M.zip [--episodes 10]

Every metric here exists because a simpler one lied at some point:

* ALWAYS many random-start episodes, never a fixed seed -- an 8-episode random eval once
  revealed 25% falls that a 3-seed eval had hidden.
* reward/step, not ep_rew: total reward rewards SURVIVING. Two runs sat at ep_len ~900
  looking like the best in the project while merely squatting in place; reward/step put
  them at 2.31 against the ~2.4 posture-only floor, versus 5.77 for a real walker.
* stride length (distance per support-leg switch), not distance: a policy that minced at
  4.6 steps/s covered 9.3 m per episode and looked fine, while the user watching it
  correctly said "barely walking".
* joint oscillation (mean |dq| per step in steady state), NOT max-min span: a span metric
  captures the one-off release of the crouch start and reports a dead-straight knee as
  "moving 1.1 rad".
* torso roll with BOTH the startup transient AND the final 25 steps dropped: every failing
  episode ends in a topple, so whole-episode peak-to-peak measures the fall, not the gait.

IMPORTANT: a policy must be evaluated on the geometry it TRAINED on. Mismatched geometry
makes a good policy look broken; that has fooled us three separate times. See BEST_README.txt.
"""

from __future__ import annotations

import argparse

import numpy as np
from stable_baselines3 import PPO

from walker.env3d import CreatureWalker3DEnv, FALL_HEIGHT_FRACTION, MIN_UPRIGHT


def evaluate(model_path: str, n_episodes: int = 10, max_steps: int = 1000) -> dict:
    model = PPO.load(model_path)
    env = CreatureWalker3DEnv(randomize=False, render_mode=None, max_episode_steps=max_steps)
    eps = []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        x0 = float(env.data.xpos[env._torso_id][0])
        terminated = truncated = False
        total_r = 0.0
        xs, rolls, knees, contacts = [], [], [], []
        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            total_r += reward
            R = env.data.xmat[env._torso_id].reshape(3, 3)
            xs.append(float(env.data.xpos[env._torso_id][0]))
            rolls.append(np.degrees(np.arctan2(R[2, 1], R[2, 2])))
            knees.append([float(env.data.qpos[k]) for k in env._knee_adr])
            contacts.append([env.data.sensordata[s] > 1.0 for s in env._sensor_adr])
        n = len(xs)
        ss = n // 5                      # drop the startup transient
        dist = xs[-1] - x0
        c = np.array(contacts)
        sup = np.where(c[ss:, 0] & ~c[ss:, 1], 0, np.where(c[ss:, 1] & ~c[ss:, 0], 1, -1))
        sup = sup[sup >= 0]
        switches = int(np.sum(np.diff(sup) != 0)) if len(sup) > 1 else 0
        k = np.degrees(np.array(knees)[ss:])
        osc = np.mean(np.abs(np.diff(k, axis=0)), axis=0) if len(k) > 2 else np.zeros(2)
        core = np.array(rolls)[ss: max(ss + 1, n - 25)]     # drop the fall too
        eps.append(dict(
            steps=n, dist=dist, reward_per_step=total_r / n, fell=bool(terminated),
            speed=dist / (n * 0.025),
            stride=dist / switches if switches else float("nan"),
            cadence=switches / (n * 0.025),
            knee_osc=osc, knee_ratio=min(osc) / max(osc) if max(osc) > 1e-6 else 0.0,
            roll=core.max() - core.min() if len(core) > 5 else float("nan"),
        ))
    env.close()

    g = lambda key: np.nanmean([e[key] for e in eps])
    print(f"\n{model_path}   ({n_episodes} random-start deterministic episodes)\n")
    print(f"{'ep':>3}{'steps':>7}{'dist_m':>8}{'m/s':>6}{'stride':>8}{'rew/st':>8}{'kneeR':>7}{'roll°':>7}  outcome")
    for i, e in enumerate(eps):
        print(f"{i:>3}{e['steps']:>7}{e['dist']:>8.2f}{e['speed']:>6.2f}{e['stride']:>8.2f}"
              f"{e['reward_per_step']:>8.2f}{e['knee_ratio']:>7.2f}{e['roll']:>7.1f}  "
              f"{'FELL' if e['fell'] else 'full'}")
    falls = sum(e["fell"] for e in eps)
    steps = np.array([e["steps"] for e in eps], dtype=float)
    print(f"\n  fall rate       {falls}/{n_episodes}")
    print(f"  episode length  {steps.mean():.0f} +- {steps.std():.0f} steps ({steps.mean()*0.025:.1f} s)")
    print(f"  distance        {g('dist'):.2f} m      speed {g('speed'):.2f} m/s")
    print(f"  stride          {g('stride'):.2f} m/step   cadence {g('cadence'):.1f} steps/s")
    print(f"  reward/step     {g('reward_per_step'):.2f}   (posture-only floor is ~2.4 -- at or below it, it is not walking)")
    print(f"  knee symmetry   {g('knee_ratio'):.2f}   (1.0 = both knees used equally; 0.1 = one leg dragged)")
    print(f"  torso roll      {g('roll'):.1f} deg p-p, steady state")
    return dict(episodes=eps, falls=falls)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model")
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--max-episode-steps", type=int, default=1000)
    a = ap.parse_args()
    evaluate(a.model, a.episodes, a.max_episode_steps)
