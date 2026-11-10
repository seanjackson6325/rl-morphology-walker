"""Loads a trained policy and shows it in the viewer.

The policy runs with deterministic=True. It then gives the mean of its action
distribution. The random part of an action helps the policy to explore during
training. It is not useful after training.
"""

from __future__ import annotations

import argparse
import time

from stable_baselines3 import PPO


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="path to a .zip checkpoint (e.g. models/fixed_body_baseline.zip)")
    parser.add_argument("--randomize", action="store_true", help="sample a new random body each episode")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--max-episode-steps", type=int, default=1000)
    parser.add_argument("--slow-factor", type=float, default=4.0, help="play back this many times slower than real time")
    parser.add_argument("--pause-between", type=float, default=1.5, help="seconds to hold the final frame between episodes")
    parser.add_argument("--env-3d", action="store_true", help="use the 3D MuJoCo env (opens the MuJoCo viewer)")
    parser.add_argument("--loop", action="store_true", help="keep replaying episodes until the viewer window is closed")
    args = parser.parse_args()

    model = PPO.load(args.model)
    if args.env_3d:
        from walker.env3d import CreatureWalker3DEnv

        env = CreatureWalker3DEnv(randomize=args.randomize, render_mode="human", max_episode_steps=args.max_episode_steps)
    else:
        from walker.env import CreatureWalkerEnv

        env = CreatureWalkerEnv(randomize=args.randomize, render_mode="human", max_episode_steps=args.max_episode_steps)
        env.metadata["render_fps"] = max(1, int(60 / args.slow_factor))

    def viewer_open() -> bool:
        viewer = getattr(env, "_viewer", None)
        return viewer is None or viewer.is_running()

    ep = 0
    while args.loop or ep < args.episodes:
        obs, info = env.reset()
        terminated = truncated = False
        total_reward = 0.0
        steps = 0
        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            if not viewer_open():
                break
        print(f"episode {ep}: steps={steps} total_reward={total_reward:.2f}")
        ep += 1
        if not viewer_open():
            break
        time.sleep(args.pause_between)

    env.close()


if __name__ == "__main__":
    main()


