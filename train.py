"""Train PPO on the creature walker.

Two modes, controlled by --randomize:
  off (default for the first run): every episode uses the same fixed body.
      This is the sanity-check training run -- if PPO can't learn to move a
      single fixed body, randomizing morphology won't help.
  on: every episode samples a new random body (walker.morphology.random_morphology),
      and the policy must condition its behavior on the morphology features in
      the observation. This is the actual "domain randomization" experiment.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

MODELS_DIR = Path("models")
LOGS_DIR = Path("runs")


def make_env(randomize: bool, max_episode_steps: int, env_3d: bool):
    def _init():
        if env_3d:
            from walker.env3d import CreatureWalker3DEnv

            return CreatureWalker3DEnv(randomize=randomize, render_mode=None, max_episode_steps=max_episode_steps)
        from walker.env import CreatureWalkerEnv

        return CreatureWalkerEnv(randomize=randomize, render_mode=None, max_episode_steps=max_episode_steps)

    return _init


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True, help="run name; used for model file + tensorboard log")
    parser.add_argument("--timesteps", type=int, default=300_000)
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--randomize", action="store_true", help="enable morphology domain randomization")
    parser.add_argument("--max-episode-steps", type=int, default=1000)
    parser.add_argument("--resume-from", type=str, default=None, help="path to a .zip checkpoint to continue training")
    parser.add_argument("--env-3d", action="store_true", help="use the 3D MuJoCo env instead of the 2D pymunk env")
    args = parser.parse_args()

    vec_env_cls = SubprocVecEnv if args.n_envs > 1 else None
    vec_env = make_vec_env(
        make_env(args.randomize, args.max_episode_steps, args.env_3d),
        n_envs=args.n_envs,
        vec_env_cls=vec_env_cls,
    )

    MODELS_DIR.mkdir(exist_ok=True)
    LOGS_DIR.mkdir(exist_ok=True)

    if args.resume_from:
        model = PPO.load(args.resume_from, env=vec_env, tensorboard_log=str(LOGS_DIR))
        # PPO.load restores the SAVED verbose flag, so a checkpoint built by a script that
        # used verbose=0 resumes silently and the run looks dead (tensorboard still logs).
        model.verbose = 1
    else:
        model = PPO(
            "MlpPolicy",
            vec_env,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=256,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.0,
            verbose=1,
            tensorboard_log=str(LOGS_DIR),
        )

    # Periodic checkpoints so a late-training collapse can't cost us a good policy
    # (learned the hard way when v12's final checkpoint caught a transient crash).
    ckpt_cb = CheckpointCallback(
        save_freq=max(1, 1_000_000 // args.n_envs),
        save_path=str(MODELS_DIR / f"{args.name}_ckpts"),
        name_prefix=args.name,
    )
    model.learn(
        total_timesteps=args.timesteps,
        tb_log_name=args.name,
        reset_num_timesteps=args.resume_from is None,
        callback=ckpt_cb,
    )

    out_path = MODELS_DIR / args.name
    model.save(out_path)
    print(f"Saved model to {out_path}.zip")


if __name__ == "__main__":
    main()
