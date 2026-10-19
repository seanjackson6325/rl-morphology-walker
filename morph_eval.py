"""Does conditioning on morphology actually help? The morphology-lottery test.

    python morph_eval.py --bodies 30

Samples random bodies and runs each one under two policies:

  SPECIALIST  - trained on a single fixed body (warmstart_morph36.zip). It receives the
                morphology inputs but never saw them vary, so it is the control: one gait,
                applied to every body.
  LOTTERY     - trained with --randomize, a new body every episode. If the premise holds it
                should degrade more gracefully as bodies move away from the trained one.

Both get the SAME bodies (same seed), so the comparison is paired. Reported against two
things that should matter: total leg length, and leg-length MISMATCH between the two legs,
which is the case the project set out to handle.
"""

from __future__ import annotations

import argparse

import numpy as np
from stable_baselines3 import PPO

import walker.mjcf as MJ
from walker.env3d import CreatureWalker3DEnv


def run_body(model, morph, max_steps=1000):
    """Evaluate one policy on one specific body by pinning fixed_morphology3d to it."""
    import walker.env3d as E
    orig = E.fixed_morphology3d
    E.fixed_morphology3d = lambda: morph
    try:
        env = E.CreatureWalker3DEnv(randomize=False, render_mode=None, max_episode_steps=max_steps)
        obs, _ = env.reset()
        x0 = float(env.data.xpos[env._torso_id][0])
        terminated = truncated = False
        n = 0
        while not (terminated or truncated):
            a, _ = model.predict(obs, deterministic=True)
            obs, r, terminated, truncated, _ = env.step(a)
            n += 1
        dist = float(env.data.xpos[env._torso_id][0]) - x0
        env.close()
        return n, dist, bool(terminated)
    finally:
        E.fixed_morphology3d = orig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bodies", type=int, default=30)
    ap.add_argument("--specialist", default="models/warmstart_morph36.zip")
    ap.add_argument("--lottery", default="models/walk3d_morph_v1_ckpts/walk3d_morph_v1_21000000_steps.zip")
    a = ap.parse_args()

    rng = np.random.default_rng(12345)          # fixed seed: both policies see identical bodies
    bodies = [MJ.random_morphology3d(rng) for _ in range(a.bodies)]
    models = {"specialist": PPO.load(a.specialist), "lottery": PPO.load(a.lottery)}

    rows = []
    for i, m in enumerate(bodies):
        legL = m.left.thigh_length + m.left.shin_length
        legR = m.right.thigh_length + m.right.shin_length
        rec = dict(i=i, leg=(legL + legR) / 2, mismatch=abs(legL - legR))
        for name, mdl in models.items():
            steps, dist, fell = run_body(mdl, m)
            rec[name] = dict(steps=steps, dist=dist, fell=fell)
        rows.append(rec)

    print(f"\n{a.bodies} random bodies, identical for both policies (seed 12345)\n")
    print(f"{'body':>5}{'legLen':>8}{'mismatch':>10}   {'SPECIALIST':>20}   {'LOTTERY':>20}")
    print(f"{'':>23}   {'steps':>8}{'dist_m':>12}   {'steps':>8}{'dist_m':>12}")
    for r in rows:
        s, l = r["specialist"], r["lottery"]
        print(f"{r['i']:>5}{r['leg']:>8.2f}{r['mismatch']:>10.3f}   "
              f"{s['steps']:>8}{s['dist']:>12.2f}   {l['steps']:>8}{l['dist']:>12.2f}")

    def agg(key, sub):
        return np.mean([r[key][sub] for r in rows])
    print()
    for name in models:
        falls = sum(r[name]["fell"] for r in rows)
        print(f"  {name:<11} fall rate {falls}/{a.bodies}  |  mean {agg(name,'steps'):.0f} steps  "
              f"|  mean {agg(name,'dist'):.2f} m  |  full episodes {sum(1 for r in rows if not r[name]['fell'])}")

    # does the gap widen on mismatched legs? split at the median
    med = float(np.median([r["mismatch"] for r in rows]))
    print(f"\n  split at median leg mismatch ({med:.3f} m):")
    for label, sel in (("well-matched legs", lambda r: r["mismatch"] <= med),
                       ("MISMATCHED legs  ", lambda r: r["mismatch"] > med)):
        sub = [r for r in rows if sel(r)]
        sp = np.mean([r["specialist"]["dist"] for r in sub])
        lo = np.mean([r["lottery"]["dist"] for r in sub])
        print(f"    {label}: specialist {sp:6.2f} m   lottery {lo:6.2f} m   "
              f"lottery advantage {lo - sp:+6.2f} m")


if __name__ == "__main__":
    main()
