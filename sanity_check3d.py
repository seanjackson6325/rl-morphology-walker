"""Visual/headless sanity check for the 3D MJCF biped.

Build the fixed-morphology body, drop it under gravity with all actuators idle,
and confirm it compiles, stands (or settles) without exploding or sinking
through the floor. Pass --view to open the interactive MuJoCo viewer.

This is the 3D analogue of sanity_check.py -- physics/geometry validation
BEFORE any RL, so we don't burn training time on a broken model.
"""

from __future__ import annotations

import argparse

import mujoco
import numpy as np

from walker.mjcf import build_mjcf, fixed_morphology3d


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--view", action="store_true", help="open the interactive MuJoCo viewer")
    parser.add_argument("--seconds", type=float, default=5.0)
    args = parser.parse_args()

    morph = fixed_morphology3d()
    xml = build_mjcf(morph)
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)

    print(f"spawn standing_height = {morph.standing_height():.3f} m")
    print(f"nq={model.nq} nv={model.nv} nu={model.nu} (expect nu=6 actuators)")

    torso_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "torso")

    if args.view:
        import mujoco.viewer as mj_viewer

        with mj_viewer.launch_passive(model, data) as viewer:
            import time

            start = time.time()
            while viewer.is_running() and time.time() - start < args.seconds:
                mujoco.mj_step(model, data)
                viewer.sync()
                time.sleep(model.opt.timestep)
        return

    steps = int(args.seconds / model.opt.timestep)
    for i in range(steps):
        mujoco.mj_step(model, data)
        if i % int(0.5 / model.opt.timestep) == 0:
            z = data.xpos[torso_id][2]
            zaxis = data.xmat[torso_id].reshape(3, 3)[:, 2]
            upright = float(zaxis[2])  # 1.0 = perfectly upright, 0 = on its side
            t = i * model.opt.timestep
            print(f"t={t:4.2f}s  torso_z={z:6.3f}  upright={upright:+.3f}")
            if not np.isfinite(z) or abs(z) > 100:
                print("ERROR: simulation diverged")
                raise SystemExit(1)

    print("Sanity check finished without exploding.")


if __name__ == "__main__":
    main()
