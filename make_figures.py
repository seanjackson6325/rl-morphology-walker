"""Writes the images for README.md and docs/paper.md into docs/img/.

    python make_figures.py

Two kinds of figure: MuJoCo renders of the bodies and the gait, and charts of the
results and the training runs.

The model file in walker/mjcf.py has no lights and no materials, because training
does not need them. A render of it is almost black. render_xml() adds a light, a
floor texture and a colour for the figures only.
"""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np
from PIL import Image

import walker.mjcf as MJ

OUT = Path("docs/img")
OUT.mkdir(parents=True, exist_ok=True)

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#8a897f"
GRID = "#e6e5e0"
S1 = "#2a78d6"        # categorical slot 1
S2 = "#eb6834"        # categorical slot 2

W, H = 640, 480


def style(ax):
    """Light axes and grid, so the data stands out."""
    ax.set_facecolor(SURFACE)
    ax.figure.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9, length=0)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


VISUAL = """
  <visual>
    <headlight ambient="0.55 0.55 0.55" diffuse="0.45 0.45 0.45" specular="0.15 0.15 0.15"/>
    <map znear="0.05"/>
  </visual>
  <asset>
    <texture name="sky" type="skybox" builtin="flat" rgb1="0.99 0.99 0.98" rgb2="0.99 0.99 0.98"
             width="64" height="64"/>
    <texture name="grid" type="2d" builtin="checker" rgb1="0.90 0.90 0.88" rgb2="0.84 0.84 0.82"
             width="300" height="300"/>
    <material name="gridmat" texture="grid" texrepeat="8 8" reflectance="0"/>
  </asset>"""


def render_xml(morph):
    """Adds a light, a floor texture and a colour. The training file has none of
    these, so a render of it is almost black."""
    xml = MJ.build_mjcf(morph)
    xml = xml.replace('<mujoco model="biped">', '<mujoco model="biped">' + VISUAL)
    xml = xml.replace('density="', 'rgba="0.27 0.40 0.57 1" density="')
    xml = xml.replace('name="floor" type="plane"',
                      'name="floor" material="gridmat" rgba="1 1 1 1" type="plane"')
    light = ('<light pos="1.5 -2.5 4" dir="-0.3 0.5 -1" directional="true"'
             ' diffuse="0.55 0.55 0.55" specular="0.1 0.1 0.1"/>')
    xml = xml.replace("<worldbody>", "<worldbody>" + light)
    return xml


def render(model, data, dist=2.7, azim=90, elev=-7, look=(0, 0, 0.74)):
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(model, cam)
    cam.distance, cam.azimuth, cam.elevation = dist, azim, elev
    cam.lookat[:] = look
    r = mujoco.Renderer(model, H, W)
    r.update_scene(data, cam)
    img = r.render().copy()
    r.close()
    return img


def pose(morph, crouch=True):
    """Builds a body and sets the start pose, with the feet on the floor."""
    import walker.env3d as E
    model = mujoco.MjModel.from_xml_string(render_xml(morph))
    data = mujoco.MjData(model)
    if crouch:
        q = data.qpos
        q[7], q[9], q[10] = E.HIP_PITCH_CROUCH, E.KNEE_CROUCH, E.ANKLE_CROUCH
        q[11], q[13], q[14] = E.HIP_PITCH_CROUCH, E.KNEE_CROUCH, E.ANKLE_CROUCH
        mujoco.mj_forward(model, data)
        gids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, n) for n in ("left_foot", "right_foot")]
        low = min(float(data.geom_xpos[g][2]) - MJ.FOOT_HEIGHT / 2 for g in gids)
        q[2] += 0.01 - low
    mujoco.mj_forward(model, data)
    return model, data


def label(img, text):
    """Adds a caption under a render."""
    fig, ax = plt.subplots(figsize=(W / 100, (H + 34) / 100), dpi=100)
    ax.imshow(img, extent=(0, W, 34, H + 34))
    ax.text(W / 2, 14, text, ha="center", va="center", color=INK2, fontsize=11)
    ax.set_xlim(0, W); ax.set_ylim(0, H + 34); ax.axis("off")
    fig.patch.set_facecolor(SURFACE)
    fig.tight_layout(pad=0.1)
    return fig


# 1. the body
def fig_body():
    m, d = pose(MJ.fixed_morphology3d())
    side = render(m, d, azim=90)
    front = render(m, d, azim=0)
    quarter = render(m, d, azim=55, elev=-14)
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.0), dpi=110)
    for ax, img, name in zip(axes, (side, front, quarter), ("side", "front", "three-quarter")):
        ax.imshow(img); ax.axis("off")
        ax.set_title(name, color=INK2, fontsize=11, pad=6)
    fig.patch.set_facecolor(SURFACE)
    fig.suptitle("The creature, in the stance it starts each episode from",
                 color=INK, fontsize=13, y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(OUT / "body.png", facecolor=SURFACE)
    plt.close(fig)
    print("body.png")


# 2. the joint diagram
def fig_joints():
    f = MJ.fixed_morphology3d()
    th, sh = f.left.thigh_length, f.left.shin_length
    hip_z = f.torso_height / 2
    fig, ax = plt.subplots(figsize=(6.6, 7.2), dpi=110)
    # torso
    ax.add_patch(plt.Rectangle((-MJ.TORSO_DEPTH / 2, hip_z - f.torso_height),
                               MJ.TORSO_DEPTH, f.torso_height,
                               facecolor="#dcdbd4", edgecolor=MUTED, linewidth=1.5))
    # one leg, drawn straight for clarity
    pts = [(0, hip_z - f.torso_height), (0, hip_z - f.torso_height - th),
           (0, hip_z - f.torso_height - th - sh)]
    ax.plot([p[0] for p in pts], [p[1] for p in pts], color=MUTED, linewidth=7,
            solid_capstyle="round", zorder=1)
    foot_y = pts[2][1]
    ax.add_patch(plt.Rectangle((-MJ.FOOT_LENGTH / 2 + MJ.FOOT_FORWARD, foot_y - MJ.FOOT_HEIGHT),
                               MJ.FOOT_LENGTH, MJ.FOOT_HEIGHT,
                               facecolor="#dcdbd4", edgecolor=MUTED, linewidth=1.5))
    ax.axhline(foot_y - MJ.FOOT_HEIGHT, color=GRID, linewidth=2)
    joints = [
        (pts[0], "hip pitch + hip roll", "2 joints: swing and sideways"),
        (pts[1], "knee", "bends backward, like a bird"),
        (pts[2], "ankle", "keeps the foot flat"),
    ]
    for (x, y), name, note in joints:
        ax.plot([x], [y], "o", markersize=12, color=S1, markeredgecolor=SURFACE,
                markeredgewidth=2, zorder=3)
        ax.annotate(f"{name}\n{note}", xy=(x, y), xytext=(x + 0.30, y),
                    color=INK, fontsize=10, va="center",
                    arrowprops=dict(arrowstyle="-", color=MUTED, linewidth=1))
    ax.text(-0.30, hip_z - f.torso_height / 2, "torso", color=INK, fontsize=10,
            ha="right", va="center")
    ax.set_xlim(-0.75, 1.15); ax.set_ylim(foot_y - 0.18, hip_z + 0.12)
    ax.set_aspect("equal"); ax.axis("off")
    fig.patch.set_facecolor(SURFACE)
    ax.set_title("One leg. Four joints per leg, eight actuators in total.",
                 color=INK, fontsize=12, pad=10)
    fig.tight_layout()
    fig.savefig(OUT / "joints.png", facecolor=SURFACE)
    plt.close(fig)
    print("joints.png")


# 3. the random body set
def fig_random_bodies():
    rng = np.random.default_rng(7)
    morphs = [MJ.fixed_morphology3d()] + [MJ.random_morphology3d(rng) for _ in range(5)]
    fig, axes = plt.subplots(1, 6, figsize=(16.5, 3.6), dpi=110)
    for i, (ax, mo) in enumerate(zip(axes, morphs)):
        m, d = pose(mo)
        ax.imshow(render(m, d, azim=72, dist=2.85, look=(0, 0, 0.74)))
        ax.axis("off")
        legL = mo.left.thigh_length + mo.left.shin_length
        legR = mo.right.thigh_length + mo.right.shin_length
        name = "the trained body" if i == 0 else f"random body {i}"
        ax.set_title(f"{name}\nlegs {legL:.2f} m / {legR:.2f} m",
                     color=INK2, fontsize=9.5, pad=5)
    fig.patch.set_facecolor(SURFACE)
    fig.suptitle("The morphology lottery: a new body at the start of every episode",
                 color=INK, fontsize=13, y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(OUT / "random_bodies.png", facecolor=SURFACE)
    plt.close(fig)
    print("random_bodies.png")


# 4. the gait filmstrip
def fig_gait(model_path="models/warmstart_morph36.zip", n_frames=6, gap=13):
    from stable_baselines3 import PPO
    import walker.env3d as E
    policy = PPO.load(model_path)
    env = E.CreatureWalker3DEnv(randomize=False, render_mode=None, max_episode_steps=1000)
    obs, _ = env.reset()
    for _ in range(180):                      # settle into the steady gait first
        a, _ = policy.predict(obs, deterministic=True)
        obs, _, t, d, _ = env.step(a)
        if t or d:
            obs, _ = env.reset()
    # The environment model has no lights. Build a lit copy of the same body and copy the
    # state into it each frame. The geometry matches, so the qpos layout matches.
    lit = mujoco.MjModel.from_xml_string(render_xml(env.morphology))
    lit_data = mujoco.MjData(lit)
    r = mujoco.Renderer(lit, H, W)
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(lit, cam)
    cam.distance, cam.azimuth, cam.elevation = 2.8, 90, -7
    frames = []
    for _ in range(n_frames):
        for _ in range(gap):
            a, _ = policy.predict(obs, deterministic=True)
            obs, _, t, d, _ = env.step(a)
        lit_data.qpos[:] = env.data.qpos
        lit_data.qvel[:] = env.data.qvel
        mujoco.mj_forward(lit, lit_data)
        cam.lookat[:] = [float(env.data.xpos[env._torso_id][0]), 0, 0.74]
        r.update_scene(lit_data, cam)
        frames.append(r.render().copy())
    r.close(); env.close()
    fig, axes = plt.subplots(1, n_frames, figsize=(2.6 * n_frames, 3.1), dpi=110)
    for i, (ax, im) in enumerate(zip(axes, frames)):
        ax.imshow(im); ax.axis("off")
        ax.set_title(f"+{i * gap * 0.025:.2f} s", color=MUTED, fontsize=9, pad=4)
    fig.patch.set_facecolor(SURFACE)
    fig.suptitle("One stride of the trained gait. Both knees bend; the step is 1.07 m long.",
                 color=INK, fontsize=12.5, y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(OUT / "gait.png", facecolor=SURFACE)
    plt.close(fig)
    print("gait.png")


# 5. the results
def fig_results():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.6, 4.3), dpi=110,
                                   gridspec_kw={"width_ratios": [1, 1.15]})
    # left: distance on 24 unseen bodies
    names = ["fixed-body\npolicy", "random-body\npolicy"]
    vals = [0.90, 18.64]
    bars = ax1.bar(names, vals, color=[S1, S2], width=0.38)
    for b, v in zip(bars, vals):
        ax1.text(b.get_x() + b.get_width() / 2, v + 0.5, f"{v:.2f} m",
                 ha="center", color=INK, fontsize=11, fontweight="bold")
    ax1.set_ylabel("mean distance (m)", color=INK2, fontsize=10)
    ax1.set_ylim(0, 22)
    ax1.set_title("24 bodies that neither policy had seen", color=INK, fontsize=11.5, pad=8)
    style(ax1)
    # right: effect of leg mismatch
    groups = ["small leg\nmismatch", "large leg\nmismatch"]
    fixed = [1.00, 0.80]
    rand = [30.25, 7.03]
    x = np.arange(2); w = 0.24
    b1 = ax2.bar(x - w / 2 - 0.012, fixed, w, label="fixed-body policy", color=S1)
    b2 = ax2.bar(x + w / 2 + 0.012, rand, w, label="random-body policy", color=S2)
    for bars_, vals_ in ((b1, fixed), (b2, rand)):
        for b, v in zip(bars_, vals_):
            ax2.text(b.get_x() + b.get_width() / 2, v + 0.7, f"{v:.2f}",
                     ha="center", color=INK, fontsize=10)
    ax2.set_xticks(x); ax2.set_xticklabels(groups)
    ax2.set_ylabel("mean distance (m)", color=INK2, fontsize=10)
    ax2.set_ylim(0, 35)
    ax2.set_title("A leg mismatch makes the body harder to walk",
                  color=INK, fontsize=11.5, pad=8)
    leg = ax2.legend(frameon=False, fontsize=9.5, loc="upper right")
    for t in leg.get_texts():
        t.set_color(INK2)
    style(ax2)
    fig.patch.set_facecolor(SURFACE)
    fig.tight_layout()
    fig.savefig(OUT / "results.png", facecolor=SURFACE)
    plt.close(fig)
    print("results.png")


# 6. the training curves
def fig_training():
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    import glob

    def smooth(y, k):
        """Moving average. np.convolve with mode same adds zeros at each end, which
        pulls the first and last points down and shows a fall that did not happen."""
        if k < 2 or len(y) < k:
            return y
        pad = k // 2
        yp = np.r_[np.full(pad, y[0]), y, np.full(pad, y[-1])]
        return np.convolve(yp, np.ones(k) / k, mode="valid")[: len(y)]

    series = [
        ("fixed body", ["runs/walk3d_bothknee_v5_*"], S1),
        ("random bodies", ["runs/walk3d_morph_v1_*", "runs/walk3d_morph_v2_*"], S2),
    ]
    fig, ax = plt.subplots(figsize=(9.0, 4.6), dpi=110)
    for name, pats, colour in series:
        ev = []
        for pat in pats:                      # a run resumed from another continues its x axis
            dirs = sorted(glob.glob(pat))
            if dirs:
                ea = EventAccumulator(dirs[-1]); ea.Reload()
                ev += ea.Scalars("rollout/ep_len_mean")
        if not ev:
            continue
        ev.sort(key=lambda e: e.step)
        xs = np.array([e.step for e in ev]) / 1e6
        ys = smooth(np.array([e.value for e in ev]), max(1, len(ev) // 60))
        ax.plot(xs, ys, color=colour, linewidth=2, label=name, solid_capstyle="round")
        ax.annotate(name, xy=(xs[-1], ys[-1]), xytext=(7, 0), textcoords="offset points",
                    color=colour, fontsize=10.5, va="center", fontweight="bold")

    ax.axhline(1000, color=MUTED, linewidth=1, linestyle=(0, (4, 4)))
    ax.text(0.4, 1015, "full episode (1000 steps)", color=MUTED, fontsize=9, va="bottom")
    ax.annotate("PPO lost the policy here.\nThe 33M checkpoint was kept.",
                xy=(34.8, 120), xytext=(21.0, 45), color=INK2, fontsize=9,
                arrowprops=dict(arrowstyle="->", color=MUTED, linewidth=1))
    ax.set_xlabel("training steps (millions)", color=INK2, fontsize=10)
    ax.set_ylabel("mean episode length (steps)", color=INK2, fontsize=10)
    ax.set_title("How long the creature stays up during training", color=INK, fontsize=12.5, pad=10)
    style(ax)
    ax.set_xlim(0, 46); ax.set_ylim(0, 1120)
    fig.patch.set_facecolor(SURFACE)
    fig.text(0.012, 0.012,
             "The fixed-body curve continues an earlier run, so it starts at 16M steps.",
             color=MUTED, fontsize=8.5)
    fig.tight_layout(rect=(0, 0.045, 1, 1))
    fig.savefig(OUT / "training.png", facecolor=SURFACE)
    plt.close(fig)
    print("training.png")


# 7. the AT-ST body (historical)
# This body is no longer in walker/mjcf.py. The plain tall box replaced it in
# June 2026. It is built here again for the paper.
ATST_THICK, ATST_LEN, ATST_WIDE = 0.26, 0.38, 0.34
ATST_HIP, ATST_RADIUS, ATST_SEG = 0.26, 0.07, 0.40


def atst_xml():
    hz = -ATST_THICK / 2
    body = f'{ATST_LEN / 2} {ATST_WIDE / 2} {ATST_THICK / 2}'

    def leg(side, sign):
        return f"""
      <body name="{side}_thigh" pos="0 {sign * ATST_HIP / 2} {hz}">
        <joint name="{side}_hip" type="hinge" axis="0 1 0" range="-1.4 1.4"/>
        <geom type="capsule" fromto="0 0 0 0 0 {-ATST_SEG}" size="{ATST_RADIUS}" density="900"/>
        <body name="{side}_shin" pos="0 0 {-ATST_SEG}">
          <joint name="{side}_knee" type="hinge" axis="0 1 0" range="-2.4 0"/>
          <geom type="capsule" fromto="0 0 0 0 0 {-ATST_SEG}" size="{ATST_RADIUS}" density="900"/>
          <body name="{side}_foot" pos="0 0 {-ATST_SEG}">
            <joint name="{side}_ankle" type="hinge" axis="0 1 0" range="-0.8 0.8"/>
            <geom type="box" pos="0.05 0 -0.025" size="0.11 0.07 0.025" density="900"/>
          </body>
        </body>
      </body>"""

    return f"""<mujoco model="atst">
  <compiler angle="radian"/>
  <option timestep="0.005" gravity="0 0 -9.81"/>
  {VISUAL}
  <default>
    <geom rgba="0.27 0.40 0.57 1" contype="2" conaffinity="1" friction="1 .1 .1"/>
  </default>
  <worldbody>
    <light pos="1.5 -2.5 4" dir="-0.3 0.5 -1" directional="true" diffuse="0.55 0.55 0.55"/>
    <geom name="floor" material="gridmat" rgba="1 1 1 1" type="plane" size="50 50 .1"
          contype="1" conaffinity="1"/>
    <body name="torso" pos="0 0 0.95">
      <freejoint/>
      <geom type="box" size="{body}" density="400"/>
      <geom type="box" pos="0 {ATST_WIDE / 2 + 0.03} 0.04" size="0.055 0.035 0.055" density="200"/>
      <geom type="box" pos="0 {-(ATST_WIDE / 2 + 0.03)} 0.04" size="0.055 0.035 0.055" density="200"/>
      <geom type="box" pos="{ATST_LEN / 2 + 0.01} 0 0.035" size="0.02 0.115 0.04" density="200"/>
      <geom type="box" pos="{ATST_LEN / 2 - 0.04} 0 -0.105" size="0.065 0.085 0.032" density="200"/>
{leg('left', 1)}
{leg('right', -1)}
    </body>
  </worldbody>
</mujoco>"""


def fig_atst():
    m = mujoco.MjModel.from_xml_string(atst_xml())
    d = mujoco.MjData(m)
    d.qpos[7], d.qpos[8], d.qpos[9] = 0.55, -1.15, 0.55      # left hip, knee, ankle
    d.qpos[10], d.qpos[11], d.qpos[12] = 0.55, -1.15, 0.55   # right
    mujoco.mj_forward(m, d)
    low = min(float(d.geom_xpos[g][2]) for g in range(m.ngeom) if m.geom_bodyid[g] != 0)
    d.qpos[2] += 0.12 - low
    mujoco.mj_forward(m, d)
    views = [(72, -8), (90, -8), (20, -6)]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.0), dpi=110)
    for ax, (az, el) in zip(axes, views):
        ax.imshow(render(m, d, dist=1.95, azim=az, elev=el, look=(0, 0, 0.58)))
        ax.axis("off")
    fig.patch.set_facecolor(SURFACE)
    fig.suptitle("The AT-ST body. A wide cockpit, sensor pods, a visor and backward knees.",
                 color=INK, fontsize=13, y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(OUT / "atst.png", facecolor=SURFACE)
    plt.close(fig)
    print("atst.png")


if __name__ == "__main__":
    fig_body()
    fig_joints()
    fig_random_bodies()
    fig_results()
    fig_training()
    fig_gait()
    fig_atst()
