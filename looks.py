"""Random looks for a Meta-World scene: object colors, table and background textures, lighting.

randomize(env, rng, split) edits the env's model after reset and before vision.Cameras
(which uploads the textures). The two splits share nothing, so that "test" looks are
never trained on:
- colors: hues from alternate 30-degree bins of the color wheel (train the even bins,
  test the odd ones), for the task objects' materials and geoms, the visible sites (goal
  markers), the sky, the floor and the table's walls;
- the table: a texture made up here, wood grain tinted, solid or striped for train,
  checkered or mottled for test;
- lighting: the lights and headlight scaled by 0.75-1.25 and nearly white for train,
  by 0.45-0.6 or 1.4-1.6 and tinted for test.
The robot keeps its look. The scene's roots other than the floor (world), the table,
its walls, the robot (base) and the mocap are the task's objects.
"""
import colorsys

import mujoco
import numpy as np

SCENE_ROOTS = {"world", "tablelink", "RetainingWall", "base", "mocap"}
SPLITS = ("train", "test")


def _color(rng, split, s=(0.45, 0.9), v=(0.35, 0.9)):
    """An RGB color in [0, 1] whose hue is in the split's bins."""
    b = 2 * rng.integers(6) + (split == "test")
    h = (b * 30 + rng.uniform(0, 30)) / 360
    return np.array(colorsys.hsv_to_rgb(h, rng.uniform(*s), rng.uniform(*v)))


def _table(rng, split, h, w, base):
    """A table texture (h, w, 3) uint8."""
    c1, c2 = _color(rng, split), _color(rng, split)
    y, x = np.mgrid[0:h, 0:w]
    kind = rng.choice(["wood", "solid", "stripes"] if split == "train" else ["checker", "mottled"])
    if kind == "wood":
        img = base / 255.0 * (0.5 + c1)[None, None]
    elif kind == "solid":
        img = np.broadcast_to(c1, (h, w, 3)).copy()
    elif kind == "stripes":
        period = rng.integers(24, 96)
        img = np.where(((x + y * rng.uniform(-0.3, 0.3)) // period % 2)[..., None] == 0, c1, c2)
    elif kind == "checker":
        period = rng.integers(24, 96)
        img = np.where(((x // period + y // period) % 2)[..., None] == 0, c1, c2)
    else:  # mottled: smooth noise between two colors
        coarse = rng.random((h // 32 + 2, w // 32 + 2))
        t = np.kron(coarse, np.ones((32, 32)))[:h, :w]
        img = t[..., None] * c1 + (1 - t[..., None]) * c2
    return (np.clip(img, 0, 1) * 255).astype(np.uint8)


def randomize(env, rng, split):
    m = env.unwrapped.model
    assert split in SPLITS
    name = lambda t, i: mujoco.mj_id2name(m, t, i)

    def root(body):
        while m.body_parentid[body] != 0:
            body = m.body_parentid[body]
        return name(mujoco.mjtObj.mjOBJ_BODY, body)

    # object colors: one color per material of the task objects, and per plain geom
    # (a geom's own rgba, when not MuJoCo's default gray, wins over its material's, so both are set)
    mat_color = {}
    for g in range(m.ngeom):
        if root(m.geom_bodyid[g]) in SCENE_ROOTS or m.geom_rgba[g][3] == 0:
            continue
        i = int(m.geom_matid[g])
        if i >= 0:
            if i not in mat_color:
                mat_color[i] = m.mat_rgba[i][:3] = _color(rng, split)
            m.geom_rgba[g][:3] = mat_color[i]
        else:
            m.geom_rgba[g][:3] = _color(rng, split)
    for s in range(m.nsite):  # goal markers and other visible sites off the robot
        if m.site_rgba[s][3] > 0 and root(m.site_bodyid[s]) != "base":
            m.site_rgba[s][:3] = _color(rng, split)

    # table texture, and the sky, floor and walls
    for t in range(m.ntex):
        tname = name(mujoco.mjtObj.mjOBJ_TEXTURE, t)
        h, w, c = m.tex_height[t], m.tex_width[t], m.tex_nchannel[t]
        data = m.tex_data[m.tex_adr[t]:m.tex_adr[t] + h * w * c].reshape(h, w, c)
        if tname == "T_table":
            data[..., :3] = _table(rng, split, h, w, data[..., :3].astype(float))
        elif m.tex_type[t] == mujoco.mjtTexture.mjTEXTURE_SKYBOX:
            data[..., :3] = (_color(rng, split, s=(0.2, 0.7), v=(0.3, 0.8)) * 255).astype(np.uint8)
    for mat in ("basic_floor", "wall_metal"):
        i = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_MATERIAL, mat)
        if i >= 0:
            m.mat_rgba[i][:3] = 0.4 + 0.6 * _color(rng, split)

    # lighting
    if split == "train":
        scale, tint = rng.uniform(0.75, 1.25), 1 + rng.uniform(-0.05, 0.05, 3)
    else:
        scale = rng.choice([rng.uniform(0.45, 0.6), rng.uniform(1.4, 1.6)])
        tint = 1 + rng.uniform(-0.25, 0.25, 3)
    m.light_diffuse[:] = np.clip(m.light_diffuse * scale * tint, 0, 1)
    m.vis.headlight.diffuse[:] = np.clip(m.vis.headlight.diffuse * scale * tint, 0, 1)
    m.vis.headlight.ambient[:] = np.clip(m.vis.headlight.ambient * scale * tint, 0, 1)
