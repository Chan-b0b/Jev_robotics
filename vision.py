"""Image prompts: the task in text, the state only as camera frames (no coordinates).

Cameras, by default corner (a fixed view of the table and the robot) at 448 px and
gripperPOV (looks down from the gripper, the fingers in view) at 320 px;
behindGripper (behind the gripper, looking forward) can be added at 320 px. All are
multiples of Qwen's 32 px merged patch (448: 196 image tokens, 320: 100). Frames
come from mujoco.Renderer on the env's own model and data. Only corner renders
upside down (as in Meta-World's offscreen render) and is rotated 180 degrees; the
gripper cameras come out upright: behindGripper has the arm above and the table
below, and gripperPOV has +y (away from the robot) up and +x to the right.

The task text names the objects by how they look, and the procedure of tasks.py in
words, without its numbers. The images go first and the question last, so the
questions of a step share a prefix.

apply() changes the scene's look for a robustness check at evaluation (the model
was trained on the default look): the goal's color (with the text's color word
changed to match, or dropped), the table's color, or the floor and sky; or
goal_far, which places the goal (for drawer-open, the drawer) at x in +-[0.2, 0.3]
instead of the [-0.1, 0.1] the tasks and so the training data sample from.
"""
import base64
import io

import mujoco
import numpy as np
from PIL import Image

RES = {"corner": 448, "gripperPOV": 320, "behindGripper": 320}
UPSIDE_DOWN = {"corner"}
DEFAULT = ("corner", "gripperPOV")
CAMERA_TEXT = {"corner": "a fixed camera at the corner of the table.",
               "gripperPOV": "a camera on the gripper, looking down between the fingers.",
               "behindGripper": "a camera behind the gripper, looking forward past it."}
TEXT = {
    "reach-v3": ("Task: Move the gripper to the goal. The goal is the small red sphere; it may be in the air. "
                 "The dark red cylinder on the table is not the goal.\n"
                 "Success: the gripper is within 0.05 m of the goal."),
    "push-v3": ("Task: Push the puck (the dark red cylinder) along the table to the goal (the small green sphere).\n"
                "Procedure: move above the puck, lower the gripper down around it and close the gripper, "
                "then move along the table to the goal, keeping the gripper pressed down.\n"
                "Success: the puck is within 0.05 m of the goal."),
    "pick-place-v3": ("Task: Pick up the puck (the dark red cylinder) and place it at the goal "
                      "(the small blue sphere; it may be in the air).\n"
                      "Procedure: move above the puck, lower the gripper around it and close the gripper, "
                      "wait until the gripper has closed, then carry the puck to the goal.\n"
                      "Success: the puck is within 0.07 m of the goal."),
    "drawer-open-v3": ("Task: Open the drawer (the green box) by pulling its white handle toward the robot (-y).\n"
                       "Procedure: move high above the handle, lower the gripper to the handle, "
                       "then pull toward -y. Keep the gripper open.\n"
                       "Success: the drawer is pulled open."),
}


CONDITIONS = ("goal_color", "goal_color_nocolor", "table_color", "background", "goal_far")
FAR_X = (0.2, 0.3)
# goal color per task: (the color word in TEXT, the new color word, the new rgba)
NEW_GOAL = {"reach-v3": ("red", "yellow", (0.9, 0.8, 0.0, 1)), "push-v3": ("green", "purple", (0.6, 0.0, 0.8, 1)),
            "pick-place-v3": ("blue", "orange", (1.0, 0.5, 0.0, 1)), "drawer-open-v3": (None, "purple", (0.6, 0.0, 0.8, 1))}
TABLE_RGBA = (0.5, 0.5, 0.5, 1)  # solid gray instead of the wood texture
FLOOR_RGBA, SKY_RGB = (0.15, 0.25, 0.7, 1), (40, 60, 140)  # dark blue


def apply(env, task_name, condition, obs, seed):
    """Change env's scene for a robustness condition, after reset and before Cameras (which
    uploads the textures). Returns the task text to show and the observation."""
    m, text = env.unwrapped.model, TEXT[task_name]
    if condition == "goal_far":
        return text, place_far(env, task_name, seed)
    mat = lambda name: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_MATERIAL, name)
    if condition in ("goal_color", "goal_color_nocolor"):
        old, new, rgba = NEW_GOAL[task_name]
        m.site_rgba[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "goal")] = rgba
        if old:
            text = text.replace(f"small {old} sphere", f"small {new} sphere" if condition == "goal_color" else "small sphere")
            assert text != TEXT[task_name]
    elif condition == "table_color":
        m.mat_texid[mat("table_wood")] = -1
        m.mat_rgba[mat("table_wood")] = TABLE_RGBA
    elif condition == "background":
        m.mat_rgba[mat("basic_floor")] = FLOOR_RGBA
        for t in range(m.ntex):
            if m.tex_type[t] == mujoco.mjtTexture.mjTEXTURE_SKYBOX:
                n = m.tex_width[t] * m.tex_height[t] * m.tex_nchannel[t]
                m.tex_data[m.tex_adr[t]:m.tex_adr[t] + n].reshape(-1, m.tex_nchannel[t])[:, :3] = SKY_RGB
    else:
        raise KeyError(condition)
    return text, obs


def place_far(env, task_name, seed):
    """Reset again with the goal's x (drawer-open: the drawer's) moved to +-FAR_X. The
    task's random vector is (object xyz, goal xyz), or the drawer's xyz; the side is
    random for reach and drawer-open, and away from the puck for push and pick-place,
    so the puck stays the 0.15 m from the goal in xy that their reset asks for."""
    u, rng = env.unwrapped, np.random.default_rng(seed)
    vec = u._last_rand_vec.copy()
    x = rng.uniform(*FAR_X)
    if task_name == "drawer-open-v3":
        vec[0] = x * rng.choice([-1, 1])
    else:
        side = -np.sign(vec[0]) if task_name in ("push-v3", "pick-place-v3") and vec[0] != 0 else rng.choice([-1, 1])
        vec[3] = x * side
    u._last_rand_vec = vec  # frozen by set_task, so the reset below reads it
    obs, _ = u.reset()
    return obs


class Cameras:
    def __init__(self, env, cameras=DEFAULT, upside_down=UPSIDE_DOWN):
        model = env.unwrapped.model
        side = max(RES[c] for c in cameras)
        model.vis.global_.offwidth = max(model.vis.global_.offwidth, side)
        model.vis.global_.offheight = max(model.vis.global_.offheight, side)
        self.data, self.upside_down = env.unwrapped.data, set(upside_down)
        self.renderers = [(c, mujoco.Renderer(model, RES[c], RES[c])) for c in cameras]

    def frames(self):
        out = []
        for cam, r in self.renderers:
            r.update_scene(self.data, camera=cam)
            frame = r.render()
            out.append(np.ascontiguousarray(np.rot90(frame, 2) if cam in self.upside_down else frame))
        return out

    def close(self):
        for _, r in self.renderers:
            r.close()


def user_content(task_name, question, cameras=DEFAULT, text=None):
    """Chat content with an {"type": "image"} placeholder per camera, in the order given.
    text replaces the task's TEXT (apply() returns it)."""
    parts = []
    for i, cam in enumerate(cameras):
        parts += [{"type": "text", "text": f"Image {i + 1}: {CAMERA_TEXT[cam]}"}, {"type": "image"}]
    parts.append({"type": "text", "text": "\n".join([text or TEXT[task_name], "", question, "Reply with only the letter."])})
    return parts


def encode(frame):
    """A frame as the base64 PNG data URL vLLM's chat API takes."""
    buf = io.BytesIO()
    Image.fromarray(frame).save(buf, format="PNG", compress_level=1)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def for_vllm(content, frames):
    """Fill the placeholders with image_url parts. frames are arrays, or data URLs from
    encode(), so that a step's questions share one encoding of its frames."""
    frames, out = iter(frames), []
    for p in content:
        if p["type"] == "image":
            f = next(frames)
            p = {"type": "image_url", "image_url": {"url": f if isinstance(f, str) else encode(f)}}
        out.append(p)
    return out
