"""Per-task prompts, questions and labels, one task at a time.

Each task gives its own goal text, the state it shows, and the questions it asks.
The gripper position is the fingertip center (env.tcp_center), not the hand
position Meta-World puts in the observation (about 0.045 m higher), because the
task success checks measure from the fingertips and a camera sees them.

Labels are decrease / hold / increase of (target - fingertips) per axis, hold
within DEADBAND, as in stage 2, but measured from the fingertips.
"""
import numpy as np

DEADBAND = 0.005  # m
MAG = 0.5  # action per non-hold label: 5 mm

SYSTEM = (
    "You control a robot gripper above a table. Positions are (x, y, z) in meters. "
    "The robot's base is at the origin: +y points away from the robot, +x to the robot's right, "
    "and z is height; the table surface is at z = 0. The gripper position is the point between "
    "the fingertips. Each step, each axis moves 0.005 m toward decrease or increase, or holds.")
# for image prompts of the mt50 split: the axes only, no numbers (the positions are not shown, and a
# soft task moves by how strongly an answer is chosen)
SYSTEM_IMAGE = (
    "You control a robot gripper above a table. +x is to the robot's right, +y points away from the robot, "
    "and +z is up. Each step, choose for each axis whether to decrease, hold or increase, and how to set the fingers.")
LETTERS = ["A", "B", "C"]
AXES = {"dx": "x", "dy": "y", "dz": "z"}


def state_of(env, obs):
    """What a prompt may show, read from the simulator. The goal is the env's own: the
    observation clips it to the task's goal range, which a moved goal (vision.place_far)
    leaves; inside the range the two are equal."""
    return {"tip": np.array(env.unwrapped.tcp_center), "opening": float(obs[3]),
            "obj": np.array(obs[4:7]), "goal": np.array(env.unwrapped._target_pos), "obs": np.array(obs)}


def fmt(v):
    return "(" + ", ".join(f"{float(x):.4f}" for x in v) + ")"


class Reach:
    name = "reach-v3"
    keys = ("dx", "dy", "dz")  # the gripper stays open; no grip question
    text = ("Task: Move the gripper to the goal. The goal may be in the air.\n"
            "Success: the gripper is within 0.05 m of the goal.")

    def state_lines(self, s):
        return [f"gripper position: {fmt(s['tip'])}", f"goal position: {fmt(s['goal'])}"]

    def target(self, s):
        return s["goal"], False  # (target for the fingertips, close the gripper)


HAND = 0.045  # m: the experts' hand point is this far above the fingertips
LIFT = {"push-v3": 0.2, "pick-place-v3": 0.1}  # expert's approach height above the puck


def xy(a, b):
    return float(np.linalg.norm(a[:2] - b[:2]))


class Push:
    """The expert's rules (metaworld.policies), moved from the hand to the fingertips."""
    name = "push-v3"
    keys = ("dx", "dy", "dz", "grip")
    text = ("Task: Push the puck along the table to the goal.\n"
            "Success: the puck is within 0.05 m of the goal.\n"
            "Procedure: Let P be the puck position shifted by -0.005 in x and lowered by 0.045 in z.\n"
            "1. If the gripper's xy distance to P is more than 0.02, the target is P raised by 0.20 in z.\n"
            "2. Otherwise, if the gripper's z differs from P's z by more than 0.04, the target is P raised by 0.03 in z.\n"
            "3. Otherwise, the target is the goal lowered by 0.045 in z (the gripper presses on the table).\n"
            "Close the gripper when its xy distance to the puck is at most 0.02 and its z is at most 0.055 above "
            "the puck's z; otherwise keep it open.")

    def state_lines(self, s):
        return [f"gripper position: {fmt(s['tip'])}", f"puck position: {fmt(s['obj'])}",
                f"goal position: {fmt(s['goal'])}"]

    def target(self, s):
        tip, obj = s["tip"], s["obj"]
        p = obj + [-0.005, 0, -HAND]
        close = xy(tip, obj) <= 0.02 and tip[2] - obj[2] <= 0.055
        if xy(tip, p) > 0.02:
            return p + [0, 0, 0.2], close
        if abs(tip[2] - p[2]) > 0.04:
            return p + [0, 0, 0.03], close
        return s["goal"] + [0, 0, -HAND], close


class PickPlace:
    name = "pick-place-v3"
    keys = ("dx", "dy", "dz", "grip")
    text = ("Task: Pick up the puck and place it at the goal. The goal may be in the air.\n"
            "Success: the puck is within 0.07 m of the goal.\n"
            "Procedure: Let P be the puck position shifted by -0.005 in x and lowered by 0.045 in z.\n"
            "1. If the gripper's xy distance to P is more than 0.02, the target is P raised by 0.10 in z.\n"
            "2. Otherwise, if the gripper's z differs from P's z by more than 0.05 and the puck's z is below 0.04, "
            "the target is P raised by 0.03 in z.\n"
            "3. Otherwise, if the gripper opening is more than 0.73, the target is the gripper's current "
            "position (wait for the gripper to close).\n"
            "4. Otherwise, the target is the goal.\n"
            "Close the gripper when the point 0.045 above the gripper is less than 0.07 from the puck; "
            "otherwise keep it open.")

    def state_lines(self, s):
        return [f"gripper position: {fmt(s['tip'])}", f"gripper opening: {s['opening']:.3f} (1 is fully open)",
                f"puck position: {fmt(s['obj'])}", f"goal position: {fmt(s['goal'])}"]

    def target(self, s):
        tip, obj = s["tip"], s["obj"]
        p = obj + [-0.005, 0, -HAND]
        close = float(np.linalg.norm(tip + [0, 0, HAND] - obj)) < 0.07
        if xy(tip, p) > 0.02:
            return p + [0, 0, 0.1], close
        if abs(tip[2] - p[2]) > 0.05 and obj[2] < 0.04:
            return p + [0, 0, 0.03], close
        if s["opening"] > 0.73:
            return tip, close
        return s["goal"], close


class DrawerOpen:
    name = "drawer-open-v3"
    keys = ("dx", "dy", "dz")  # the gripper stays open
    text = ("Task: Open the drawer by pulling its handle toward -y.\n"
            "Success: the drawer is pulled open.\n"
            "Procedure: Let H be the handle position lowered by 0.065 in z.\n"
            "1. If the gripper's xy distance to H is more than 0.06, the target is H raised by 0.30 in z.\n"
            "2. Otherwise, if the gripper's z differs from H's z by more than 0.04, the target is H.\n"
            "3. Otherwise, the target is H moved by -0.06 in y.")

    def state_lines(self, s):
        return [f"gripper position: {fmt(s['tip'])}", f"handle position: {fmt(s['obj'])}"]

    def target(self, s):
        tip, h = s["tip"], s["obj"] + [0, 0, -0.02 - HAND]
        if xy(tip, h) > 0.06:
            return h + [0, 0, 0.3], False
        if abs(tip[2] - h[2]) > 0.04:
            return h, False
        return h + [0, -0.06, 0], False


TASKS = {t.name: t for t in (Reach(), Push(), PickPlace(), DrawerOpen())}


GRIP_LEVELS = (-1.0, 0.65, 1.0)  # soft tasks' grip options: open, grasp, close tightly
SOFT_MAG = 1.0  # soft tasks: an axis moves (P(increase) - P(decrease)) * SOFT_MAG, up to 10 mm, as the experts do


class Expert:
    """A Meta-World task labelled by its scripted expert (metaworld.policies), softly: each
    axis's label is the expert's action v in [-1, 1] split over decrease / hold / increase
    (|v| on the side of its sign, 1 - |v| on hold), so that P(increase) - P(decrease) = v;
    the grip label is the expert's grab effort rounded to GRIP_LEVELS. For image prompts only."""
    keys = ("dx", "dy", "dz", "grip")
    soft = True

    def __init__(self, name):
        self.name, self._policy = name, None

    @property
    def policy(self):  # made on first use, so tasks.py imports without metaworld (lora_train.py's venv)
        if self._policy is None:
            from metaworld.policies import ENV_POLICY_MAP
            self._policy = ENV_POLICY_MAP[self.name]()
        return self._policy

    def soft_labels(self, s):
        a = self.policy.get_action(s["obs"].copy())  # some experts (door-open, door-close) edit the obs in place
        out = {}
        for k, v in zip(("dx", "dy", "dz"), np.clip(a[:3], -1, 1)):
            out[k] = np.array([max(-v, 0), 1 - abs(v), max(v, 0)])
        g = 0 if a[3] <= 0 else 2 if a[3] >= 0.9 else 1
        out["grip"] = np.eye(3)[g]
        return out

    def target(self, s):
        """The expert's subgoal for the fingertips where it has one (for traces), else the goal."""
        if hasattr(self.policy, "_desired_pos"):
            obs = s["obs"]
            return self.policy._desired_pos(self.policy._parse_obs(obs.copy())) + (s["tip"] - obs[:3]), None
        return s["goal"], None


# Meta-World's ML10 split
ML10_TRAIN = ("reach-v3", "push-v3", "pick-place-v3", "door-open-v3", "drawer-close-v3", "button-press-topdown-v3",
              "peg-insert-side-v3", "window-open-v3", "sweep-v3", "basketball-v3")
ML10_TEST = ("drawer-open-v3", "door-close-v3", "shelf-place-v3", "sweep-into-v3", "lever-pull-v3")
ML10 = {n: Expert(n) for n in ML10_TRAIN + ML10_TEST}
# all of Meta-World's MT50, labelled by their experts the same way (for other splits)
MT50_NAMES = ("assembly-v3 basketball-v3 bin-picking-v3 box-close-v3 button-press-topdown-v3 button-press-topdown-wall-v3 "
              "button-press-v3 button-press-wall-v3 coffee-button-v3 coffee-pull-v3 coffee-push-v3 dial-turn-v3 "
              "disassemble-v3 door-close-v3 door-lock-v3 door-open-v3 door-unlock-v3 drawer-close-v3 drawer-open-v3 "
              "faucet-close-v3 faucet-open-v3 hammer-v3 hand-insert-v3 handle-press-side-v3 handle-press-v3 "
              "handle-pull-side-v3 handle-pull-v3 lever-pull-v3 peg-insert-side-v3 peg-unplug-side-v3 pick-out-of-hole-v3 "
              "pick-place-v3 pick-place-wall-v3 plate-slide-back-side-v3 plate-slide-back-v3 plate-slide-side-v3 "
              "plate-slide-v3 push-back-v3 push-v3 push-wall-v3 reach-v3 reach-wall-v3 shelf-place-v3 soccer-v3 "
              "stick-pull-v3 stick-push-v3 sweep-into-v3 sweep-v3 window-close-v3 window-open-v3").split()
MT50 = {n: ML10.get(n) or Expert(n) for n in MT50_NAMES}
SUITES = {"default": TASKS, "ml10": ML10, "mt50": MT50}
SYSTEMS = {"default": SYSTEM, "ml10": SYSTEM, "mt50": SYSTEM_IMAGE}


def user_text(task, s, question):
    return "\n".join([task.text, "", "State:", *task.state_lines(s), "", question, "Reply with only the letter."])


def questions(task):
    """(key, question text, number of options). Axis options are decrease, hold,
    increase; the grip options are close, open."""
    out = []
    for key in task.keys:
        if key == "grip" and getattr(task, "soft", False):
            out.append((key, "Question: What should the gripper do with its fingers?\nA: open the gripper\n"
                             "B: close the gripper to grasp\nC: close the gripper tightly", 3))
        elif key == "grip":
            out.append((key, "Question: What should the gripper do with its fingers?\nA: close the gripper\n"
                             "B: open the gripper", 2))
        else:
            axis = AXES[key]
            out.append((key, f"Question: What should the gripper do along {axis}?\nA: decrease {axis}\n"
                             f"B: hold {axis}\nC: increase {axis}", 3))
    return out


def labels(task, s):
    """True option index per question key (for a soft task, the most likely option)."""
    if getattr(task, "soft", False):
        return {k: int(np.argmax(p)) for k, p in task.soft_labels(s).items()}
    target, close = task.target(s)
    err = target - s["tip"]
    out = {k: int(0 if e < -DEADBAND else 2 if e > DEADBAND else 1) for k, e in zip(("dx", "dy", "dz"), err)}
    out["grip"] = 0 if close else 1
    return {k: out[k] for k in task.keys}


def action_of(task, probs):
    """Axis moves by (P(increase) - P(decrease)) * MAG; the gripper closes when
    P(close) > 0.5, and stays open for a task without a grip question. A soft task moves
    by SOFT_MAG and takes the most likely grip level."""
    if getattr(task, "soft", False):
        move = [(probs[k][2] - probs[k][0]) * SOFT_MAG for k in ("dx", "dy", "dz")]
        return np.array([*move, GRIP_LEVELS[int(np.argmax(probs["grip"]))]], np.float32)
    move = [(probs[k][2] - probs[k][0]) * MAG if k in probs else 0.0 for k in ("dx", "dy", "dz")]
    grip = 1.0 if "grip" in probs and probs["grip"][0] > 0.5 else -1.0
    return np.array([*move, grip], np.float32)
