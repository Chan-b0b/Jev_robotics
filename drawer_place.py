"""Drawer place: open the drawer, pick up the puck and put it in the drawer.

A new task, not in the training data: drawer-open-v3's scene with pick-place-v3's
puck on the table beside the drawer (assets/sawyer_xyz/sawyer_drawer_place.xml).
The drawer is placed as in drawer-open-v3; the puck is PUCK_DX to either side of
it, at pick-place-v3's y range.

The harness gives one instruction per phase and moves on when the state says the
phase is done:
1. open: drawer-open-v3's own prompt (vision.TEXT), until the drawer is open.
2. lift: lift the gripper straight up out of the handle; done above LIFT_DONE_Z,
   wherever it is in xy (the instruction names no height). The labels aim at LIFT_Z.
3. pick: pick up the puck and lift it, to LIFT_Z. The approach goes over at LIFT_Z
   first, so the arm stays clear of the drawer.
4. place: carry the puck PLACE_Z above the open drawer's middle, gripper closed.
5. release: open the gripper, for RELEASE_STEPS steps.
Success: the puck is at rest inside the drawer. Phases 2-5 are new instructions;
drawer-open's green goal sphere is hidden after phase 1, and no other sphere is shown.

--policy labels acts on the phases' labels (checks the task can be done);
--policy qwen asks Qwen with the camera frames, as closed_loop.py --vision does.
"""
import argparse
import asyncio
import json
import os

import imageio.v2 as imageio
import numpy as np
from metaworld.envs.sawyer_drawer_open_v3 import SawyerDrawerOpenEnvV3
from PIL import Image, ImageDraw, ImageFont

import vision
from closed_loop import FONT, Qwen, panel
from tasks import HAND, TASKS, action_of, labels, questions, xy

XML = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "sawyer_xyz", "sawyer_drawer_place.xml")
PUCK_DX = (0.28, 0.32)  # m, puck x from the drawer's, either side: far enough that the carry clears the walls
PUCK_Y = (0.6, 0.7)  # pick-place-v3's puck y range
LIFT_Z = 0.25  # fingertips' height to lift to, and to move over the table at
LIFT_DONE_Z = 0.15  # clear of the handle (z 0.09)
PLACE_Z = 0.25  # above the drawer's middle; its walls top out at z 0.15
RELEASE_STEPS = 40
INSIDE_XY = (0.084, 0.066)  # half size of the drawer's inside (walls' inner faces)
WALL_TOP = 0.15


class SawyerDrawerPlaceEnvV3(SawyerDrawerOpenEnvV3):
    @property
    def model_name(self):
        return XML

    def _get_pos_objects(self):  # (handle, puck); the handle stays first, as drawer-open's reward reads it
        return np.hstack([super()._get_pos_objects(), self.get_body_com("obj")])

    def _get_quat_objects(self):
        return np.hstack([super()._get_quat_objects(), self.data.body("obj").xquat])

    def reset_model(self):
        super().reset_model()  # hand, drawer, drawer-open's goal
        side = self.np_random.choice([-1, 1])
        puck = [self.obj_init_pos[0] + side * self.np_random.uniform(*PUCK_DX), self.np_random.uniform(*PUCK_Y), 0.02]
        adr = self.model.joint("objjoint").qposadr[0]
        qpos, qvel = self.data.qpos.copy(), self.data.qvel.copy()
        qpos[adr:adr + 7] = [*puck, 1, 0, 0, 0]
        qvel[self.model.joint("objjoint").dofadr[0]:][:6] = 0
        self.set_state(qpos, qvel)
        return self._get_obs()


def make_env(seed, camera=None):
    env = SawyerDrawerPlaceEnvV3(render_mode="rgb_array" if camera else None, camera_name=camera,
                                 width=320, height=320)
    env._set_task_called, env._freeze_rand_vec, env.seeded_rand_vec = True, False, True
    env.max_path_length = 10 ** 6
    env.seed(seed)
    return env


def state(env, obs):
    link = np.array(env.get_body_com("drawer_link"))
    return {"tip": np.array(env.tcp_center), "opening": float(obs[3]), "handle": np.array(obs[4:7]),
            "puck": np.array(env.get_body_com("obj")), "link": link,
            "drop": np.array([link[0], link[1], PLACE_Z])}


def inside(s):
    d = s["puck"] - s["link"]
    return abs(d[0]) < INSIDE_XY[0] and abs(d[1]) < INSIDE_XY[1] and s["puck"][2] < WALL_TOP


class Open:
    name, keys, text = "open", TASKS["drawer-open-v3"].keys, vision.TEXT["drawer-open-v3"]

    def target(self, s):
        return TASKS["drawer-open-v3"].target({"tip": s["tip"], "obj": s["handle"]})

    def done(self, env, s):  # drawer-open-v3's success
        return float(np.linalg.norm(s["handle"] - env._target_pos)) <= 0.03


class Lift:
    name, keys = "lift", ("dx", "dy", "dz")
    text = "Task: Lift the gripper straight up, out of the drawer handle."

    def target(self, s):
        return np.array([*s["lift_xy"], LIFT_Z]), False

    def done(self, env, s):
        return s["tip"][2] > LIFT_DONE_Z


class Pick:
    """pick-place-v3's rules up to the grasp, with the approach and the lift at LIFT_Z."""
    name, keys = "pick", ("dx", "dy", "dz", "grip")
    text = ("Task: Pick up the puck (the dark red cylinder).\n"
            "Procedure: move above the puck, lower the gripper around it and close the gripper, "
            "wait until the gripper has closed, then lift the puck.")

    def target(self, s):
        tip, puck = s["tip"], s["puck"]
        p = puck + [-0.005, 0, -HAND]
        close = float(np.linalg.norm(tip + [0, 0, HAND] - puck)) < 0.07
        if xy(tip, p) > 0.02:
            return np.array([p[0], p[1], LIFT_Z]), close
        if abs(tip[2] - p[2]) > 0.05 and puck[2] < 0.04:
            return p + [0, 0, 0.03], close
        if s["opening"] > 0.73:
            return tip, close
        return np.array([p[0], p[1], LIFT_Z]), close

    def done(self, env, s):
        return s["puck"][2] > LIFT_Z - 0.05


class Place:
    name, keys = "place", ("dx", "dy", "dz", "grip")
    # out/drawer_place/all_2cam was run with "Task: Put the puck into the open drawer: the tray pulled
    # out toward the robot, in front of the green box. Keep the gripper closed and carry the puck above
    # the middle of the tray." There Qwen opened the gripper before it got there.
    text = ("Task: Hold the puck in the air above the open drawer: the tray pulled out toward the robot, in "
            "front of the green box. Keep the gripper closed and carry the puck through the air until it is "
            "above the middle of the tray.")

    def target(self, s):
        return s["drop"], True

    def done(self, env, s):
        return float(np.linalg.norm(s["drop"] - s["tip"])) < 0.01


class Release:
    name, keys = "release", ("grip",)
    text = "Task: Drop the puck into the drawer: open the gripper."

    def target(self, s):
        return s["tip"], False

    def done(self, env, s):
        return False  # the episode ends RELEASE_STEPS steps in


PHASES = [Open(), Lift(), Pick(), Place(), Release()]


async def episode(seed, policy, qwen, max_steps, video):
    env = make_env(seed)
    obs, _ = env.reset(seed=seed)
    cams = vision.Cameras(env)
    k, lift_xy, release, frames, trace, matched, reads = 0, None, 0, [], [], 0, 0
    phase_start = {"open": 0}
    for t in range(max_steps):
        s = state(env, obs)
        if lift_xy is not None:
            s["lift_xy"] = lift_xy
        if PHASES[k].done(env, s):
            k += 1
            phase_start[PHASES[k].name] = t
            if PHASES[k].name == "lift":
                env.model.site("goal").rgba = (0, 0, 0, 0)  # drawer-open's goal sphere, not part of the rest
                lift_xy = s["tip"][:2].copy()
                s["lift_xy"] = lift_xy
        phase = PHASES[k]
        truth = labels(phase, s)
        qs = questions(phase)
        shots = cams.frames()
        if policy == "qwen":
            users = [vision.for_vllm(vision.user_content(phase.name, text, text=phase.text), shots) for _, text, _ in qs]
            ps = await asyncio.gather(*[qwen.read(u, n) for u, (_, _, n) in zip(users, qs)])
            probs = {key: p for (key, _, _), p in zip(qs, ps)}
        else:
            probs = {key: np.eye(n)[truth[key]] for key, _, n in qs}
        matched += sum(int(probs[key].argmax() == truth[key]) for key in probs)
        reads += len(probs)
        action = action_of(phase, probs)
        trace.append({"seed": seed, "t": t, "phase": phase.name, "tip": s["tip"].round(4).tolist(),
                      "puck": s["puck"].round(4).tolist(), "handle": s["handle"].round(4).tolist(),
                      "probs": {key: p.round(4).tolist() for key, p in probs.items()}, "truth": truth})
        if video:
            side = max(f.shape[0] for f in shots)
            frame = np.concatenate([np.array(Image.fromarray(f).resize((side, side))) for f in shots], axis=1)
            img = Image.fromarray(panel(frame, f"drawer-place [{phase.name}]", t, probs, truth, action))
            d, font = ImageDraw.Draw(img), ImageFont.truetype(FONT, 22)
            label = f"phase {k + 1}/{len(PHASES)}: {phase.name}"
            d.rectangle([0, 0, 12 + d.textlength(label, font=font), 34], fill="white")
            d.text((6, 5), label, fill="black", font=font)
            frames.append(np.array(img))
        obs, *_ = env.step(action)
        release += phase.name == "release"
        if release >= RELEASE_STEPS:
            break
    s = state(env, obs)
    cams.close()
    env.close()
    if video:
        imageio.mimsave(video, frames, fps=10, macro_block_size=1)
    return {"seed": seed, "success": bool(release >= RELEASE_STEPS and inside(s)), "steps": t + 1,
            "reached": PHASES[k].name, "phase_start": phase_start, "puck": s["puck"].round(4).tolist(),
            "match": matched / reads}, trace


async def main(args):
    qwen = None
    if args.policy == "qwen":
        qwen = Qwen(args.url, args.model, args.concurrency)
        await qwen.setup()
    os.makedirs(args.outdir, exist_ok=True)
    runs = [episode(args.seed + i, args.policy, qwen, args.max_steps,
                    os.path.join(args.outdir, f"drawer-place_{args.policy}_{args.seed + i}.mp4") if i < args.video else None)
            for i in range(args.episodes)]
    done = await asyncio.gather(*runs)
    results = [r for r, _ in done]
    with open(os.path.join(args.outdir, f"closed_{args.policy}.jsonl"), "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    with open(os.path.join(args.outdir, f"steps_{args.policy}.jsonl"), "w") as f:
        for _, trace in done:
            for row in trace:
                f.write(json.dumps(row) + "\n")
    mean = lambda k: np.mean([r[k] for r in results])
    print(f"policy {args.policy} ({args.model if qwen else 'labels'}), {args.episodes} episodes, "
          f"at most {args.max_steps} steps -> {args.outdir}")
    reached = [p.name for p in PHASES]
    reached = {name: np.mean([reached.index(r["reached"]) >= i for r in results]) for i, name in enumerate(reached)}
    print("reached " + "  ".join(f"{name} {v:.2f}" for name, v in reached.items()))
    print(f"success {mean('success'):.2f}  steps {mean('steps'):.0f}  label match {mean('match'):.2f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", choices=["labels", "qwen"], default="qwen")
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--max-steps", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=5000)
    ap.add_argument("--video", type=int, default=0, help="save the first N episodes as mp4")
    ap.add_argument("--url", default="http://127.0.0.1:8001")
    ap.add_argument("--model", default="vmulti")
    ap.add_argument("--concurrency", type=int, default=256)
    ap.add_argument("--outdir", default="out/drawer_place")
    asyncio.run(main(ap.parse_args()))
