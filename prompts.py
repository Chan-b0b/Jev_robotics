"""Task texts for the ML10 image prompts: several descriptions and short procedures per task.

Training draws one description per row and adds one procedure half the time, so the
model reads a task in many wordings, with and without the steps. Some descriptions
leave out colors. The procedures follow the order of each expert's subgoals
(metaworld.policies) and its gripper use (grasp, keep closed to push, keep open),
without numbers. Test tasks (tasks.ML10_TEST) are never trained on; evaluation asks
with a description only (a) or with a procedure too (b).
"""
import numpy as np

DESCRIPTIONS = {
    # ---- train
    "reach-v3": [
        "Move the gripper to the goal, the small red sphere; it may be in the air.",
        "Bring the fingertips to the red ball floating above the table.",
        "Reach for the small red sphere. The dark red cylinder on the table is not the target.",
        "Go to the red marker; nothing needs to be picked up.",
        "Touch the small red sphere with the gripper.",
        "Put the gripper at the red goal sphere and stay there.",
        "Move the tip of the gripper onto the red dot in the air.",
        "Move the gripper to the goal marker; it may be in the air.",
        "Reach the small floating sphere with the gripper.",
        "Bring the gripper to the goal point marked by a small sphere.",
    ],
    "push-v3": [
        "Push the puck (the dark red cylinder) across the table to the goal, the small green sphere.",
        "Slide the dark red puck along the table until it reaches the green sphere.",
        "Move the red cylinder on the table over to the green marker by pushing it.",
        "The goal is the small green sphere. Push the puck there without lifting it.",
        "Get the dark red puck to the green sphere by sliding it along the tabletop.",
        "Shove the puck across the table so that it ends up at the green goal marker.",
        "Bring the red cylinder to the green sphere, keeping it on the table the whole way.",
        "Push the puck across the table to the goal marker.",
        "Slide the puck along the tabletop until it sits at the goal.",
        "Move the cylinder on the table to the small sphere by pushing it.",
    ],
    "pick-place-v3": [
        "Pick up the puck (the dark red cylinder) and place it at the goal, the small blue sphere.",
        "Grab the dark red puck and carry it to the blue sphere; the goal may be in the air.",
        "Lift the red cylinder off the table and bring it to the blue marker.",
        "Take the puck from the table and hold it at the small blue sphere.",
        "Grasp the dark red cylinder and move it to where the blue ball is.",
        "Carry the red puck through the air to the blue goal sphere.",
        "Pick the puck up and put it at the blue marker.",
        "Pick up the puck and place it at the goal marker; the goal may be in the air.",
        "Grasp the cylinder on the table and carry it to the small sphere.",
        "Lift the puck and bring it to the goal point.",
    ],
    "door-open-v3": [
        "Open the door of the dark gray cabinet by pulling its handle.",
        "Swing the dark gray cabinet door open.",
        "Pull the handle of the gray cabinet so that the door opens.",
        "The dark gray cabinet has a closed door. Open it.",
        "Open the gray safe's door by dragging its handle sideways.",
        "Pull the cabinet door open until it reaches the green sphere.",
        "Get the door of the dark cabinet to swing open.",
        "Open the cabinet door by pulling its handle.",
        "Swing the door open using its handle.",
        "Pull the door of the cabinet open.",
    ],
    "drawer-close-v3": [
        "Close the drawer: push the open drawer of the green box shut.",
        "The green cabinet's drawer is pulled out. Push it back in until it is closed.",
        "Shut the drawer by pushing its white handle away from the robot.",
        "Slide the open drawer back into the green box.",
        "Push the drawer of the green chest closed.",
        "The drawer is sticking out of the green box. Close it.",
        "Close the green drawer by pressing its handle toward the box.",
        "Close the open drawer.",
        "Push the drawer back in until it is shut.",
        "Shut the drawer by pushing its handle away from the robot.",
    ],
    "button-press-topdown-v3": [
        "Press the red button on top of the yellow box.",
        "Push the red button down from above.",
        "Press down on the red button of the yellow and black box.",
        "The yellow box has a red button on top. Push it down.",
        "Press the button on the yellow box straight down.",
        "Push the red knob on the box down until it clicks.",
        "Press the red button from above with the gripper.",
        "Press the button on top of the box.",
        "Push the button down from above.",
        "Press down on the button until it is pushed in.",
    ],
    "peg-insert-side-v3": [
        "Insert the green peg into the hole in the side of the box.",
        "Pick up the green peg and push it into the hole in the box's red face.",
        "Grab the green stick and slide it sideways into the hole of the wooden box.",
        "Put the green peg into the side hole of the box.",
        "Lift the green peg and insert it into the opening on the red side of the box.",
        "Take the green peg from the table and fit it into the hole in the box.",
        "Grasp the green rod and push it into the side of the box through its hole.",
        "Insert the peg into the hole in the side of the box.",
        "Pick up the peg and slide it into the box's side hole.",
        "Grab the peg and fit it into the opening in the box.",
    ],
    "window-open-v3": [
        "Open the window: slide the pane in the dark red frame sideways.",
        "Slide the window in the red frame open by its handle.",
        "Push the window's handle sideways so that the window opens.",
        "The window in the dark red frame is closed. Slide it open.",
        "Open the sliding window by moving its handle along the frame.",
        "Slide the pane of the red window aside.",
        "Move the window handle sideways until the window is open.",
        "Slide the window open.",
        "Open the window by sliding its handle sideways.",
        "Push the sliding window open along its frame.",
    ],
    "sweep-v3": [
        "Sweep the small brown cube off the side of the table.",
        "Move the brown cube to the edge of the table and drop it off.",
        "Get the small brown block off the table at its side.",
        "Carry the brown cube along the table and let it fall off the edge.",
        "Clear the brown cube off the table to the side.",
        "Push the little brown box over the edge of the table.",
        "Take the brown cube to the table's edge and let go of it there.",
        "Sweep the cube off the side of the table.",
        "Move the small block to the edge of the table and drop it off.",
        "Get the cube off the table at its side.",
    ],
    "basketball-v3": [
        "Put the orange ball through the hoop.",
        "Pick up the orange basketball and dunk it into the hoop with the white backboard.",
        "Grab the orange ball and drop it into the basket.",
        "Score: lift the orange ball and put it in the hoop.",
        "Take the orange ball from the table and place it through the basketball hoop.",
        "Carry the orange ball up to the hoop and drop it in.",
        "Lift the orange ball into the ring in front of the white board.",
        "Put the ball through the hoop.",
        "Pick up the ball and dunk it into the basket.",
        "Grab the ball and drop it in the hoop.",
    ],
    # ---- test
    "drawer-open-v3": [
        "Open the drawer of the green box by pulling its white handle toward the robot.",
        "Pull the drawer out of the green cabinet.",
        "Open the drawer by pulling its handle.",
    ],
    "door-close-v3": [
        "Close the open door of the dark gray cabinet.",
        "Swing the gray cabinet's door shut.",
        "Close the cabinet door.",
    ],
    "shelf-place-v3": [
        "Pick up the blue cube and put it on the wooden shelf.",
        "Place the blue block on the shelf at the green sphere.",
        "Put the cube on the shelf.",
    ],
    "sweep-into-v3": [
        "Sweep the small brown cube into the hole marked in blue on the table.",
        "Move the brown cube into the blue hole in the table.",
        "Get the cube into the hole in the table.",
    ],
    "lever-pull-v3": [
        "Pull the lever up: lift the dark lever arm on the gray stand so that it points upward.",
        "Raise the lever on the gray post until it swings up.",
        "Pull the lever up.",
    ],
}

PROCEDURES = {
    # ---- train
    "reach-v3": [
        "Move straight to the goal with the gripper open.",
        "Keep the gripper open and go to the sphere.",
        "Head for the goal and stop there.",
        "Go directly to the target; do not close the gripper.",
        "Move to the goal and hold still.",
    ],
    "push-v3": [
        "Move above the puck, lower onto it and close the gripper, then push it along the table to the goal.",
        "Get over the puck, go down around it, grip it, and slide it to the goal while staying low.",
        "Line up above the puck, descend and grasp it, then drag it to the goal.",
        "Go over the puck, drop down, close the fingers, and move it to the goal on the table.",
        "Position above the puck, lower and close the gripper, then push to the goal.",
    ],
    "pick-place-v3": [
        "Move above the puck, lower and close the gripper, wait until it has closed, then carry it to the goal.",
        "Get over the puck, go down and grasp it, then lift it to the goal.",
        "Line up above the puck, descend and grip it, then move it to the goal.",
        "Go over the puck, drop down, close the fingers, and bring it to the goal.",
        "Position above the puck, lower and grasp it, then carry it to the goal.",
    ],
    "door-open-v3": [
        "Keep the gripper closed. Move above and beside the handle, lower to it, then pull the door open.",
        "With the fingers closed, go to the handle from above, drop down to it, and drag it to open the door.",
        "Close the gripper, get next to the handle, lower to handle height, and pull the door open.",
        "Keep the gripper closed, reach the handle from above, then pull it sideways.",
        "Go above the handle with a closed gripper, come down to it, and pull the door open.",
    ],
    "drawer-close-v3": [
        "Keep the gripper closed, get in front of the handle, and push the drawer in.",
        "With the fingers closed, come to the handle from above and push the drawer shut.",
        "Close the gripper, move in front of the handle, lower to it, and push forward.",
        "Keep the gripper closed. Go high, then down in front of the handle, and push the drawer in.",
        "Close the gripper, get to the front of the handle, and push the drawer closed.",
    ],
    "button-press-topdown-v3": [
        "Keep the gripper closed, move above the button, then push straight down.",
        "With the fingers closed, go over the button and press down.",
        "Close the gripper, line up above the button, and lower onto it.",
        "Get above the button with a closed gripper and push down.",
        "Keep the gripper closed, move over the button, and press it down.",
    ],
    "peg-insert-side-v3": [
        "Move above the peg, lower and grasp it, then carry it level with the hole and push it in.",
        "Get over the peg, go down and grip it, then bring it to the hole and slide it in.",
        "Grasp the peg from above, move it in line with the hole, then insert it.",
        "Go over the peg, lower, close the fingers, and push the peg into the hole.",
        "Pick up the peg, align it with the hole, and slide it in.",
    ],
    "window-open-v3": [
        "Keep the gripper closed, move above the handle, lower to it, then push it sideways.",
        "With the fingers closed, come down to the handle and slide it along the frame.",
        "Close the gripper, get to the handle from above, and push it to the side.",
        "Keep the gripper closed. Go over the handle, drop to it, and slide the window open.",
        "Close the gripper, lower onto the handle, and move it sideways.",
    ],
    "sweep-v3": [
        "Move above the cube, lower and grasp it, carry it to the table's edge, then open the gripper.",
        "Get over the cube, go down and grip it, move it to the edge, and let it go.",
        "Grasp the cube from above, bring it to the side of the table, and release it.",
        "Go over the cube, lower, close the fingers, and take it off the edge.",
        "Pick up the cube, move it past the edge, and drop it.",
    ],
    "basketball-v3": [
        "Move above the ball, lower and grasp it, lift it to the hoop's height, then move it into the hoop.",
        "Get over the ball, go down and grip it, raise it, and bring it to the hoop.",
        "Grasp the ball from above, lift it up, and carry it into the hoop.",
        "Go over the ball, lower, close the fingers, rise, and move to the hoop.",
        "Pick up the ball, raise it to the hoop, and put it in.",
    ],
    # ---- test
    "drawer-open-v3": [
        "Keep the gripper open, move high above the handle, lower to it, then pull toward the robot.",
        "Go above the handle with an open gripper, come down to it, and pull the drawer out.",
    ],
    "door-close-v3": [
        "Keep the gripper closed. Get beside the open door from above, lower to it, then push it shut.",
        "With the fingers closed, come down next to the door and push it closed.",
    ],
    "shelf-place-v3": [
        "Move above the cube, lower and grasp it, lift it to the shelf's height, then move it onto the shelf.",
        "Grasp the cube from above, raise it, and carry it onto the shelf.",
    ],
    "sweep-into-v3": [
        "Move above the cube, lower and grasp it, then carry it along the table to the hole.",
        "Grasp the cube from above and bring it to the hole.",
    ],
    "lever-pull-v3": [
        "Keep the gripper closed. Move to just below the lever's end, rise up to it, then push forward and up.",
        "With the fingers closed, go under the end of the lever and lift it.",
    ],
}


def text(task_name, rng, procedure):
    """A task text: a random description, with a random procedure when procedure is True."""
    lines = [f"Task: {rng.choice(DESCRIPTIONS[task_name])}"]
    if procedure:
        lines.append(f"Procedure: {rng.choice(PROCEDURES[task_name])}")
    return "\n".join(lines)
