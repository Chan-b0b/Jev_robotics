# mt50 split: task texts (prompts_mt50.py)

Objects are named by kind and shape, not color, because looks.py recolors them. Training draws one description per row and adds one procedure half the time. Test tasks are never trained on; they are evaluated with (a) a description only and (b) a description and a procedure.

## Train (38)

### reach-v3

Descriptions:
1. Move the gripper to the goal, the small sphere; it may be in the air.
2. Bring the fingertips to the small ball floating above the table.
3. Reach for the small sphere. The cylinder on the table is not the target.
4. Go to the goal marker; nothing needs to be picked up.
5. Touch the small sphere with the gripper.
6. Put the gripper at the goal sphere and stay there.
7. Move the tip of the gripper onto the small ball in the air.
8. Reach the small floating sphere with the gripper.
9. Bring the gripper to the goal point marked by a small sphere.
10. Move the gripper to the marker and hold it there.

Procedures:
1. Move straight to the goal with the gripper open.
2. Keep the gripper open and go to the sphere.
3. Head for the goal and stop there.
4. Go directly to the target; do not close the gripper.
5. Move to the goal and hold still.

### reach-wall-v3

Descriptions:
1. Reach the small sphere behind the wall with the gripper.
2. Move the gripper to the goal on the far side of the low wall.
3. Get the fingertips to the small ball beyond the wall, going over the wall.
4. The goal sphere is behind a wall. Reach it without hitting the wall.
5. Bring the gripper over the wall to the small sphere.
6. Touch the goal marker on the other side of the wall.
7. Reach over the wall to the small ball.
8. Move the gripper past the wall to the goal point.
9. Go around the top of the wall and reach the sphere.
10. Put the gripper at the small sphere that sits behind the wall.

Procedures:
1. Keep the gripper open, rise above the wall, then move to the goal.
2. Go up over the wall and come down at the sphere.
3. Move toward the goal, climbing over the wall on the way.
4. Lift the gripper above the wall, pass it, then reach the goal.
5. Head for the goal, going over the wall rather than into it.

### push-v3

Descriptions:
1. Push the puck across the table to the goal, the small sphere.
2. Slide the puck along the table until it reaches the goal marker.
3. Move the cylinder on the table over to the small sphere by pushing it.
4. The goal is the small sphere. Push the puck there without lifting it.
5. Get the puck to the goal by sliding it along the tabletop.
6. Shove the puck across the table so that it ends up at the goal marker.
7. Bring the cylinder to the small sphere, keeping it on the table the whole way.
8. Push the puck across the table to the goal.
9. Slide the puck along the tabletop until it sits at the goal.
10. Move the short cylinder to the small sphere by pushing it on the table.

Procedures:
1. Move above the puck, lower onto it and close the gripper, then push it along the table to the goal.
2. Get over the puck, go down around it, grip it, and slide it to the goal while staying low.
3. Line up above the puck, descend and grasp it, then drag it to the goal.
4. Go over the puck, drop down, close the fingers, and move it to the goal on the table.
5. Position above the puck, lower and close the gripper, then push to the goal.

### push-wall-v3

Descriptions:
1. Push the puck around the wall to the goal.
2. Slide the puck along the table past the wall to the small sphere.
3. There is a wall between the puck and the goal. Push the puck around it.
4. Get the cylinder to the goal marker, going around the wall.
5. Move the puck on the table to the goal without hitting the wall.
6. Push the puck to the goal sphere behind the wall.
7. Slide the cylinder around the side of the wall to the goal.
8. Bring the puck past the wall to the small sphere.
9. Push the puck along the table, around the wall, to the marker.
10. Steer the puck around the wall to the goal.

Procedures:
1. Move above the puck, lower and grip it, then push it around the side of the wall to the goal.
2. Grasp the puck from above, steer it around the wall, then on to the goal.
3. Get over the puck, go down and close the gripper, go around the wall, then to the goal.
4. Grip the puck, move it sideways past the wall, then forward to the goal.
5. Lower onto the puck and grasp it; push it around the wall and to the goal.

### push-back-v3

Descriptions:
1. Push the puck back toward the robot to the goal.
2. Slide the puck along the table toward the robot until it reaches the goal marker.
3. Bring the cylinder back to the small sphere near the robot.
4. Move the puck toward the robot's side of the table, to the goal.
5. Pull the puck along the table back to the goal.
6. The goal is closer to the robot. Slide the puck there.
7. Drag the puck across the table toward the robot, to the marker.
8. Get the cylinder back to the goal sphere.
9. Move the puck backward along the table to the goal.
10. Slide the puck to the goal on the robot's side.

Procedures:
1. Move above the puck, lower and grip it, then move it back toward the robot to the goal.
2. Grasp the puck from above and drag it toward the robot to the goal.
3. Get over the puck, go down and close the gripper, then pull it to the goal.
4. Grip the puck and slide it backward along the table to the goal.
5. Lower onto the puck, grasp it, and bring it back to the goal.

### pick-place-v3

Descriptions:
1. Pick up the puck and place it at the goal, the small sphere; the goal may be in the air.
2. Grab the puck and carry it to the goal marker.
3. Lift the cylinder off the table and bring it to the small sphere.
4. Take the puck from the table and hold it at the goal sphere.
5. Grasp the cylinder and move it to where the small ball is.
6. Carry the puck through the air to the goal.
7. Pick the puck up and put it at the marker.
8. Pick up the puck and place it at the goal point.
9. Grasp the cylinder on the table and carry it to the small sphere.
10. Lift the puck and bring it to the goal.

Procedures:
1. Move above the puck, lower and close the gripper, wait until it has closed, then carry it to the goal.
2. Get over the puck, go down and grasp it, then lift it to the goal.
3. Line up above the puck, descend and grip it, then move it to the goal.
4. Go over the puck, drop down, close the fingers, and bring it to the goal.
5. Position above the puck, lower and grasp it, then carry it to the goal.

### pick-place-wall-v3

Descriptions:
1. Pick up the puck, lift it over the wall and place it at the goal.
2. Carry the puck over the wall to the small sphere.
3. Grab the cylinder and bring it to the goal on the other side of the wall.
4. Lift the puck high enough to clear the wall, then put it at the goal.
5. Take the puck over the wall to the goal marker.
6. Grasp the puck and move it past the wall to the goal.
7. Pick up the cylinder and carry it over the wall to the small ball.
8. The goal is behind a wall. Lift the puck over it and place it there.
9. Pick up the puck and bring it over the wall to the goal point.
10. Carry the puck through the air, over the wall, to the goal.

Procedures:
1. Move above the puck, lower and grasp it, lift it straight up over the wall, then carry it to the goal.
2. Grasp the puck from above, raise it above the wall, then bring it to the goal.
3. Get over the puck, go down and grip it, go up high, move past the wall, then to the goal.
4. Pick up the puck, lift it clear of the wall, then place it at the goal.
5. Grip the puck, rise over the wall, and move to the goal.

### bin-picking-v3

Descriptions:
1. Move the cube from one bin to the other.
2. Pick up the small cube in the bin and drop it into the other bin.
3. Take the block out of its bin and put it in the empty bin.
4. Transfer the cube between the two bins.
5. Lift the cube from the first bin and place it in the second.
6. Grab the small block and carry it over to the other bin.
7. Put the cube into the other bin.
8. Move the block across to the empty bin.
9. Pick the cube out of the bin and drop it in the bin beside it.
10. Carry the small cube to the other container.

Procedures:
1. Move above the cube, lower and grasp it, lift it, carry it over the other bin, then lower it in.
2. Grasp the cube from above, raise it, move over the empty bin, and put it down there.
3. Get over the cube, go down and grip it, rise, go to the other bin, and lower.
4. Pick up the cube, bring it above the second bin, and lower it into the bin.
5. Grip the cube, lift it out of its bin, and set it into the other bin.

### basketball-v3

Descriptions:
1. Put the ball through the hoop.
2. Pick up the basketball and dunk it into the hoop in front of the backboard.
3. Grab the ball and drop it into the basket.
4. Score: lift the ball and put it in the hoop.
5. Take the ball from the table and place it through the basketball hoop.
6. Carry the ball up to the hoop and drop it in.
7. Lift the ball into the ring in front of the board.
8. Pick up the ball and dunk it into the basket.
9. Grab the ball and drop it in the hoop.
10. Raise the ball to the hoop and put it through.

Procedures:
1. Move above the ball, lower and grasp it, lift it to the hoop's height, then move it into the hoop.
2. Get over the ball, go down and grip it, raise it, and bring it to the hoop.
3. Grasp the ball from above, lift it up, and carry it into the hoop.
4. Go over the ball, lower, close the fingers, rise, and move to the hoop.
5. Pick up the ball, raise it to the hoop, and put it in.

### soccer-v3

Descriptions:
1. Kick the ball into the goal net.
2. Push the soccer ball along the table into the net.
3. Score: roll the ball into the goal.
4. Drive the ball into the small goal on the table.
5. Get the ball into the net by pushing it.
6. Knock the ball into the goal.
7. Push the ball across the table so it goes into the net.
8. Move the ball into the soccer goal.
9. Shoot the ball into the net.
10. Roll the ball along the table into the goal frame.

Procedures:
1. Keep the gripper closed. Get behind the ball, staying low, then push the ball toward the goal.
2. With the fingers closed, move to the side of the ball away from the goal and drive it into the net.
3. Close the gripper, go behind the ball, and push it straight at the goal.
4. Keep the gripper closed, line up behind the ball, then push it into the net.
5. Get to the far side of the ball from the goal, then push through the ball toward the goal.

### sweep-v3

Descriptions:
1. Sweep the small cube off the side of the table.
2. Move the cube to the edge of the table and drop it off.
3. Get the small block off the table at its side.
4. Carry the cube along the table and let it fall off the edge.
5. Clear the cube off the table to the side.
6. Push the little box over the edge of the table.
7. Take the cube to the table's edge and let go of it there.
8. Sweep the cube off the table.
9. Move the small block to the edge and drop it off.
10. Get the cube off the side of the table.

Procedures:
1. Move above the cube, lower and grasp it, carry it to the table's edge, then open the gripper.
2. Get over the cube, go down and grip it, move it to the edge, and let it go.
3. Grasp the cube from above, bring it to the side of the table, and release it.
4. Go over the cube, lower, close the fingers, and take it off the edge.
5. Pick up the cube, move it past the edge, and drop it.

### sweep-into-v3

Descriptions:
1. Sweep the small cube into the hole in the table.
2. Move the cube into the hole marked on the tabletop.
3. Get the small block into the hole in the table.
4. Carry the cube along the table and drop it into the hole.
5. Put the cube in the hole on the table.
6. Slide the little box into the opening in the table.
7. Bring the cube to the hole and let it fall in.
8. Sweep the cube into the hole.
9. Move the block over to the hole and drop it in.
10. Get the cube into the opening on the tabletop.

Procedures:
1. Move above the cube, lower and grasp it, then carry it along the table to the hole.
2. Grasp the cube from above and bring it to the hole.
3. Get over the cube, go down and grip it, and move it over the hole.
4. Go over the cube, lower, close the fingers, and take it to the hole.
5. Pick up the cube and move it to the hole.

### shelf-place-v3

Descriptions:
1. Pick up the cube and put it on the shelf.
2. Place the block on the shelf at the goal marker.
3. Lift the cube off the table and set it on the shelf.
4. Put the small block onto the shelf.
5. Carry the cube up to the shelf and leave it there.
6. Grab the block and place it on the wooden shelf.
7. Move the cube from the table onto the shelf.
8. Put the cube on the shelf.
9. Take the block and set it on the shelf's level.
10. Place the cube on the shelf at the small sphere.

Procedures:
1. Move above the cube, lower and grasp it, lift it to the shelf's height, then move it onto the shelf.
2. Grasp the cube from above, raise it, and carry it onto the shelf.
3. Get over the cube, go down and grip it, rise to the shelf, and push it in.
4. Pick up the cube, bring it in line with the shelf, lift it, and set it on the shelf.
5. Grip the cube, lift it up to the shelf level, and move it forward onto the shelf.

### hammer-v3

Descriptions:
1. Use the hammer to drive the nail into the box.
2. Pick up the hammer and hit the nail in.
3. Grab the hammer by its handle and hammer the nail.
4. Hammer the nail into the wall of the box.
5. Lift the hammer and knock the nail in.
6. Take the hammer and drive in the nail sticking out of the box.
7. Hit the nail with the hammer until it goes in.
8. Hammer in the nail.
9. Pick up the hammer and strike the nail.
10. Use the hammer on the nail.

Procedures:
1. Move above the hammer's handle, lower and grasp it, raise it to the nail's height, then swing it into the nail.
2. Grasp the hammer from above, lift it, line it up with the nail, and drive it in.
3. Get over the handle, go down and grip it, move to the nail, and hit it.
4. Pick up the hammer, bring it level with the nail, then push into the nail.
5. Grip the hammer's handle, raise it, and strike the nail.

### assembly-v3

Descriptions:
1. Put the ring onto the peg.
2. Pick up the wrench and fit its ring over the peg.
3. Place the ring handle on the peg.
4. Lift the wrench and drop its ring onto the post.
5. Assemble: put the ring of the wrench around the peg.
6. Grab the wrench by its handle and slide the ring down the peg.
7. Carry the ring over to the peg and lower it on.
8. Put the wrench's ring on the post.
9. Fit the ring over the peg.
10. Move the ring onto the peg.

Procedures:
1. Move above the wrench, lower and grasp it, lift it to the peg's height, move over the peg, then lower the ring onto it.
2. Grasp the wrench from above, raise it, line the ring up above the peg, and lower it down.
3. Get over the wrench, go down and grip it, carry it above the peg, and drop the ring on.
4. Pick up the wrench, bring its ring over the peg, and lower it.
5. Grip the wrench, lift it, go over the peg, and put the ring down around it.

### disassemble-v3

Descriptions:
1. Take the ring off the peg.
2. Lift the wrench's ring up off the post.
3. Pull the ring handle off the peg.
4. Disassemble: remove the wrench from the peg.
5. Grab the wrench and lift it off the peg.
6. Pull the ring up and off the post.
7. Remove the ring from the peg.
8. Lift the wrench off the peg.
9. Take the ring off the post.
10. Get the wrench's ring off the peg.

Procedures:
1. Move above the wrench, lower and grasp it, then lift it straight up off the peg.
2. Grasp the wrench from above and pull it up off the peg.
3. Get over the wrench, go down and grip it, and raise it.
4. Go over the wrench, lower, close the fingers, and lift up.
5. Grip the wrench and pull the ring straight up.

### box-close-v3

Descriptions:
1. Put the lid on the box.
2. Pick up the lid by its handle and close the box with it.
3. Close the box: place the lid on top.
4. Lift the lid and set it onto the box.
5. Cover the box with its lid.
6. Grab the lid's handle and put the lid on the box.
7. Carry the lid over to the box and put it on.
8. Close the box with the lid.
9. Place the lid on top of the box.
10. Put the cover on the box.

Procedures:
1. Move above the lid, lower and grip its handle, lift it, carry it over the box, then put it down.
2. Grasp the lid's handle from above, raise it, and place it on the box.
3. Get over the lid, go down and close the gripper on the handle, move above the box, and lower.
4. Pick up the lid by its handle, bring it over the box, and set it on.
5. Grip the lid, lift it, go to the box, and put the lid on top.

### peg-insert-side-v3

Descriptions:
1. Insert the peg into the hole in the side of the box.
2. Pick up the peg and push it into the hole in the box's side.
3. Grab the stick and slide it sideways into the hole of the box.
4. Put the peg into the side hole of the box.
5. Lift the peg and insert it into the opening on the side of the box.
6. Take the peg from the table and fit it into the hole in the box.
7. Grasp the rod and push it into the side of the box through its hole.
8. Insert the peg into the box's side hole.
9. Pick up the peg and slide it into the hole.
10. Grab the peg and fit it into the opening in the box.

Procedures:
1. Move above the peg, lower and grasp it, then carry it level with the hole and push it in.
2. Get over the peg, go down and grip it, then bring it to the hole and slide it in.
3. Grasp the peg from above, move it in line with the hole, then insert it.
4. Go over the peg, lower, close the fingers, and push the peg into the hole.
5. Pick up the peg, align it with the hole, and slide it in.

### hand-insert-v3

Descriptions:
1. Put the object into the hole in the table.
2. Pick up the object and drop it into the hole.
3. Move the object on the table into the hole.
4. Grab the object and insert it into the opening in the table.
5. Carry the object to the hole and put it in.
6. Place the object in the hole.
7. Lift the object and lower it into the hole in the table.
8. Insert the object into the hole.
9. Take the object to the opening and put it in.
10. Put the block into the hole in the tabletop.

Procedures:
1. Move above the object, lower and grasp it, carry it over the hole, then lower it in.
2. Grasp the object from above, bring it above the hole, and put it down into it.
3. Get over the object, go down and grip it, move over the hole, and lower.
4. Pick up the object, carry it to the hole, and lower it in.
5. Grip the object, move it above the hole, and drop it in.

### button-press-v3

Descriptions:
1. Press the button on the front of the box.
2. Push the button in, from the robot's side.
3. Press the button that faces the robot.
4. Push the button on the box's front face.
5. Press in the button, moving away from the robot.
6. The box has a button facing you. Press it.
7. Push the button forward until it is pressed.
8. Press the button.
9. Push in the button on the box.
10. Press the front button.

Procedures:
1. Keep the gripper open, line up with the button, then push forward into it.
2. Get in front of the button at its height, then move straight toward it.
3. Align with the button from the robot's side, then press forward.
4. Move to the button's height in front of it and push it in.
5. Face the button, then push forward until it is pressed.

### button-press-topdown-v3

Descriptions:
1. Press the button on top of the box.
2. Push the button down from above.
3. Press down on the button on the box.
4. The box has a button on top. Push it down.
5. Press the button on the box straight down.
6. Push the knob on the box down until it clicks.
7. Press the button from above with the gripper.
8. Press the top button.
9. Push the button down.
10. Press down on the button until it is pushed in.

Procedures:
1. Move above the button, then push straight down.
2. Go over the button and press down.
3. Line up above the button and lower onto it.
4. Get above the button and push down.
5. Move over the button and press it down.

### button-press-wall-v3

Descriptions:
1. Press the button behind the wall.
2. Reach over the wall and push the button on the box.
3. Go over the wall and press the front button.
4. The button is behind a wall. Press it.
5. Get past the wall and push the button in.
6. Press the box's button, going around the wall.
7. Push the button that sits behind the wall.
8. Clear the wall and press the button.
9. Press the button beyond the wall.
10. Reach the button behind the wall and push it in.

Procedures:
1. Go up high, move over the wall, come down in front of the button, then push it with the gripper open.
2. Line up with the button, rise over the wall, drop to the button's height, and press forward.
3. Clear the wall from above, get in front of the button, and push it in.
4. Move high over the wall to the button, lower, open the fingers, and press forward.
5. Rise above the wall, move beyond it to the button, and push the button in.

### button-press-topdown-wall-v3

Descriptions:
1. Press the button on top of the box behind the wall.
2. Push the top button down; there is a wall in front of it.
3. Reach over the wall and press the button down from above.
4. Press down on the button behind the wall.
5. The box behind the wall has a button on top. Push it down.
6. Press the top button, past the wall.
7. Go above the button behind the wall and press it down.
8. Push down the button beyond the wall.
9. Press the button on the box from above.
10. Push the top button in, clearing the wall.

Procedures:
1. Keep the gripper open, move above the button past the wall, then push straight down.
2. Go over the wall to above the button and press down.
3. Line up above the button behind the wall and lower onto it.
4. Get above the button, clearing the wall, and push down.
5. Move over the button beyond the wall and press it down.

### door-open-v3

Descriptions:
1. Open the cabinet door by pulling its handle.
2. Swing the cabinet door open.
3. Pull the handle of the cabinet so that the door opens.
4. The cabinet has a closed door. Open it.
5. Open the safe's door by dragging its handle sideways.
6. Pull the door open until it reaches the goal marker.
7. Get the door of the cabinet to swing open.
8. Open the door with its handle.
9. Swing the door open using its handle.
10. Pull the door of the cabinet open.

Procedures:
1. Keep the gripper closed. Move above and beside the handle, lower to it, then pull the door open.
2. With the fingers closed, go to the handle from above, drop down to it, and drag it to open the door.
3. Close the gripper, get next to the handle, lower to handle height, and pull the door open.
4. Keep the gripper closed, reach the handle from above, then pull it sideways.
5. Go above the handle with a closed gripper, come down to it, and pull the door open.

### door-lock-v3

Descriptions:
1. Lock the door by turning the lock knob.
2. Turn the lock on the cabinet door to lock it.
3. Push the lock knob on the door down to lock it.
4. Lock the cabinet door.
5. Turn the knob on the door to the locked position.
6. Rotate the door's lock so the door is locked.
7. Press the lock lever on the door down.
8. Lock the door using its knob.
9. Turn the lock knob to lock the door.
10. Set the door's lock.

Procedures:
1. Keep the gripper open, move above the lock knob, lower to it, then push it sideways and down.
2. Go over the lock, drop to it, and press it down to the side.
3. Get above the knob, lower onto it, and turn it by pushing sideways and down.
4. Move to the lock from above and push it down sideways.
5. Reach the knob from above, then press it down and aside.

### door-unlock-v3

Descriptions:
1. Unlock the door by turning the lock knob.
2. Turn the lock on the cabinet door to unlock it.
3. Push the lock knob on the door sideways to unlock it.
4. Unlock the cabinet door.
5. Turn the knob on the door to the unlocked position.
6. Rotate the door's lock so the door is unlocked.
7. Release the lock on the door.
8. Unlock the door using its knob.
9. Turn the lock knob to unlock the door.
10. Open the door's lock.

Procedures:
1. Keep the gripper closed, go to the lock knob, then push it sideways.
2. With the fingers closed, move to the knob and push it to the side.
3. Close the gripper, reach the lock, and turn it by pushing sideways.
4. Get to the knob with a closed gripper and push it aside.
5. Keep the gripper closed, move onto the lock, and push it sideways.

### drawer-open-v3

Descriptions:
1. Open the drawer by pulling its handle toward the robot.
2. Pull the drawer out of the cabinet.
3. Slide the drawer open with its handle.
4. The drawer is closed. Pull it open.
5. Pull the handle so the drawer comes out toward the robot.
6. Open the box's drawer.
7. Draw the drawer out by its handle.
8. Open the drawer.
9. Pull the drawer open.
10. Pull the handle to open the drawer.

Procedures:
1. Keep the gripper open, move high above the handle, lower to it, then pull toward the robot.
2. Go above the handle with an open gripper, come down to it, and pull the drawer out.
3. With the fingers open, get over the handle, drop down, and pull it toward the robot.
4. Rise above the handle, lower onto it, and draw the drawer out.
5. Keep the gripper open, go down to the handle from above, and pull back.

### window-open-v3

Descriptions:
1. Open the window: slide the pane sideways.
2. Slide the window open by its handle.
3. Push the window's handle sideways so that the window opens.
4. The window is closed. Slide it open.
5. Open the sliding window by moving its handle along the frame.
6. Slide the pane of the window aside to open it.
7. Move the window handle sideways until the window is open.
8. Slide the window open.
9. Open the window by sliding its handle.
10. Push the sliding window open along its frame.

Procedures:
1. Keep the gripper closed, move above the handle, lower to it, then push it sideways.
2. With the fingers closed, come down to the handle and slide it along the frame.
3. Close the gripper, get to the handle from above, and push it to the side.
4. Keep the gripper closed. Go over the handle, drop to it, and slide the window open.
5. Close the gripper, lower onto the handle, and move it sideways.

### window-close-v3

Descriptions:
1. Close the window: slide the pane back.
2. Slide the window shut by its handle.
3. Push the window's handle sideways so that the window closes.
4. The window is open. Slide it closed.
5. Close the sliding window by moving its handle along the frame.
6. Slide the pane of the window back to close it.
7. Move the window handle sideways until the window is shut.
8. Slide the window closed.
9. Close the window by sliding its handle.
10. Push the sliding window shut along its frame.

Procedures:
1. Keep the gripper closed, move above the handle, lower to it, then push it back sideways.
2. With the fingers closed, come down beside the handle and slide it the other way along the frame.
3. Close the gripper, get to the handle from above, and push it back to close the window.
4. Keep the gripper closed. Go over the handle, drop to it, and slide the window shut.
5. Close the gripper, lower beside the handle, and push it sideways to close.

### handle-press-v3

Descriptions:
1. Press the handle down.
2. Push the handle on the box all the way down.
3. Press down on the lever handle.
4. The handle sticks up. Push it down.
5. Press the handle of the box downward.
6. Push down on the handle until it stops.
7. Lower the handle by pressing on it.
8. Press the handle down from above.
9. Push the handle down.
10. Press down the handle.

Procedures:
1. Keep the gripper open, move above the handle, then push straight down.
2. Go high over the handle and press it down.
3. Line up above the handle and lower onto it until it is down.
4. Get above the handle and push down.
5. Move over the handle and press it all the way down.

### handle-press-side-v3

Descriptions:
1. Press down the handle on the side of the box.
2. Push the side handle of the box down.
3. Press down on the handle sticking out of the box's side.
4. The box has a handle on its side. Push it down.
5. Lower the side handle by pressing on it.
6. Press the side handle all the way down.
7. Push down on the handle at the side of the box.
8. Press the side handle down.
9. Push the handle on the side down.
10. Press down the box's side handle.

Procedures:
1. Keep the gripper closed, move high above the side handle, then push straight down.
2. With the fingers closed, go over the side handle and press it down.
3. Close the gripper, line up above the side handle, and lower onto it.
4. Get above the side handle with a closed gripper and push down.
5. Keep the gripper closed, move over the side handle, and press it down.

### handle-pull-v3

Descriptions:
1. Pull the handle up.
2. Lift the handle on the box all the way up.
3. Pull up on the lever handle.
4. The handle is down. Pull it up.
5. Raise the handle of the box.
6. Pull the handle upward until it stops.
7. Lift the handle by pulling it up.
8. Pull up the handle.
9. Raise the handle.
10. Lift the handle up.

Procedures:
1. Keep the gripper closed, move under the handle's end, then pull straight up.
2. With the fingers closed, get just below the handle and lift it up.
3. Close the gripper, go to the handle at its height, and raise it.
4. Get under the handle with a closed gripper and push up.
5. Keep the gripper closed, reach below the handle, and lift.

### plate-slide-v3

Descriptions:
1. Slide the plate into the goal cage.
2. Push the plate across the table into the net.
3. Slide the flat plate away from the robot into the goal.
4. Get the plate into the cage by sliding it.
5. Push the plate along the table into the goal frame.
6. Move the plate into the net, sliding it on the table.
7. Shove the plate into the cage.
8. Slide the plate into the goal.
9. Push the plate into the net.
10. Slide the disk across the table into the cage.

Procedures:
1. Keep the gripper open, move behind the plate on the robot's side, lower, then push it forward into the cage.
2. Get to the near side of the plate, drop down, and slide it away into the net.
3. Go above the side of the plate nearest the robot, lower, and push it forward.
4. Lower behind the plate and push it away from the robot into the goal.
5. Get behind the plate, go down, and slide it forward into the cage.

### plate-slide-back-v3

Descriptions:
1. Slide the plate out of the cage back toward the robot.
2. Pull the plate out of the net to the goal.
3. Drag the plate back out of the goal cage.
4. Get the plate out of the cage by sliding it back.
5. Pull the plate along the table away from the net.
6. Move the plate out of the cage toward the robot.
7. Slide the plate back out of the goal frame.
8. Pull the plate out of the cage.
9. Slide the plate back to the goal.
10. Drag the disk out of the net.

Procedures:
1. Keep the gripper open, move to the robot's side of the plate, lower, then pull it back toward the robot.
2. Get over the near side of the plate, drop down, and drag it back out of the cage.
3. Go above the plate's near edge, lower, and pull it toward the robot, then to the goal.
4. Lower at the near side of the plate and pull it back out of the net.
5. Reach the plate's near edge from above and drag it back to the goal.

### plate-slide-side-v3

Descriptions:
1. Slide the plate sideways into the goal cage.
2. Push the plate to the side into the net.
3. Slide the flat plate sideways along the table into the goal.
4. Get the plate into the cage at the side by sliding it.
5. Push the plate sideways into the goal frame.
6. Move the plate into the net on its side.
7. Shove the plate sideways into the cage.
8. Slide the plate into the goal at the side.
9. Push the plate sideways into the net.
10. Slide the disk sideways into the cage.

Procedures:
1. Keep the gripper closed, move beside the plate, lower, then push it sideways into the cage.
2. With the fingers closed, get to the plate's side, drop down, and slide it into the net.
3. Close the gripper, go above the far side of the plate, lower, and push it sideways.
4. Lower beside the plate with a closed gripper and push it to the side into the goal.
5. Keep the gripper closed, get next to the plate, go down, and slide it sideways.

### lever-pull-v3

Descriptions:
1. Pull the lever up: lift the lever arm on the stand so that it points upward.
2. Raise the lever on the post until it swings up.
3. The lever sticks out from the stand with a ball at its end. Rotate it upward.
4. Lift the lever up.
5. Swing the lever arm upward.
6. Pull the lever on the stand up.
7. Raise the lever until it points up.
8. Pull the lever up.
9. Lift the lever arm.
10. Turn the lever upward.

Procedures:
1. Keep the gripper closed. Move to just below the lever's end, rise up to it, then push forward and up.
2. With the fingers closed, go under the end of the lever and lift it.
3. Close the gripper, get below the lever, come up to it, and push it upward.
4. Keep the gripper closed, reach under the lever's end, and raise it.
5. Go below the lever with a closed gripper and lift it up and away.

### dial-turn-v3

Descriptions:
1. Turn the dial.
2. Rotate the knob on the table.
3. Spin the dial around.
4. Turn the round knob.
5. Rotate the dial with the gripper.
6. Twist the dial on the table.
7. Turn the knob to a new position.
8. Rotate the dial.
9. Spin the knob.
10. Turn the dial knob.

Procedures:
1. Keep the gripper closed, move above the edge of the dial, lower, then push sideways to turn it.
2. With the fingers closed, go over the side of the knob, drop down, and sweep sideways.
3. Close the gripper, get above the dial's edge, lower, and push across it.
4. Keep the gripper closed, come down at the edge of the dial, and move sideways.
5. Lower onto the dial's edge with a closed gripper and push it around.

### stick-push-v3

Descriptions:
1. Use the stick to push the container to the goal.
2. Pick up the stick and push the thermos with it.
3. Grab the stick and shove the container along the table.
4. Push the container to the goal using the stick.
5. Take the stick and use it to move the thermos away.
6. With the stick, push the container forward.
7. Pick up the stick and push the container to the marker.
8. Push the thermos with the stick.
9. Use the stick as a tool to push the container.
10. Grab the stick and push the container with it.

Procedures:
1. Move above the stick, lower and grasp it, bring it in line with the container, then push the container to the goal.
2. Grasp the stick from above, line it up with the container, and push the container away.
3. Get over the stick, go down and grip it, move to the container, and shove it to the goal.
4. Pick up the stick, align it with the container, then drive the container forward.
5. Grip the stick, bring it to the container, and push the container with it.

## Test, T-comp: seen scene, motion seen elsewhere (4)

### drawer-close-v3

Descriptions:
1. Close the drawer: push the open drawer shut.
2. Push the drawer back into the cabinet.
3. Close the drawer.

Procedures:
1. Keep the gripper closed, get in front of the handle, and push the drawer in.
2. With the fingers closed, come to the handle from above and push the drawer shut.

### door-close-v3

Descriptions:
1. Close the open door of the cabinet.
2. Swing the cabinet's door shut.
3. Close the cabinet door.

Procedures:
1. Keep the gripper closed. Get beside the open door from above, lower to it, then push it shut.
2. With the fingers closed, come down next to the door and push it closed.

### handle-pull-side-v3

Descriptions:
1. Pull up the handle on the side of the box.
2. Lift the side handle of the box.
3. Pull the side handle up.

Procedures:
1. Move above the side handle, lower and grasp it, then pull straight up.
2. Grasp the side handle from above and lift it up.

### plate-slide-back-side-v3

Descriptions:
1. Slide the plate sideways out of the goal cage.
2. Pull the plate out of the net to the side.
3. Slide the plate out of the cage sideways.

Procedures:
1. Keep the gripper closed, move beside the plate, lower, then push it sideways out of the cage.
2. With the fingers closed, get to the plate's side and slide it out of the net sideways.

## Test, T-obj: scenes never seen (6)

### faucet-open-v3

Descriptions:
1. Open the faucet: turn its handle.
2. Turn the tap handle to open the faucet.
3. Open the faucet.

Procedures:
1. Keep the gripper closed, move above one side of the handle, lower, then push it sideways to turn it.
2. With the fingers closed, come down beside the faucet handle and push it around.

### faucet-close-v3

Descriptions:
1. Close the faucet: turn its handle back.
2. Turn the tap handle to close the faucet.
3. Close the faucet.

Procedures:
1. Keep the gripper closed, move above the other side of the handle, lower, then push it sideways to turn it back.
2. With the fingers closed, come down beside the faucet handle and push it the other way.

### coffee-button-v3

Descriptions:
1. Press the button on the coffee machine.
2. Push the coffee machine's button.
3. Press the coffee button.

Procedures:
1. Keep the gripper open, line up with the button on the machine, then push forward into it.
2. Get in front of the machine's button at its height and press forward.

### coffee-push-v3

Descriptions:
1. Push the mug under the coffee machine.
2. Slide the cup to the coffee machine's spout.
3. Move the mug to the coffee machine.

Procedures:
1. Move above the mug, lower and grasp it, then carry it low to the machine's spout.
2. Grasp the mug from above and slide it under the machine.

### coffee-pull-v3

Descriptions:
1. Pull the mug out from the coffee machine.
2. Slide the cup away from the coffee machine toward the robot.
3. Move the mug away from the coffee machine.

Procedures:
1. Move above the mug, lower and grasp it, then pull it away from the machine to the goal.
2. Grasp the mug from above and bring it back toward the robot.

### stick-pull-v3

Descriptions:
1. Use the stick to pull the container toward the goal.
2. Pick up the stick, hook the thermos and drag it.
3. Pull the container with the stick.

Procedures:
1. Move above the stick, lower and grasp it, hook it into the container's handle, then pull the container to the goal.
2. Grasp the stick, bring it to the container, hook it, and drag the container.
