# requires opencv 
#   pip insytall opencv-python
# and mediapipe
#  pip install mediapipe
from controller import Robot
from naomotion import NaoMotion
import cv2
from pose import Pose

robot = Robot()
timestep = int(robot.getBasicTimeStep())
motion = NaoMotion(robot)

# parameters
min_detection_confidence = 0.5
min_tracking_confidence = 0.5
model_complexity = 1
video_source = 0  # Default webcam
# end parameters

pose = Pose(min_detection_confidence, min_tracking_confidence, 
            model_complexity, video_source)

speed=1
while robot.step(timestep) != -1:
    # this code just show the different moves
    motion.forward(speed)
    motion.backward(speed)
    motion.turn("left", speed)
    motion.turn("right", speed)
    motion.sidestep("left", speed)
    motion.sidestep("right", speed)
    speed = (speed % 3) + 1 
    # here is where the model can be plugged
    pose_vec = pose.getPose()
    # a processed version of vec should go as input of the model
    print(pose_vec)