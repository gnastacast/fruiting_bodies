# From Python
# It requires OpenCV installed for Python
from __future__ import print_function, division
import sys
import cv2
import os
import pickle
from sys import platform

# Remember to add your installation path here
# Option a
openpose_dir = '/home/biomed/openpose'
if platform == "win32": sys.path.append(os.path.join(openpose_dir, 'build', 'python'));
sys.path.append(os.path.join(openpose_dir, 'build', 'python'));
# Option b
# If you run `make install` (default path is `/usr/local/python` for Ubuntu), you can also access the OpenPose/python module from there. This will install OpenPose and the python library at your desired installation path. Ensure that this is in your python path in order to use it.
# sys.path.append('/usr/local/python')

# Parameters for OpenPose. Take a look at C++ OpenPose example for meaning of components. Ensure all below are filled
try:
    from openpose import *
except:
    raise Exception('Error: OpenPose library could not be found. Did you enable `BUILD_PYTHON` in CMake and have this Python script in the right folder?')
params = dict()
params["logging_level"] = 3
params["output_resolution"] = "-1x-1"
params["net_resolution"] = "320x176"#"-1x368"
params["model_pose"] = "BODY_25"
params["alpha_pose"] = 0.6
params["scale_gap"] = 0.3
params["scale_number"] = 1
params["render_threshold"] = 0.05
# If GPU version is built, and multiple GPUs are available, set the ID here
params["num_gpu_start"] = 0
params["disable_blending"] = False
# Ensure you point to the correct path where models are located
params["default_model_folder"] = os.path.join(openpose_dir, "models","")
# Construct OpenPose object allocates GPU memory
openpose = OpenPose(params)
dir_path = os.path.dirname(os.path.realpath(__file__))
print(dir_path)
with open(os.path.join(dir_path,'camera_calib','calib.pkl'), 'rb') as f:
    calib = pickle.load(f)
    K, d = [calib[key] for key in ['M1', 'dist1']]
print(K, d)
cap = cv2.VideoCapture(0)

twoVideos = True

while 1:
    # Capture frame-by-frame
    ret, frame = cap.read()

    if twoVideos:
        frame = frame[:,0:frame.shape[1]//2].copy()

    keypoints, output_image = openpose.forward(frame, True)

    # Our operations on the frame come here
    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Display the resulting frame
    cv2.imshow('frame',output_image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroAllWindows()

    # # Read new image
    # img = cv2.imread("/home/biomed/openpose/examples/media/COCO_val2014_000000000192.jpg")
    # # Output keypoints and the image with the human skeleton blended on it
    # keypoints, output_image = openpose.forward(img, True)
    # # Print the human pose keypoints, i.e., a [#people x #keypoints x 3]-dimensional numpy object with the keypoints of all the people on that image
    # print(keypoints)
    # # Display the image
    # cv2.imshow("output", output_image)
    # cv2.waitKey(15)
