# From Python
# It requires OpenCV installed for Python
from __future__ import print_function, division
import sys
import cv2
import numpy as np
import os
import pickle
from sys import platform
import vtk
import vtktools
from scipy.spatial import distance
from scipy.optimize import linear_sum_assignment




colors =[(1.0,  0,  0),
         (0  ,1.0,  0),
         (0  ,0  ,1.0),
         (0  ,1.0,1.0),
         (1.0,1.0,  0)]

dir_path = os.path.dirname(os.path.realpath(__file__))

# https://stackoverflow.com/questions/19448078/python-opencv-access-webcam-maximum-resolution
def set_res(cap, x,y):
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(x))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(y))
    return int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

def draw_axis(img, r, t, K, d):
    size = 0.1
    axis = np.float32([[size,0,0], [0,size,0], [0,0,size],[0,0,0]]).reshape(-1,3)
    imgpts, jac = cv2.projectPoints(axis, r, t, K, d)
    corner = tuple(imgpts[3].ravel())
    img = cv2.line(img, corner, tuple(imgpts[2].ravel()), (255,0,0), 5) # X
    img = cv2.line(img, corner, tuple(imgpts[0].ravel()), (0,0,255), 5) # Z
    img = cv2.line(img, corner, tuple(imgpts[1].ravel()), (0,255,0), 5) # Y
    return img

class Organs(object):
    def __init__(self, camMatrix, imgDims, n=5):
        # vtkWarpLens
        filename = "FaceTest.stl"
 
        reader = vtk.vtkSTLReader()
        reader.SetFileName(filename)

        mapper = vtk.vtkPolyDataMapper()
        if vtk.VTK_MAJOR_VERSION <= 5:
            mapper.SetInput(reader.GetOutput())
        else:
            mapper.SetInputConnection(reader.GetOutputPort())

        # Create a rendering window and renderer
        ren = vtk.vtkRenderer()
        self.renWin = vtk.vtkRenderWindow()
        self.renWin.SetOffScreenRendering(1)
        self.renWin.SetAlphaBitPlanes(1)  # Enable usage of alpha channel
        self.renWin.AddRenderer(ren)

        self.renWin.SetSize(imgDims[0],imgDims[1])

        # Assume camera matrix has no offset
        dims = (camMatrix[0,2] * 2, camMatrix[1,2]*2)

        camera = vtktools.cameraFromMatrix(camMatrix,dims, (imgDims[0],imgDims[1]))
        ren.SetActiveCamera(camera)
        # Set camera to look forward from center

        self.w2if = vtk.vtkWindowToImageFilter()
        self.w2if.SetInput(self.renWin)
        self.w2if.SetInputBufferTypeToRGBA()

        self.actors = [vtk.vtkActor() for i in range(n)];
        for i in range(n):
            self.actors[i].SetMapper(mapper)
            self.actors[i].GetProperty().SetColor(colors[i])
            ren.AddActor(self.actors[i])

        self.w2if.ReadFrontBufferOff()

    def render(self):
        self.renWin.Render()
        # self.renWin.WaitForCompletion()
        self.w2if.Modified()
        self.w2if.Update()
        # self.w2if.WaitForCompletion()
        return vtktools.vtkImageToNumpy(self.w2if.GetOutput())

    def move_organ(self, organ_id, r, t):
        transform = vtk.vtkTransform()
        angle = np.linalg.norm(r);
        transform.RotateWXYZ(angle * 180 / np.pi, r/angle)
        rot = transform.GetOrientation()
        self.actors[organ_id].SetPosition(t)
        self.actors[organ_id].SetOrientation(rot)

def order_points(last_ts, last_rs, new_ts, new_rs):
    retval_t = last_ts
    retval_r = last_rs

    diffs = distance.cdist(last_ts, new_ts)
    rows, cols = linear_sum_assignment(diffs)
    print(rows, cols, new_ts[:,0].T)
    retval_t[rows,:] = new_ts[cols,:]
    retval_r[rows,:] = new_rs[cols,:]

    retval_t[np.logical_and(retval_t == -1000, last_ts != -1000)] = last_ts[np.logical_and(retval_t == -1000, last_ts != -1000)]

    print(retval_t[:,0].T)
    return retval_t, retval_r

def main():
    twoVideos = True
    # twoVideos = False
    cap = cv2.VideoCapture('video.webm')
    res = set_res(cap, 2560//2, 960//2)
    ret, frame = cap.read()
    
    if twoVideos:
        res = (res[0] // 2, res[1])
        frame = frame[:,0:frame.shape[1]//2].copy()

    with open(os.path.join(dir_path,'camera_calib','calib.pkl'), 'rb') as f:
        calib = pickle.load(f)
        K, d = [calib[key] for key in ['M1', 'dist1']]
        ratios = (res[0] / (K[0,2] * 2), res[1] / (K[1,2] * 2))
        K[[0, 0], [0, 2]] = K[[0, 0], [0, 2]] * ratios[0]
        K[[1, 1], [2, 1]] = K[[1, 1], [2, 1]] * ratios[1]
    
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
        import openpose.pyopenpose as op
    except ImportError as e:
        print('Error: OpenPose library could not be found. Did you enable `BUILD_PYTHON` in CMake and have this Python script in the right folder?')
        raise e

    # Custom Params (refer to include/openpose/flags.hpp for more parameters)
    params = dict()
    params["model_folder"] = os.path.join(openpose_dir, "models","")
    # params["face"] = True
    params["model_pose"] = "BODY_25"
    # params["body_disable"] = True
    params["net_resolution"] = "160x160"
    # params["face_net_resolution"] = "160x160"

    # Starting OpenPose
    opWrapper = op.WrapperPython()
    opWrapper.configure(params)
    opWrapper.start()

    datum = op.Datum()

    stl_points = np.array([[ 0.000000, 0.000000, 0.000000], # nose
                           [ 0.025817,-0.035204, 0.022871], # left eye
                           [-0.025817,-0.035204, 0.022871], # right eye
                           [ 0.062054,-0.103087, 0.010866], # left ear
                           [-0.062053,-0.103087, 0.010866]])# right ear

    # Initial guess for head transform
    r_guess = np.array([0,1,-1]).reshape(3,1)
    r_guess = r_guess / np.linalg.norm(r_guess) * np.pi
    t_guess = np.array([0.0,0,0.5]).reshape(3,1)

    organs = Organs(K, res)

    counter = 0
    skip = 5

    large_n = -1000


    last_ts = np.array([large_n, large_n, large_n] * len(organs.actors)).reshape(-1, 3)
    last_rs = np.array([large_n, large_n, large_n] * len(organs.actors)).reshape(-1, 3)

    while 1:
        # Capture frame-by-frame
        ret, frame = cap.read()
        counter = counter + 1
        if counter % (skip+1) != 0:
            continue

        if twoVideos:
            frame = frame[:,0:frame.shape[1]//2].copy()

        # Process Image
        # datum.cvInputData = cv2.resize(cv2.undistort(frame, K, d, K), (320,240))
        datum.cvInputData = frame
        opWrapper.emplaceAndPop([datum])
        # print(dir(datum))
        # quit()

        img = cv2.resize(datum.cvOutputData, res)
        img = img // 2
        # print(datum.poseKeypoints.shape)

        # print(stl_points[1])
        # print(K[0,:])

        ts = np.ones((len(organs.actors), 3)) * large_n
        rs = np.ones((len(organs.actors), 3)) * large_n

        if len(datum.poseKeypoints.shape):
            for i, pose in enumerate(list(datum.poseKeypoints)):
                head = pose[[0, 15, 16, 17, 18]].copy()
                order = np.argsort(-head[:,2])
                if np.sum(head[:, 2] > 0.05) > 3:
                    head_filt = head[order[0:4]][:,0:2].astype(np.float32).reshape((4,1,2))
                    stl_filt = stl_points[order[0:4]].astype(np.float32).reshape((4,1,3))
                    rt, r, t = cv2.solvePnP(stl_filt, head_filt, K, np.zeros(4), r_guess.copy(), t_guess.copy(), useExtrinsicGuess = True)
                    verts = cv2.projectPoints(stl_filt, r, t, K, np.zeros(4))[0].reshape(-1, 2)
                    # for v in verts:
                    #     cv2.circle(img, tuple(v), 5, (255,255,0))
                    if(np.sum(np.abs(verts - head_filt.reshape(4,2))) < 30):
                        ts[i] = t.T
                        rs[i] = r.T

        ts, rs = order_points(last_ts, last_rs, ts, rs)
        last_ts = ts
        last_rs = rs
        for i, t, r in zip(range(len(ts)), ts, rs):
            organs.move_organ(i, r.T, t.T)
            # img = draw_axis(img, r, t, K, np.zeros(4))

        # if len(datum.poseKeypoints.shape):
        #     for i, pose in enumerate(list(datum.poseKeypoints)):
        #         head = pose[[0, 15, 16, 17, 18]].copy()
        #         for v in head[:,0:2]:
        #                 cv2.circle(img, tuple(v), 5, tuple(np.multiply(colors[i], 255)),-1)
        #         # print(head)
        #         order = np.argsort(-head[:,2])
        #         if np.sum(head[:, 2] > 0.05) > 3:
        #             head_filt = head[order[0:4]][:,0:2].astype(np.float32).reshape((4,1,2))
        #             stl_filt = stl_points[order[0:4]].astype(np.float32).reshape((4,1,3))
        #             rt, r, t = cv2.solvePnP(stl_filt, head_filt, K, np.zeros(4), r_guess.copy(), t_guess.copy(), useExtrinsicGuess = True)
        #             verts = cv2.projectPoints(stl_filt, r, t, K, np.zeros(4))[0].reshape(-1, 2)
        #             for v in verts:
        #                 cv2.circle(img, tuple(v), 5, (255,255,0))
        #             print(np.sum(np.abs(verts - head_filt.reshape(4,2))))
        #             if(np.sum(np.abs(verts - head_filt.reshape(4,2))) < 30):
        #                 organs.move_organ(i, r, t)
        #                 img = draw_axis(img, r, t, K, np.zeros(4))
        # quit()

        render = organs.render()
        img = np.multiply(img.astype(float), 1 - render[:,:,3,np.newaxis] / 255).astype(np.uint8);
        img = img + np.multiply(render[:,:,0:3].astype(float), render[:,:,3,np.newaxis] / 255).astype(np.uint8);
        cv2.imshow('frame',img)
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()

# params = dict()
# params["logging_level"] = 3
# params["output_resolution"] = "-1x-1"
# params["net_resolution"] = "320x176"#"-1x368"
# params["model_pose"] = "BODY_25"
# params["alpha_pose"] = 0.6
# params["face"] = True
# params["scale_gap"] = 0.3
# params["scale_number"] = 1
# params["render_threshold"] = 0.05
# # If GPU version is built, and multiple GPUs are available, set the ID here
# params["num_gpu_start"] = 0
# params["disable_blending"] = False
# # Ensure you point to the correct path where models are located
# params["default_model_folder"] = os.path.join(openpose_dir, "models","")
# # Construct OpenPose object allocates GPU memory
# op = openpose.OpenPose(params)
# dir_path = os.path.dirname(os.path.realpath(__file__))
# print(dir_path)
# with open(os.path.join(dir_path,'camera_calib','calib.pkl'), 'rb') as f:
#     calib = pickle.load(f)
#     K, d = [calib[key] for key in ['M1', 'dist1']]
# print(K, d)
# cap = cv2.VideoCapture(0)

# twoVideos = True

# while 1:
#     # Capture frame-by-frame
#     ret, frame = cap.read()

#     if twoVideos:
#         frame = frame[:,0:frame.shape[1]//2].copy()

#     keypoints, frame = op.forward(frame, True)
#     # keypoints = op.forward(frame, False)

#     print(keypoints)

#     face = {'nose':0, }

#     # cv2.solvePnP()

#     # Our operations on the frame come here
#     # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     # frame = cv2.circle(frame, tuple(keypoints[0,0,0:2]), 3, (255,255,0), -1)

#     # Display the resulting frame
#     cv2.imshow('frame',frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroAllWindows()

#     # # Read new image
#     # img = cv2.imread("/home/biomed/openpose/examples/media/COCO_val2014_000000000192.jpg")
#     # # Output keypoints and the image with the human skeleton blended on it
#     # keypoints, output_image = openpose.forward(img, True)
#     # # Print the human pose keypoints, i.e., a [#people x #keypoints x 3]-dimensional numpy object with the keypoints of all the people on that image
#     # print(keypoints)
#     # # Display the image
#     # cv2.imshow("output", output_image)
#     # cv2.waitKey(15)
