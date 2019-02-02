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
from iou_tracker import IouTracker
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(dir_path, 'randomcolor-py'))
import randomcolor
from tentacle import Tentacle, TentacleController, axis_angle_to_mat
from tentacle_vtk import vtkTentacle
from bullet_vtk_scene import Scene
import pybullet as p
import transformations as tf
import time
from multiprocessing import Process, Pipe, Queue
# Parameters for OpenPose. Take a look at C++ OpenPose example for meaning of components. Ensure all below are filled
try:
    # Windows Import
    if platform == "win32":
        _openpose_dir = 'I:\\Work\\Art\\openpose'
        sys.path.append(os.path.join(_openpose_dir, 'build', 'python'));
        # Change these variables to point to the correct folder (Release/x64 etc.) 
        sys.path.append(os.path.join(_openpose_dir, 'build', 'python', 'openpose','Release'))
        os.environ['PATH']  = os.environ['PATH'] + ';' + os.path.join(_openpose_dir, 'build', 'x64', 'Release') + ';' + os.path.join(_openpose_dir, 'build', 'bin') + ';'
        import pyopenpose as op
    else:
        _openpose_dir = '/home/biomed/openpose'
        sys.path.append(os.path.join(_openpose_dir, 'build', 'python'));
        import openpose.pyopenpose as op 
except ImportError as e:
    print('Error: OpenPose library could not be found. Did you enable `BUILD_PYTHON` in CMake and have this Python script in the right folder?')
    raise e

rand_color = randomcolor.RandomColor(42)

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
    def __init__(self, camMatrix, imgDims, scene, rot90, n=5):
        filename_head = "tubby_head.obj"
        reader_head = vtk.vtkOBJReader()
        reader_head.SetFileName(filename_head)
        reader_head.Update()

        filename_face = "tubby_face.obj"
        reader_face = vtk.vtkOBJReader()
        reader_face.SetFileName(filename_face)
        reader_face.Update()

        imgReader = vtk.vtkPNGReader()
        imgReader.SetFileName("tubby_COMBINED.png")
        imgReader.Update()

        self.texture = vtk.vtkTexture()
        self.texture.SetInputData(imgReader.GetOutput())

        self.texmapper = vtk.vtkPolyDataMapper()
        self.texmapper.SetInputConnection(reader_face.GetOutputPort())

        self.glslmapper = vtk.vtkPolyDataMapper()

        with open("perlin.glsl", "r") as file:
            shader_str = file.read()

        self.glslmapper.SetVertexShaderCode(
            "//VTK::System::Dec\n"  # always start with this line
            "attribute vec4 vertexMC;\n" +
            shader_str +
            # use the default normal decl as the mapper
            # will then provide the normalMatrix uniform
            # which we use later on
            "//VTK::Normal::Dec\n"
            "varying vec4 myVertexMC;\n"
            "varying vec3 myNormalMCVSOutput;\n"
            "uniform mat4 MCDCMatrix;\n"
            "void main () {\n"
            "  normalVCVSOutput = normalMatrix * normalMC;\n"
            # do something weird with the vertex positions
            # this will mess up your head if you keep
            # rotating and looking at it, very trippy
            # "  vec4 tmpPos = MCDCMatrix * vertexMC;\n"
            "  float disp = 0.00;\n"
            "  vec4 tmpPos = MCDCMatrix * vec4(vertexMC.x+normalMC.x*disp, vertexMC.y+normalMC.y*disp, vertexMC.z+normalMC.z*disp, 1.0);\n"
            # "  gl_Position = tmpPos*vec4(0.2+0.8*abs(tmpPos.x),0.2+0.8*abs(tmpPos.y),1.0,1.0);\n"
            "  gl_Position = tmpPos;\n"
            "  myVertexMC = vertexMC;\n"
            "}\n"
        );

        # self.glslmapper.AddShaderReplacement(
        #     vtk.vtkShader.Vertex,
        #     "//VTK::Normal::Dec", # replace the normal block
        #     True, # before the standard replacements
        #     "//VTK::Normal::Dec\n" # we still want the default
        #     "  varying vec3 myNormalMCVSOutput;\n" #but we add this
        #     "  out vec4 myVertexMC;\n",
        #     False # only do it once
        # )
        self.glslmapper.AddShaderReplacement(
            vtk.vtkShader.Vertex,
            "//VTK::Normal::Impl", # replace the normal block
            True, # before the standard replacements
            "//VTK::Normal::Impl\n" # we still want the default
            "  myNormalMCVSOutput = normalMC;\n" #but we add this
            "  myVertexMC = vertexMC;\n",
            False # only do it once
        )

        # Add the code to generate noise
        # These functions need to be defined outside of main. Use the System::Dec
        # to declare and implement
        self.glslmapper.AddShaderReplacement(
            vtk.vtkShader.Fragment,
            "//VTK::System::Dec", 
            False, # after the standard replacements
            "//VTK::System::Dec\n" + shader_str,
            False # only do it once
        )

        # now modify the fragment shader
        self.glslmapper.AddShaderReplacement(
            vtk.vtkShader.Fragment,  # in the fragment shader
            "//VTK::Normal::Dec", # replace the normal block
            True, # before the standard replacements
            "//VTK::Normal::Dec\n" # we still want the default
            "  varying vec3 myNormalMCVSOutput;\n" #but we add this
            "  varying vec4 myVertexMC;\n"
            "  uniform float k = 1.0;\n",
            False # only do it once
        )
        # # self.glslmapper.AddShaderReplacement(
        # #     vtk.vtkShader.Fragment,  # in the fragment shader
        # #     "//VTK::Normal::Impl", # replace the normal block
        # #     True, # before the standard replacements
        # #     "//VTK::Normal::Impl\n" # we still want the default calc
        # #     "  diffuseColor = abs(myNormalMCVSOutput);\n", #but we add this
        # #     False # only do it once
        # # )
        # self.glslmapper.AddShaderReplacement(
        #     vtk.vtkShader.Fragment,  # in the fragment shader
        #     "//VTK::Normal::Impl", # replace the normal block
        #     True, # before the standard replacements
        #     "//VTK::Normal::Impl\n" # we still want the default calc
        #     "  diffuseColor = abs(myNormalMCVSOutput);\n", #but we add this
        #     False # only do it once
        # )

        self.glslmapper.AddShaderReplacement(
            vtk.vtkShader.Fragment,  # in the fragment shader
            "//VTK::Light::Impl", # replace the light block
            False, # after the standard replacements
            "//VTK::Light::Impl\n" # we still want the default calc
            "#define pnoise(x) ((noise(x) + 1.0) / 2.0)\n"
            "  vec3 noisyColor;\n"
            "  noisyColor.r = noise(k * 100.0 * myVertexMC);\n"
            "  noisyColor.g = noise(k * 11.0 * myVertexMC);\n"
            "  noisyColor.b = noise(k * 12.0 * myVertexMC);\n"
            "  fragOutput0.rgb = opacity * vec3(ambientColor + diffuse);\n"
            "  fragOutput0.rgb = vec3(1, 1, 1) - fragOutput0.r * (noisyColor.b/10.0 + 0.9);"
            # "  fragOutput0.rgb = opacity * vec3(ambientColor + diffuse + specular);\n"
            "  fragOutput0.g = fragOutput0.g / 2.0 * (1.0 - noisyColor.r/5.0 + 0.8);"
            "  fragOutput0.b = fragOutput0.r / 2.0 * (1.0 - noisyColor.r/5.0 + 0.8);"
            "  fragOutput0.r = fragOutput0.b * (1.0 - noisyColor.r/5.0 + 0.8);"
            "  fragOutput0.rgb = fragOutput0.rgb * 0.8 + abs(noisyColor.r) * specular * 2;"
            "  fragOutput0.a = opacity;\n",
            False # only do it once
        );


        self.glslmapper.SetInputConnection(reader_head.GetOutputPort())

        # Get renderer
        self.ren = scene.ren

        self.camMatrix = camMatrix

        # Assume camera matrix has no offset
        dims = (self.camMatrix[0,2] * 2, self.camMatrix[1,2]*2)

        camera = vtktools.cameraFromMatrix(self.camMatrix, dims, (imgDims[0],imgDims[1]))
        self.ren.SetActiveCamera(camera)

        self.actors = {}
        self.poses = {}


        self.rot90 = rot90

        ############ Set up tentacles ################

        sphereRadius = 0.03
        # pybullet stuff        
        p.setPhysicsEngineParameter(numSolverIterations=10)
        p.setPhysicsEngineParameter(contactBreakingThreshold=0.001)
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING,0)
        if self.rot90:
            p.setGravity(-1,0,0)
        else:
            p.setGravity(0,-1,0)

        distance = 0.1
        n_tentacles = 8
        if self.rot90:
            self.centers = [(0, (distance * i - distance * n_tentacles / 2) ,-2) for i in range(n_tentacles)]
        else:
            self.centers = [(distance * i - distance * n_tentacles / 2, .5 ,-2) for i in range(n_tentacles)]

        self.tentacles = [Tentacle(7, c, sphereRadius) for c in self.centers]
        self.tentacleIndex = np.array([-1] * n_tentacles)
        self.vtk_tentacles = []
        self.controller = TentacleController(intensity=10)
        for tent_id, tentacle in enumerate(self.tentacles):
            self.controller.add_tentacle(tentacle)
            self.vtk_tentacles.append(vtkTentacle())
            self.ren.AddActor(self.vtk_tentacles[-1])
            if self.rot90:
                tentacle.reset(self.centers[tent_id],[ 0, -0.7071068, 0, 0.7071068 ])
            else:
                tentacle.reset(self.centers[tent_id],[ 0.7071068, 0, 0, 0.7071068 ])
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING,1)

    def getPose(self, keypoints):
        scale = 0.95
        stl_points = scale * np.array([[ 0.000000, 0.000000, 0.000000], # nose
                                       [ 0.025817,-0.035204, 0.022871], # left eye
                                       [-0.025817,-0.035204, 0.022871], # right eye
                                       [ 0.062054,-0.103087, 0.010866], # left ear
                                       [-0.062053,-0.103087, 0.010866]])# right ear

        if self.rot90:
            # Initial guess for head transform
            r_guess = np.reshape([ -1, -1, 1 ],(3,1))
            r_guess = r_guess / np.linalg.norm(r_guess) * np.pi * (2/3)
            t_guess = np.array([0.0,0,0.5]).reshape(3,1)
            stl_points = np.array((stl_points[0], stl_points[2], stl_points[1], stl_points[4], stl_points[3]))
        else:
            # Initial guess for head transform
            r_guess = np.array([0,1,-1]).reshape(3,1)
            r_guess = r_guess / np.linalg.norm(r_guess) * np.pi
            t_guess = np.array([0.0,0,0.5]).reshape(3,1)
        head = keypoints[[0, 15, 16, 17, 18]].copy()
        order = np.argsort(-head[:,2])

        if np.sum(head[:, 2] > 0.05) > 3:
            head_filt = head[order[0:4]][:,0:2].astype(np.float32).reshape((4,1,2))
            stl_filt = stl_points[order[0:4]].astype(np.float32).reshape((4,1,3))
            rt, r, t = cv2.solvePnP(stl_filt, head_filt, self.camMatrix, np.zeros(4), r_guess.copy(), t_guess.copy(), useExtrinsicGuess = True)
            verts = cv2.projectPoints(stl_filt, r, t, self.camMatrix, np.zeros(4))[0].reshape(-1, 2)
            # for v in verts:
            #     cv2.circle(img, tuple(v), 5, (255,255,0))
            if(np.sum(np.abs(verts - head_filt.reshape(4,2))) < 30):
                return t, r

        return None, None

    def addActor(self, track_id):
        # Check if we have any tentacles to spare
        if np.all(self.tentacleIndex != -1):
            return False

        self.actors[track_id] = [vtk.vtkActor(), vtk.vtkActor()]
        self.actors[track_id][0].GetProperty().SetAmbientColor(0.2, 0.2, 0.2)
        self.actors[track_id][0].GetProperty().SetDiffuseColor(1.0, 1.0, 1.0)
        self.actors[track_id][0].GetProperty().SetSpecularColor(1.0, 1.0, 1.0)
        self.actors[track_id][0].GetProperty().SetSpecular(0.5)
        self.actors[track_id][0].GetProperty().SetDiffuse(0.7)
        self.actors[track_id][0].GetProperty().SetAmbient(0.1)
        self.actors[track_id][0].GetProperty().SetSpecularPower(100.0)
        self.actors[track_id][0].GetProperty().SetOpacity(1.0)
        self.actors[track_id][0].GetProperty().BackfaceCullingOn()
        # self.actors[track_id].SetTexture(self.texture)\
        # self.actors[track_id].GetProperty().LightingOff()
        self.actors[track_id][0].SetMapper(self.glslmapper)
        color = rand_color.generate(hue="pink", luminosity="bright", format_="rgb")
        # self.actors[track_id].GetProperty().SetColor([float(i) / 255 for i in str.split(color[0][5:-1],',')])
        self.actors[track_id][1] = vtk.vtkActor()
        self.actors[track_id][1].SetMapper(self.texmapper)
        self.actors[track_id][1].SetTexture(self.texture)
        self.actors[track_id][1].GetProperty().SetAmbientColor(0.4, 0.2, 0.2)
        self.actors[track_id][1].GetProperty().SetDiffuseColor(1.0, 1.0, 1.0)
        self.actors[track_id][1].GetProperty().SetSpecularColor(1.0, 1.0, 1.0)
        # self.actors[track_id][1].GetProperty().SetSpecular(1.0)
        self.actors[track_id][1].GetProperty().SetDiffuse(1.0)
        self.actors[track_id][1].GetProperty().SetAmbient(0.3)
        self.actors[track_id][1].GetProperty().SetSpecularPower(100.0)
        self.actors[track_id][1].GetProperty().SetOpacity(1.0)
        self.actors[track_id][1].GetProperty().BackfaceCullingOn()
        self.ren.AddActor(self.actors[track_id][0])
        self.ren.AddActor(self.actors[track_id][1])
        self.tentacleIndex[np.argwhere(self.tentacleIndex == -1)[0]] = track_id
        return True


    def update(self, tracks, pose_keypoints):
        new_ids = [tr['id'] for tr in tracks]
        # Get rid of actors that are no longer tracked
        remove = [k for k in self.actors if k not in new_ids]
        for k in remove:
            self.ren.RemoveActor(self.actors[k][0])
            self.ren.RemoveActor(self.actors[k][1])
            self.tentacleIndex[self.tentacleIndex == k] = -1
            del self.actors[k]

        # Update actors
        for idx, key in enumerate(new_ids):
            pose_id = tracks[idx]['pose_id']
            if pose_id == -1:
                continue
            if key not in self.actors.keys():
                if not self.addActor(key):
                    continue
            t, r = self.getPose(pose_keypoints[pose_id])
            if t is not None:
                self.move_organ(key, r, t)

        self.controller.update(self.tentacleIndex != -1)

        for center, tentacle, vtk_tentacle, idx in zip(self.centers, self.tentacles, self.vtk_tentacles, self.tentacleIndex):
            if idx == -1:
                tentacle.reset(center,[ 0, -0.7071068, 0, 0.7071068 ])
                vtk_tentacle.update(tentacle)
                continue
            rotA = np.array(self.actors[idx][0].GetOrientationWXYZ())
            matA = tf.quaternion_matrix(tf.quaternion_about_axis((rotA[0] % 360) / 180 * np.pi, rotA[1:]))
            matB = tf.euler_matrix(0,0,np.pi/2)
            rot = np.dot(matA, matB)
            rot = tf.quaternion_from_matrix(rot)
            rot = [rot[1], rot[2], rot[3], rot[0]]
            pos = self.actors[idx][0].GetPosition() + np.dot(matA, np.array([[0],[-.1],[0.07], [1]]))[0:3].reshape(3,)
            if np.allclose(tentacle.get_position(), center):
                tentacle.reset(pos, rot)
            else:
                tentacle.move(pos, rot)
            vtk_tentacle.update(tentacle)

    def move_organ(self, organ_id, r, t):
        oldRot = self.actors[organ_id][0].GetOrientationWXYZ()
        if oldRot[0] > 1:
            oldR = np.array(oldRot[1:]) * (((oldRot[0] + 180)%360 - 180) / 180 * np.pi)
            diff = r.flatten() - oldR
            if np.linalg.norm(diff) < 50:
                r = oldR + diff / 2
        angle = np.linalg.norm(r)
        transform = vtk.vtkTransform()
        transform.RotateWXYZ(angle * 180 / np.pi, r/angle)
        rot = transform.GetOrientation()
        pos = np.array(self.actors[organ_id][0].GetPosition())
        if np.linalg.norm(t.flatten() - pos) < 0.2:
            self.actors[organ_id][0].SetPosition((t.flatten()+pos) * 0.5)
            self.actors[organ_id][1].SetPosition((t.flatten()+pos) * 0.5)
        else:
            self.actors[organ_id][0].SetPosition(t.flatten())
            self.actors[organ_id][1].SetPosition(t.flatten())
        self.actors[organ_id][0].SetOrientation(rot)
        self.actors[organ_id][1].SetOrientation(rot)

def openpose_update(in_q, out_q, q_conn, rot90):
    # Custom Params (refer to include/openpose/flags.hpp for more parameters)
    params = dict()
    params["model_folder"] = os.path.join(_openpose_dir, "models","")
    # params["face"] = True
    params["model_pose"] = "BODY_25"
    # params["body_disable"] = True
    # params["net_resolution"] = "320x320"
    params["render_pose"] = 0
    # params["face_net_resolution"] = "160x160"

    # Starting OpenPose
    opWrapper = op.WrapperPython()
    opWrapper.configure(params)
    opWrapper.start()

    datum = op.Datum()

    while True:
        try:
            q_conn.get(0.01)
            print("ALSKFJSADL:FKJSDL:KSAJDL:KASJD:LKAJ")
            opWrapper.stop()
            return
        except Exception:
            pass
        if in_q.empty():
            time.sleep(0.01)
        # Process Image
        frame = in_q.get()
        if rot90:
            datum.cvInputData = cv2.transpose(frame)
            opWrapper.emplaceAndPop([datum])
        else:
            datum.cvInputData = frame
            opWrapper.emplaceAndPop([datum])
        out_q.put([frame, datum.poseKeypoints.copy(), datum.poseScores.copy()])

class fakeDatum():
    def __init__(self):
        self.poseKeypoints = []
        self.poseScores = []

def main():
    cap = cv2.VideoCapture(0)
    # res = set_res(cap, 2560//2, 960//2)
    res = set_res(cap, 1280//2, 720//2)
    ret, frame = cap.read()

    with open(os.path.join(dir_path,'camera_calib','calib.pkl'), 'rb') as f:
        calib = pickle.load(f)
        K, d = [calib[key] for key in ['M1', 'dist1']]
        ratios = (res[0] / (K[0,2] * 2), res[1] / (K[1,2] * 2))
        K[[0, 0], [0, 2]] = K[[0, 0], [0, 2]] * ratios[0]
        K[[1, 1], [2, 1]] = K[[1, 1], [2, 1]] * ratios[1]

    rot90 = True

    window_name = 'frame'
    cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN) 
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    scene = Scene(res[1], res[0], window_name, fov=60)
    organs = Organs(K, res, scene, rot90)

    counter = 0
    skip = 1

    large_n = -1000.0
    tracker =  IouTracker()
    started = False
    frame_time  = time.time()
    in_queue =  Queue(1)
    out_queue = Queue(1)
    in_conn, out_conn = Pipe()
    openpose_process = Process(target=openpose_update, args=(in_queue, out_queue, out_conn, rot90))
    openpose_process.start()

    datum = fakeDatum()
    a, b ,tracks = [], [], []
    img = frame

    field = cv2.imread('field.png', cv2.IMREAD_UNCHANGED)

    while 1:
        # Capture frame-by-frame
        ret, frame = cap.read()
        counter = counter + 1
        # if counter % (skip+1) != 0:
        #     continue
        if not started:
            scene.start()
            started = True

        if not in_queue.full():
            in_queue.put(frame.copy())

        if not out_queue.empty():
            [img, a, b] = out_queue.get()
            if len(a.shape):
                datum.poseKeypoints, datum.poseScores = a, b
            else:
                datum.poseKeypoints, datum.poseScores = [], []

            msec = cap.get(cv2.CAP_PROP_POS_MSEC)
            cap_time = msec / 1000
            # print(cap_time)
            if len(datum.poseKeypoints) and rot90:
                keypoints = datum.poseKeypoints.copy()
                poseScores = datum.poseScores.copy()
                for i in range(len(keypoints)):
                    keypoints[i][:,0:2] = np.dstack( (keypoints[i][:,1],
                                                      keypoints[i][:,0]))
                datum.poseKeypoints = keypoints
                tracker.update(datum, cap_time)
            else:
                tracker.update(datum, cap_time)
            tracks = tracker.get_tracks()

        # # # font = cv2.FONT_HERSHEY_SIMPLEX
        # # # for t in tracks:
        # # #     i = int(t['id'])
        # # #     x, y = int(t['center'][0]), int(t['center'][1])
        # # #     cv2.putText(img, "%i %.3f \n %i"% (i, t['score'], t['pose_id']), (x,y), font, 0.8, (max(0, 255-100*i),max(0, 255-100*i),70*i))

        organs.update(tracks, datum.poseKeypoints)

        ### Render scene and overlay it
        render = scene.render()
        render, alpha = render[:,:,[2,1,0]], render[:,:,3,np.newaxis] > 50
        if (img.shape[0] > img.shape[1]):
            img = cv2.transpose(img)
        # img[:,:,0] = img[:,:,0] / 2
        # img[:,:,1] = img[:,:,1] / 1.5
        # img[:,:,0] = img[:,:,0] / 2
        # img[:,:,1] = img[:,:,1] / 1.5
        img = np.where(alpha, render, img)
        img = np.where(field[:,:,3,np.newaxis] > 50, field[:,:,0:3], img)

        # img = np.multiply(frame, 1 - alpha / 255).astype(np.uint8);
        # img = img + np.multiply(render.astype(float), alpha / 255).astype(np.uint8);
        # # img = frame.copy()

        if rot90:
            img = cv2.transpose(img)

        # fps = 1 / (time.time() - frame_time)
        # frame_time = time.time()
        # font = cv2.FONT_HERSHEY_SIMPLEX
        # cv2.putText(img, "%.1f frames per second" % (fps), (20,20), font, 0.8, (255,0,0))

        cv2.imshow('frame',img)
        cv2.imshow(window_name,img)

        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

    scene.stop()
    start_time = time.time()
    in_conn.send(True)
    while openpose_process.exitcode is None:
        out_conn.close()
        out_queue.close()
        timeout = time.time()-start_time
        print("QUITTING", timeout)
        if timeout > 0.5:
            break
        time.sleep(0.1)
    openpose_process.terminate()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()