#using the eglRendererPlugin (hardware OpenGL acceleration)
#using EGL on Linux and default OpenGL window on Win32.

#make sure to compile pybullet with PYBULLET_USE_NUMPY enabled
#otherwise use testrender.py (slower but compatible without numpy)
#you can also use GUI mode, for faster OpenGL rendering (instead of TinyRender CPU)

import numpy as np
import cv2
import pybullet as p
import time
import pkgutil
import os
from threading import Thread
import vtk
import vtktools
# import multiprocessing

# https://stackoverflow.com/questions/19448078/python-opencv-access-webcam-maximum-resolution
def set_res(cap, x,y):
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(x))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(y))
    return int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

class Scene(object):
    def __init__(self, height, width, window_name, fov=60):
        camPos = [0,0,0]
        camUpVector = [0,-1,0]
        camTargetPos = [0,0,1]
        nearPlane = 0.01
        farPlane = 100

        self.window_name = window_name
        self.height = height
        self.width = width

        ##############    VTK stuff   ##############

        # Create a rendering window and renderer
        self.ren = vtk.vtkRenderer()
        self.renWin = vtk.vtkRenderWindow()
        self.renWin.SetOffScreenRendering(1)
        self.renWin.SetAlphaBitPlanes(1)  # Enable usage of alpha channel
        self.renWin.AddRenderer(self.ren)

        self.renWin.SetSize(width, height)

        self.camera = self.ren.GetActiveCamera()
        self.camera.SetPosition(camPos)
        self.camera.SetViewUp(camUpVector)
        self.camera.SetFocalPoint(camTargetPos)
        self.camera.SetViewAngle(fov)

        self.w2if = vtk.vtkWindowToImageFilter()
        self.w2if.SetInput(self.renWin)
        self.w2if.SetInputBufferTypeToRGBA()

        ############## pybullet stuff ##############
        
        p.connect(p.DIRECT)
        # conid = p.connect(p.SHARED_MEMORY)
        # if (conid<0):
        #     p.connect(p.GUI)
        p.resetSimulation()
        p.setGravity(0,0,-10)
        # p.setPhysicsEngineParameter(numSolverIterations=5)
        # p.setPhysicsEngineParameter(fixedTimeStep=1./240.)
        # p.setPhysicsEngineParameter(numSubSteps=10)

        self.viewMatrix = p.computeViewMatrix(camPos, camTargetPos, camUpVector)
        aspect = width / height;
        self.projectionMatrix = p.computeProjectionMatrixFOV(fov, aspect, nearPlane, farPlane);

        egl = pkgutil.get_loader('eglRenderer')
        if (egl):
            p.loadPlugin(egl.get_filename(), "_eglRendererPlugin")
        else:
            p.loadPlugin("eglRendererPlugin")

        self.play = True
        self.physics_thread = Thread(target=self.update_physics, args=())
        self.physics_thread.daemon = True
        print("READY")

    def update_physics(self):
        stop = time.time()
        t = 0
        while True:
            try:
                    if cv2.getWindowProperty(self.window_name,1) == -1 :
                            raise Exception
            except:
                    time.sleep(0.1)
                    continue
            if not self.play:
                    return
            t = (t + time.time() - stop) / 2
            if t < 1/60:
                time.sleep(1/60 - t)
                t = 1/60
            p.setTimeStep(t)
            p.stepSimulation()
            stop = time.time()

    def start(self):
        self.physics_thread.start()


    def stop(self):
        self.play = False
        p.resetSimulation()

    def render(self, vtk_render=True):
        if not vtk_render:
            img_arr = p.getCameraImage(self.width, self.height, self.viewMatrix,
                                       self.projectionMatrix, shadow=1,
                                       lightDirection=[1,1,1], renderer=p.ER_BULLET_HARDWARE_OPENGL)
            w=img_arr[0] #width of the image, in pixels
            h=img_arr[1] #height of the image, in pixels
            rgb=img_arr[2] #color data RGBA
            dep=img_arr[3] #depth data
            np_img_arr = np.reshape(rgb, (h, w, 4))
            return np_img_arr*(1./255.)

        self.renWin.Render()
        self.w2if.Modified()
        self.w2if.Update()
        return vtktools.vtkImageToNumpy(self.w2if.GetOutput())



def main():

    pixelWidth, pixelHeight = 1920 // 2, 1080 // 2

    window_name = 'frame'

    scn = Scene(pixelHeight, pixelWidth, window_name)

    # VTK STUFF
    sphere = vtk.vtkSphereSource()
    sphere.SetRadius(0.01 / 2)
    mapper = vtk.vtkPolyDataMapper()
    if vtk.VTK_MAJOR_VERSION <= 5:
            mapper.SetInput(sphere.GetOutput())
    else:
            mapper.SetInputConnection(sphere.GetOutputPort())

    actors = {}
    poses = {}

    dataDir = '/home/biomed/Art/bullet3/data'

    #p.loadPlugin("eglRendererPlugin")
    p.loadURDF(os.path.join(dataDir, "plane.urdf"),[0,0,.4])
    # p.loadURDF(os.path.join(dataDir, "r2d2.urdf"))

    p.configureDebugVisualizer(p.COV_ENABLE_RENDERING,0)
    ob = {}
    count = 0
    for i in range (5):
        for j in range (5):
            for k in range (5):
                ob[count] = p.loadURDF(os.path.join(dataDir, "sphere_1cm.urdf"),[0.02*i + np.random.random_sample(1) * 0.02,0.02*j + np.random.random_sample(1) * 0.01,0.02*k],useMaximalCoordinates=True)
                actors[count] = vtk.vtkActor()
                actors[count].SetMapper(mapper)
                scn.ren.AddActor(actors[count])
                count = count + 1

    main_start = time.time()
    cap = cv2.VideoCapture(0)
    res = set_res(cap, 1280//2, 960//2)

    p.configureDebugVisualizer(p.COV_ENABLE_RENDERING,1)

    count = 0
    scn.start()

    while True:
        ret, frame = cap.read()

        for k in actors.keys():
            pos, rot = p.getBasePositionAndOrientation(ob[k])
            actors[k].SetPosition(pos)

        cv2.imshow(window_name,scn.render(count%2))
        count = count + 1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            play = False
            break
        stop = time.time()

    scn.stop()
    cap.release()
    cv2.destroyAllWindows()

    p.resetSimulation()

if __name__ == '__main__':
    main()
