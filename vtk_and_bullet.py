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

camPos = [0,0,0]
camUpVector = [0,-1,0]
camTargetPos = [0,0,1]
pixelWidth = 1920//2
pixelHeight = 1080 // 2
nearPlane = 0.01
farPlane = 100
fov = 60

# VTK STUFF
sphere = vtk.vtkSphereSource()
sphere.SetRadius(0.01 / 2)
mapper = vtk.vtkPolyDataMapper()
if vtk.VTK_MAJOR_VERSION <= 5:
    mapper.SetInput(sphere.GetOutput())
else:
    mapper.SetInputConnection(sphere.GetOutputPort())

# Create a rendering window and renderer
ren = vtk.vtkRenderer()
renWin = vtk.vtkRenderWindow()
renWin.SetOffScreenRendering(1)
renWin.SetAlphaBitPlanes(1)  # Enable usage of alpha channel
renWin.AddRenderer(ren)

pixelWidth = 1920//2
pixelHeight = 1080//2
renWin.SetSize(pixelWidth, pixelHeight)

# camMatrix = camMatrix

# Assume camera matrix has no offset
# dims = (camMatrix[0,2] * 2, camMatrix[1,2]*2)

# camera = vtktools.cameraFromMatrix(camMatrix, dims, (imgDims[0],imgDims[1]))
# ren.SetActiveCamera(camera)
camera = ren.GetActiveCamera()
camera.SetPosition(camPos)
camera.SetViewUp(camUpVector)
camera.SetFocalPoint(camTargetPos)
camera.SetViewAngle(fov)
# Set camera to look forward from center

w2if = vtk.vtkWindowToImageFilter()
w2if.SetInput(renWin)
w2if.SetInputBufferTypeToRGBA()

actors = {}
poses = {}

# https://stackoverflow.com/questions/19448078/python-opencv-access-webcam-maximum-resolution
def set_res(cap, x,y):
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(x))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(y))
    return int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
p.connect(p.DIRECT)

egl = pkgutil.get_loader('eglRenderer')
if (egl):
    p.loadPlugin(egl.get_filename(), "_eglRendererPlugin")
else:
    p.loadPlugin("eglRendererPlugin")

dataDir = '/home/biomed/Art/bullet3/data'

#p.loadPlugin("eglRendererPlugin")
p.loadURDF(os.path.join(dataDir, "plane.urdf"),[0,0,.4])
# p.loadURDF(os.path.join(dataDir, "r2d2.urdf"))

p.setGravity(0,0,-10)
p.setPhysicsEngineParameter(numSolverIterations=5)
p.setPhysicsEngineParameter(fixedTimeStep=1./240.)
p.setPhysicsEngineParameter(numSubSteps=10)


play = True
window_name = 'frame'

def update_physics():
    global play
    stop = time.time()
    t = 1/240
    while True:
        try:
            if cv2.getWindowProperty(window_name,1) == -1 :
                raise Exception
        except:
            time.sleep(0.1)
            continue
        if not play:
            return
        t = (t + time.time() - stop) / 2
        p.setTimeStep(t)
        p.stepSimulation()
        stop = time.time()
        print(t)

p.configureDebugVisualizer(p.COV_ENABLE_RENDERING,0)
ob = {}
count = 0
for i in range (5):
    for j in range (5):
        for k in range (5):
            ob[count] = p.loadURDF(os.path.join(dataDir, "sphere_1cm.urdf"),[0.02*i + np.random.random_sample(1) * 0.02,0.02*j + np.random.random_sample(1) * 0.01,0.02*k],useMaximalCoordinates=True)
            actors[count] = vtk.vtkActor()
            actors[count].SetMapper(mapper)
            ren.AddActor(actors[count])
            count = count + 1

main_start = time.time()
cap = cv2.VideoCapture(0)
res = set_res(cap, 1280//2, 960//2)

t = Thread(target=update_physics, args=())
t.daemon = True
# t = multiprocessing.Process(target=update_physics)
t.start()

p.configureDebugVisualizer(p.COV_ENABLE_RENDERING,1)

count = 0

while play:
  # for yaw in range (0,360,10):
      yaw = 0
      start = time.time()
      ret, frame = cap.read()
      # p.stepSimulation()
  #   start = time.time()
    
      # viewMatrix = p.computeViewMatrixFromYawPitchRoll(camTargetPos, camDistance, yaw, pitch, roll, upAxisIndex)
      viewMatrix = p.computeViewMatrix(camPos, camTargetPos, camUpVector)
      aspect = pixelWidth / pixelHeight;
      projectionMatrix = p.computeProjectionMatrixFOV(fov, aspect, nearPlane, farPlane);
      img_arr = p.getCameraImage(pixelWidth, pixelHeight, viewMatrix,projectionMatrix, shadow=1,lightDirection=[1,1,1],renderer=p.ER_BULLET_HARDWARE_OPENGL)
  #   stop = time.time()
  #   print ("renderImage %f" % (stop - start))

      for k in actors.keys():
        pos, rot = p.getBasePositionAndOrientation(ob[k])
        actors[k].SetPosition(pos)

      w=img_arr[0] #width of the image, in pixels
      h=img_arr[1] #height of the image, in pixels
      rgb=img_arr[2] #color data RGB
      dep=img_arr[3] #depth data

  #   print ('width = %d height = %d' % (w,h))

  #   #note that sending the data to matplotlib is really slow

  #   #reshape is not needed
      np_img_arr = np.reshape(rgb, (h, w, 4))
      np_img_arr = np_img_arr*(1./255.)

      renWin.Render()
      w2if.Modified()
      w2if.Update()


      print(camera.GetPosition())
      print(camera.GetViewUp())

      if count % 2 == 0:
        cv2.imshow(window_name,vtktools.vtkImageToNumpy(w2if.GetOutput()))
      else:
        cv2.imshow(window_name,np_img_arr)
      # count = count + 1
      if cv2.waitKey(1) & 0xFF == ord('q'):
        play = False
        break
      stop = time.time()
      print ("renderImage %f" % (stop - start))

cap.release()
cv2.destroyAllWindows()

main_stop = time.time()

print ("Total time %f" % (main_stop - main_start))

p.resetSimulation()
