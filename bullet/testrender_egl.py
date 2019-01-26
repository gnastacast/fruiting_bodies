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
p.loadURDF(os.path.join(dataDir, "plane.urdf"),[0,0,0])
p.loadURDF(os.path.join(dataDir, "r2d2.urdf"))

camTargetPos = [0,0,0]
cameraUp = [0,0,1]
cameraPos = [1,1,2]
p.setGravity(0,0,-10)


play = True

def update_physics():
    global play
    stop = time.time()
    t = 1/240
    while True:
        if not play:
            return
        t = (t + time.time() - stop) / 2
        p.setTimeStep(t)
        p.stepSimulation()
        stop = time.time()
        time.sleep(0.01)
        print(t)

p.createCollisionShape(p.GEOM_PLANE)
p.createMultiBody(0,0)

sphereRadius = 0.05
colSphereId = p.createCollisionShape(p.GEOM_SPHERE,radius=sphereRadius)
colBoxId = p.createCollisionShape(p.GEOM_BOX,halfExtents=[sphereRadius,sphereRadius,sphereRadius])

mass = 1
visualShapeId = -1

    
    
link_Masses=[1]
linkCollisionShapeIndices=[colBoxId]
linkVisualShapeIndices=[-1]
linkPositions=[[0,0,0.11]]
linkOrientations=[[0,0,0,1]]
linkInertialFramePositions=[[0,0,0]]
linkInertialFrameOrientations=[[0,0,0,1]]
indices=[0]
jointTypes=[p.JOINT_REVOLUTE]
axis=[[0,0,1]]

for i in range (3):
    for j in range (3):
        for k in range (3):
            basePosition = [1+i*5*sphereRadius,1+j*5*sphereRadius,1+k*5*sphereRadius+1]
            baseOrientation = [0,0,0,1]
            if (k&2):
                sphereUid = p.createMultiBody(mass,colSphereId,visualShapeId,basePosition,baseOrientation)
            else:
                sphereUid = p.createMultiBody(mass,colBoxId,visualShapeId,basePosition,baseOrientation,linkMasses=link_Masses,linkCollisionShapeIndices=linkCollisionShapeIndices,linkVisualShapeIndices=linkVisualShapeIndices,linkPositions=linkPositions,linkOrientations=linkOrientations,linkInertialFramePositions=linkInertialFramePositions, linkInertialFrameOrientations=linkInertialFrameOrientations,linkParentIndices=indices,linkJointTypes=jointTypes,linkJointAxis=axis)            
            
            p.changeDynamics(sphereUid,-1,spinningFriction=0.001, rollingFriction=0.001,linearDamping=0.0)
            for joint in range (p.getNumJoints(sphereUid)):
                p.setJointMotorControl2(sphereUid,joint,p.VELOCITY_CONTROL,targetVelocity=1,force=10)

pitch = -10.0

roll=0
upAxisIndex = 2
camDistance = 4
pixelWidth = 1920//2
pixelHeight = 1080 // 2
nearPlane = 0.01
farPlane = 100

fov = 60

main_start = time.time()
cap = cv2.VideoCapture(0)
res = set_res(cap, 1280//2, 960//2)

t = Thread(target=update_physics, args=())
t.daemon = True
t.start()
while play:
  # for yaw in range (0,360,10):
      yaw = 0
      start = time.time()
      ret, frame = cap.read()
      # p.stepSimulation()
  #   start = time.time()
    
      viewMatrix = p.computeViewMatrixFromYawPitchRoll(camTargetPos, camDistance, yaw, pitch, roll, upAxisIndex)
      aspect = pixelWidth / pixelHeight;
      projectionMatrix = p.computeProjectionMatrixFOV(fov, aspect, nearPlane, farPlane);
      img_arr = p.getCameraImage(pixelWidth, pixelHeight, viewMatrix,projectionMatrix, shadow=1,lightDirection=[1,1,1],renderer=p.ER_BULLET_HARDWARE_OPENGL)
  #   stop = time.time()
  #   print ("renderImage %f" % (stop - start))

      w=img_arr[0] #width of the image, in pixels
      h=img_arr[1] #height of the image, in pixels
      rgb=img_arr[2] #color data RGB
      dep=img_arr[3] #depth data

  #   print ('width = %d height = %d' % (w,h))

  #   #note that sending the data to matplotlib is really slow

  #   #reshape is not needed
      np_img_arr = np.reshape(rgb, (h, w, 4))
      np_img_arr = np_img_arr*(1./255.)
    
  #   #show
  #   #plt.imshow(np_img_arr,interpolation='none',extent=(0,1600,0,1200))
  #   #image = plt.imshow(np_img_arr,interpolation='none',animated=True,label="blah")

  #   image.set_data(np_img_arr)
  #   ax.plot([0])
  #   #plt.draw()
  #   #plt.show()
  #   plt.pause(0.01)
  #   #image.draw()
      cv2.imshow('frame',np_img_arr)
      if cv2.waitKey(5) & 0xFF == ord('q'):
        play = False
        break
      stop = time.time()
      print ("renderImage %f" % (stop - start))
cap.release()
cv2.destroyAllWindows()

main_stop = time.time()

print ("Total time %f" % (main_stop - main_start))

p.resetSimulation()
