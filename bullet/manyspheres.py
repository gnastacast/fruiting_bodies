import pybullet as p
import time
import os
from threading import Thread

import vtk


dataDir = '/home/biomed/Art/bullet3/data'

conid = p.connect(p.SHARED_MEMORY)
if (conid<0):
	p.connect(p.GUI)
	
p.setInternalSimFlags(0)
p.resetSimulation()
	
p.loadURDF(os.path.join(dataDir, "plane.urdf"),useMaximalCoordinates=True)
p.loadURDF(os.path.join(dataDir, "tray/traybox.urdf"),useMaximalCoordinates=True)

gravXid = p.addUserDebugParameter("gravityX",-10,10,0)
gravYid = p.addUserDebugParameter("gravityY",-10,10,0)
gravZid = p.addUserDebugParameter("gravityZ",-10,10,-10)
p.setPhysicsEngineParameter(numSolverIterations=10)
p.setPhysicsEngineParameter(contactBreakingThreshold=0.001)
p.configureDebugVisualizer(p.COV_ENABLE_RENDERING,0)
for i in range (10):
    for j in range (10):
        for k in range (10):
            ob = p.loadURDF(os.path.join(dataDir, "sphere_1cm.urdf"),[0.02*i,0.02*j,0.2+0.02*k],useMaximalCoordinates=True)

p.configureDebugVisualizer(p.COV_ENABLE_RENDERING,1)
p.setGravity(0,0,-10)

# p.setRealTimeSimulation(1)

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

t = Thread(target=update_physics, args=())
t.daemon = True
t.start()

while True:
    gravX = p.readUserDebugParameter(gravXid)
    gravY = p.readUserDebugParameter(gravYid)
    gravZ = p.readUserDebugParameter(gravZid)
    p.setGravity(gravX,gravY,gravZ)
    time.sleep(0.01)

play = False