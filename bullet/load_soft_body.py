import pybullet as p
from time import sleep
import os

dataDir = '/home/biomed/Art/bullet3/data'
physicsClient = p.connect(p.GUI)

p.setGravity(0,0,-10)
planeId = p.loadURDF(os.path.join(dataDir, "plane.urdf"))
bunnyId = p.loadSoftBody(os.path.join(dataDir, "bunny.obj"))
p.loadURDF(os.path.join(dataDir, "cube_small.urdf"),[1,0,1])
useRealTimeSimulation = 1

print(help(p.loadSoftBody))

if (useRealTimeSimulation):
	p.setRealTimeSimulation(1)

while p.isConnected():
	p.setGravity(0,0,-10)
	if (useRealTimeSimulation):

		sleep(0.01) # Time in seconds.
		#p.getCameraImage(320,200,renderer=p.ER_BULLET_HARDWARE_OPENGL )
	else:
		p.stepSimulation()