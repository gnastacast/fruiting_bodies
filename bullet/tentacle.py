import pybullet as p
import time
import os
from threading import Thread
import numpy as np
import vtk

class Tentacle(object):
    def __init__(self, n_joints, length):
        self.particles = [None] * n_joints
        base_shape = p.createCollisionShape(p.GEOM_PLANE, radius=0.01, halfExtents=[0.2, 0.2,0.1])
        self.base = p.createMultiBody(0, base_shape, -1, [0,0,-.2])
        for i in range (n_joints):
            self.particles[i] = p.createMultiBody(mass,colSphereId,visualShapeId,[0,0,(i+1)*4*sphereRadius])
            if i > 0:
                p.createConstraint(self.particles[i-1], -1, self.particles[i], -1, p.JOINT_FIXED,[0,0,0], [0,0, 4*sphereRadius], [0,0,0], [0,0,0,1])
            else:
                p.createConstraint(self.base, -1, self.particles[i], -1, p.JOINT_FIXED, [0,0,0], [0,0,.1], [0,0,0])
            p.changeDynamics(self.particles[i],-1, mass=1, linearDamping=1)
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING,1)

    def move(self, position, orientation):
        pos = position
        pos[2] = pos[2] - 0.1
        p.resetBasePositionAndOrientation(self.base, pos, orientation)
        # _  , rot = particleOrient = p.getBasePositionAndOrientation(self.particles[0])
        # pos, _   = p.multiplyTransforms([0,0,0], [0, 0, 0, 1], position, orientation)
        # p.resetBasePositionAndOrientation(self.particles[0], pos, rot)


useMaximalCoordinates = 0

dataDir = '/home/biomed/Art/bullet3/data'

conid = p.connect(p.SHARED_MEMORY)
if (conid<0):
	p.connect(p.GUI)
	
p.setInternalSimFlags(0)
p.resetSimulation()
	
# plane = p.loadURDF(os.path.join(dataDir, "plane.urdf"),useMaximalCoordinates=useMaximalCoordinates)

sphereRadius = 0.05
colSphereId = p.createCollisionShape(p.GEOM_SPHERE,radius=sphereRadius)

gravXid = p.addUserDebugParameter("gravityX",-10,10,0)
gravYid = p.addUserDebugParameter("gravityY",-10,10,0)
gravZid = p.addUserDebugParameter("gravityZ",-10,10,5)
rotBase = p.addUserDebugParameter("rotateBase",-3,3,0)
movBase = p.addUserDebugParameter("moveBase",-3,3,0)
p.setPhysicsEngineParameter(numSolverIterations=10)
p.setPhysicsEngineParameter(contactBreakingThreshold=0.001)
p.configureDebugVisualizer(p.COV_ENABLE_RENDERING,0)

mass = 1
visualShapeId = -1



# p.setRealTimeSimulation(1)

play = True

def update_physics():
    global play
    stop = time.time()
    t = 1/240
    p.setGravity(0,0,5)
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

tentacle = Tentacle(5, 1)
tentacle2 = Tentacle(4, 1)

started = False
while True:
    if not started:
        start_time = time.time()
        started = True
    # p.resetBasePositionAndOrientation(ob[0], [np.sin(time.time() - start_time) * 0.5,np.sin((time.time() - start_time)*4)* 0.5,0.5], [ 0.8509035, 0, 0, 0.525322])
    tentacle.move([0,0,p.readUserDebugParameter(movBase)], p.getQuaternionFromEuler([p.readUserDebugParameter(rotBase),0,0]))
    tentacle2.move([-0.5,0,p.readUserDebugParameter(movBase)], p.getQuaternionFromEuler([p.readUserDebugParameter(rotBase),0,0]))
    gravX = p.readUserDebugParameter(gravXid)
    gravY = p.readUserDebugParameter(gravYid)
    gravZ = p.readUserDebugParameter(gravZid)
    p.setGravity(gravX,gravY,gravZ)
    time.sleep(0.01)

play = False