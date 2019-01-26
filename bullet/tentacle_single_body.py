import pybullet as p
import time
import os
from threading import Thread
import numpy as np

import vtk

useMaximalCoordinates = 0

dataDir = '/home/biomed/Art/bullet3/data'

conid = p.connect(p.SHARED_MEMORY)
if (conid<0):
	p.connect(p.GUI)
	
p.setInternalSimFlags(0)
p.resetSimulation()
	
plane = p.loadURDF(os.path.join(dataDir, "plane.urdf"),useMaximalCoordinates=useMaximalCoordinates)

sphereRadius = 0.05
colSphereId = p.createCollisionShape(p.GEOM_SPHERE,radius=sphereRadius)

gravXid = p.addUserDebugParameter("gravityX",-10,10,0)
gravYid = p.addUserDebugParameter("gravityY",-10,10,0)
gravZid = p.addUserDebugParameter("gravityZ",-10,10,-5)
p.setPhysicsEngineParameter(numSolverIterations=10)
p.setPhysicsEngineParameter(contactBreakingThreshold=0.001)
p.configureDebugVisualizer(p.COV_ENABLE_RENDERING,0)

centers = 0

mass = 1
visualShapeId = -1

n_links = 5

link_Masses=[1] * n_links
linkCollisionShapeIndices=[colSphereId] * n_links
linkVisualShapeIndices=[-1] * n_links
linkPositions=[[0,0,0.11 * (j+1)] for j in range(n_links)]
linkOrientations=[[0,0,0,1]] * n_links
linkInertialFramePositions=[[0,0,0]] * n_links
linkInertialFrameOrientations=[[0,0,0,1]] * n_links
indices=[0] * n_links
jointTypes= [p.JOINT_FIXED] * n_links
axis=[[0,0,0]] * n_links

ob = {}
for i in range (5):
    ob[i] = p.createMultiBody(mass,colSphereId,visualShapeId,[0,(i+1)*4*sphereRadius,0.4], [0,0,0,1],
                              linkMasses=link_Masses,
                              linkCollisionShapeIndices=linkCollisionShapeIndices,
                              linkVisualShapeIndices=linkVisualShapeIndices,
                              linkPositions=linkPositions,
                              linkOrientations=linkOrientations,
                              linkInertialFramePositions=linkInertialFramePositions,
                              linkInertialFrameOrientations=linkInertialFrameOrientations,
                              linkParentIndices=indices,
                              linkJointTypes=jointTypes,
                              linkJointAxis=axis)
                              # linkMasses= [[1]] * n_links,
                              # linkPositions = [[0, (j+1) * 4 * sphereRadius, 0] for j in range(n_links)],
                              # linkCollisionShapeIndices = [[colSphereId]] * n_links,
                              # linkVisualShapeIndices = [[-1]] * n_links)
                              # linkOrientations = [[0,0,0,1]] * n_links,
                              # linkInertialFramePositions = [[0, 0, 0]] * n_links, 
                              # linkInertialFrameOrientations = [[0, 0, 0, 1]] * n_links,
                              # linkParentIndices = [j for j in range(n_links)],
                              # linkJointTypes = [[p.JOINT_FIXED]] * n_links,
                              # linkJointAxis = [[0,0,0]] * n_links,
                              # useMaximalCoordinates = useMaximalCoordinates)
    # if i > 0:
    #     p.createConstraint(ob[i-1], -1, ob[i], -1, p.JOINT_FIXED,[0,0,0], [0,0, 4*sphereRadius], [0,0,0], [0,0,0,1])
    # else:
    #     p.createConstraint(plane, -1, ob[i], -1, p.JOINT_FIXED,[0,0,0], [0,0, .4], [0,0,0])
    p.changeDynamics(ob[i],-1, mass=1, linearDamping=1)
p.configureDebugVisualizer(p.COV_ENABLE_RENDERING,1)

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

started = False
while True:
    if not started:
        start_time = time.time()
        started = True
    # p.resetBasePositionAndOrientation(ob[0], [np.sin(time.time() - start_time) * 0.5,np.sin((time.time() - start_time)*4)* 0.5,0.5], [ 0.8509035, 0, 0, 0.525322])
    # p.resetBasePositionAndOrientation(ob[0], [0,0,0.5], [ 0.7071068, 0, 0, 0.7071068 ])
    gravX = p.readUserDebugParameter(gravXid)
    gravY = p.readUserDebugParameter(gravYid)
    gravZ = p.readUserDebugParameter(gravZid)
    p.setGravity(gravX,gravY,gravZ)
    time.sleep(0.01)

play = False