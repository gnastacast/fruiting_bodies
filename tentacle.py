import pybullet as p
import numpy as np
from perlin import perlin_2d

def axis_angle_to_mat(axis, angle):
    if abs(angle) <= 0.001:
        return(np.eye(3))
    axis = axis / np.linalg.norm(axis)
    # print(angle)
    c = np.cos(angle)
    s = np.sin(angle)
    t = 1-c
    x, y, z = axis
    R = np.array([[t*x*x + c, t*x*y - z*s, t*x*z + y*s],
                  [t*x*y + z*s, t*y*y + c, t*y*z - x*s],
                  [t*x*z - y*s, t*y*z + x*s, t*z*z + c]])
    return R

class Tentacle(object):
    def __init__(self, n_joints, pos, radius, parent=None):
        '''
        Create a pybullet tentacle
        '''
        mass = 1
        visualShapeId = -1
        self.particles = [None] * n_joints
        self.radius = radius
        if parent is None:
            base_shape = p.createCollisionShape(p.GEOM_PLANE)
            self.base = p.createMultiBody(0, base_shape, -1, pos)
        else:
            self.base = parent

        colSphereId = p.createCollisionShape(p.GEOM_SPHERE,radius=self.radius)
        p.setCollisionFilterGroupMask(self.base,-1,0,0)
        for i in range (n_joints):
            self.particles[i] = p.createMultiBody(mass, colSphereId ,visualShapeId,
                                                  [pos[0], pos[1], pos[2] + i*2*self.radius + self.radius])
            if i > 0:
                p.createConstraint(self.particles[i-1], -1, self.particles[i], -1, p.JOINT_FIXED,[0,0,0], [0,0, 2*self.radius], [0,0,0])
            else:
                p.setCollisionFilterPair(self.base, self.particles[i], -1, -1, True)
                p.createConstraint(self.base, -1, self.particles[i], -1, p.JOINT_FIXED, [0,0,0], [0,0,self.radius], [0,0,0], [0,0,0,1])
            p.changeDynamics(self.particles[i],-1, mass=1, linearDamping=1)
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING,1)

    def move(self, position, orientation):
        '''
        Set tentacle base position (xyz meters) and orientation (xyzw quaternion)
        '''
        p.resetBasePositionAndOrientation(self.base, position, orientation)
        base_pos_inv, base_rot_inv = p.invertTransform(position, orientation)
        part_pos, part_rot = p.multiplyTransforms(position, orientation, [0,0,self.radius], [0,0,0,1])
        p.resetBasePositionAndOrientation(self.particles[0], part_pos, part_rot)


    def get_particle_positions(self):
        return [p.getBasePositionAndOrientation(part)[0] for part in self.particles]

    def get_particle_rotations(self):
        return [p.getBasePositionAndOrientation(part)[1] for part in self.particles]

    def get_local_particle_positions(self):
        positions = []
        base_pos, base_rot = p.getBasePositionAndOrientation(self.base)
        base_pos_inv, base_rot_inv = p.invertTransform(base_pos, base_rot)
        for part in self.particles:
            pos, rot = p.getBasePositionAndOrientation(part)
            pos, rot = p.multiplyTransforms(base_pos_inv, base_rot_inv, pos, rot)
            positions.append(pos)
        return positions

    def get_local_particle_rotations(self):
        rotations = []
        base_pos, base_rot = p.getBasePositionAndOrientation(self.base)
        base_pos_inv, base_rot_inv = p.invertTransform(base_pos, base_rot)
        for part in self.particles:
            pos, rot = p.getBasePositionAndOrientation(part)
            pos, rot = p.multiplyTransforms(base_pos_inv, base_rot_inv, pos, rot)
            rotations.append(rot)
        return rotations

    def get_position(self):
        return p.getBasePositionAndOrientation(self.base)[0]

    def get_rotation(self):
        return p.getBasePositionAndOrientation(self.base)[1]

    def reset(self, position, orientation):
        base_pos, base_rot = p.getBasePositionAndOrientation(self.base)
        base_pos_inv, base_rot_inv = p.invertTransform(base_pos, base_rot)
        p.resetBasePositionAndOrientation(self.base, position, orientation)
        for part in self.particles:
            part_pos, part_rot = p.getBasePositionAndOrientation(part)
            part_pos, part_rot = p.multiplyTransforms(base_pos_inv, base_rot_inv, part_pos, part_rot)
            part_pos, part_rot = p.multiplyTransforms(position, orientation, part_pos, part_rot)
            p.resetBasePositionAndOrientation(part, part_pos, part_rot)

class TentacleController(object):
    def __init__(self, intensity=100, offset_scale=2, perlinSize=100):
        lin = np.linspace(0,5,perlinSize,endpoint=False)
        x,y = np.meshgrid(lin,lin)
        self.perlin = perlin_2d(x, y, 2)
        self.x, self.y = self.perlin.shape[0], self.perlin.shape[1]
        self.tentacles = []
        self.offsets = []
        self.offset_scale = offset_scale
        self.intensity = intensity

    def update(self, tentacleMask=None):
        if tentacleMask is None:
            tentacleMask = [True] * len(self.tentacles)
        ends = [p.getBasePositionAndOrientation(tentacle.particles[-1])[0] for tentacle in self.tentacles]
        for tentacle, end, offset, maskVal in zip(self.tentacles, ends, self.offsets, tentacleMask):
            if not maskVal:
                continue
            diff = np.subtract(np.array(ends)[tentacleMask], end)
            norm = np.linalg.norm(diff, axis=1)
            diff = diff[norm != 0]
            norm = norm[norm != 0]
            if len(norm > 0):
                idx = [np.argmin(norm)]
                direction = diff[idx] / norm[idx]
                axis = np.subtract(end, p.getBasePositionAndOrientation(tentacle.particles[-2])[0])
                axis = axis / np.linalg.norm(axis)
                self.x = (self.x + np.random.standard_normal())
                self.y = (self.y + np.random.standard_normal())
                x, y = [int(self.x + offset[0]) % self.perlin.shape[0], int(self.y + offset[1]) % self.perlin.shape[0]]
                # print(x,y)
                angle = np.sign(self.perlin[x, y]) * 1
                rot = axis_angle_to_mat(axis, angle)
                direction = np.dot(rot, direction.T)
                direction = direction / np.linalg.norm(direction) ** 2
                # print(direction)
                p.applyExternalForce(tentacle.particles[-1], -1, direction * self.intensity, end, p.WORLD_FRAME)

    def add_tentacle(self, tentacle):
        self.tentacles.append(tentacle)
        self.offsets.append([np.random.standard_normal() * self.offset_scale - self.offset_scale / 2 for i in range(2)])

def main():
    import time
    from threading import Thread

    conid = p.connect(p.SHARED_MEMORY)
    if (conid<0):
    	p.connect(p.GUI)
    	
    p.setInternalSimFlags(0)
    p.resetSimulation()

    sphereRadius = 0.03

    gravXid = p.addUserDebugParameter("gravityX",-10,10,0)
    gravYid = p.addUserDebugParameter("gravityY",-10,10,0)
    gravZid = p.addUserDebugParameter("gravityZ",-10,10,5)
    rotBase = p.addUserDebugParameter("rotateBase",-3,3,0)
    movBase = p.addUserDebugParameter("moveBase",-3,3,0)
    p.setPhysicsEngineParameter(numSolverIterations=10)
    p.setPhysicsEngineParameter(contactBreakingThreshold=0.001)
    p.configureDebugVisualizer(p.COV_ENABLE_RENDERING,0)

    # p.setRealTimeSimulation(1)


    distance = 0.1
    centers = [(distance * i , 0 ,0) for i in range(5)]
    tentacles = [Tentacle(7, c, sphereRadius) for c in centers]
    controller = TentacleController()
    for tentacle in tentacles:
        controller.add_tentacle(tentacle)

    p.configureDebugVisualizer(p.COV_ENABLE_RENDERING,1)

    play = True

    def update_physics():
        stop = time.time()
        t = 1/240
        p.setGravity(0,0,1)
        while True:
            if not play:
                return
            t = (t + time.time() - stop) / 2
            p.setTimeStep(t)
            p.stepSimulation()
            stop = time.time()
            time.sleep(0.01)

    t = Thread(target=update_physics, args=())
    t.daemon = True
    t.start()

    started = False
    while True:
        if not started:
            start_time = time.time()
            started = True

        # Move tentacles to match orientation of planes
        for tentacle, c in zip(tentacles, centers):
            tentacle.move([c[0],c[1],p.readUserDebugParameter(movBase)], p.getQuaternionFromEuler([p.readUserDebugParameter(rotBase),0,0]))

        controller.update()

        # p.applyExternalForce(tentacle.particles[-1], -1, [0,0,-50], pos, p.WORLD_FRAME)
        gravX = p.readUserDebugParameter(gravXid)
        gravY = p.readUserDebugParameter(gravYid)
        gravZ = p.readUserDebugParameter(gravZid)
        p.setGravity(gravX,gravY,gravZ)
        time.sleep(0.01)

    play = False

if __name__ == '__main__':
    main()