import numpy as np
import vtk
import transformations as tf

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

def quaternion_to_axis_angle(rot):
    qw = rot[3]
    if abs(qw) != 1:
        angle = np.arccos(qw) * 114.6   # 360 / pi  (to degrees)
        value = np.sqrt(1 - qw * qw)
        axis = tuple([i / value for i in rot[:3]])
        return axis, angle
    return [0,0,1], 0.0

def mkVtkIdList(it):
    """
    Makes a vtkIdList from a Python iterable. I'm kinda surprised that
     this is necessary, since I assumed that this kind of thing would
     have been built into the wrapper and happen transparently, but it
     seems not.

    :param it: A python iterable.
    :return: A vtkIdList
    """
    vil = vtk.vtkIdList()
    for i in it:
        vil.InsertNextId(int(i))
    return vil

class vtkTentacle(vtk.vtkActor):
    def __init__(self):
        super().__init__()
        verts = np.loadtxt('tentacle_verts.txt')
        faces = np.loadtxt('tentacle_polys.txt')
        bone_dat = np.loadtxt('tentacle_weights.txt')
        bones = np.unique(bone_dat[:,0])
        bones = np.unique(bone_dat[:,0])

        # We'll create the building blocks of polydata including data attributes.
        polyData = vtk.vtkPolyData()
        points = vtk.vtkPoints()
        polys = vtk.vtkCellArray()
        scalars = vtk.vtkFloatArray()

        weights = np.zeros((len(verts), len(bones)))
        # Load the point, cell, and data attributes.
        for v in verts:
            points.InsertPoint(int(v[0]), v[1:])
        for pt in faces:
            polys.InsertNextCell(mkVtkIdList(pt[1:].astype(int)))
        for bone in bones:
            idx =  np.round(bone_dat[np.where(bone_dat[:,0] == bone)][:,1]).astype(int)
            weights[idx, int(bone)] = bone_dat[np.where(bone_dat[:,0] == bone)][:,2]

        polyData.SetPoints(points)
        polyData.SetPolys(polys)

        normalGenerator = vtk.vtkPolyDataNormals()
        normalGenerator.SetInputData(polyData)
        normalGenerator.ComputePointNormalsOn()
        normalGenerator.ComputeCellNormalsOn()
        normalGenerator.Update()

        tfarray = vtk.vtkFloatArray()
        npoints = polyData.GetNumberOfPoints()
        bounds = polyData.GetBounds()
        tfarray.SetNumberOfComponents(len(bones))
        tfarray.SetNumberOfTuples(npoints)

        for i, weight in enumerate(weights):
            for j in range(len(bones)):
                tfarray.SetComponent(i, j, weight[j])

        tfarray.SetName("weights")
        polyData.GetPointData().AddArray(tfarray)

        # Now, for the weighted transform stuff
        weightedTrans = vtk.vtkWeightedTransformFilter();
        weightedTrans.SetNumberOfTransforms(len(bones))

        self.transforms = [vtk.vtkTransform() for bone in bones]
        for i in range(len(self.transforms)):
            self.transforms[i].Identity()
            # self.transforms[i].RotateX(i*10)
            # self.transforms[i].Translate(0,0,1*i)
            weightedTrans.SetTransform(self.transforms[i], i)


        # Which data array should the filter use ?
        weightedTrans.SetWeightArray("weights")

        weightedTrans.SetInputConnection(normalGenerator.GetOutputPort())

        smoothFilter = vtk.vtkSmoothPolyDataFilter()
        smoothFilter.SetInputConnection(weightedTrans.GetOutputPort())
        smoothFilter.SetNumberOfIterations(15)
        smoothFilter.SetRelaxationFactor(0.2)
        # smoothFilter.FeatureEdgeSmoothingOn()
        # smoothFilter.BoundarySmoothingOn()

        weightedTransMapper = vtk.vtkPolyDataMapper()
        weightedTransMapper.SetInputConnection(smoothFilter.GetOutputPort())

        with open("perlin.glsl", "r") as file:
            shader_str = file.read()

        weightedTransMapper.SetVertexShaderCode(
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

        # weightedTransMapper.AddShaderReplacement(
        #     vtk.vtkShader.Vertex,
        #     "//VTK::Normal::Dec", # replace the normal block
        #     True, # before the standard replacements
        #     "//VTK::Normal::Dec\n" # we still want the default
        #     "  varying vec3 myNormalMCVSOutput;\n" #but we add this
        #     "  out vec4 myVertexMC;\n",
        #     False # only do it once
        # )
        weightedTransMapper.AddShaderReplacement(
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
        weightedTransMapper.AddShaderReplacement(
            vtk.vtkShader.Fragment,
            "//VTK::System::Dec", 
            False, # after the standard replacements
            "//VTK::System::Dec\n" + shader_str,
            False # only do it once
        )

        # now modify the fragment shader
        weightedTransMapper.AddShaderReplacement(
            vtk.vtkShader.Fragment,  # in the fragment shader
            "//VTK::Normal::Dec", # replace the normal block
            True, # before the standard replacements
            "//VTK::Normal::Dec\n" # we still want the default
            "  varying vec3 myNormalMCVSOutput;\n" #but we add this
            "  varying vec4 myVertexMC;\n"
            "  uniform float k = 1.0;\n",
            False # only do it once
        )
        # # weightedTransMapper.AddShaderReplacement(
        # #     vtk.vtkShader.Fragment,  # in the fragment shader
        # #     "//VTK::Normal::Impl", # replace the normal block
        # #     True, # before the standard replacements
        # #     "//VTK::Normal::Impl\n" # we still want the default calc
        # #     "  diffuseColor = abs(myNormalMCVSOutput);\n", #but we add this
        # #     False # only do it once
        # # )
        # weightedTransMapper.AddShaderReplacement(
        #     vtk.vtkShader.Fragment,  # in the fragment shader
        #     "//VTK::Normal::Impl", # replace the normal block
        #     True, # before the standard replacements
        #     "//VTK::Normal::Impl\n" # we still want the default calc
        #     "  diffuseColor = abs(myNormalMCVSOutput);\n", #but we add this
        #     False # only do it once
        # )

        weightedTransMapper.AddShaderReplacement(
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

        self.SetMapper(weightedTransMapper)
        self.GetProperty().SetColor(1,1,0)
        self.GetProperty().SetAmbientColor(0.2, 0.2, 0.2);
        self.GetProperty().SetDiffuseColor(1.0, 1.0, 1.0);
        self.GetProperty().SetSpecularColor(1.0, 1.0, 1.0);
        self.GetProperty().SetSpecular(0.5);
        self.GetProperty().SetDiffuse(0.7);
        self.GetProperty().SetAmbient(0.1);
        self.GetProperty().SetSpecularPower(100.0);
        self.GetProperty().SetOpacity(1.0);
        self.GetProperty().BackfaceCullingOn();

    def update(self, tentacle):
        # Move actor
        self.SetPosition(tentacle.get_position())
        self.SetOrientation(0,0,0)
        axis, angle = quaternion_to_axis_angle(tentacle.get_rotation())
        # print(np.round(axis, 3), int(angle))
        # self.SetOrientation(tf.euler_from_quaternion(tentacle.get_rotation(), axes='sxyz'))
        self.RotateWXYZ(angle, axis[0], axis[1], axis[2])
        # Move particles
        positions = tentacle.get_local_particle_positions()
        rotations = tentacle.get_local_particle_rotations()
        for i, pos, rot in zip(range(len(positions)), positions, rotations):
            if i < len(self.transforms) and i > 0:
                self.transforms[i].Identity()
                self.transforms[i].PostMultiply()
                self.transforms[i].Translate([0,0, -i * tentacle.radius * self.GetScale()[0] * 2])
                x, y ,z = tf.euler_from_quaternion(rot, axes='sxyz')
                # self.transforms[i].RotateZ(z)
                # self.transforms[i].RotateY(y)
                # self.transforms[i].RotateX(x)
                axis, angle = quaternion_to_axis_angle(rot)
                # print(np.round(axis, 3), int(angle))
                self.transforms[i].RotateWXYZ(angle, axis)
                self.transforms[i].Translate(pos[0] * self.GetScale()[0], pos[1] * self.GetScale()[1], pos[2] * self.GetScale()[2])


def main():
    import time
    import cv2
    from tentacle import Tentacle, TentacleController, axis_angle_to_mat
    from bullet_vtk_scene import Scene
    import pybullet as p

    pixelWidth, pixelHeight = 1920 // 2, 1080 // 2

    window_name = 'frame'

    scene = Scene(pixelHeight, pixelWidth, window_name, fov=60)

    sphereRadius = 0.03

    # # VTK STUFF
    # sphere = vtk.vtkSphereSource()
    # sphere.SetRadius(sphereRadius)
    # mapper = vtk.vtkPolyDataMapper()
    # if vtk.VTK_MAJOR_VERSION <= 5:
    #         mapper.SetInput(sphere.GetOutput())
    # else:
    #         mapper.SetInputConnection(sphere.GetOutputPort())

    # pybullet stuff        
    p.setPhysicsEngineParameter(numSolverIterations=10)
    p.setPhysicsEngineParameter(contactBreakingThreshold=0.001)
    p.configureDebugVisualizer(p.COV_ENABLE_RENDERING,0)
    p.setGravity(0,-1,0)

    distance = 0.1
    n_tentacles = 5
    centers = [(distance * i - distance * n_tentacles / 2, .5 ,1) for i in range(n_tentacles)]
    tentacles = [Tentacle(7, c, sphereRadius) for c in centers]
    tentacle_spheres = []
    vtk_tentacles = []
    controller = TentacleController(intensity=30)
    for tent_id, tentacle in enumerate(tentacles):
        controller.add_tentacle(tentacle)
        # tentacle_spheres.append([None] * len(tentacle.particles))
        # for i in range(len(tentacle.particles)):
        #     tentacle_spheres[-1][i] = vtk.vtkActor()
        #     tentacle_spheres[-1][i].SetMapper(mapper)
        #     tentacle_spheres[-1][i].GetProperty().SetColor(1-i/len(tentacle.particles),0,i/len(tentacle.particles))
        #     scene.ren.AddActor(tentacle_spheres[-1][i])
        vtk_tentacles.append(vtkTentacle())
        vtk_tentacles[-1].SetScale(sphereRadius / 0.05)
        # vtk_tentacles[-1].RotateX(90)
        # vtk_tentacles[-1]. SetPosition(centers[tent_id])
        scene.ren.AddActor(vtk_tentacles[-1])
        tentacle.reset(centers[tent_id],[ 0.7071068, 0, 0, 0.7071068 ])

    p.configureDebugVisualizer(p.COV_ENABLE_RENDERING,1)

    scene.start()
    while True:
        # Move tentacles to match orientation of planes
        # for tentacle, c in zip(tentacles, centers):
        #     tentacle.move([c[0],c[1],0/10], p.getQuaternionFromEuler([0/10,0,0]))
        #     pos, rot = p.getBasePositionAndOrientation(tentacle.particles[-1])

        controller.update()

        # Rotate camera
        # scene.camera.SetPosition(np.sin(time.time()), 0, -np.cos(time.time())+1)

        # for tentacle, spheres, vtk_tentacle in zip(tentacles, tentacle_spheres, vtk_tentacles):
        #     positions = tentacle.get_particle_positions()
        #     # vtk_tentacle.SetPosition(positions[0][0], positions[0][1], positions[0][2])
        #     for pos, sphere in zip(positions, spheres):
        #         sphere.SetPosition(pos)
        for tentacle, vtk_tentacle, center in zip(tentacles, vtk_tentacles, centers):
            # tentacle.move([center[0] + np.sin(time.time()) * 0.5, center[1], center[2]], [ 0.7071068, 0, 0, 0.7071068 ])
            vtk_tentacle.update(tentacle)

        cv2.imshow(window_name,scene.render(True))

        if cv2.waitKey(5) & 0xFF == ord('q'):
            play = False
            break
        stop = time.time()
        time.sleep(0.01)

    play = False

if __name__ == '__main__':
    main()