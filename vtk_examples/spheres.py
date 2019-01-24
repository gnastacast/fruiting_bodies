#!/usr/bin/env python

import vtk
import time
import numpy as np

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

def main():
    bounds = [4.0, 4.0, 8.0]
    colors = vtk.vtkNamedColors()

    coordData = np.random.random_sample((4,3))

    # x = array of 8 3-tuples of float representing the vertices of a cube:
    x = (np.array([(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0),
                      (0.0, 0.0, 1.0), (1.0, 0.0, 1.0), (1.0, 1.0, 1.0), (0.0, 1.0, 1.0)]) - 0.5) * bounds

    # pts = array of 6 4-tuples of vtkIdType (int) representing the faces
    #     of the cube in terms of the above vertices
    pts = [(3, 2, 1, 0), (4, 5, 6, 7), (0, 1, 5, 4),
           (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]

    # We'll create the building blocks of polydata including data attributes.
    cube = vtk.vtkPolyData()
    points = vtk.vtkPoints()
    polys = vtk.vtkCellArray()
    scalars = vtk.vtkFloatArray()

    # Load the point, cell, and data attributes.
    for i, xi in enumerate(x):
        points.InsertPoint(i, xi)
    for pt in pts:
        polys.InsertNextCell(mkVtkIdList(pt))
    for i, _ in enumerate(x):
        scalars.InsertTuple1(i, i)

    # We now assign the pieces to the vtkPolyData.
    cube.SetPoints(points)
    cube.SetPolys(polys)
    cube.GetPointData().SetScalars(scalars)

    # cubeNormals = vtk.vtkPolyDataNormals()
    # cubeNormals.SetInputData(cube)

    # cubeNormals.Update()

    dist = vtk.vtkImplicitPolyDataDistance()
    dist.SetInput(cube)

    # sphere = vtk.vtkQuadratic()
    # # sphere.SetRadius(0.2)

    # # implicitFunction = vtk.vtkSuperquadric()
    # # implicitFunction.SetPhiRoundness(2.5)
    # # implicitFunction.SetThetaRoundness(.5)
    noiseFunction = vtk.vtkPerlinNoise()
    noiseFunction.SetAmplitude(2)
    # noiseFunction2 = vtk.vtkPerlinNoise()
    # noiseFunction2.SetAmplitude(3)
    # noiseFunction2.SetFrequency(2,2,2)

    # window = vtk.vtkImplicitWindowFunction()
    # window.SetImplicitFunction(noiseFunction2)
    # window.SetWindowRange(0, 10)

    noiseSum = vtk.vtkImplicitSum()
    noiseSum.AddFunction(noiseFunction)
    # # noiseSum.AddFunction(window)
    # noiseSum.AddFunction(sphere)
    noiseSum.AddFunction(dist)

    # # noiseFunction.SetFrequency([5, 5, 5])
    # # window = vtk.vtkImplicitWindowFunction()
    # # window.SetImplicitFunction(halo)
    # # window.SetWindowRange(-0, 10)

    # Sample the function.
    sample = vtk.vtkSampleFunction()
    sample.SetSampleDimensions(30,30,30)
    sample.SetImplicitFunction(noiseSum)

    xmin, xmax, ymin, ymax, zmin, zmax = -bounds[0], bounds[0], -bounds[1], bounds[1], -bounds[2], bounds[2]
    sample.SetModelBounds(xmin, xmax, ymin, ymax, zmin, zmax)

    # Create the 0 isosurface.
    contours = vtk.vtkContourFilter()
    contours.SetInputConnection(sample.GetOutputPort())
    contours.GenerateValues(1, 2.0, 2.0)

    # Map the contours to graphical primitives.
    contourMapper = vtk.vtkPolyDataMapper()
    contourMapper.SetInputConnection(contours.GetOutputPort())
    contourMapper.SetScalarRange(0.0, 1.2)
    contourMapper.ScalarVisibilityOff()

    # Create an actor for the contours.
    contourActor = vtk.vtkActor()
    contourActor.SetMapper(contourMapper)
    contourActor.GetProperty().SetColor(colors.GetColor3d("salmon"))

    # Create a box around the function to indicate the sampling volume. 

    #Create outline.
    outline = vtk.vtkOutlineFilter()
    outline.SetInputConnection(sample.GetOutputPort())

    # Map it to graphics primitives.
    outlineMapper = vtk.vtkPolyDataMapper()
    outlineMapper.SetInputConnection(outline.GetOutputPort())

    # Create an actor.
    outlineActor = vtk.vtkActor()
    outlineActor.SetMapper(outlineMapper)
    outlineActor.GetProperty().SetColor(0,0,0)

    # Visualize.
    renderer = vtk.vtkRenderer()
    renderWindow = vtk.vtkRenderWindow()
    renderWindow.AddRenderer(renderer)
    interactor = vtk.vtkRenderWindowInteractor()
    interactor.SetRenderWindow(renderWindow)

    renderer.AddActor(contourActor)
    renderer.AddActor(outlineActor)
    renderer.SetBackground(colors.GetColor3d("powder_blue"))

    # Enable user interface interactor
    renderWindow.Render()

    # Sign up to receive TimerEvent
    cb = vtkTimerCallback()
    cb.noise = noiseFunction
    interactor.AddObserver('TimerEvent', cb.execute)
    timerId = interactor.CreateRepeatingTimer(1);

    interactor.Start()


class vtkTimerCallback():
    def __init__(self):
       self.timer_count = 0
       self.start_time = time.time()

    def execute(self,obj,event):
       # print(self.timer_count)
       self.noise.SetPhase(0, 0, time.time() - self.start_time)
       # self.actor.SetPosition(self.timer_count, self.timer_count,0);
       iren = obj
       iren.GetRenderWindow().Render()
       self.timer_count += 1

if __name__ == '__main__':
    main()