#!/usr/bin/env python

import vtk
import time
import numpy as np

def main():
    value = 2.0
    colors = vtk.vtkNamedColors()

    sphere = vtk.vtkSphere()
    sphere.SetRadius(0.2)

    # implicitFunction = vtk.vtkSuperquadric()
    # implicitFunction.SetPhiRoundness(2.5)
    # implicitFunction.SetThetaRoundness(.5)
    noiseFunction = vtk.vtkPerlinNoise()
    noiseFunction.SetAmplitude(2)
    noiseFunction2 = vtk.vtkPerlinNoise()
    noiseFunction2.SetAmplitude(3)
    noiseFunction2.SetFrequency(2,2,2)

    window = vtk.vtkImplicitWindowFunction()
    window.SetImplicitFunction(noiseFunction2)
    window.SetWindowRange(0, 10)

    noiseSum = vtk.vtkImplicitSum()
    noiseSum.AddFunction(noiseFunction)
    # noiseSum.AddFunction(window)
    noiseSum.AddFunction(sphere)

    # noiseFunction.SetFrequency([5, 5, 5])
    # window = vtk.vtkImplicitWindowFunction()
    # window.SetImplicitFunction(halo)
    # window.SetWindowRange(-0, 10)

    # Sample the function.
    sample = vtk.vtkSampleFunction()
    sample.SetSampleDimensions(30,30,30)
    sample.SetImplicitFunction(noiseSum)

    xmin, xmax, ymin, ymax, zmin, zmax = -value, value, -value, value, -value, value
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