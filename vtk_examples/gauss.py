#!/usr/bin/env python

"""
"""

import vtk
import time
import numpy as np
from vtk.numpy_interface import dataset_adapter as dsa


def main():
    colors = vtk.vtkNamedColors()

    colors.SetColor("PopColor", [230, 230, 230, 255])

    # fileName = get_program_parameters()

    keys = ['NUMBER_POINTS', 'MONTHLY_PAYMENT', 'INTEREST_RATE', 'LOAN_AMOUNT', 'TIME_LATE']

    # Read in the data and make an unstructured data set.
    dataSet = make_dataset()

    # pts[:,0] = pts[:,0] ** 3
    # npData.SetPoints(npData.GetPoints())

    # # Construct the pipeline for the original population.
    # popSplatter = vtk.vtkGaussianSplatter()
    # popSplatter.SetInputData(dataSet)
    # popSplatter.SetSampleDimensions(100, 100, 100)
    # popSplatter.SetRadius(0.1)
    # popSplatter.ScalarWarpingOff()

    # popSurface = vtk.vtkContourFilter()
    # popSurface.SetInputConnection(popSplatter.GetOutputPort())
    # popSurface.SetValue(0, 0.01)

    # popMapper = vtk.vtkPolyDataMapper()
    # popMapper.SetInputConnection(popSurface.GetOutputPort())
    # popMapper.ScalarVisibilityOff()

    # popActor = vtk.vtkActor()
    # popActor.SetMapper(popMapper)
    # popActor.GetProperty().SetOpacity(0.3)
    # popActor.GetProperty().SetColor(colors.GetColor3d("PopColor"))

    # Construct the pipeline for the delinquent population.
    lateSplatter = vtk.vtkGaussianSplatter()
    lateSplatter.SetInputData(dataSet)
    lateSplatter.SetSampleDimensions(150, 150, 150)
    lateSplatter.SetRadius(0.1)
    lateSplatter.SetScaleFactor(.001)

    lateSurface = vtk.vtkContourFilter()
    lateSurface.SetInputConnection(lateSplatter.GetOutputPort())
    lateSurface.SetValue(0, 0.01)

    lateMapper = vtk.vtkPolyDataMapper()
    lateMapper.SetInputConnection(lateSurface.GetOutputPort())
    lateMapper.ScalarVisibilityOff()

    lateActor = vtk.vtkActor()
    lateActor.SetMapper(lateMapper)
    lateActor.GetProperty().SetColor(colors.GetColor3d("Red"))

    # Create axes.
    lateSplatter.Update()
    bounds = lateSplatter.GetOutput().GetBounds()

    axes = vtk.vtkAxes()
    axes.SetOrigin(bounds[0], bounds[2], bounds[4])
    axes.SetScaleFactor(lateSplatter.GetOutput().GetLength() / 5)

    axesTubes = vtk.vtkTubeFilter()
    axesTubes.SetInputConnection(axes.GetOutputPort())
    axesTubes.SetRadius(axes.GetScaleFactor() / 25.0)
    axesTubes.SetNumberOfSides(6)

    axesMapper = vtk.vtkPolyDataMapper()
    axesMapper.SetInputConnection(axesTubes.GetOutputPort())

    axesActor = vtk.vtkActor()
    axesActor.SetMapper(axesMapper)

    # Visualize.
    renderer = vtk.vtkRenderer()
    renderWindow = vtk.vtkRenderWindow()
    renderWindow.AddRenderer(renderer)
    interactor = vtk.vtkRenderWindowInteractor()
    interactor.SetRenderWindow(renderWindow)

    renderer.AddActor(axesActor)
    renderer.AddActor(lateActor)
    renderer.SetBackground(colors.GetColor3d("powder_blue"))

    # Enable user interface interactor
    renderWindow.Render()

    def update(self):
        pts = dataSet.GetPoints()
        pts.SetPoint(0, [0,0,np.sin(time.time())])
        pts.Modified()
        interactor.Render()

    # Sign up to receive TimerEvent
    cb = vtkTimerCallback()
    cb.update = update

    interactor.AddObserver('TimerEvent', cb.execute)
    timerId = interactor.CreateRepeatingTimer(10);

    interactor.Start()

class vtkTimerCallback():
    def __init__(self):
       self.timer_count = 0
       self.start_time = time.time()

    def execute(self,obj,event):
       # print(self.timer_count)
       # self.noise.SetPhase(0, 0, time.time() - self.start_time)
       # self.actor.SetPosition(self.timer_count, self.timer_count,0);
       iren = obj
       iren.GetRenderWindow().Render()
       self.update(self)

    def update(self):
        pass

def make_dataset():
    n_points = 100
    xyz = np.random.random_sample((n_points,4))

    newPts = vtk.vtkPoints()
    newScalars = vtk.vtkFloatArray()
    # xyz = list(zip(res[keys[1]], res[keys[2]], res[keys[3]]))
    for i in range(0, n_points):
        # print(xyz[i])
        newPts.InsertPoint(i, xyz[i][0:3])
        newScalars.InsertValue(i, xyz[i][2]*100)

    dataset = vtk.vtkUnstructuredGrid()
    dataset.SetPoints(newPts)
    dataset.GetPointData().SetScalars(newScalars)
    return dataset


if __name__ == '__main__':
    main()