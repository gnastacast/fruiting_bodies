import vtk
import sys
import numpy as np
import time

class AnimatedPolydata(object):
    def __init__(self):
        self.ren = vtk.vtkRenderer()
        self.renWin = vtk.vtkRenderWindow()
        self.renWin.AddRenderer(self.ren)
        self.iren = vtk.vtkRenderWindowInteractor()
        self.iren.SetRenderWindow(self.renWin)
        self.iren.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())
        self.ren.ResetCamera()


        # Point coordinate data ---------------------------------
        self.coordData = np.random.random_sample((4,3))

        # Create the polydata object -----------------------------
        points = vtk.vtkPoints()
        points.SetNumberOfPoints(len(self.coordData))
        self.polydata = vtk.vtkPolyData()

        for i in range(len(self.coordData)):
            points.SetPoint(i, self.coordData[i])
        self.polydata.SetPoints(points)

        self.ptsFilter = vtk.vtkVertexGlyphFilter()
        self.ptsFilter.SetInputData(self.polydata)
        ptsMapper = vtk.vtkPolyDataMapper()
        ptsMapper.SetInputConnection(self.ptsFilter.GetOutputPort())
        ptsActor = vtk.vtkActor()
        ptsActor.SetMapper(ptsMapper)
        ptsActor.GetProperty().SetPointSize(10)

        self.ren.AddActor(ptsActor)

        # Enable user interface interactor
        self.renWin.Render()

        self.iren.AddObserver('TimerEvent', self.cb)
        timerId = self.iren.CreateRepeatingTimer(1);

        self.iren.Start()

    def cb(self, obj, event):
        points = self.polydata.GetPoints()
        coordData = self.coordData
        coordData[0,0] = np.sin(time.time())
        points.SetPoint(0, coordData[0])
        print(coordData[0])
        points.Modified()
        self.iren.Render()
        return

if __name__ == "__main__":

    window = AnimatedPolydata()
    sys.exit()
# import numpy as np
# import vtk
# from vtk.qt4.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
# from PyQt4 import QtGui
# import sys

# class ViewerWithScrollBar(QtGui.QMainWindow):
#     def __init__(self, parent=None):
#         super(ViewerWithScrollBar, self).__init__(parent)
#         # Define the renderer and Qt window ------------------------
#         self.frame = QtGui.QFrame()

#         self.hl = QtGui.QHBoxLayout()
#         self.vtkWidget = QVTKRenderWindowInteractor(self.frame)
#         self.hl.addWidget(self.vtkWidget)

#         self.ren = vtk.vtkRenderer()
#         self.vtkWidget.GetRenderWindow().AddRenderer(self.ren)
#         self.iren = self.vtkWidget.GetRenderWindow().GetInteractor()
#         self.iren.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())
#         self.ren.ResetCamera()

#         self.frame.setLayout(self.hl)
#         self.setCentralWidget(self.frame)

#         # Point coordinate data ---------------------------------
#         self.coordData = {}
#         self.coordData[0] = np.array([[0,0,0], [1,0,0], [1,1,0]])
#         self.coordData[1] = self.coordData[0] + np.array([[0.2, 0.1, -0.05], [0,0,0], [0,0,0]])
#         self.coordData[2] = self.coordData[1] + np.array([[0.2, 0.1, -0.05], [0,0,0], [0,0,0]])

#         # Define the slider bar and add it to the window ---------------
#         slider = QtGui.QSlider()
#         slider.setAccessibleName('Time index')
#         slider.setRange(0, len(self.coordData)-1)
#         slider.valueChanged.connect(self.sliderCallback)

#         self.hl.addWidget(slider)

#         # Create the polydata object -----------------------------
#         points = vtk.vtkPoints()
#         points.SetNumberOfPoints(len(self.coordData[0]))
#         self.polydata = vtk.vtkPolyData()

#         for i in range(len(self.coordData[0])):
#             points.SetPoint(i, self.coordData[0][i])
#         self.polydata.SetPoints(points)

#         self.ptsFilter = vtk.vtkVertexGlyphFilter()
#         self.ptsFilter.SetInputData(self.polydata)
#         ptsMapper = vtk.vtkPolyDataMapper()
#         ptsMapper.SetInputConnection(self.ptsFilter.GetOutputPort())
#         ptsActor = vtk.vtkActor()
#         ptsActor.SetMapper(ptsMapper)
#         ptsActor.GetProperty().SetPointSize(10)

#         self.ren.AddActor(ptsActor)

#         self.show()
#         self.iren.Initialize()

#     def sliderCallback(self):
#         index = self.sender().value() # The index that the slider bar is currently on
#         points = self.polydata.GetPoints()
#         for i in range(len(self.coordData[index])):
#             points.SetPoint(i, self.coordData[index][i])
#         self.polydata.Modified()
#         self.iren.Render()
#         return

# if __name__ == "__main__":

#     app = QtGui.QApplication(sys.argv)

#     window = ViewerWithScrollBar()

#     sys.exit(app.exec_())