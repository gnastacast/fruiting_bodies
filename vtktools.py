import numpy as np
from vtk.util import numpy_support
import vtk
import cv2

def npMatrixToVtkMatrix(matrix):
    retMat = vtk.vtkMatrix4x4()
    shape = matrix.shape
    for r in range(0,shape[0]):
        for c in range(0,shape[1]):
            retMat.SetElement(r,c,matrix[r,c])
    return retMat

def vtkMatrixtoNpMatrix(matrix):
    return np.array(matrix.GetData(), (4,4))

# Sets up a three channel empty vtkImageData object
def makeVtkImage(imgDims):
    ''' Creates a blank vtkImageData object of specified dimensions
        Input: imgDims (int, int): desired width and height
        Output: imData (vtkImageData): empty VTK image with three
                                       channels for RGB
    '''
    imData = vtk.vtkImageData()
    imData.SetDimensions(imgDims[1],imgDims[0],1) #set dimensions as necessary
    imData.SetOrigin(0,0,0) #set origin as necessary
    imData.SetSpacing(1, 1, 1) #set spacing as necessary
    # Set number of channels (3)
    if vtk.VTK_MAJOR_VERSION <= 5:
        imData.SetNumberOfScalarComponents(3)
        imData.SetScalarTypeToUnsignedChar()
        imData.Update()
    else:
        imData.AllocateScalars(vtk.VTK_UNSIGNED_CHAR, 3)
    return imData

def numpyToVtkImage(numpyData,VTKImageData):
    ''' Copies a numpy array to a vtkImageData object in place.
        Used for updating existing vtkImageData with new pixel
        data.
        https://pyscience.wordpress.com/2014/09/06/numpy-to-vtk-converting-your-numpy-arrays-to-vtk-arrays-and-files/
        was extremely useful
    '''
    if len(numpyData.shape)==3 and numpyData.shape[2] > 2 and numpyData.shape[2] <= 4:
        # Data from CV comes as BGR and vtkImageData displays as RGB so we convert
        # rgb = cv2.cvtColor(numpyData, cv2.COLOR_BGR2RGB)
        rgb = numpyData.copy()
        rgb[:,:,0:3] = cv2.cvtColor(numpyData[:,:,0:3], cv2.COLOR_BGR2RGB)
        # Get a vtkDataArray object containing numpy pixel data
        VTK_data = numpy_support.numpy_to_vtk(num_array=rgb[::-1].ravel(),
                                              deep=True, 
                                              array_type=vtk.VTK_UNSIGNED_CHAR)
        # Set number of channels in pixel data
        VTK_data.SetNumberOfComponents(numpyData.shape[2])
    else:
        #Get a vtkDataArray object containing pixel data
        VTK_data = numpy_support.numpy_to_vtk(num_array=numpyData[::-1].ravel(),
                                              deep=True, 
                                              array_type=vtk.VTK_UNSIGNED_CHAR)
    VTKImageData.GetPointData().SetScalars(VTK_data)

def vtkImageToNumpy(VTKImageData):
    ''' Copies pixel data from a vtkImageData and returns a new numpy array
    '''
    size = VTKImageData.GetDimensions()
    shape = (size[1],size[0],VTKImageData.GetNumberOfScalarComponents())
    VTK_data = VTKImageData.GetPointData().GetScalars()
    numpyData = numpy_support.vtk_to_numpy(VTK_data)
    numpyData = numpyData.reshape(shape)
    numpyData = numpyData[::-1]
    return numpyData

def actorFromStl(stlPath):
    ''' Reads in an STL file and returns a vtkActor
        object that the vtkRenderer can use
    '''
    reader = vtk.vtkSTLReader()
    reader.SetFileName(stlPath)
    mapper = vtk.vtkPolyDataMapper()
    if vtk.VTK_MAJOR_VERSION <= 5:
        mapper.SetInput(reader.GetOutput())
    else:
        mapper.SetInputConnection(reader.GetOutputPort())
    # Create actor that we will position manually
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    return actor

def setupRenWinForRegistration(renWin, bgImage, fgImage, camIntrinsic):
    ''' Sets up the render window for registration in place
        Input:
            renWin (vtkRenderWindow): render window to set up with
                                      actors and renderers
            bgImage (vtkImageData): image to put in the background
                                    that will be updated with video
                                    footage and masking feedback
            camIntrinsic (np.ndarray): 4x4 camera intrinsic matrix from
                                       OpenCV calibration
    '''


    renWin.SetNumberOfLayers(3)

    # Create main renderer for stl object
    ren = vtk.vtkRenderer()
    ren.SetLayer(1)
    renWin.AddRenderer(ren)
    imgDims = bgImage.GetDimensions()
    renWin.SetSize(imgDims[0],imgDims[1])
    camera = cameraFromMatrix(camIntrinsic,imgDims,imgDims)
    ren.SetActiveCamera(camera)

    # Create background renderer for static object
    backgroundRen = vtk.vtkRenderer()
    backgroundRen.SetLayer(0)
    backgroundRen.InteractiveOff()
    backgroundRen.SetBackground(1, 1, 1)
    # Create background object for showing video feed.
    bgCamera, backgroundActor = _makeImagePlane(bgImage)
    # Setup camera
    backgroundRen.AddActor(backgroundActor)
    backgroundRen.SetActiveCamera(bgCamera)
    # Add background renderer to window
    renWin.AddRenderer(backgroundRen)

    # Create background renderer for static object
    foregroundRen = vtk.vtkRenderer()
    foregroundRen.SetLayer(2)
    foregroundRen.InteractiveOff()
    # Create background object for showing video feed.
    fgCamera, foregroundActor = _makeImagePlane(fgImage)
    # Setup camera
    foregroundRen.AddActor(foregroundActor)
    foregroundRen.SetActiveCamera(fgCamera)
    # Add background renderer to window
    renWin.AddRenderer(foregroundRen)
    
    # Way of getting camera: renWin.GetRenderers().GetFirstRenderer()
    return ren, backgroundRen, foregroundRen

class vtkTimerCallback():
   # Callback that renders on a timer
    def __init__(self, renWin, parent=None):
        self.renderWindows = [renWin]
        self.parent = parent
    def execute(self,obj,event):
        # for renWin in self.renderWindows:
        #     if self.parent is None:
        #         renWin.Render()
        #     else:
        #         if self.parent.isVisible():
        #             renWin.Render()
        self.update()
    def addRenWin(self,renWin):
        self.renderWindows.append(renWin)
    def update(self):
        pass


def matrixFromCamInfo(camInfo):
    # Get intrinsic matrix of rectified image
    intrinsicMatrix = np.reshape(camInfo.P,(3,4))[0:3,0:3]
    # Get camera extrinsic matrix
    extrinsicMatrix = np.identity(4)
    # Get extrinsic rotation
    extrinsicMatrix [0:3,0:3] = np.reshape(camInfo.R,(3,3))
    # Get baseline translation (will usually be zero as this is for left camera)
    xBaseline = camInfo.P[3] / -camInfo.P[0]
    yBaseline = camInfo.P[7] / -camInfo.P[5]
    extrinsicMatrix [0:3,3] = [xBaseline, yBaseline, 0]

    return intrinsicMatrix, extrinsicMatrix

class zBuff:
    def __init__(self,renWin):
        self.renWin = renWin
        self.zBuffFilter = vtk.vtkWindowToImageFilter()
        self.zBuffFilter.SetInput(self.renWin)
        self.zBuffFilter.SetMagnification(1)
        self.zBuffFilter.SetInputBufferTypeToZBuffer()

        self.scaledZBuff = vtk.vtkImageShiftScale()
        self.scaledZBuff.SetOutputScalarTypeToUnsignedChar()
        self.scaledZBuff.SetInputConnection(self.zBuffFilter.GetOutputPort())
        self.scaledZBuff.SetShift(0)
        self.scaledZBuff.SetScale(-255)

    def GetOutput(self):
        # self.renWin.Render()
        self.zBuffFilter.Modified()
        self.scaledZBuff.Update()
        return self.scaledZBuff.GetOutput()

# Loads an image and returns a plane object and a camera pointed towards it
def _makeImagePlane(imageData) :
    ''' Takes a vtkImageData object and
        returns a plane textured with that image
        and a camera for the background renderer
        http://www.vtk.org/Wiki/VTK/Examples/Cxx/Images/BackgroundImage
        was the basis for this function
    '''
    imageActor = vtk.vtkImageActor()

    if vtk.VTK_MAJOR_VERSION <= 5:
        imageActor.SetInput(imageData)
    else:
        imageActor.SetInputData(imageData)

    origin = imageData.GetOrigin()
    spacing = imageData.GetSpacing()
    extent = imageData.GetExtent()

    camera = vtk.vtkCamera()
    camera.ParallelProjectionOn()

    xc = origin[0] + 0.5*(extent[0] + extent[1])*spacing[0]
    yc = origin[1] + 0.5*(extent[2] + extent[3])*spacing[1]
    yd = (extent[3] - extent[2] + 1)*spacing[1]
    d = camera.GetDistance()
    camera.SetParallelScale(0.5*yd)
    camera.SetFocalPoint(xc,yc,0.0)
    camera.SetPosition(xc,yc,d)

    return camera, imageActor

def setCameraExtrinsic(camera,matrix):
    R = matrix[0:3,0:3]
    up = np.dot(R,[0,-1,0])
    forward = np.dot(R,[0,0,1])
    camera.SetViewUp(up)
    camera.SetFocalPoint(forward)
    # eye = (-R'T), center = eye + R'(0,0,1)', up = -R[0][1]
    return

# Sets up a camera based on a camera intrinsics matrix from OpenCV
def cameraFromMatrix(camMatrix, imageDims, windowDims) :
    ''' Takes a camera matrix derived from OpenCV
        returns a camera for registration
        http://stackoverflow.com/questions/25539898/how-to-apply-the-camera-pose-transformation-computed-using-epnp-to-the-vtk-camer
        was a great help
        Input: 
            camMatrix (np.ndarray) = 4x4 camera intrinsic matrix
            imageDims ((float, float))  = width and height of image to register
            imageDims ((float, float))  = width and height of render window
        Output:
            camera (vtk.vtkCamera) = camera to use for registration
    '''
    # Initialize camera
    camera = vtk.vtkCamera()

    # Set view angle
    focalLengthY = camMatrix[1,1]
    if( windowDims[1] != imageDims[1] ):
        factor = float(windowDims[1])/float(imageDims[1])
        focalLengthY = camMatrix[1,1] * factor
    viewAngle = np.arctan((windowDims[1]/2.0)/focalLengthY)*360/np.pi
    camera.SetViewAngle(viewAngle)

    #Set window center
    py = 0
    height = 0
    if imageDims[0] != windowDims[0] or imageDims[1] != windowDims[1] :
        factor = float(windowDims[1])/float(imageDims[1])
        px = factor * camMatrix[0,2]
        width = windowDims[0]
        expectedWindowSize = np.round(factor * float(imageDims[0]))
        if( expectedWindowSize != windowDims[0] ):
            diffX = (windowDims[0] - expectedWindowSize) / 2
            px = px + diffX
        py = factor * camMatrix[1,2]
        height = windowDims[1]
    else :
        px = camMatrix[0,2]
        width = imageDims[0]
        py = camMatrix[1,2]
        height = imageDims[1]
    cx = width - px
    cy = py
    windowCenter = [0,0]
    windowCenter[0] = cx / ( (width)/2 ) - 1 
    windowCenter[1] = cy / ( (height)/2 ) - 1
    camera.SetWindowCenter(windowCenter[0],windowCenter[1])

    # Set camera to look forward from center
    camera.SetPosition(0,0,0)
    camera.SetViewUp([0, -1, 0])
    camera.SetFocalPoint(0,0,1)
    
    return camera