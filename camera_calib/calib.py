from __future__ import print_function, division
import numpy as np
import cv2
import glob
import pickle

# https://medium.com/@kennethjiang/calibrate-fisheye-lens-using-opencv-part-2-13990f1b157f
def undistort(img_path, camMat, dist, balance=0.0, dim2=None, dim3=None):
    img = cv2.imread(img_path)
    dim1 = img.shape[:2][::-1]  #dim1 is the dimension of input image to un-distort
    assert dim1[0]/dim1[1] == DIM[0]/DIM[1], "Image to undistort needs to have same aspect ratio as the ones used in calibration"
    if not dim2:
        dim2 = dim1
    if not dim3:
        dim3 = dim1
    scaled_K = K * dim1[0] / DIM[0]  # The values of K is to scale with image dimension.
    scaled_K[2][2] = 1.0  # Except that K[2][2] is always 1.0
    # This is how scaled_K, dim2 and balance are used to determine the final K used to un-distort image. OpenCV document failed to make this clear!
    new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(scaled_K, D, dim2, np.eye(3), balance=balance)
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(scaled_K, D, np.eye(3), new_K, dim3, cv2.CV_16SC2)
    undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    cv2.imshow("undistorted", undistorted_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main():
    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((6*9,3), np.float32)
    objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)
    objp = objp * 0.0254

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpointsL = [] # 2d points in image plane.
    imgpointsR = [] # 2d points in image plane.

    images = glob.glob('*.jpg')
    images.sort()
    width = 0
    height =0

    for fname in images:
        img = cv2.imread(fname)
        height, width = img.shape[0:2]
        imgL = img[:,0:width//2];
        imgR = img[:,width//2 : width];
        grayL = cv2.cvtColor(imgL,cv2.COLOR_BGR2GRAY)
        grayR = cv2.cvtColor(imgR,cv2.COLOR_BGR2GRAY)

        # Find the chess board corners
        retL, cornersL = cv2.findChessboardCorners(grayL, (9,6), None)
        retR, cornersR = cv2.findChessboardCorners(grayR, (9,6), None)

        # If found, add object points, image points (after refining them)
        if retL == True and retR == True:
            print(fname)
            objpoints.append(objp)

            corners2L = cv2.cornerSubPix(grayL,cornersL,(11,11),(-1,-1),criteria)
            imgpointsL.append(corners2L)
            corners2R = cv2.cornerSubPix(grayR,cornersR,(11,11),(-1,-1),criteria)
            imgpointsR.append(corners2R)

            # Draw and display the corners
            imgL = cv2.drawChessboardCorners(imgL, (9,6), corners2L, retL)
            offset = np.ones((len(corners2R), 1, 2)) * width//2
            cv2.imshow('imgL',imgL)
            cv2.waitKey(5)

    calibration_flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC+cv2.fisheye.CALIB_CHECK_COND+cv2.fisheye.CALIB_FIX_SKEW

    M1 = np.zeros((3, 3))
    d1 = np.zeros((4, 1))
    r1 = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(len(objpoints))]
    t1 = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(len(objpoints))]

    rms, _, _, _, _ =  cv2.fisheye.calibrate(np.reshape(objpoints, (len(objpoints), 1, -1, 3)),
                                             np.reshape(imgpointsL, (len(objpoints), 1, -1, 2)),
                                             grayL.shape[::-1],
                                             M1,
                                             d1,
                                             r1,
                                             t1,
                                             calibration_flags,
                                             (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6))

    print(rms)
    print(M1)
    print(d1)

    # m1 = np.reshape([898.8330441160133, 0, 640, 0, 898.5978305609192, 480, 0, 0, 1], (3,3))
    # d1 = np.array([-0.3984360149537951, 0.1302381183957002, 0.009177901416678803, 0.007710077375683005, 0])

    rms, M1, d1, r1, t1 = cv2.calibrateCamera(
                objpoints, imgpointsL, (width//2, height), None, None, flags=calibration_flags)
    rms, M2, d2, r2, t2 = cv2.calibrateCamera(
                objpoints, imgpointsR, (height,width//2), None, None)

    print(rms)
    print(M1)
    print(d1)

    # tot_error = 0
    # for i in xrange(len(objpoints)):
    #     imgpoints2, _ = cv2.projectPoints(objpoints[i], r1[i], t1[i], M1, d1)
    #     error = cv2.norm(imgpointsL[i],imgpoints2, cv2.NORM_L2)/len(imgpoints2)
    #     tot_error += error
    #     imgpoints2, _ = cv2.projectPoints(objpoints[i], r2[i], t2[i], M2, d2)
    #     error = cv2.norm(imgpointsR[i],imgpoints2, cv2.NORM_L2)/len(imgpoints2)
    #     tot_error += error

    # print("mean error: ", tot_error/len(objpoints)/2)

    flags = 0
    flags |= cv2.CALIB_FIX_INTRINSIC
    # flags |= cv2.CALIB_FIX_PRINCIPAL_POINT
    flags |= cv2.CALIB_USE_INTRINSIC_GUESS
    flags |= cv2.CALIB_FIX_FOCAL_LENGTH
    # flags |= cv2.CALIB_FIX_ASPECT_RATIO
    flags |= cv2.CALIB_ZERO_TANGENT_DIST
    # flags |= cv2.CALIB_RATIONAL_MODEL
    flags |= cv2.CALIB_SAME_FOCAL_LENGTH
    # flags |= cv2.CALIB_FIX_K3
    # flags |= cv2.CALIB_FIX_K4
    # flags |= cv2.CALIB_FIX_K5

    stereocalib_criteria = (cv2.TERM_CRITERIA_MAX_ITER +
                            cv2.TERM_CRITERIA_EPS, 100, 1e-5)

    ret, M1, d1, M2, d2, R, T, E, F = cv2.stereoCalibrate(
        objpoints, imgpointsL,
        imgpointsR, M1, d1, M2,
        d2, (height,width//2),
        criteria=stereocalib_criteria, flags=flags)

    # newcameramtx, roi=cv2.getOptimalNewCameraMatrix(M1,d1,(width//2, height),.99)
    # newcameramtx, roi = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(M1.astype(cv2.CV_32F), d1.astype(cv2.CV_32F), (width//2, height), np.eye(3), balance=0.5)
    # print("ROI\n", roi)
    imgL = cv2.imread(images[0])[:,0:width//2].copy()
    print(imgL.shape)
    dst = cv2.undistort(imgL, M1, d1, None, M1)
    # crop the image
    # x,y,w,h = roi
    # dst = dst[y:y+h, x:x+w]

    # mapx,mapy = cv2.initUndistortRectifyMap(M1,d1,None,M1,(width//2, height),5)
    # dst = cv2.remap(img,mapx,mapy,cv2.INTER_LINEAR)
    cv2.imwrite('calibresult.png',dst)

    cv2.destroyAllWindows()

    camera_model = dict([('M1', M1), ('M2', M2), ('dist1', d1),
                        ('dist2', d2), ('rvecs1', r1),
                        ('rvecs2', r2), ('R', R), ('T', T),
                        ('E', E), ('F', F)])

    with open('calib.pkl', 'wb') as f:
        pickle.dump(camera_model, f, pickle.HIGHEST_PROTOCOL)

    # print('Intrinsic_mtx_1\n', M1)
    # print('dist_1\n', d1)
    # print('Intrinsic_mtx_2\n', M2)
    # print('dist_2\n', d2)
    # print('R\n', R)
    # print('T\n', T)
    # print('E\n', E)
    # print('F\n', F)

    # for i in range(len(r1)):
    #     print("--- pose[", i+1, "] ---")
    #     ext1, _ = cv2.Rodrigues(r1[i])
    #     ext2, _ = cv2.Rodrigues(r2[i])
    #     print('Ext1', ext1)
    #     print('Ext2', ext2)

    # print('')

if __name__ == '__main__':
    main()
'''
import numpy as np
import cv2
import glob
import os
import argparse

# https://github.com/bvnayak/stereo_calibration/blob/master/camera_calibrate.py
class StereoCalibration(object):
    def __init__(self, filepath):
        # termination criteria
        self.criteria = (cv2.TERM_CRITERIA_EPS +
                         cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        self.criteria_cal = (cv2.TERM_CRITERIA_EPS +
                             cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5)

        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        self.objp = np.zeros((9*6, 3), np.float32)
        self.objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)

        # Arrays to store object points and image points from all the images.
        self.objpoints = []  # 3d point in real world space
        self.imgpoints_l = []  # 2d points in image plane.
        self.imgpoints_r = []  # 2d points in image plane.

        self.cal_path = filepath
        self.read_images(self.cal_path)

    def read_images(self, cal_path):
        # images_right = glob.glob(cal_path + 'RIGHT/*.JPG')
        # images_left = glob.glob(cal_path + 'LEFT/*.JPG')
        # images_left.sort()
        # images_right.sort()

        # img_shape = (0,0)
        images = glob.glob(os.path.join(cal_path, "*.jpg"))
        img_shape = (0,0)

        print(images)

        for i, fname in enumerate(images):
            img = cv2.imread(images[i])
            width = img.shape[1]
            img_l = img[:,:width//2]
            img_r = img[:,width//2:]

            gray_l = cv2.cvtColor(img_l, cv2.COLOR_BGR2GRAY)
            gray_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)

            # Find the chess board corners
            ret_l, corners_l = cv2.findChessboardCorners(gray_l, (9, 6), None)
            ret_r, corners_r = cv2.findChessboardCorners(gray_r, (9, 6), None)

            # If found, add object points, image points (after refining them)
            self.objpoints.append(self.objp)

            if ret_l is True and ret_r is True:
                rt = cv2.cornerSubPix(gray_l, corners_l, (11, 11),
                                      (-1, -1), self.criteria)
                self.imgpoints_l.append(corners_l)

                # Draw and display the corners
                ret_l = cv2.drawChessboardCorners(img_l, (9, 6),
                                                  corners_l, ret_l)
                cv2.imshow('chessboard', img_l)
                cv2.waitKey(50)

                rt = cv2.cornerSubPix(gray_r, corners_r, (11, 11),
                                      (-1, -1), self.criteria)
                self.imgpoints_r.append(corners_r)

                # Draw and display the corners
                ret_r = cv2.drawChessboardCorners(img_r, (9, 6),
                                                  corners_r, ret_r)
                cv2.imshow('chessboard', img_r)
                cv2.waitKey(50)
            img_shape = gray_l.shape[::-1]

        rt, self.M1, self.d1, self.r1, self.t1 = cv2.calibrateCamera(
            self.objpoints, self.imgpoints_l, img_shape, None, None)
        rt, self.M2, self.d2, self.r2, self.t2 = cv2.calibrateCamera(
            self.objpoints, self.imgpoints_r, img_shape, None, None)

        self.camera_model = self.stereo_calibrate(img_shape)

    def stereo_calibrate(self, dims):
        flags = 0
        flags |= cv2.CALIB_FIX_INTRINSIC
        # flags |= cv2.CALIB_FIX_PRINCIPAL_POINT
        flags |= cv2.CALIB_USE_INTRINSIC_GUESS
        flags |= cv2.CALIB_FIX_FOCAL_LENGTH
        # flags |= cv2.CALIB_FIX_ASPECT_RATIO
        flags |= cv2.CALIB_ZERO_TANGENT_DIST
        # flags |= cv2.CALIB_RATIONAL_MODEL
        # flags |= cv2.CALIB_SAME_FOCAL_LENGTH
        # flags |= cv2.CALIB_FIX_K3
        # flags |= cv2.CALIB_FIX_K4
        # flags |= cv2.CALIB_FIX_K5

        stereocalib_criteria = (cv2.TERM_CRITERIA_MAX_ITER +
                                cv2.TERM_CRITERIA_EPS, 100, 1e-5)
        ret, M1, d1, M2, d2, R, T, E, F = cv2.stereoCalibrate(
            self.objpoints, self.imgpoints_l,
            self.imgpoints_r, self.M1, self.d1, self.M2,
            self.d2, dims,
            criteria=stereocalib_criteria, flags=flags)

        print('Intrinsic_mtx_1', M1)
        print('dist_1', d1)
        print('Intrinsic_mtx_2', M2)
        print('dist_2', d2)
        print('R', R)
        print('T', T)
        print('E', E)
        print('F', F)

        # for i in range(len(self.r1)):
        #     print("--- pose[", i+1, "] ---")
        #     self.ext1, _ = cv2.Rodrigues(self.r1[i])
        #     self.ext2, _ = cv2.Rodrigues(self.r2[i])
        #     print('Ext1', self.ext1)
        #     print('Ext2', self.ext2)

        print('')

        camera_model = dict([('M1', M1), ('M2', M2), ('dist1', d1),
                            ('dist2', d2), ('rvecs1', self.r1),
                            ('rvecs2', self.r2), ('R', R), ('T', T),
                            ('E', E), ('F', F)])

        cv2.destroyAllWindows()
        return camera_model

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('filepath', help='String Filepath')
    args = parser.parse_args()
    cal_data = StereoCalibration(args.filepath)
'''