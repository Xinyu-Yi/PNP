r"""
    Detect aruco patterns' pose. See https://blog.csdn.net/dgut_guangdian/article/details/107814300
"""

import cv2
import cv2.aruco as aruco
import numpy as np


markerLength = 0.12  # meter
cameraMatrix = np.array([[635.33, 0.,     316.41],
                         [0.,     635.24, 237.14],
                         [0.,     0.,     1.    ]])  # camera intrinsic 640x480
distCoeffs = np.array([0.04431522, -0.24497552, 0.00103816, -0.0019876, 0.32108645])  # camera distortion
objPoints = np.array([[-markerLength/2,  markerLength/2, 0],
                      [ markerLength/2,  markerLength/2, 0],
                      [ markerLength/2, -markerLength/2, 0],
                      [-markerLength/2, -markerLength/2, 0]])  # in frame at the center of the marker

cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
ARUCO_PARAMETERS = aruco.DetectorParameters()
ARUCO_DICT = aruco.getPredefinedDictionary(aruco.DICT_6X6_50)  # 6X6, id 0~49
detector = aruco.ArucoDetector(ARUCO_DICT, ARUCO_PARAMETERS)

while True:
    ret, frame = cap.read()
    corners, ids, rejectedCandidates = detector.detectMarkers(frame)
    if ids is not None:
        aruco.drawDetectedMarkers(frame, corners, ids)
        for i, corner in zip(ids, corners):
            _, rvec, tvec = cv2.solvePnP(objPoints, corner, cameraMatrix, distCoeffs, flags=cv2.SOLVEPNP_IPPE_SQUARE)  # Rco, tco
            cv2.drawFrameAxes(frame, cameraMatrix, distCoeffs, rvec, tvec, 0.1)

    cv2.imshow('frame', frame)
    cv2.waitKey(1)
