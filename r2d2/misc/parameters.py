from cv2 import aruco

# Robot Params #
nuc_ip = "172.16.0.4"
robot_ip = "172.16.0.2"
sudo_password = "iloverobots"
robot_type = "panda"  # 'panda' or 'fr3'
robot_serial_number = "295341-1324688"

# Camera ID's #
hand_camera_id = "17225336"
varied_camera_1_id = "25047636" #left
varied_camera_2_id = "24013089" #right

#hand_camera_id = ""
#varied_camera_1_id = ""
#varied_camera_2_id = ""

# Charuco Board Params #
CHARUCOBOARD_ROWCOUNT = 9
CHARUCOBOARD_COLCOUNT = 14
CHARUCOBOARD_CHECKER_SIZE = 0.020
CHARUCOBOARD_MARKER_SIZE = 0.016
ARUCO_DICT = aruco.Dictionary_get(aruco.DICT_5X5_100)

# Code Version [DONT CHANGE] #
r2d2_version = "1.3"
