"""
Created by Elias Obreque
Date: 17-10-2023
email: els.obrq@gmail.com
"""

SATELLITE_NAME = "SUCHAI-3"
FORCE_CALCULATION = False
OBC_DATA = "S3-gyros-13112023.xlsx"
OBC_DATA_STEP = 1.0
VIDEO_FPS = 30
VIDEO_DATA = ["attitude-10.mp4", "attitude-11.mp4", "attitude-12.mp4", "attitude-13.mp4"]  # reference unit time
VIDEO_TIME_LAST_FRAME = ["2023/11/13 15:49:05.24", "2023/11/13 15:50:53.97", "2023/11/13 15:53:19.48", "2023/11/13 15:55:28.32"]
VIDEO_CORRECTION_TIME = 0.5
IMAGEN_DATA = None
SIMULATION = False
# data wind time
WINDOW_TIME = {'Start': '2023/11/13 15:48:47',
               'Stop': '2023/11/13 16:05:26',
               'STEP': 1.0,
               'FLAG': True}  # if true use the manual time, else use the gyro data time
TIME_FORMAT = "%Y/%m/%d %H:%M:%S"

ONLINE_MAG_CALIBRATION = True
CREATE_FRAME = False
GET_VECTOR_FROM_PICTURE = True
EKF_SETUP = 'NORMAL'
