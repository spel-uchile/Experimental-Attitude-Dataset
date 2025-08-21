"""
Created by Elias Obreque
Date: 17-10-2023
email: els.obrq@gmail.com
"""

SATELLITE_NAME = "SUCHAI-3"
FORCE_CALCULATION = False
OBC_DATA = "gyros-S3-040923.xlsx"
OBC_DATA_STEP = 1.0
VIDEO_FPS = 30
VIDEO_DATA = ["20230904-video-att6-clip.mp4", "20230904-video-att7 - clip.mp4", "20230904-video-att9-clip.mp4"]  # reference unit time
VIDEO_TIME_LAST_FRAME = ["2023/09/04 14:49:04", "2023/09/04 16:21:21", "2023/09/04 16:22:40"]
VIDEO_CORRECTION_TIME = 0.5
IMAGEN_DATA = None
# data wind time
WINDOW_TIME = {'Start': '2023/09/04 14:48:23',
               'Stop': '2023/09/04 16:22:40',
               'STEP': 1.0,
               'FLAG': True}  # if true use the manual time, else use the gyro data time
TIME_FORMAT = "%Y/%m/%d %H:%M:%S"

ONLINE_MAG_CALIBRATION = True
CREATE_FRAME = False
GET_VECTOR_FROM_PICTURE = True
EKF_SETUP = 'NORMAL'
