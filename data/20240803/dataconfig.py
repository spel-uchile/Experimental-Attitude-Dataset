"""
Created by Elias Obreque
Date: 17-10-2023
email: els.obrq@gmail.com
"""
FORCE_CALCULATION = False
OBC_DATA = "obc-20240322.xlsx"
VIDEO_DATA = None # reference unit time
VIDEO_TIME_LAST_FRAME = None
IMAGEN_DATA = None
# data wind time
WINDOW_TIME = {'Start': '2024/03/22 15:46:41',
               'Stop': '2024/03/22 15:48:37',
               'STEP': 1,
               'FLAG': True}  # if true use the manual time, else use the gyro data time
TIME_FORMAT = "%Y/%m/%d %H:%M:%S"

ONLINE_MAG_CALIBRATION = False
CREATE_FRAME = False
GET_VECTOR_FROM_PICTURE = False
EKF_SETUP = 'NORMAL'
