"""
Created by Elias Obreque
Date: 17-10-2023
email: els.obrq@gmail.com
"""

OBC_DATA = "obc_adcs.xlsx"
VIDEO_DATA = None  # reference unit time
VIDEO_TIME_LAST_FRAME = None
IMAGEN_DATA = None
# data wind time
WINDOW_TIME = {'Start': '2024/04/08 15:40:46',
               'Stop': '2024/04/08 15:45:45',
               'STEP': 1,
               'FLAG': True}  # if true use the manual time, else use the gyro data time
TIME_FORMAT = "%Y/%m/%d %H:%M:%S"

ONLINE_MAG_CALIBRATION = True
CREATE_FRAME = False
GET_VECTOR_FROM_PICTURE = False
EKF_SETUP = 'NORMAL'
