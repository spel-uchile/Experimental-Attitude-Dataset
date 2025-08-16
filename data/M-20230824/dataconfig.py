"""
Created by Elias Obreque
Date: 17-10-2023
email: els.obrq@gmail.com
"""
SIMULATION = True
SATELLITE_NAME = "SUCHAI-3"
FORCE_CALCULATION = False
OBC_DATA = "gyros-S3-240823-full.xlsx"
VIDEO_DATA = "20230824-att1-original.mp4"  # reference unit time
VIDEO_TIME_LAST_FRAME = "2023/08/24 14:49:04" #?????????????
IMAGEN_DATA = None
VIDEO_TIME_LAST_FRAME = None
OBC_DATA_STEP = 1
# data wind time
WINDOW_TIME = {'Start': '2023/08/24 14:44:09',
               'Stop': '2023/08/24 16:17:28',
               'STEP': 1,
               'FLAG': True}
TIME_FORMAT = "%Y/%m/%d %H:%M:%S"

ONLINE_MAG_CALIBRATION = True
CREATE_FRAME = False
GET_VECTOR_FROM_PICTURE = False
EKF_SETUP = 'NORMAL'
