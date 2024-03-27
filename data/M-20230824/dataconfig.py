"""
Created by Elias Obreque
Date: 17-10-2023
email: els.obrq@gmail.com
"""

OBC_DATA = "gyros-S3-240823.xlsx"
VIDEO_DATA = "20230824-att1-original.mp4"  # reference unit time
IMAGEN_DATA = None
# data wind time
WINDOW_TIME = {'Start': '2023/08/24 13:30:09',
               'Stop': '2023/08/24 15:10:43',
               'STEP': 1,
               'FLAG': True}
TIME_FORMAT = "%Y/%m/%d %H:%M:%S"

ONLINE_MAG_CALIBRATION = True
CREATE_FRAME = False
GET_VECTOR_FROM_PICTURE = True
EKF_SETUP = 'NORMAL'
