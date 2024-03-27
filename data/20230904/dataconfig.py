"""
Created by Elias Obreque
Date: 17-10-2023
email: els.obrq@gmail.com
"""

OBC_DATA = "gyros-S3-040923.xlsx"
VIDEO_DATA = "20230904-video-att6-clip.mp4"  # reference unit time
VIDEO_TIME_LAST_FRAME = "2023/09/04 14:49:04"
IMAGEN_DATA = None
# data wind time
WINDOW_TIME = {'Start': '2023/09/04 14:48:23',
               'Stop': '2023/09/04 16:18:22',
               'STEP': 1,
               'FLAG': True}  # if true use the manual time, else use the gyro data time
TIME_FORMAT = "%Y/%m/%d %H:%M:%S"

ONLINE_MAG_CALIBRATION = False
CREATE_FRAME = False
GET_VECTOR_FROM_PICTURE = False
EKF_SETUP = 'NORMAL'
