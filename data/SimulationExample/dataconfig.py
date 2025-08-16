"""
Created by Elias Obreque
Date: 17-10-2023
email: els.obrq@gmail.com
"""
SIMULATION = True
SATELLITE_NAME = "SUCHAI-3"
FORCE_CALCULATION = False
OBC_DATA = "imu-sim.xlsx"
VIDEO_DATA = ["sim-att1.avi", "sim-att2.avi", "sim-att3.avi"]  # reference unit time
IMAGEN_DATA = None
VIDEO_TIME_LAST_FRAME = ["2023/08/24 14:05:00", "2023/08/24 14:15:00", "2023/08/24 15:50:00"]
OBC_DATA_STEP = 1
# data wind time 15:40
WINDOW_TIME = {'Start': '2023/08/24 14:00:00',
               'Stop': '2023/08/24 16:00:00',
               # 'STEP': 1.0,
               'FLAG': True}
TIME_FORMAT = "%Y/%m/%d %H:%M:%S"

ONLINE_MAG_CALIBRATION = True
CREATE_FRAME = False
GET_VECTOR_FROM_PICTURE = True
EKF_SETUP = 'NORMAL'
