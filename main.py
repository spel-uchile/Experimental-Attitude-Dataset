"""
Created by Elias Obreque
Date: 04-09-2023
email: els.obrq@gmail.com
"""
from data_process import RealData
from data.value_mag_calibration import get_s3_mag_cal
from get_video_frame import save_frame
import numpy as np
import datetime

# CONFIG
PROJECT_FOLDER = "data/M-20230824/"
OBC_DATA = "gyros-S3-240823.xlsx"
VIDEO_DATA = "20230824-att1-original.mp4"    # reference unit time


# data wind time
WINDOW_TIME = {'Start': '2023/08/24 10:44:09',
               'Stop': '2023/08/24 11:40:43',
               'STEP': 1,
               'FLAG': True}
TIME_FORMAT = "%Y/%m/%d %H:%M:%S"


CREATE_FRAME = False

if __name__ == '__main__':
    if CREATE_FRAME:
        save_frame(PROJECT_FOLDER, VIDEO_DATA)
    D, bias = get_s3_mag_cal()

    # create data with datetime
    sensor_data = RealData(PROJECT_FOLDER + OBC_DATA)
    sensor_data.create_datetime_from_timestamp(TIME_FORMAT)
    # mag calibration
    sensor_data.calibrate_mag(scale=D, bias=bias)
    # show window time
    if WINDOW_TIME['FLAG']:
        sensor_data.set_window_time(WINDOW_TIME['Start'], WINDOW_TIME['Stop'], TIME_FORMAT)
    else:
        sensor_data.set_window_time()

    # plot
    sensor_data.plot_key(['mag_x', 'mag_y', 'mag_z'])
    sensor_data.plot_key(['acc_x', 'acc_y', 'acc_z'])
    sensor_data.plot_key(['sun3', 'sun2', 'sun4'], show=True)

    # TIME
    dt = WINDOW_TIME['STEP']
    start_datetime = datetime.datetime.strptime(WINDOW_TIME['Start'], TIME_FORMAT)
    stop_datetime = datetime.datetime.strptime(WINDOW_TIME['Stop'], TIME_FORMAT)
    print(start_datetime.timestamp(), stop_datetime.timestamp())


    current_time = 0


