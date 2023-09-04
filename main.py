"""
Created by Elias Obreque
Date: 04-09-2023
email: els.obrq@gmail.com
"""
import matplotlib.pyplot as plt

from data.value_mag_calibration import get_s3_mag_cal
from get_video_frame import save_frame
import pandas as pd
import numpy as np

# CONFIG
PROJECT_FOLDER = "data/M-20230824/"
OBC_DATA = "gyros-S3-240823.xlsx"
VIDEO_DATA = "20230824-att1-original.mp4"    # reference unit time
ts2022 = 1640995200 / 86400  # day
jd2022 = 2459580.50000


CREATE_FRAME = False

if __name__ == '__main__':
    if CREATE_FRAME:
        save_frame(PROJECT_FOLDER, VIDEO_DATA)
    D, bias = get_s3_mag_cal()

    # Real datal
    sensor_data = pd.read_excel(PROJECT_FOLDER + OBC_DATA)
    sensor_data.sort_values(by=['timestamp'], inplace=True)
    sensor_data['jd'] = sensor_data['timestamp'].values / 86400 - ts2022 + jd2022
    sensor_data[['mag_x', 'mag_y', 'mag_z']] = np.matmul(sensor_data[['mag_x', 'mag_y', 'mag_z']], (np.eye(3) + D)) - bias

    sensor_data[['mag_x', 'mag_y', 'mag_z']].plot()
    plt.show()
