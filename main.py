"""
Created by Elias Obreque
Date: 04-09-2023
email: els.obrq@gmail.com
"""
import numpy as np
import datetime

from data.value_mag_calibration import get_s3_mag_cal

from src.data_process import RealData
from src.dynamics.dynamics_kinematics import *
from src.dynamics.Quaternion import Quaternions
from src.dynamics.MagEnv import MagEnv

from tools.get_video_frame import save_frame
from tools.monitor import Monitor

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

    # create data with datetime, and near tle
    sensors = RealData(PROJECT_FOLDER + OBC_DATA)
    sensors.create_datetime_from_timestamp(TIME_FORMAT)
    # mag calibration
    sensors.calibrate_mag(scale=D, bias=bias)
    # show window time
    if WINDOW_TIME['FLAG']:
        sensors.set_window_time(WINDOW_TIME['Start'], WINDOW_TIME['Stop'], TIME_FORMAT)
    else:
        sensors.set_window_time()
    line1, line2 = sensors.search_nearly_tle()

    # plot
    # sensors.plot_key(['mag_x', 'mag_y', 'mag_z'])
    # sensors.plot_key(['acc_x', 'acc_y', 'acc_z'])
    # sensors.plot_key(['sun3', 'sun2', 'sun4'], show=True)

    # TIME
    dt = WINDOW_TIME['STEP']
    start_datetime = datetime.datetime.strptime(WINDOW_TIME['Start'], TIME_FORMAT)
    stop_datetime = datetime.datetime.strptime(WINDOW_TIME['Stop'], TIME_FORMAT)
    print(start_datetime.timestamp(), stop_datetime.timestamp())

    # SIMULATION
    current_time = 0
    c_jd = sensors.data['jd'].values[0]
    tend = stop_datetime.timestamp() - start_datetime.timestamp()
    mag_model = MagEnv()

    sat_pos, sat_vel = calc_sat_pos_i(line1, line2, c_jd)
    sun_pos = calc_sun_pos_i(c_jd)
    lat, lon, alt, sideral = calc_geod_lat_lon_alt(sat_pos, c_jd)
    mag_i = mag_model.calc_mag(c_jd, sideral, lat, lon, alt)
    omega_b = sensors.data[['acc_x', 'acc_y', 'acc_z']].values[0]
    q_i2b = Quaternions.get_from_two_v(mag_i, sensors.data[['mag_x', 'mag_y', 'mag_z']].values[0])()

    channels = {'time': [current_time],
                'sat_pos_i': [sat_pos],
                'lonlat': [np.array([lon, lat])],
                'sat_vel_i': [sat_vel],
                'q_i2b': [q_i2b],
                'omega_b': [omega_b],
                'mag_i': [mag_i],
                'sun_i': [sun_pos],
                'sideral': [sideral]}

    k = 0
    while current_time < tend * 0.1:
        # # integration
        if k < len(sensors.data):
            omega_b = sensors.data[['acc_x', 'acc_y', 'acc_z']].values[k]

        q_i2b = calc_quaternion(q_i2b, omega_b, dt)
        omega_b = calc_omega_b(omega_b, dt)

        # update time
        current_time = np.round(current_time + dt, 5)
        c_jd += dt/86400
        k += 1
        # update position without integration
        sat_pos, sat_vel = calc_sat_pos_i(line1, line2, c_jd)
        sun_pos = calc_sun_pos_i(c_jd)
        lat, lon, alt, sideral = calc_geod_lat_lon_alt(sat_pos, c_jd)
        mag_i = mag_model.calc_mag(c_jd, sideral, lat, lon, alt)

        # save data
        channels['time'].append(current_time)
        channels['sat_pos_i'].append(sat_pos)
        channels['sat_vel_i'].append(sat_vel)
        channels['lonlat'].append(np.array([lon, lat]))
        channels['q_i2b'].append(q_i2b)
        channels['omega_b'].append(omega_b)
        channels['mag_i'].append(mag_i)
        channels['sun_i'].append(sun_pos)
        channels['sideral'].append(sideral)

        print(current_time, tend, k)

    monitor = Monitor(channels)
    monitor.set_position('sat_pos_i')
    monitor.set_quaternion('q_i2b')
    monitor.set_sideral('sideral')
    monitor.add_vector('sun_i', color='yellow')
    monitor.add_vector('mag_i', color='red')

    monitor.plot(x_dataset='time', y_dataset='mag_i')
    monitor.plot(x_dataset='time', y_dataset='lonlat')
    monitor.plot(x_dataset='time', y_dataset='sun_i')
    monitor.plot(x_dataset='time', y_dataset='q_i2b')
    monitor.plot(x_dataset='time', y_dataset='omega_b')

    monitor.plot3d()

    monitor.show_monitor()



