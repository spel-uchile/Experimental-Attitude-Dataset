"""
Created by Elias Obreque
Date: 04-09-2023
email: els.obrq@gmail.com
"""
import matplotlib.pyplot as plt
import numpy as np
import datetime
import sys
from src.kalman_filter.ekf_multy import MEKF
from src.kalman_filter.ekf_mag_calibration import MagEKF
from src.kalman_filter.ekf_full import MEKF_FULL
from src.data_process import RealData
from src.dynamics.dynamics_kinematics import *
from src.dynamics.Quaternion import Quaternions
from src.dynamics.MagEnv import MagEnv

from tools.get_video_frame import save_frame
from tools.monitor import Monitor
import importlib.util

# CONFIG
PROJECT_FOLDER = "data/M-20230824/"
module_name = "dataconfig"

# Cargamos el módulo desde la ruta
spec = importlib.util.spec_from_file_location(module_name, PROJECT_FOLDER + module_name + ".py")
myconfig = importlib.util.module_from_spec(spec)
spec.loader.exec_module(myconfig)

CREATE_FRAME = myconfig.CREATE_FRAME
VIDEO_DATA = myconfig.VIDEO_DATA
GET_VECTOR_FROM_PICTURE = myconfig.GET_VECTOR_FROM_PICTURE
OBC_DATA = myconfig.OBC_DATA
TIME_FORMAT = myconfig.TIME_FORMAT
WINDOW_TIME = myconfig.WINDOW_TIME
ONLINE_MAG_CALIBRATION = myconfig.ONLINE_MAG_CALIBRATION
EKF_SETUP = myconfig.EKF_SETUP
IMAGEN_DATA = myconfig.IMAGEN_DATA


if __name__ == '__main__':
    if CREATE_FRAME and VIDEO_DATA is not None:
        save_frame(PROJECT_FOLDER, VIDEO_DATA)

    if GET_VECTOR_FROM_PICTURE and IMAGEN_DATA is not None:
        pass

    # create data with datetime, and near tle
    sensors = RealData(PROJECT_FOLDER, OBC_DATA)
    sensors.create_datetime_from_timestamp(TIME_FORMAT)

    # show window time
    if WINDOW_TIME['FLAG']:
        sensors.set_window_time(WINDOW_TIME['Start'], WINDOW_TIME['Stop'], TIME_FORMAT)
    else:
        sensors.set_window_time()
    line1, line2 = sensors.search_nearly_tle()

    # TIME
    dt = WINDOW_TIME['STEP']
    start_datetime = datetime.datetime.strptime(WINDOW_TIME['Start'], TIME_FORMAT)
    stop_datetime = datetime.datetime.strptime(WINDOW_TIME['Stop'], TIME_FORMAT)
    print(start_datetime.timestamp(), stop_datetime.timestamp())

    # Inertial Parameters
    mag_model = MagEnv()
    time_vector = sensors.data['jd'].values
    sat_state = np.asarray([calc_sat_pos_i(line1, line2, cj) for cj in time_vector])
    sat_pos = sat_state[:, 0, :]
    sat_vel = sat_state[:, 1, :]
    sun_pos = np.asarray([calc_sun_pos_i(c_j) for c_j in time_vector])
    sat_lla = np.asarray([calc_geod_lat_lon_alt(sat_pos_, cj) for sat_pos_, cj in zip(sat_pos, time_vector)])
    lat, lon, alt, sideral = sat_lla[:, 0], sat_lla[:, 1], sat_lla[:, 2], sat_lla[:, 3]
    mag_i_e = np.asarray([mag_model.calc_mag(c_j, s_, lat_, lon_, alt_)
                          for c_j, s_, lat_, lon_, alt_ in zip(time_vector, sideral, lat, lon, alt)])
    sun_sc_i = sun_pos - sat_pos
    mag_i = mag_i_e[:, 0, :]
    mag_ned = mag_i_e[:, 1, :]
    # calibration
    ekf_mag_cal = None
    new_sensor = []
    if not ONLINE_MAG_CALIBRATION:
        # sensors.calibrate_mag(mag_i=mag_i)
        sensors.calibrate_mag(by_file=True)
    else:
        # sensors.calibrate_mag(by_file=True)
        sensors.calibrate_mag(by_file=True)
        ekf_mag_cal = MagEKF()
        for mag_i_, mag_b_ in zip(mag_i, sensors.data[['mag_x', 'mag_y', 'mag_z']].values):
            ekf_mag_cal.update_state(mag_b_, mag_i_, cov_sensor_=100)
            bias_, D_scale = ekf_mag_cal.get_calibration()
            new_sensor.append((np.eye(3) + D_scale) @ mag_b_ - bias_)
        ekf_mag_cal.plot(np.asarray(new_sensor) - mag_i)

    # plot
    sensors.plot_key(['mag_x', 'mag_y', 'mag_z'])
    sensors.plot_key(['acc_x', 'acc_y', 'acc_z'])
    sensors.plot_key(['sun3', 'sun2', 'sun4'])

    # SIMULATION
    current_time = 0
    c_jd = sensors.data['jd'].values[0]
    tend = stop_datetime.timestamp() - start_datetime.timestamp()

    omega_b = sensors.data[['acc_x', 'acc_y', 'acc_z']].values[0]
    q_i2b = Quaternions.get_from_two_v(mag_i[0], sensors.data[['mag_x', 'mag_y', 'mag_z']].values[0])()
    mag_b = Quaternions(q_i2b).frame_conv(mag_i[0])

    channels = {'full_time': time_vector,
                'sim_time': [current_time],
                'sat_pos_i': sat_pos,
                'lonlat': np.array([lon, lat]).T * RAD2DEG,
                'sat_vel_i': sat_vel,
                'q_i2b': [q_i2b],
                'omega_b': [omega_b],
                'mag_i': mag_i,
                'mag_ned': mag_ned,
                'mag_b': [mag_b],
                'sun_i': sun_pos,
                'sun_i_sc': sun_sc_i,
                'sideral': sideral}

    if EKF_SETUP == 'NORMAL':
        # MEKF
        P = np.diag([0.5, 0.5, 0.5, 0.1, 0.1, 0.1])
        ekf_model = MEKF(inertia, P=P, Q=np.zeros((6, 6)), R=np.zeros((3, 3)))
        ekf_model.sigma_bias = 0.001
        ekf_model.sigma_omega = 0.005
        ekf_model.current_bias = np.array([0.0, 0.0, 0])
    elif EKF_SETUP == 'FULL':
        # MEKF
        P = np.diag([0.5, 0.5, 0.5, 0.001, 0.001, 0.001, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01])
        ekf_model = MEKF_FULL(inertia, P=P, Q=np.zeros((15, 15)), R=np.zeros((3, 3)))
        ekf_model.sigma_bias = 0.0005
        ekf_model.sigma_omega = 0.0005

    ekf_model.set_quat(q_i2b, save=True)
    ekf_model.get_observer_prediction(None, mag_i[0])
    omega_b_model = omega_b
    ekf_model.set_gyro_measure(omega_b)
    k = 1
    ekf_model.historical['sun_b_est'] = [Quaternions(q_i2b).frame_conv(sun_sc_i[0])]
    for t_, omega_gyro_, mag_ref_, body_vec_, sun_sc_i_ in zip(time_vector[1:],
                                                               sensors.data[['acc_x', 'acc_y', 'acc_z']].values[1:],
                                                               mag_i[1:],
                                                               sensors.data[['mag_x', 'mag_y', 'mag_z']].values[1:],
                                                               sun_sc_i[1:]):
        # # integration
        ekf_model.propagate(dt)

        q_i2b = calc_quaternion(q_i2b, omega_b_model, dt)
        omega_b_model = calc_omega_b(omega_b_model, dt)

        ekf_model.inject_vector(body_vec_, mag_ref_, sigma2=0.001)
        ekf_model.historical['sun_b_est'].append(Quaternions(ekf_model.current_quaternion).frame_conv(sun_sc_i_))
        ekf_model.reset_state()
        omega_gyro_[2] = 0.0
        ekf_model.set_gyro_measure(omega_gyro_)

        # save data
        channels['q_i2b'].append(q_i2b)
        channels['omega_b'].append(omega_b_model)

    channels = {**channels, **ekf_model.historical}
    monitor = Monitor(channels)
    monitor.set_position('sat_pos_i')
    monitor.set_quaternion('q_est')
    monitor.set_sideral('sideral')

    monitor.add_vector('sun_i', color='yellow')
    monitor.add_vector('mag_i', color='red')

    monitor.plot(x_dataset='full_time', y_dataset='mag_i')
    # monitor.plot(x_dataset='full_time', y_dataset='lonlat')
    monitor.plot(x_dataset='full_time', y_dataset='sun_i_sc')
    # monitor.plot(x_dataset='full_time', y_dataset='sat_pos_i')
    monitor.plot(x_dataset='full_time', y_dataset='q_i2b')
    monitor.plot(x_dataset='full_time', y_dataset='omega_b')
    # ekf
    monitor.plot(x_dataset='full_time', y_dataset='b_est')
    monitor.plot(x_dataset='full_time', y_dataset='q_est')
    monitor.plot(x_dataset='full_time', y_dataset='omega_est')
    monitor.plot(x_dataset='full_time', y_dataset='mag_est')
    monitor.plot(x_dataset='full_time', y_dataset='sun_b_est')
    monitor.plot(x_dataset='full_time', y_dataset='p_cov')
    if EKF_SETUP == 'FULL':
        monitor.plot(x_dataset='full_time', y_dataset='scale')
        monitor.plot(x_dataset='full_time', y_dataset='ku')
        monitor.plot(x_dataset='full_time', y_dataset='kl')
    monitor.show_monitor()
    monitor.plot3d()
