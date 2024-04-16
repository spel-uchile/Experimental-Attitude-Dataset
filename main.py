"""
Created by Elias Obreque
Date: 04-09-2023
email: els.obrq@gmail.com
"""
import matplotlib.pyplot as plt
import numpy as np
import pickle
import datetime
import sys
import os
import pandas as pd
from sklearn.metrics import mean_squared_error
import cv2
from src.kalman_filter.ekf_multy import MEKF
from src.kalman_filter.ekf_mag_calibration import MagEKF
from src.kalman_filter.ukf_propagation import UKF
from src.kalman_filter.ekf_full import MEKF_FULL
from src.kalman_filter.ekf_mag_calibration import MagUKF
from src.data_process import RealData
from src.dynamics.dynamics_kinematics import *
from src.dynamics.Quaternion import Quaternions
from src.dynamics.MagEnv import MagEnv

from tools.get_video_frame import save_frame
from tools.get_point_vector_from_picture import get_vector
from tools.monitor import Monitor
import importlib.util

# CONFIG
PROJECT_FOLDER = "./data/20240804/"
# PROJECT_FOLDER = "./data/20230904/"
module_name = "dataconfig"

# Cargamos el módulo desde la ruta
spec = importlib.util.spec_from_file_location(module_name, PROJECT_FOLDER + module_name + ".py")
myconfig = importlib.util.module_from_spec(spec)
spec.loader.exec_module(myconfig)

CREATE_FRAME = myconfig.CREATE_FRAME
VIDEO_DATA = myconfig.VIDEO_DATA
VIDEO_TIME_LAST_FRAME = myconfig.VIDEO_TIME_LAST_FRAME
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

    if GET_VECTOR_FROM_PICTURE:
        height = 440
        list_file = [elem for elem in os.listdir(PROJECT_FOLDER + VIDEO_DATA.split('.')[0]) if 'png' in elem]
        num_list = [int(elem.split(".")[0].replace("frame", "")) for elem in list_file if 'png' in elem]
        datalist = pd.DataFrame({'filename': list_file, 'id': num_list})
        datalist.sort_values(by='id', inplace=True)
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        video_salida = cv2.VideoWriter(PROJECT_FOLDER + "att_.avi", fourcc, 10.0, (100, 100))
        for filename in datalist['filename'].values:
            edge_, img_cv2_, p_, r_ = get_vector(PROJECT_FOLDER + VIDEO_DATA.split('.')[0] + "/" + filename, height)
            if img_cv2_ is not None:
                video_salida.write(img_cv2_)
                print("added")
        video_salida.release()

    # create data with datetime, and near tle
    sensors = RealData(PROJECT_FOLDER, OBC_DATA)
    sensors.create_datetime_from_timestamp(TIME_FORMAT)

    # show window time
    if WINDOW_TIME['FLAG']:
        sensors.set_window_time(WINDOW_TIME['Start'], WINDOW_TIME['Stop'], TIME_FORMAT)
    else:
        sensors.set_window_time()
    line1, line2 = sensors.search_nearly_tle()
    print(line1, line2)
    # TIME
    dt = WINDOW_TIME['STEP']
    start_datetime = datetime.datetime.strptime(WINDOW_TIME['Start'], TIME_FORMAT)
    stop_datetime = datetime.datetime.strptime(WINDOW_TIME['Stop'], TIME_FORMAT)
    print(start_datetime.timestamp(), stop_datetime.timestamp())
    # SIMULATION
    current_time = 0
    c_jd = sensors.data['jd'].values[0]
    tend = stop_datetime.timestamp() - start_datetime.timestamp()

    # if not exist file channels
    if os.path.exists(PROJECT_FOLDER + "channels.p"):
        with open(PROJECT_FOLDER + "channels.p", 'rb') as fp:
            channels = pickle.load(fp)
    else:
        # Inertial Parameters
        mag_model = MagEnv()
        time_vector = sensors.data['jd'].values
        sat_state = np.asarray([calc_sat_pos_i(line1, line2, cj) for cj in time_vector])
        sat_pos = sat_state[:, 0, :]
        sat_vel = sat_state[:, 1, :]
        sun_pos = np.asarray([calc_sun_pos_i(c_j) for c_j in time_vector])
        moon_pos = np.asarray([calc_moon_pos_i(c_j) for c_j in time_vector])
        sat_lla = np.asarray([calc_geod_lat_lon_alt(sat_pos_, cj) for sat_pos_, cj in zip(sat_pos, time_vector)])
        lat, lon, alt, sideral = sat_lla[:, 0], sat_lla[:, 1], sat_lla[:, 2], sat_lla[:, 3]
        mag_i_e = np.asarray([mag_model.calc_mag(c_j, s_, lat_, lon_, alt_)
                              for c_j, s_, lat_, lon_, alt_ in zip(time_vector, sideral, lat, lon, alt)])
        sun_sc_i = sun_pos - sat_pos
        moon_sc_i = moon_pos - sat_pos
        mag_i = mag_i_e[:, 0, :] * 0.01  # nT to mG
        mag_ned = mag_i_e[:, 1, :] * 0.01  # nT to mG

        channels = {'full_time': time_vector,
                    'sim_time': [current_time],
                    'sat_pos_i': sat_pos,
                    'lonlat': np.array([lon, lat]).T * RAD2DEG,
                    'sat_vel_i': sat_vel,
                    'q_i2b_pred': [],
                    'omega_b_pred': [],
                    'time_pred': [],
                    'mag_i': mag_i,
                    'mag_ned': mag_ned,
                    'sun_i': sun_pos,
                    'sun_sc_i': sun_sc_i,
                    'moon_sc_i': moon_sc_i,
                    'sideral': sideral}
        # save channels as json
        with open(PROJECT_FOLDER + 'channels.p', 'wb') as file_:
            pickle.dump(channels, file_)

    plt.figure()
    plt.plot(channels['full_time'], channels['mag_i'][:, 0], 'o-', color='blue', label='mag_x [mG]')
    plt.plot(channels['full_time'], channels['mag_i'][:, 1], 'o-', color='orange', label='mag_y [mG]')
    plt.plot(channels['full_time'], channels['mag_i'][:, 2], 'o-', color='green', label='mag_z [mG]')
    plt.grid()
    plt.legend()
    sensors.plot_key(['mag_x', 'mag_y', 'mag_z'], color=['blue', 'orange', 'green'],
                     label=['x [mG]', 'y [mG]', 'z [mG]'])

    # calibration
    ekf_mag_cal = None
    new_sensor = []
    sensors.plot_key(['mag_x', 'mag_y', 'mag_z'], color=['blue', 'orange', 'green'], label=['x [mG]', 'y [mG]', 'z [mG]'])
    # sensors.calibrate_mag(mag_i=channels['mag_i'])
    # sensors.calibrate_mag(mag_i=channels['mag_i'], force=True)
    sensors.plot_key(['mag_x'], color=['blue'], label=['x [mG]'])
    sensors.plot_key(['mag_y'], color=['orange'], label=['y [mG]'])
    sensors.plot_key(['mag_z'], color=['green'], label=['z [mG]'])
    sensors.plot_key(['sun3'], color=['blue'], label=['-x [mA]'])
    sensors.plot_key(['sun2'], color=['orange'], label=['-y [mA]'])
    sensors.plot_key(['sun4'], color=['green'], label=['-z [mA]'])

    error_mag_ts = np.linalg.norm(channels['mag_i'], axis=1) - np.linalg.norm(sensors.data[['mag_x', 'mag_y', 'mag_z']], axis=1)
    mse_ts = mean_squared_error(np.linalg.norm(channels['mag_i'], axis=1), np.linalg.norm(sensors.data[['mag_x', 'mag_y', 'mag_z']], axis=1))
    plt.figure()
    plt.title("Two Step Magnitude Error")
    plt.plot(error_mag_ts, label='RMSE: {:2f}'.format(np.sqrt(mse_ts)))
    plt.legend()
    plt.grid()

    if ONLINE_MAG_CALIBRATION:
        ukf = MagUKF(alpha=1)
        D_est = np.zeros(6) + 1e-7
        b_est = np.zeros(3) + 10
        P_est = np.diag([*b_est, *D_est])  # (uT)^2
        ukf.pCov_x = P_est
        ukf.pCov_zero = P_est / 2
        new_sensor_ukf = []
        stop_k = 0
        flag_t = True
        for mag_i_, mag_b_ in zip(channels['mag_i'], sensors.data[['mag_x', 'mag_y', 'mag_z']].values):
            ukf.save()
            # if len(ukf.historical['error']) > 0:
            #     if ukf.historical['error'][-1] > 5000 and flag_t:
            #         print("Before: ", stop_k, ukf.historical['error'][-1])
            #         P_est = np.diag([*b_est, *D_est])  # (uT)^2
            #         ukf.pCov_x = P_est
            #         flag_t = False
            #     else:
            #         flag_t = True
            ukf.run(mag_b_, mag_i_, 200)#, 5000, 30)
            bias_, D_scale = ukf.get_calibration()
            new_sensor_ukf.append((np.eye(3) + D_scale) @ mag_b_ - bias_)
            stop_k += 1
        # ukf.plot(np.linalg.norm(np.asarray(new_sensor_ukf), axis=1), np.linalg.norm(mag_i, axis=1))

        # plot
        sensors.plot_key(['mag_x', 'mag_y', 'mag_z'])
        # sensors.plot_key(['acc_x', 'acc_y', 'acc_z'])
        # sensors.plot_key(['sun3', 'sun2', 'sun4'])

        D_ukf_vector = ukf.historical['scale'][-1]
        b_ukf = ukf.historical['bias'][-1]
        print(b_ukf, D_ukf_vector)
        D_ukf = np.array([[D_ukf_vector[0], D_ukf_vector[3], D_ukf_vector[4]],
                          [D_ukf_vector[3], D_ukf_vector[1], D_ukf_vector[5]],
                          [D_ukf_vector[4], D_ukf_vector[5], D_ukf_vector[2]]])

        # mag_ukf = (np.eye(3) + D_ukf).dot(sensors.data[['mag_x', 'mag_y', 'mag_z']].values.T).T - b_ukf.reshape(-1, 1).T
        mag_ukf = np.asarray(new_sensor_ukf)
        sensors.data[['mag_x', 'mag_y', 'mag_z']] = mag_ukf
        ukf.plot(np.linalg.norm(mag_ukf, axis=1), np.linalg.norm(channels['mag_i'], axis=1))
        plt.figure()
        plt.plot(mag_ukf[:, 0], label='new_mag_x')
        plt.plot(mag_ukf[:, 1], label='new_mag_y')
        plt.plot(mag_ukf[:, 2], label='new_mag_z')
        plt.legend()
        plt.grid()

        # plt.figure()
        # plt.plot(channels['full_time'], channels['mag_i'][:, 0], label='mag_x')
        # plt.plot(channels['full_time'], channels['mag_i'][:, 1], label='mag_y')
        # plt.plot(channels['full_time'], channels['mag_i'][:, 2], label='mag_z')
        # plt.legend()
        plt.show()
    # exit()
    omega_b = sensors.data[['acc_x', 'acc_y', 'acc_z']].values[0]
    q_i2b = Quaternions.get_from_two_v(channels['mag_i'][0], sensors.data[['mag_x', 'mag_y', 'mag_z']].values[0])()
    mag_b = Quaternions(q_i2b).frame_conv(channels['mag_i'][0])
    moon_b = Quaternions(q_i2b).frame_conv(channels['moon_sc_i'][0])

    if EKF_SETUP == 'NORMAL':
        # MEKF
        P = np.diag([0.5, 0.5, 0.5, 0.01, 0.01, 0.01])
        ekf_model = MEKF(inertia, P=P, Q=np.zeros((6, 6)), R=np.zeros((3, 3)))
        ekf_model.sigma_bias = 0.0001
        ekf_model.sigma_omega = 0.0001
        ekf_model.current_bias = np.array([0.0, 0.0, 0])
    elif EKF_SETUP == 'FULL':
        # MEKF
        P = np.diag([0.5, 0.5, 0.5, 0.01, 0.01, 0.01, 1e-7, 1e-7, 1e-7, 1e-7, 1e-7, 1e-7, 1e-7, 1e-7, 1e-7]) * 100
        ekf_model = MEKF_FULL(inertia, P=P, Q=np.zeros((15, 15)), R=np.zeros((3, 3)))
        ekf_model.sigma_bias = 0.0005
        ekf_model.sigma_omega = 0.0001

    ekf_model.set_quat(q_i2b, save=True)
    ekf_model.set_gyro_measure(omega_b)
    ekf_model.save_vector(name='mag_est', vector=sensors.data[['mag_x', 'mag_y', 'mag_z']].values[0])
    ekf_model.save_vector(name='css_est', vector=sensors.data[['sun3', 'sun2', 'sun4']].values[0])
    ekf_model.save_vector(name='sun_b_est', vector=Quaternions(q_i2b).frame_conv(channels['sun_sc_i'][0]))

    Imax = 930
    pred_step_sec = 60

    # ukf_model = UKF(P, dt=1)
    # ukf_model.set_quat(q_i2b, save=True)
    # ukf_model.save_vector(name='mag_est', vector=sensors.data[['mag_x', 'mag_y', 'mag_z']].values[0])
    # ukf_model.save_vector(name='css_est', vector=sensors.data[['sun3', 'sun2', 'sun4']].values[0])
    # ukf_model.save_vector(name='sun_b_est', vector=Quaternions(q_i2b).frame_conv(sun_sc_i[0]))
    # ukf_model.set_gyro_measure(omega_b)
    # ukf_model.set_Q(ekf_model.sigma_omega, ekf_model.sigma_bias)
    k = 1
    moon_sc_b = [moon_b]
    t0 = channels['full_time'][0]
    sensor_step = 1
    for t_, omega_gyro_, mag_ref_, body_vec_, sun_sc_i_, css_3_ in zip(channels['full_time'][1:],
                                                                       sensors.data[['acc_x', 'acc_y', 'acc_z']].values[
                                                                       1:],
                                                                       channels['mag_i'][1:],
                                                                       sensors.data[['mag_x', 'mag_y', 'mag_z']].values[
                                                                       1:],
                                                                       channels['sun_sc_i'][1:],
                                                                       sensors.data[['sun3', 'sun2', 'sun4']].values[
                                                                       1:]):
        # Optimization of R and Q
        # ekf_model.optimize_R_Q(body_vec_, mag_ref_, dt)

        omega_b_pred = ekf_model.get_calibrate_omega()
        q_i2b_pred = ekf_model.current_quaternion

        for _ in range(int(pred_step_sec / 0.1)):
            q_i2b_pred = calc_quaternion(q_i2b_pred, omega_b_pred, 0.1)
            omega_b_pred = calc_omega_b(omega_b_pred, 0.1)
        t_pred = t_ + pred_step_sec

        # # integration
        dt = np.round((t_ - t0) * 86400, 3)
        while dt - sensor_step >= 0.0:
            print(k, dt)
            ekf_model.propagate(sensor_step)
            dt -= sensor_step
        t0 = t_
        # ukf_model.predict()
        # mag
        mag_est = ekf_model.inject_vector(body_vec_, mag_ref_, sigma2=200, sensor='mag')

        # mag_est_ukf = ukf_model.inject_vector(body_vec_, mag_ref_, sigma2=5000, sensor='mag')
        # css
        css_est = np.zeros(3)
        is_dark = shadow_zone(channels['sat_pos_i'][k], channels['sun_i'][k])
        if not is_dark and np.linalg.norm(css_3_) > 100:
            css_est = ekf_model.inject_vector(css_3_, sun_sc_i_, gain=-Imax * np.eye(3), sigma2=200, sensor='css')
        ekf_model.save_vector(name='css_est', vector=css_est)
        ekf_model.save_vector(name='mag_est', vector=mag_est)
        ekf_model.save_vector(name='sun_b_est', vector=Quaternions(ekf_model.current_quaternion).frame_conv(sun_sc_i_))
        ekf_model.reset_state()
        moon_sc_b.append(Quaternions(ekf_model.current_quaternion).frame_conv(channels['moon_sc_i'][k]))
        ekf_model.set_gyro_measure(omega_gyro_)
        # save data
        channels['time_pred'].append(t_pred)
        channels['q_i2b_pred'].append(q_i2b_pred)
        channels['omega_b_pred'].append(omega_b_pred)
        k += 1

    channels = {**channels, **ekf_model.historical}
    error_mag = channels['mag_est'] - sensors.data[['mag_x', 'mag_y', 'mag_z']].values
    error_pred = [(Quaternions(Quaternions(q_p).conjugate()) * Quaternions(q_kf)).get_angle()
                  for q_p, q_kf in zip(channels['q_i2b_pred'], channels['q_est'][pred_step_sec:])]

    q_est = np.array(channels['q_est'])

    # q_train = q_est[:int(0.1 * len(q_est))]
    # theta_r = np.arccos(q_train[:, 3]) * 2
    # quat_vec = q_train[:, :3] / np.sin(theta_r * 0.5)
    plt.figure()
    plt.title("Error Mag")
    plt.plot(error_mag)
    plt.grid()

    plt.figure()
    plt.title("Error prediction [deg]")
    plt.xlabel("Step")
    plt.plot(np.array(error_pred) * 180 / np.pi)
    plt.grid()

    fov = 48 * np.deg2rad(1) / 2
    view_of_moon = np.cos(fov)
    moon_sc_b = np.asarray(moon_sc_b) / np.linalg.norm(moon_sc_b, axis=1).reshape(-1, 1)
    fig, axes = plt.subplots(nrows=3, ncols=1)
    plt.title("Moon View")
    axes[0].plot(channels['full_time'], moon_sc_b[:, 0], color='blue', label='x')
    axes[1].plot(channels['full_time'], moon_sc_b[:, 1], color='orange', label='y')
    axes[1].hlines(view_of_moon, channels['full_time'][0], channels['full_time'][-1], color='red')
    axes[2].plot(channels['full_time'], moon_sc_b[:, 2], color='green', label='z')
    axes[0].grid()
    axes[0].legend()
    axes[1].grid()
    axes[1].legend()
    axes[2].grid()
    axes[2].legend()

    # channels['q_est'] = [np.array([0, 0, 0, 1]) for elem in channels['sat_pos_i']]
    monitor = Monitor(channels)
    monitor.set_position('sat_pos_i')
    monitor.set_quaternion('q_est')
    monitor.set_sideral('sideral')

    monitor.add_vector('sun_sc_i', color='yellow')
    monitor.add_vector('mag_i', color='orange')
    monitor.add_vector('moon_sc_i', color='white')

    # monitor.plot(x_dataset='full_time', y_dataset='mag_i')
    # monitor.plot(x_dataset='full_time', y_dataset='lonlat')
    # monitor.plot(x_dataset='full_time', y_dataset='sun_i_sc')
    # monitor.plot(x_dataset='full_time', y_dataset='sat_pos_i')
    monitor.plot(x_dataset='time_pred', y_dataset='q_i2b_pred')
    monitor.plot(x_dataset='time_pred', y_dataset='omega_b_pred')
    # ekf
    monitor.plot(x_dataset='full_time', y_dataset='b_est')
    monitor.plot(x_dataset='full_time', y_dataset='q_est')
    monitor.plot(x_dataset='full_time', y_dataset='omega_est')
    monitor.plot(x_dataset='full_time', y_dataset='mag_est')
    monitor.plot(x_dataset='full_time', y_dataset='sun_b_est')
    monitor.plot(x_dataset='full_time', y_dataset='p_cov')

    fig, axes = plt.subplots(nrows=3, ncols=1, sharex=True)
    plt.title("Magnetometer estimation [mG]")
    axes[0].plot(channels['full_time'], np.array(channels['mag_est'])[:, 0], color='blue', label='-x')
    axes[1].plot(channels['full_time'], np.array(channels['mag_est'])[:, 1], color='orange', label='-y')
    axes[2].plot(channels['full_time'], np.array(channels['mag_est'])[:, 2], color='green', label='-z')
    axes[0].grid()
    axes[0].legend()
    axes[1].grid()
    axes[1].legend()
    axes[2].grid()
    axes[2].legend()
    axes[2].set_xlabel("step")

    fig, axes = plt.subplots(nrows=3, ncols=1, sharex=True)
    plt.title("Coarse sun sensor estimation [mA]")
    axes[0].plot(channels['full_time'], np.array(channels['css_est'])[:, 0], color='blue', label='-x')
    axes[1].plot(channels['full_time'], np.array(channels['css_est'])[:, 1], color='orange', label='-y')
    axes[2].plot(channels['full_time'], np.array(channels['css_est'])[:, 2], color='green', label='-z')
    axes[0].grid()
    axes[0].legend()
    axes[1].grid()
    axes[1].legend()
    axes[2].grid()
    axes[2].legend()
    axes[2].set_xlabel("step")
    if EKF_SETUP == 'FULL':
        monitor.plot(x_dataset='full_time', y_dataset='scale')
        monitor.plot(x_dataset='full_time', y_dataset='ku')
        monitor.plot(x_dataset='full_time', y_dataset='kl')
    monitor.show_monitor()
    monitor.plot3d()
