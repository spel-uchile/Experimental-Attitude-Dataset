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

import cv2
from src.kalman_filter.ekf_multy import MEKF
# from src.kalman_filter.ekf_mag_calibration import MagEKF
# from src.kalman_filter.ukf_propagation import UKF
from src.kalman_filter.ekf_full import MEKF_FULL
from src.kalman_filter.ekf_mag_calibration import MagUKF
from src.data_process import RealData
from src.dynamics.Quaternion import Quaternions
from src.dynamics.dynamics_kinematics import Dynamics, calc_quaternion, calc_omega_b, shadow_zone
from tools.get_video_frame import save_frame
from tools.get_point_vector_from_picture import get_vector
from tools.monitor import Monitor
from tools.mathtools import julian_to_datetime, timestamp_to_julian
import importlib.util
import matplotlib as mpl

mpl.rcParams['font.size'] = 12
# mpl.rcParams['font.family'] = 'Arial'   # Set the default font family

# CONFIG
# PROJECT_FOLDER = "./data/20240804/"
# PROJECT_FOLDER = "./data/M-20230824/"
PROJECT_FOLDER = "./data/20230904/"
module_name = "dataconfig"

# Cargamos el módulo desde la ruta
spec = importlib.util.spec_from_file_location(module_name, PROJECT_FOLDER + module_name + ".py")
myconfig = importlib.util.module_from_spec(spec)
spec.loader.exec_module(myconfig)

FORCE_CALCULATION = myconfig.FORCE_CALCULATION
CREATE_FRAME = myconfig.CREATE_FRAME
VIDEO_DATA = myconfig.VIDEO_DATA
VIDEO_TIME_LAST_FRAME = myconfig.VIDEO_TIME_LAST_FRAME
GET_VECTOR_FROM_PICTURE = myconfig.GET_VECTOR_FROM_PICTURE
OBC_DATA = myconfig.OBC_DATA
OBC_DATA_STEP = myconfig.OBC_DATA_STEP
TIME_FORMAT = myconfig.TIME_FORMAT
WINDOW_TIME = myconfig.WINDOW_TIME
ONLINE_MAG_CALIBRATION = myconfig.ONLINE_MAG_CALIBRATION
EKF_SETUP = myconfig.EKF_SETUP
IMAGEN_DATA = myconfig.IMAGEN_DATA

if __name__ == '__main__':
    # LOAD LAST RESULT ------------------------------------------------------------------------------------------------
    ekf_channels = None
    if not os.path.exists(PROJECT_FOLDER + "results/"):
        os.mkdir(PROJECT_FOLDER + "results/")
    if os.path.exists(PROJECT_FOLDER + "estimation_results.p") and not FORCE_CALCULATION:
        with open(PROJECT_FOLDER + "estimation_results.p", 'rb') as fp:
            ekf_channels = pickle.load(fp)

    # REAL DATA and TIME ----------------------------------------------------------------------------------------------
    # create data with datetime, and near tle
    sensors = RealData(PROJECT_FOLDER, OBC_DATA)
    # sensors.set_gyro_bias(-3.846, 0.1717, -0.6937, unit='deg')
    sensors.create_datetime_from_timestamp(TIME_FORMAT)
    inertia = np.array([38478.678, 38528.678, 6873.717, 0, 0, 0]) * 1e-6
    # sensors.estimate_inertia_matrix(guess=inertia)
    sensors.set_inertia(inertia)
    # show window time
    if WINDOW_TIME['FLAG']:
        sensors.set_window_time(WINDOW_TIME['Start'], WINDOW_TIME['Stop'], TIME_FORMAT)
    else:
        sensors.set_window_time()
    line1, line2 = sensors.search_nearly_tle()
    print(line1, line2)
    # TIME
    dt_obc = OBC_DATA_STEP
    dt_sim = WINDOW_TIME['STEP']
    start_datetime = datetime.datetime.strptime(WINDOW_TIME['Start'], TIME_FORMAT)
    stop_datetime = datetime.datetime.strptime(WINDOW_TIME['Stop'], TIME_FORMAT)
    print(start_datetime.timestamp(), stop_datetime.timestamp())

    # SIMULATION ------------------------------------------------------------------------------------------------------
    # if not exist file channels
    Imax = 930
    pred_step_sec = 60
    time_vector = sensors.data['jd'].values
    dynamic_orbital = Dynamics(time_vector, line1, line2)
    if os.path.exists(PROJECT_FOLDER + "channels.p") and not FORCE_CALCULATION:
        with open(PROJECT_FOLDER + "channels.p", 'rb') as fp:
            channels = pickle.load(fp)
        dynamic_orbital.load_data(channels)
        dynamic_orbital.calc_mag()
        channels = dynamic_orbital.channels
    else:
        # Inertial Parameters
        channels = dynamic_orbital.get_dynamics()
        # save channels as json
        with open(PROJECT_FOLDER + 'channels.p', 'wb') as file_:
            pickle.dump(channels, file_)

    dynamic_orbital.plot_gt(PROJECT_FOLDER + 'results/gt')
    dynamic_orbital.plot_mag(PROJECT_FOLDER + 'results/mag_model_igrf13')
    dynamic_orbital.plot_sun_sc(PROJECT_FOLDER + 'results/sun_pos_from_sc')

    sensors.plot_main_data(channels)
    # calibrate gyro
    pcov_gyro = sensors.calibrate_gyro()
    print(pcov_gyro)
    sensors.plot_key(['acc_x', 'acc_y', 'acc_z'], color=['blue', 'orange', 'green'],
                      name="cal_gyro_sensor_dps", title="Calibrate Gyro Sensor [rad/s]",
                      label=['x [mG]', 'y [mG]', 'z [mG]', '||mag||'], drawstyle=['steps-post'] * 4, marker=['.'] * 4)

    # calibration using two-step and channels
    sensors.plot_mag_error(channels, 'before')
    sensors.calibrate_mag(mag_i=channels['mag_i'])
    sensors.plot_key(['mag_x', 'mag_y', 'mag_z', '||mag||'], color=['blue', 'orange', 'green', 'black'],
                     name="mag_sensor_mg_two_step", title="TWO STEP Calibration - Mag [mG]",
                     label=['x [mG]', 'y [mG]', 'z [mG]', '||mag||'], drawstyle=['steps-post'] * 4, marker=['.'] * 4)
    sensors.show_mag_geometry("TWO STEP Method")
    sensors.plot_mag_error(channels, 'after')


    # VIDEO -----------------------------------------------------------------------------------------------------------
    if CREATE_FRAME and VIDEO_DATA is not None:
        # width, height
        frame_shape = save_frame(PROJECT_FOLDER, VIDEO_DATA, VIDEO_TIME_LAST_FRAME)

    if GET_VECTOR_FROM_PICTURE:
        list_file = [elem for elem in os.listdir(PROJECT_FOLDER + VIDEO_DATA.split('.')[0]) if 'png' in elem]
        num_list = [float(elem[:-4]) for elem in list_file if 'png' in elem]
        datalist = pd.DataFrame({'filename': list_file, 'id': num_list})
        datalist.sort_values(by='id', inplace=True)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_salida = cv2.VideoWriter(PROJECT_FOLDER + f"att_process_{VIDEO_DATA}.avi", fourcc, 10.0,
                                       (frame_shape[1], frame_shape[0]))
        rot_info = {'time_picture':[], 'pitch': [], 'roll': []}
        for filename, ts_i in datalist.values:
            height = dynamic_orbital.get_altitude(ts_i)
            edge_, img_cv2_, p_, r_ = get_vector(PROJECT_FOLDER + VIDEO_DATA.split('.')[0] + "/" + filename, height)
            rot_info['pitch'].append(p_)
            rot_info['roll'].append(r_)
            rot_info['time_picture'].append(timestamp_to_julian(ts_i) - 2400000.5)
            if img_cv2_ is not None:
                video_salida.write(img_cv2_)
                print(f" - filename {filename} added")
        video_salida.release()

        plt.figure()
        plt.ylabel("Angle rotation [deg]")
        plt.ylabel("MJD [sec]")
        plt.plot(rot_info['time_picture'], rot_info['pitch'], '.', label='Pitch Angle')
        plt.plot(rot_info['time_picture'], rot_info['roll'], '.', label='Roll Angle')
        plt.legend()
        plt.grid()
        plt.show()

    # UKF MAG CALIBRATION ----------------------------------------------------------------------------------------------
    if ONLINE_MAG_CALIBRATION:
        D_est = np.zeros(6) + 1e-9
        b_est = np.zeros(3) + 10
        ukf = MagUKF(b_est, D_est, alpha=0.1)
        mag_ukf = ukf.calibrate(channels['mag_i'], sensors.data[['mag_x', 'mag_y', 'mag_z']].values)
        ukf.plot(np.linalg.norm(mag_ukf, axis=1), np.linalg.norm(channels['mag_i'], axis=1), sensors.data['mjd'],
                 PROJECT_FOLDER + 'results/')
        sensors.data[['mag_x', 'mag_y', 'mag_z']] = mag_ukf
        sensors.data['||mag||'] = np.linalg.norm(mag_ukf, axis=1)
        sensors.plot_key(['mag_x', 'mag_y', 'mag_z', '||mag||'], color=['blue', 'orange', 'green', 'black'],
                         name="mag_sensor_mg_ukf", show=False, title="UKF Calibration - Mag [mG]",
                         label=['x [mG]', 'y [mG]', 'z [mG]', '||mag||'], drawstyle=['steps-post'] * 4, marker=['.'] * 4)

    sensors.show_mag_geometry("UKF Method")
    omega_b = sensors.data[['acc_x', 'acc_y', 'acc_z']].values[0]
    q_i2b = Quaternions.get_from_two_v(channels['mag_i'][0], sensors.data[['mag_x', 'mag_y', 'mag_z']].values[0])()
    mag_b = Quaternions(q_i2b).frame_conv(channels['mag_i'][0])
    moon_b = Quaternions(q_i2b).frame_conv(channels['moon_sc_i'][0])

    prediction_dict = {'q_i2b_pred': [],
                       'omega_b_pred': [],
                       'time_pred': [],}

    if not os.path.exists(PROJECT_FOLDER + "estimation_results.p"):
        if EKF_SETUP == 'NORMAL':
            # MEKF
            P = np.diag([0.5, 0.5, 0.5, 0.01, 0.01, 0.01])  # * 10
            ekf_model = MEKF(inertia, P=P, Q=np.zeros((6, 6)), R=np.zeros((3, 3)))
            ekf_model.sigma_bias = 1e-3
            ekf_model.sigma_omega = 1e-4
            ekf_model.current_bias = np.array([0.0, 0.0, 0])
        elif EKF_SETUP == 'FULL':
            # MEKF
            P = np.diag([0.5, 0.5, 0.5, 0.01, 0.01, 0.01, 1e-7, 1e-7, 1e-7, 1e-7, 1e-7, 1e-7, 1e-7, 1e-7, 1e-7]) * 100
            ekf_model = MEKF_FULL(inertia, P=P, Q=np.zeros((15, 15)), R=np.zeros((3, 3)))
            ekf_model.sigma_bias = 0.0005
            ekf_model.sigma_omega = 1e-6

        ekf_model.set_quat(q_i2b, save=True)
        ekf_model.set_gyro_measure(omega_b)
        ekf_model.save_vector(name='mag_est', vector=sensors.data[['mag_x', 'mag_y', 'mag_z']].values[0])
        ekf_model.save_vector(name='css_est', vector=sensors.data[['sun3', 'sun2', 'sun4']].values[0])
        ekf_model.save_vector(name='sun_b_est', vector=Quaternions(q_i2b).frame_conv(channels['sun_sc_i'][0]))

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
            mag_est = ekf_model.inject_vector(body_vec_, mag_ref_, sigma2=2.8 ** 2, sensor='mag')

            # mag_est_ukf = ukf_model.inject_vector(body_vec_, mag_ref_, sigma2=5000, sensor='mag')
            # css
            css_est = np.zeros(3)
            is_dark = shadow_zone(channels['sat_pos_i'][k], channels['sun_i'][k])
            if not is_dark and False:
                css_3_[css_3_ < 50] = 0.0
                css_est = ekf_model.inject_vector(css_3_, sun_sc_i_, gain=-Imax * np.eye(3), sigma2=10, sensor='css')
            ekf_model.save_vector(name='css_est', vector=css_est)
            ekf_model.save_vector(name='mag_est', vector=mag_est)
            ekf_model.save_vector(name='sun_b_est', vector=Quaternions(ekf_model.current_quaternion).frame_conv(sun_sc_i_))
            ekf_model.reset_state()
            moon_sc_b.append(Quaternions(ekf_model.current_quaternion).frame_conv(channels['moon_sc_i'][k]))
            ekf_model.set_gyro_measure(omega_gyro_)
            # save data
            prediction_dict['time_pred'].append(t_pred)
            prediction_dict['q_i2b_pred'].append(q_i2b_pred)
            prediction_dict['omega_b_pred'].append(omega_b_pred)
            k += 1
        ekf_channels = {**prediction_dict, **ekf_model.historical}
        with open(PROJECT_FOLDER + 'estimation_results.p', 'wb') as file_:
            pickle.dump(ekf_channels, file_)

    data_text = ["{}".format(julian_to_datetime(jd_)) for jd_ in channels['full_time']]
    channels['DateTime'] = data_text
    channels = {**channels, **ekf_channels}
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

    fig = plt.figure()
    plt.title("Error prediction [deg]")
    plt.xlabel("Step")
    plt.plot(np.array(error_pred) * 180 / np.pi)
    plt.grid()

    fov = 48 * np.deg2rad(1) / 2
    view_of_moon = np.cos(fov)
    moon_sc_b = [Quaternions(ekf_q).frame_conv(vec_) for ekf_q, vec_ in zip(channels['q_est'], channels['moon_sc_i'])]
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

    sensors.plot_key(['mag_x'], color=['blue'], label=['x [mG]'])
    sensors.plot_key(['mag_y'], color=['orange'], label=['y [mG]'])
    sensors.plot_key(['mag_z'], color=['green'], label=['z [mG]'])
    sensors.plot_key(['sun3'], color=['blue'], label=['-x [mA]'])
    sensors.plot_key(['sun2'], color=['orange'], label=['-y [mA]'])
    sensors.plot_key(['sun4'], color=['green'], label=['-z [mA]'])

    monitor.add_vector('sun_sc_i', color='yellow')
    monitor.add_vector('mag_i', color='orange')
    monitor.add_vector('moon_sc_i', color='white')

    monitor.plot(x_dataset='full_time', y_dataset='mag_i')
    # monitor.plot(x_dataset='full_time', y_dataset='lonlat')
    # monitor.plot(x_dataset='full_time', y_dataset='sun_i_sc')
    # monitor.plot(x_dataset='full_time', y_dataset='sat_pos_i')
    monitor.plot(x_dataset='time_pred', y_dataset='q_i2b_pred')
    monitor.plot(x_dataset='time_pred', y_dataset='omega_b_pred')
    # ekf
    monitor.plot(x_dataset='full_time', y_dataset='b_est')
    monitor.plot(x_dataset='full_time', y_dataset='q_est')
    monitor.plot(x_dataset='full_time', y_dataset='omega_est')
    # monitor.plot(x_dataset='full_time', y_dataset='mag_est')
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
