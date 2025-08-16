"""
Created by Elias Obreque
Date: 04-09-2023
email: els.obrq@gmail.com
"""
import matplotlib.pyplot as plt
import numpy as np
import pickle
import datetime
import os
from tqdm import tqdm
import pandas as pd
import cv2
import importlib.util
import matplotlib as mpl

from src.kalman_filter.ekf_multy import MEKF
from src.kalman_filter.ekf_mag_calibration import MagUKF
from src.data_process import RealData
from src.dynamics.quaternion import Quaternions
from src.dynamics.dynamics_kinematics import Dynamics, calc_quaternion, calc_omega_b, shadow_zone, _MJD_1858, RAD2DEG
from tools.get_video_frame import save_frame
from tools.get_point_vector_from_picture import get_vector_v2
from tools.monitor import Monitor
from tools.camera_sensor import CamSensor
from tools.mathtools import julian_to_datetime, timestamp_to_julian, get_lvlh2b

mpl.rcParams['font.size'] = 12
# mpl.rcParams['font.family'] = 'Arial'   # Set the default font family

# CONFIG
# PROJECT_FOLDER = "./data/20240804/"
# PROJECT_FOLDER = "./data/M-20230824/"
# PROJECT_FOLDER = "./data/20230904/"
PROJECT_FOLDER = "./data/SimulationExample/"

PROJECT_FOLDER = os.path.abspath(PROJECT_FOLDER) + "/"
module_name = "dataconfig"

# load configuration
spec = importlib.util.spec_from_file_location(module_name, PROJECT_FOLDER + module_name + ".py")
myconfig = importlib.util.module_from_spec(spec)
spec.loader.exec_module(myconfig)
SATELLITE_NAME = myconfig.SATELLITE_NAME
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
if "SIMULATION" in list(myconfig.__dict__):
    SIMULATION = myconfig.SIMULATION
else:
    SIMULATION = False
VIDEO_FPS = 30
VIDEO_DT = 1 / VIDEO_FPS
if "VIDEO_FPS" in list(myconfig.__dict__):
    VIDEO_FPS = float(myconfig.VIDEO_FPS)
    VIDEO_DT = 1 / VIDEO_FPS

# samples. None to use all the samples.
MAX_SAMPLES = 600
#================================================#
FORCE_CALCULATION = myconfig.FORCE_CALCULATION
FORCE_ESTIMATION = True
#================================================#

# long time prediction
pred_step_sec = 5

if __name__ == '__main__':
    # LOAD LAST RESULT ------------------------------------------------------------------------------------------------
    ekf_channels = None
    if not os.path.exists(PROJECT_FOLDER + "results/"):
        os.mkdir(PROJECT_FOLDER + "results/")
    if os.path.exists(PROJECT_FOLDER + "estimation_results.pkl") and not FORCE_CALCULATION:
        with open(PROJECT_FOLDER + "estimation_results.pkl", 'rb') as fp:
            ekf_channels = pickle.load(fp)

    # REAL DATA and TIME ----------------------------------------------------------------------------------------------
    # create data with datetime, and near tle
    sensors = RealData(PROJECT_FOLDER, OBC_DATA)
    # sensors.set_gyro_bias(-3.846, 0.1717, -0.6937, unit='deg')
    sensors.create_datetime_from_timestamp(TIME_FORMAT)
    # INERTIA definition
    inertia = np.array([37540.678, 38550.678, 6873.717, -100.0, -50.0, 50.0]) * 1e-6
    # inertia = sensors.estimate_inertia_matrix(guess=inertia)
    sensors.set_inertia(inertia)
    # show window time
    if WINDOW_TIME['FLAG']:
        sensors.set_window_time(WINDOW_TIME['Start'], WINDOW_TIME['Stop'], TIME_FORMAT, OBC_DATA_STEP, SIMULATION)
    else:
        sensors.set_window_time()

    line1, line2 = sensors.search_nearly_tle(SATELLITE_NAME)
    print(line1, line2)
    # TIME
    MAX_SAMPLES = len(sensors.data['jd']) if MAX_SAMPLES is None else MAX_SAMPLES
    dt_obc = OBC_DATA_STEP
    dt_sim = dt_obc # WINDOW_TIME['STEP'] # TODO: use an external dt for simulations
    start_datetime = datetime.datetime.strptime(WINDOW_TIME['Start'], TIME_FORMAT)
    stop_datetime = datetime.datetime.strptime(WINDOW_TIME['Stop'], TIME_FORMAT)
    print(start_datetime.timestamp(), stop_datetime.timestamp())
    sensors.MAX_SAMPLES = MAX_SAMPLES

    # MODEL ------------------------------------------------------------------------------------------------------
    # if not exist file channels
    dynamic_orbital = Dynamics(WINDOW_TIME['Start'], WINDOW_TIME['Stop'], dt_sim, line1, line2, TIME_FORMAT)
    if os.path.exists(PROJECT_FOLDER + "channels.pkl") and not FORCE_CALCULATION:
        with open(PROJECT_FOLDER + "channels.pkl", 'rb') as fp:
            channels = pickle.load(fp)
        dynamic_orbital.load_data(channels)
        dynamic_orbital.calc_mag()
        channels = dynamic_orbital.channels
    else:
        # Inertial Parameters
        channels = dynamic_orbital.get_dynamics(SIMULATION, sensors.sc_inertia)
        # save channels as json
        with open(PROJECT_FOLDER + 'channels.pkl', 'wb') as file_:
            pickle.dump(channels, file_)

    if SIMULATION:
        # Create synthetic data in sensor
        dynamic_orbital.set_inertia(sensors.sc_inertia)
        if not os.path.exists(PROJECT_FOLDER + OBC_DATA):
            dynamic_orbital.update_attitude(np.array([0, 1, 1, 0]) / np.sqrt(2),
                                            np.array([-30 * np.deg2rad(1), -20 * np.deg2rad(1), 0.1]))
            sensors.create_sim_data(channels)
        sensors.create_sim_video(channels, VIDEO_DATA, VIDEO_TIME_LAST_FRAME, VIDEO_FPS)

    dynamic_orbital.plot_gt(PROJECT_FOLDER + 'results/gt')
    dynamic_orbital.plot_mag(PROJECT_FOLDER + 'results/mag_model_igrf13')
    dynamic_orbital.plot_sun_sc(PROJECT_FOLDER + 'results/sun_pos_from_sc')
    dynamic_orbital.plot_darkness(PROJECT_FOLDER + 'results/is_dark_sc')

    # sensors.data['is_dark'] = channels['is_dark']
    sensors.plot_main_data(SIMULATION)
    # calibrate gyro
    # pcov_gyro = sensors.calibrate_gyro()
    # print(pcov_gyro)
    # sensors.plot_key(['acc_x', 'acc_y', 'acc_z'], color=['blue', 'orange', 'green'],
    #                  name="cal_gyro_sensor_dps", title="Calibrate Gyro Sensor [rad/s]",
    #                  label=['x [mG]', 'y [mG]', 'z [mG]', r'$||\cdot||$'], drawstyle=['steps-post'] * 4, marker=['.'] * 4,
    #                  show=True)

    # calibration using two-step and channels
    mag_i_on_obc = [channels['mag_i'][np.argmin(np.abs(channels['full_time'] - jd_obc))] for jd_obc in sensors.data['jd']]
    sensors.plot_mag_error(mag_i_on_obc, 'before')
    sensors.calibrate_mag(mag_i=mag_i_on_obc)
    sensors.plot_key(['mag_x', 'mag_y', 'mag_z', '||mag||'], color=['blue', 'orange', 'green', 'black'],
                     name="mag_sensor_mg_two_step", title="TWO STEP Calibration - Mag", y_name='Magnetic field [mG]',
                     label=['x', 'y', 'z', r'$||\cdot||$'], drawstyle=['steps-post'] * 4, marker=['.'] * 4)
    sensors.show_mag_geometry("TWO STEP Method")
    sensors.plot_mag_error(mag_i_on_obc, 'after')

    # VIDEO -----------------------------------------------------------------------------------------------------------
    if CREATE_FRAME and VIDEO_DATA is not None:
        # width, height
        if isinstance(VIDEO_DATA, str):
            VIDEO_DATA = [VIDEO_DATA]
        if isinstance(VIDEO_TIME_LAST_FRAME, str):
            VIDEO_TIME_LAST_FRAME = [VIDEO_TIME_LAST_FRAME]
        for data_frame, time_frame in zip(VIDEO_DATA, VIDEO_TIME_LAST_FRAME):
            frame_shape = save_frame(PROJECT_FOLDER, data_frame, time_frame)

    data_video_list = {}
    if GET_VECTOR_FROM_PICTURE:
        focal_length = CamSensor.focal_length

        channels_video_list = {}
        for data_frame in VIDEO_DATA:
            VIDEO_FOLDER = PROJECT_FOLDER + data_frame.split('.')[0] + "/"
            vide_name = data_frame.split('.')[0]
            if not os.path.exists(VIDEO_FOLDER + "results/"):
                os.makedirs(VIDEO_FOLDER + "results/")
            if not os.path.exists(VIDEO_FOLDER + "results/" + f'pitch_roll_LVLH_{vide_name}.xlsx'):
                list_file = [elem for elem in os.listdir(VIDEO_FOLDER + "/frames/") if 'png' in elem]
                frame_shape = cv2.imread(VIDEO_FOLDER + "/frames/" + list_file[0]).shape
                num_list = [float(elem[:-4]) for elem in list_file if 'png' in elem]
                datalist = pd.DataFrame({'filename': list_file, 'id': num_list})
                datalist.sort_values(by='id', inplace=True)
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                video_salida = cv2.VideoWriter(VIDEO_FOLDER + "results/" + f"att_process_lvlh_{vide_name}.avi", fourcc, 10.0,
                                               (frame_shape[1], frame_shape[0]))
                rot_info = {'MJD':[], 'pitch': [], 'roll': [], 'e_b_x': [], 'e_b_y': [], 'e_b_z': [], 's_b_x': [], 's_b_y': [], 's_b_z': [], 'timestamp': []}
                frame_correction_time = 0.5

                pixel_size_width = CamSensor.sensor_width_v / frame_shape[0]
                pixel_size_height = CamSensor.sensor_width_h / frame_shape[1]
                # 22 +36 + 15+26+18
                for filename, ts_i in tqdm(datalist.values, desc=f"Computing vector from video {vide_name}... ",
                                           total=len(datalist.values)):
                    #1693838944.0
                    #1693838934.3
                    height_e = dynamic_orbital.get_altitude(ts_i)
                    height_sun = dynamic_orbital.get_distance_sun(ts_i)
                    edge_, img_cv2_, p_, r_, e_b_ = get_vector_v2(VIDEO_FOLDER + "/frames/" + filename, height_e,
                                                                  height_sun, pixel_size_height, pixel_size_width, focal_length)
                    rot_info['pitch'].append(p_)
                    rot_info['roll'].append(r_)
                    rot_info['timestamp'].append(ts_i)
                    rot_info['MJD'].append(timestamp_to_julian(ts_i - frame_correction_time) - _MJD_1858)
                    rot_info['e_b_x'].append(e_b_['Earth_c'][0])
                    rot_info['e_b_y'].append(e_b_['Earth_c'][1])
                    rot_info['e_b_z'].append(e_b_['Earth_c'][2])
                    rot_info['s_b_x'].append(e_b_['Sun_c'][0])
                    rot_info['s_b_y'].append(e_b_['Sun_c'][1])
                    rot_info['s_b_z'].append(e_b_['Sun_c'][2])
                    if img_cv2_ is not None:
                        video_salida.write(img_cv2_)
                        print(f" - filename {filename} added")
                video_salida.release()
                data_video = pd.DataFrame(rot_info)
                data_video.to_excel(VIDEO_FOLDER + "results/" + f'pitch_roll_LVLH_{vide_name}.xlsx', index=False)
            else:
                data_video = pd.read_excel(VIDEO_FOLDER + "results/" + f'pitch_roll_LVLH_{vide_name}.xlsx')

            data_video_list[vide_name] = data_video
            earth_b_camera = data_video[['e_b_x', 'e_b_y', 'e_b_z']].values
            # start_str, stop_str, step, line1, line2, format_time
            dynamic_video = Dynamics(jd_array=data_video['MJD'].values + _MJD_1858, line1=line1, line2=line2)
            if os.path.exists(VIDEO_FOLDER + "results/" + "channels_camera.p") and not FORCE_CALCULATION:
                with open(VIDEO_FOLDER + "results/" + "channels_camera.p", 'rb') as fp:
                    channels_video = pickle.load(fp)
                dynamic_video.load_data(channels_video)
                dynamic_video.calc_mag()
                channels_video = dynamic_video.channels
            else:
                channels_video = dynamic_video.get_dynamics()
                # save channels as json
                with open(VIDEO_FOLDER + "results/" + 'channels_camera.p', 'wb') as file_:
                    pickle.dump(channels_video, file_)

            channels_video_list[vide_name] = channels_video

            dynamic_video.plot_gt(VIDEO_FOLDER + "results/" + 'gt_video')
            dynamic_video.plot_mag(VIDEO_FOLDER + "results/" + 'mag_model_igrf13_video')
            dynamic_video.plot_sun_sc(VIDEO_FOLDER + "results/" + 'sun_pos_from_sc_video')
            dynamic_video.plot_earth_vector(VIDEO_FOLDER + "results/", earth_b_camera)
            # earth_point_inertial = -dynamic_video.get_unit_vector("sat_pos_i")
            sensors.plot_video_data(data_video, vide_name, VIDEO_FOLDER)
            plt.close("all")

        sensors.plot_gt_full_videos(PROJECT_FOLDER + "/results/" + 'gt_plus_videos.png', channels, channels_video_list)
        sensors.plot_windows(PROJECT_FOLDER)


    # UKF MAG CALIBRATION ----------------------------------------------------------------------------------------------
    if ONLINE_MAG_CALIBRATION:
        D_est = np.zeros(6) + 1e-9
        b_est = np.zeros(3) + 100
        ukf = MagUKF(b_est, D_est, alpha=0.2)
        mag_ukf = ukf.calibrate(mag_i_on_obc, sensors.data[['mag_x', 'mag_y', 'mag_z']].values)
        ukf.plot(np.linalg.norm(mag_ukf, axis=1), np.linalg.norm(mag_i_on_obc, axis=1), sensors.data['mjd'],
                 PROJECT_FOLDER + 'results/')
        sensors.data[['mag_x', 'mag_y', 'mag_z']] = mag_ukf
        sensors.data['||mag||'] = np.linalg.norm(mag_ukf, axis=1)
        sensors.plot_key(['mag_x', 'mag_y', 'mag_z', '||mag||'], color=['blue', 'orange', 'green', 'black'],
                         name="mag_sensor_mg_ukf", show=False, title="UKF Calibration - Mag [mG]",
                         label=['x [mG]', 'y [mG]', 'z [mG]', '||mag||'], drawstyle=['steps-post'] * 4, marker=['.'] * 4)

    sensors.show_mag_geometry("UKF Method")
    # ----------------------------------------------------------------------------------------------

    # Prediction using MEKF ----------------------------------------------------------------------------------------------

    omega_b = sensors.data[['acc_x', 'acc_y', 'acc_z']].values[0]
    q_i2b_0 = Quaternions.get_from_two_v(mag_i_on_obc[0], sensors.data[['mag_x', 'mag_y', 'mag_z']].values[0])()
    mag_b = Quaternions(q_i2b_0).frame_conv(mag_i_on_obc[0])
    moon_b = Quaternions(q_i2b_0).frame_conv(channels['moon_sc_i'][0])

    prediction_dict = {'q_i2b_pred': [],
                       'omega_b_pred': [],
                       'time_pred': [],}
    aux_data = {'q_lvlh2b': [],
                'ypr_lvlh2b': [],}
    if not os.path.exists(PROJECT_FOLDER + "estimation_results.pkl") or FORCE_ESTIMATION:
        # MEKF
        P = np.diag([1.0, 1.0, 1.0, 1.0, 1, 1]) * 1e-1
        ekf_model = MEKF(inertia, P=P, Q=np.zeros((6, 6)), R=np.zeros((3, 3)))
        ekf_model.sigma_bias = 1e-3 # gyro noise standard deviation [rad/s]
        ekf_model.sigma_omega = 1e-3 # gyro random walk standard deviation [rad/s*s^0.5]
        ekf_model.current_bias = np.array([0.0, 0.0, 0])

        q_i2b = np.array([0, 0, 0, 1])

        # Save initial data
        ekf_model.set_quat(q_i2b, save=True)
        ekf_model.set_gyro_measure(omega_b)
        ekf_model.save_time(channels['mjd'][0])
        ekf_model.save_vector(name='mag_est', vector=sensors.data[['mag_x', 'mag_y', 'mag_z']].values[0])
        ekf_model.save_vector(name='css_est', vector=sensors.data[['sun3', 'sun2', 'sun4']].values[0])
        ekf_model.save_vector(name='sun_b_est', vector=Quaternions(q_i2b).frame_conv(channels['sun_sc_i'][0]))

        # sensors_idx = 1
        moon_sc_b = [moon_b]
        t0 = channels['full_time'][0]
        ekf_sensors_step = 0.1
        flag_css = False

        q_lvlhl_, ypr_lvlh_ = get_lvlh2b(channels['sat_pos_i'][0], channels['sat_vel_i'][0], q_i2b)
        aux_data['q_lvlh2b'].append(q_lvlhl_)
        aux_data['ypr_lvlh2b'].append(ypr_lvlh_)

        for ch_idx, t_jd in tqdm(enumerate(channels['full_time'][1:MAX_SAMPLES]), total=MAX_SAMPLES - 1, desc="Main loop Estimation"):
            ch_idx += 1
            mag_ref_ = channels['mag_i'][ch_idx]
            sun_sc_i_ = channels['sun_sc_i'][ch_idx]
            sat_pos_i_ = channels['sat_pos_i'][ch_idx]
            sat_vel_i_ = channels['sat_vel_i'][ch_idx]
            sun_pos_i_ = channels['sun_i'][ch_idx]
            moon_pos_i_ = channels['moon_sc_i'][ch_idx]
            body_vec_ = sensors.data[['mag_x', 'mag_y', 'mag_z']].values[ch_idx]
            css_3_ = sensors.data[['sun3', 'sun2', 'sun4']].values[ch_idx]
            omega_gyro_ = sensors.data[['acc_x', 'acc_y', 'acc_z']].values[ch_idx]

            t_sec = np.round((t_jd -  channels['full_time'][1]) * 86400, 3)
            dt_sec = np.round((t_jd - t0) * 86400, 3)

            # ======================================================================================================== #
            # PREDICTION with base time (Orbit and OBC @ 1 sec) in a big Windows

            omega_b_pred = ekf_model.get_calibrate_omega()
            q_i2b_pred = ekf_model.current_quaternion
            for _ in tqdm(range(int(pred_step_sec / 0.1)), total=int(pred_step_sec / 0.1), desc="Prediction calculation"):
                q_i2b_pred = calc_quaternion(q_i2b_pred, omega_b_pred, 0.1)
                sat_pos_b_ = Quaternions(q_i2b_pred).frame_conv(sat_pos_i_) * 1e3
                omega_b_pred = calc_omega_b(omega_b_pred, 0.1, inertia_=sensors.sc_inertia, rb=sat_pos_b_)
            t_pred = t_sec + pred_step_sec

            prediction_dict['time_pred'].append(t_pred)
            prediction_dict['q_i2b_pred'].append(q_i2b_pred)
            prediction_dict['omega_b_pred'].append(omega_b_pred)

            # # integration
            ekf_model.propagate(dt_obc)

            # ukf_model.predict()
            # mag
            mag_est = ekf_model.inject_vector(body_vec_, mag_ref_, sigma2=sensors.std_rn_mag ** 2, sensor='mag')

            # mag_est_ukf = ukf_model.inject_vector(body_vec_, mag_ref_, sigma2=5000, sensor='mag')
            # css
            css_est = np.zeros(3)
            is_dark = shadow_zone(sat_pos_i_, sun_pos_i_)
            error_mag = np.linalg.norm(mag_est - body_vec_)
            if not is_dark and error_mag < 10 or flag_css: # mG
                css_3_[css_3_ < 50] = 0.0
                css_est = ekf_model.inject_vector(css_3_, sun_sc_i_, gain=-sensors.I_max * np.eye(3), sigma2=1 ** 2, sensor='css')
                flag_css = True
                if error_mag > 100:
                    flag_css = False
            ekf_model.save_vector(name='css_est', vector=css_est)
            ekf_model.save_vector(name='mag_est', vector=mag_est)
            ekf_model.save_vector(name='sun_b_est', vector=Quaternions(ekf_model.current_quaternion).frame_conv(sun_sc_i_))
            ekf_model.reset_state()
            moon_sc_b.append(Quaternions(ekf_model.current_quaternion).frame_conv(moon_pos_i_))
            ekf_model.set_gyro_measure(omega_gyro_)
            q_lvlh2b, ypr_lvlh2b = get_lvlh2b(sat_pos_i_, sat_vel_i_, ekf_model.current_quaternion)
            aux_data['q_lvlh2b'].append(q_lvlh2b)
            aux_data['ypr_lvlh2b'].append(ypr_lvlh2b)


        ekf_channels = {**prediction_dict, **ekf_model.historical, **aux_data}
        with open(PROJECT_FOLDER + 'estimation_results.pkl', 'wb') as file_:
            pickle.dump(ekf_channels, file_)

    data_text = ["{}".format(julian_to_datetime(jd_)) for jd_ in channels['full_time']]
    channels['DateTime'] = data_text
    channels_temp = channels.copy()
    channels = {k: v[:MAX_SAMPLES] for k, v in channels_temp.items()}
    channels = {**channels, **ekf_channels}
    error_mag = channels['mag_est'] - sensors.data[['mag_x', 'mag_y', 'mag_z']].values[:MAX_SAMPLES]
    error_pred = [(Quaternions(Quaternions(q_p).conjugate()) * Quaternions(q_kf)).get_angle(error_flag=True)
                  for q_p, q_kf in zip(channels['q_i2b_pred'], channels['q_est'][pred_step_sec:])]

    q_est = np.array(channels['q_est'])
    if SIMULATION:
        error_est = []
        for q_kf, q_p in zip(sensors.data[['q_i2b_x', 'q_i2b_y', 'q_i2b_z', 'q_i2b_r']].values[:MAX_SAMPLES],
                             channels['q_est']):
            q_error = Quaternions(q_p).conjugate_class() * Quaternions(q_kf)
            error_est.append(q_error.get_angle(error_flag=True) * np.rad2deg(1))

        error_est = np.array(error_est)
        # error_est[error_est > 180.0] = 360.0 - error_est[error_est > 180.0]
        channels_true = {'error_q_true': error_est,
                         'error_w_true': sensors.data[['w_x', 'w_y', 'w_z']].values[:MAX_SAMPLES] - np.array(channels['omega_est'])}
        channels = {**channels, **channels_true}

    # q_train = q_est[:int(0.1 * len(q_est))]
    # theta_r = np.arccos(q_train[:, 3]) * 2
    # quat_vec = q_train[:, :3] / np.sin(theta_r * 0.5)

    error_mag_std = np.std(error_mag, axis=0)

    color_lists = ['blue', 'orange', 'green']
    name_lists = ['x', 'y', 'z']
    fig = plt.figure()
    plt.title("Magnetic field estimation error")
    plt.plot(np.abs(error_mag), '.-', alpha=0.7)
    plt.ylabel("Magnetic field [mG]")
    plt.xlabel("Steps [s]")
    plt.grid()
    plt.legend([
        f'x (std: {error_mag_std[0]:.2f})',
        f'y (std: {error_mag_std[1]:.2f})',
        f'z (std: {error_mag_std[2]:.2f})'
    ])
    plt.xticks(rotation=15)
    plt.ticklabel_format(useOffset=False)
    plt.tight_layout()
    plt.yscale('log')
    fig.savefig(PROJECT_FOLDER + "results/" + "error_mag_est.png")

    fig = plt.figure()
    plt.title(f"Predicted quaternion angular error (forward: {pred_step_sec})")
    plt.ylabel("Angular error [deg]")
    plt.xlabel("Steps [s]")
    plt.plot(np.array(error_pred) * 180 / np.pi, '.-', alpha=0.7)
    plt.grid()
    plt.xticks(rotation=15)
    plt.ticklabel_format(useOffset=False)
    plt.tight_layout()
    fig.savefig(PROJECT_FOLDER + "results/" + f"error_prediction_quat_{pred_step_sec}.png")

    fov = 48 * np.deg2rad(1) / 2
    view_of_moon = np.cos(fov)
    moon_sc_b = [Quaternions(ekf_q).frame_conv(vec_) for ekf_q, vec_ in zip(channels['q_est'], channels['moon_sc_i'])]
    moon_sc_b = np.asarray(moon_sc_b) / np.linalg.norm(moon_sc_b, axis=1).reshape(-1, 1)

    fig, axes = plt.subplots(nrows=3, ncols=1, sharex=True)
    fig.suptitle("Moon Position estimation @ BodyFrame")
    axes[1].hlines(view_of_moon, channels['mjd'][0], channels['mjd'][-1], color='red')
    for i in range(3):
        axes[i].plot(channels['mjd'], moon_sc_b[:, i], '.-', color=color_lists[i], alpha=0.7)
        axes[i].grid()
        axes[i].set_ylabel(f"{name_lists[i]} [km]")
    axes[2].set_xlabel("MJD")
    plt.tight_layout()
    fig.savefig(PROJECT_FOLDER + "results/" + f"moon_estimation_fov.png")

    # channels['q_est'] = [np.array([0, 0, 0, 1]) for elem in channels['sat_pos_i']]
    monitor = Monitor(channels, PROJECT_FOLDER + "results/")
    monitor.set_position('sat_pos_i')
    monitor.set_quaternion('q_est')
    monitor.set_sideral('sideral')

    # sensors.plot_key(['mag_x'], color=['blue'], label=['x [mG]'])
    # sensors.plot_key(['mag_y'], color=['orange'], label=['y [mG]'])
    # sensors.plot_key(['mag_z'], color=['green'], label=['z [mG]'])
    # sensors.plot_key(['sun3'], color=['blue'], label=['-x [mA]'])
    # sensors.plot_key(['sun2'], color=['orange'], label=['-y [mA]'])
    # sensors.plot_key(['sun4'], color=['green'], label=['-z [mA]'])

    monitor.add_vector('sun_sc_i', color='yellow')
    monitor.add_vector('mag_i', color='orange')
    monitor.add_vector('moon_sc_i', color='white')

    monitor.plot(x_dataset='full_time', y_dataset='mag_i')
    # monitor.plot(x_dataset='full_time', y_dataset='lonlat')
    # monitor.plot(x_dataset='full_time', y_dataset='sun_i_sc')
    # monitor.plot(x_dataset='full_time', y_dataset='sat_pos_i')
    monitor.plot(x_dataset='time_pred', y_dataset='q_i2b_pred', xname="MJD", yname="Quaternion i2b",
                 title="Predicted Quaternion", legend_list=["x", "y", "z", "s"])
    monitor.plot(x_dataset='time_pred', y_dataset='omega_b_pred', xname="MJD", yname="Angular velocity [rad/s]",
                 title="Predicted Angular velocity", legend_list=["x", "y", "z"])
    # ekf
    monitor.plot(x_dataset='mjd', y_dataset='b_est', xname="MJD", yname="Bias [rad/s]",
                 title="Gyroscope bias estimation", legend_list=["x", "y", "z"])
    monitor.plot(x_dataset='mjd', y_dataset='q_est', xname="MJD", yname="Quaternion i2b",
                 title="Quaternion estimation", legend_list=["x", "y", "z", "s"])
    monitor.plot(x_dataset='mjd', y_dataset='omega_est', xname="MJD", yname="Angular velocity [rad/s]",
                 title="Angular velocity estimation", legend_list=["x", "y", "z"])
    # monitor.plot(x_dataset='full_time', y_dataset='mag_est')
    monitor.plot(x_dataset='mjd', y_dataset='sun_b_est')
    monitor.plot(x_dataset='mjd', y_dataset='p_cov', xname="MJD", yname="Diagonal Covariance",
                 title="State error covariance matrix", legend_list=[r"$\delta \theta_x$", r"$\delta \theta_y$",
                                                                     r"$\delta \theta_z$", r"$\Delta b_x$", r"$\Delta b_y$", r"$\Delta b_z$"])
    monitor.plot(x_dataset='mjd', y_dataset='q_lvlh2b', xname="MJD", yname="Quaternion LVLH2b",
                 title="Quaternion estimation LVLH", legend_list=["x", "y", "z", "s"])
    monitor.plot(x_dataset='mjd', y_dataset='ypr_lvlh2b', xname="MJD", yname="YPR LVLH2b",
                 title="Yaw-Pitch-Roll estimation LVLH", legend_list=["yaw", "pitch", "roll"])

    if SIMULATION:
        monitor.plot(x_dataset='mjd', y_dataset='error_q_true', log_scale=True)
        monitor.plot(x_dataset='mjd', y_dataset='error_w_true')

    if GET_VECTOR_FROM_PICTURE:
        fig_picture, axes = plt.subplots(nrows=2, ncols=1, sharex=True)
        axes[0].grid()
        axes[0].set_xlabel("MJD")
        axes[1].grid()
        axes[1].set_xlabel("MJD")
        axes[0].set_ylabel("Roll [deg]")
        axes[1].set_ylabel("Pitch [deg]")
        for key, data in data_video_list.items():
            axes[0].plot(data['MJD'], data['roll'] * RAD2DEG, 'o', label="Video {}".format(key))
            axes[1].plot(data['MJD'], data['pitch'] * RAD2DEG, 'o', label="Video {}".format(key))
        axes[0].plot(channels['mjd'], np.array(channels['ypr_lvlh2b'])[:, 2] * RAD2DEG, label="EKF")
        axes[1].plot(channels['mjd'], np.array(channels['ypr_lvlh2b'])[:, 1] * RAD2DEG, label="EKF")
        axes[0].legend()
        axes[1].legend()
        plt.xlim(np.min(channels['mjd']), np.max(channels['mjd']))
        fig_picture.savefig(PROJECT_FOLDER + "results/" + f"ypr_estimation_lvlh.png")

    fig, axes = plt.subplots(nrows=3, ncols=1, sharex=True)
    fig.suptitle("Magnetometer estimation @ BodyFrame")
    for i in range(3):
        axes[i].plot(channels['mjd'], np.array(channels['mag_est'])[:, i], '.-', color=color_lists[i], alpha=0.3)
        axes[i].grid()
        axes[i].set_ylabel(f"{name_lists[i]} [mG]")
    axes[2].set_xlabel("MJD")
    plt.tight_layout()
    fig.savefig(PROJECT_FOLDER + "results/" + "magnetometer_estimation.png")

    fig, axes = plt.subplots(nrows=3, ncols=1, sharex=True)
    fig.suptitle("Coarse sun sensor estimation @ BodyFrame")
    for i in range(3):
        axes[i].plot(channels['mjd'], np.array(channels['css_est'])[:, i], '.-', color=color_lists[i], alpha=0.3)
        axes[i].grid()
        axes[i].set_ylabel(f"{name_lists[i]} [mA]")
    axes[2].set_xlabel("MJD")
    plt.tight_layout()
    fig.savefig(PROJECT_FOLDER + "results/" + "coarse_sun_sensor_estimation.png")

    plt.show()
    plt.close("all")
    #monitor.show_monitor()
    #monitor.plot3d()
