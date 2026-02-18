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
from sklearn.metrics import mean_squared_error

from src.kalman_filter.simulation_estimation import run_main_estimation
from src.kalman_filter.ekf_mag_calibration import MagUKF
from src.data_process import RealData
from src.dynamics.quaternion import Quaternions
from src.dynamics.dynamics_kinematics import Dynamics, MJD_1858
from tools.get_video_frame import save_frame
from tools.get_point_vector_from_picture import get_vector_v2, ROT_CAM2BODY
from tools.monitor import Monitor
from tools.camera_sensor import CamSensor
from tools.mathtools import julian_to_datetime, timestamp_to_julian

mpl.rcParams['font.size'] = 12
# mpl.rcParams['font.family'] = 'Arial'   # Set the default font family

# CONFIG
# PROJECT_FOLDER = "./data/20240804/"
# PROJECT_FOLDER = "./data/M-20230824/"
PROJECT_FOLDER = "./data/20230904/"
# PROJECT_FOLDER = "./data/SimulationExample/"

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
VIDEO_CORRECTION_TIME = myconfig.VIDEO_CORRECTION_TIME

if "SIMULATION" in list(myconfig.__dict__):
    SIMULATION = myconfig.SIMULATION
else:
    SIMULATION = False

# VIEW
SHOW_BASIC_PLOT = False
SHOW_EKF_PLOT = False
SHOW_VIDEO_PLOT = True
VIDEO_FPS = 30
VIDEO_DT = 1 / VIDEO_FPS
if "VIDEO_FPS" in list(myconfig.__dict__):
    VIDEO_FPS = float(myconfig.VIDEO_FPS)
    VIDEO_DT = 1 / VIDEO_FPS

# samples. None to use all the samples.
MAX_SAMPLES = 500
#================================================#
FORCE_CALCULATION = myconfig.FORCE_CALCULATION
FORCE_ESTIMATION = True
#================================================#

# long time prediction
pred_step_sec = 30

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
    #sensors.set_gyro_bias(-3.846, 0.1717, -0.6937, unit='deg')
    sensors.create_datetime_from_timestamp(TIME_FORMAT)
    # INERTIA definition
    inertia = np.array([37540.678, 38550.678, 6873.717, -0.0, -0.0, 0.0]) * 1e-6
    inertia = np.array([38478.678, 38528.678, 6873.717, -0.0, -0.0, 0.0]) * 1e-6
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
    dt_sim = dt_obc # WINDOW_TIME['STEP'] # TODO: use an external dt for simulations ??
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
        sensors.create_sim_video(channels, VIDEO_DATA, VIDEO_TIME_LAST_FRAME, VIDEO_FPS, VIDEO_CORRECTION_TIME)

        print(f"Gyro bias: [-0.07, -0.01, 0.04]")
    # dynamic_orbital.plot_gt(PROJECT_FOLDER + 'results/gt')
    # dynamic_orbital.plot_mag(PROJECT_FOLDER + 'results/mag_model_igrf13')
    # dynamic_orbital.plot_sun_sc(PROJECT_FOLDER + 'results/sun_pos_from_sc')
    # dynamic_orbital.plot_darkness(PROJECT_FOLDER + 'results/is_dark_sc')

    # sensors.data['is_dark'] = channels['is_dark']
    if SHOW_BASIC_PLOT:
        sensors.plot_main_data(SIMULATION)
        # calibrate gyro
        pcov_gyro = sensors.calibrate_gyro()
        print(pcov_gyro)
        sensors.plot_key(['acc_x', 'acc_y', 'acc_z'], color=['blue', 'orange', 'green'],
                         name="cal_gyro_sensor_dps", title="Calibrate Gyro Sensor [rad/s]",
                         label=['x [mG]', 'y [mG]', 'z [mG]', r'$||\cdot||$'], drawstyle=['steps-post'] * 4, marker=['.'] * 4,
                         show=True)

    # calibration using two-step and channels
    mag_i_on_obc = [channels['mag_i'][np.argmin(np.abs(channels['full_time'] - jd_obc))] for jd_obc in sensors.data['jd']]
    if SHOW_BASIC_PLOT:
        sensors.plot_mag_error(mag_i_on_obc, 'before')
    sensors.calibrate_mag(mag_i=np.array(mag_i_on_obc))
    if SHOW_BASIC_PLOT:
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
                path_frame = VIDEO_FOLDER + "/frames/"
                list_file = [elem for elem in os.listdir(path_frame) if 'png' in elem]
                frame_shape = cv2.imread(path_frame + list_file[0]).shape
                num_list = [float(elem[:-4]) for elem in list_file if 'png' in elem]
                datalist = pd.DataFrame({'filename': list_file, 'id': num_list})
                datalist.sort_values(by='id', inplace=True)
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                video_salida = cv2.VideoWriter(VIDEO_FOLDER + "results/" + f"att_process_lvlh_{vide_name}.avi", fourcc, 10.0,
                                               (frame_shape[1], frame_shape[0]))
                rot_info = {'MJD':[], 'pitch': [], 'roll': [], 'e_b_x': [], 'e_b_y': [], 'e_b_z': [], 's_b_x': [], 's_b_y': [], 's_b_z': [], 'timestamp': []}

                pixel_size_width = CamSensor.sensor_width_v / frame_shape[0]
                pixel_size_height = CamSensor.sensor_width_h / frame_shape[1]
                # 22 +36 + 15+26+18
                for filename, ts_i in tqdm(datalist.values, desc=f"Computing vector from video {vide_name}... ",
                                           total=len(datalist.values)):
                    #1693838944.0
                    #1693838934.3
                    height_e = dynamic_orbital.get_altitude(ts_i)
                    height_sun = dynamic_orbital.get_distance_sun(ts_i)
                    edge_, img_cv2_, p_, r_, e_c_ = get_vector_v2(path_frame + filename, height_e,
                                                                  height_sun, pixel_size_height, pixel_size_width, focal_length)
                    rot_info['pitch'].append(p_)
                    rot_info['roll'].append(r_)
                    rot_info['timestamp'].append(ts_i - VIDEO_CORRECTION_TIME)
                    rot_info['MJD'].append(timestamp_to_julian(ts_i - VIDEO_CORRECTION_TIME) - MJD_1858)
                    e_b_ = ROT_CAM2BODY @ e_c_['Earth_c']
                    s_b_ = ROT_CAM2BODY @ e_c_['Sun_c']

                    rot_info['e_b_x'].append(e_b_[0])
                    rot_info['e_b_y'].append(e_b_[1])
                    rot_info['e_b_z'].append(e_b_[2])
                    rot_info['s_b_x'].append(s_b_[0])
                    rot_info['s_b_y'].append(s_b_[1])
                    rot_info['s_b_z'].append(s_b_[2])
                    if img_cv2_ is not None:
                        video_salida.write(img_cv2_)
                        print(f" - filename {filename} added")
                video_salida.release()
                data_video = pd.DataFrame(rot_info)
                data_video.to_excel(VIDEO_FOLDER + "results/" + f'pitch_roll_LVLH_{vide_name}.xlsx', index=False)
            else:
                data_video = pd.read_excel(VIDEO_FOLDER + "results/" + f'pitch_roll_LVLH_{vide_name}.xlsx')

            if SIMULATION:
                data_video['MJD'] += VIDEO_CORRECTION_TIME / 86400 # check with dataconfig
            data_video_list[vide_name] = data_video
            earth_b_camera = data_video[['e_b_x', 'e_b_y', 'e_b_z']].values
            # start_str, stop_str, step, line1, line2, format_time
            dynamic_video = Dynamics(jd_array=data_video['MJD'].values + MJD_1858, line1=line1, line2=line2)
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

            # if SHOW_VIDEO_PLOT:
            #     dynamic_video.plot_gt(VIDEO_FOLDER + "results/" + 'gt_video')
            #     dynamic_video.plot_mag(VIDEO_FOLDER + "results/" + 'mag_model_igrf13_video')
            #     dynamic_video.plot_sun_sc(VIDEO_FOLDER + "results/" + 'sun_pos_from_sc_video')
            #     dynamic_video.plot_earth_vector(VIDEO_FOLDER + "results/", earth_b_camera)
            #     # earth_point_inertial = -dynamic_video.get_unit_vector("sat_pos_i")
            #     sensors.plot_video_data(data_video, vide_name, VIDEO_FOLDER)

        # if SHOW_VIDEO_PLOT:
        #     sensors.plot_gt_full_videos(PROJECT_FOLDER + "/results/" + 'gt_plus_videos.png', channels, channels_video_list)
        #     sensors.plot_windows(PROJECT_FOLDER)

    # ==================================================================================================================
    # UKF MAG CALIBRATION ----------------------------------------------------------------------------------------------
    if not ONLINE_MAG_CALIBRATION:
        D_est = np.zeros(6) + 1e-9
        b_est = np.zeros(3) + 100
        ukf = MagUKF(b_est, D_est, alpha=0.1)
        mag_raw = sensors.data[['mag_x', 'mag_y', 'mag_z']].values
        mag_ukf = ukf.calibrate(mag_i_on_obc, mag_raw)
        ukf.plot(np.linalg.norm(mag_ukf, axis=1), np.linalg.norm(mag_i_on_obc, axis=1), sensors.data['mjd'],
                 PROJECT_FOLDER + 'results/')
        sensors.data[['mag_x', 'mag_y', 'mag_z']] = mag_ukf
        sensors.data['||mag||'] = np.linalg.norm(mag_ukf, axis=1)

        sensors.plot_key(['mag_x', 'mag_y', 'mag_z', '||mag||'], color=['blue', 'orange', 'green', 'black'],
                         name="mag_sensor_mg_ukf", show=False, title="UKF Calibration - Mag [mG]",
                         label=['x [mG]', 'y [mG]', 'z [mG]', '||mag||'], drawstyle=['steps-post'] * 4, marker=['.'] * 4)

        sensors.show_mag_geometry("UKF Method", show_plot=True)
        sensors.plot_key(['mag_x', 'mag_y', 'mag_z', '||mag||'], color=['blue', 'orange', 'green', 'black'],
                         name="mag_sensor_mg_ukf", show=False, title="UKF Calibration - Mag [mG]",
                         label=['x [mG]', 'y [mG]', 'z [mG]', '||mag||'], drawstyle=['steps-post'] * 4, marker=['.'] * 4)
    # ----------------------------------------------------------------------------------------------
    # ==================================================================================================================

    # ==================================================================================================================
    # Prediction using MEKF ----------------------------------------------------------------------------------------------
    # ==================================================================================================================

    if not os.path.exists(PROJECT_FOLDER + "estimation_results.pkl") or FORCE_ESTIMATION:
        if SIMULATION:
            gain_sigma = 1
            sigma_omega = gain_sigma * np.deg2rad(0.3) # 0.01 # V
            sigma_bias = 1e-4 # gain_sigma * np.deg2rad(4.2e-3) # 1e-3 # U
            sigma_mag = 2.8
            sigma_css = 10 #5 * 930 * (1 - np.cos(np.deg2rad(3.5)))       # MEKF
        else:
            gain_sigma = 0.5
            sigma_omega = gain_sigma * np.deg2rad(0.3) # 0.01 #
            sigma_bias = gain_sigma * 1e-4  # 1e-3
            sigma_mag = 5
            sigma_css = 930 * (1 - np.cos(np.deg2rad(3.5)))      # MEKF
        print(f"\nSigma values. bias: {sigma_bias}, omega: {sigma_omega}, mag: {sigma_mag}, css: {sigma_css}\n")

        ekf_channels, rmse_pred_error = run_main_estimation(inertia, channels, sensors, mag_i_on_obc,
                                                            max_samples=MAX_SAMPLES,
                                                            online_mag_calibration=ONLINE_MAG_CALIBRATION,
                                                            pred_step_sec=pred_step_sec,
                                                            sigma_bias=sigma_bias,
                                                            sigma_omega=sigma_omega,
                                                            mag_sig=sigma_mag,
                                                            css_sig=sigma_css)
        
        error_mag_rmse = mean_squared_error(np.linalg.norm(ekf_channels['mag_ref_est'], axis=1),
                                            np.linalg.norm(ekf_channels['mag_ukf'], axis=1))
        print("Error prediction:", rmse_pred_error, "Erros RMSE MAG: ", error_mag_rmse)
        with open(PROJECT_FOLDER + 'estimation_results.pkl', 'wb') as file_:
            pickle.dump(ekf_channels, file_)

    # ==================================================================================================================
    # ==================================================================================================================

    data_text = ["{}".format(julian_to_datetime(jd_)) for jd_ in channels['full_time']]
    channels['DateTime'] = data_text
    channels_temp = channels.copy()
    channels = {k: v[:MAX_SAMPLES] for k, v in channels_temp.items()}

    channels = {**channels, **ekf_channels}
    error_mag = np.array(channels['mag_ref_est']) - np.array(channels['mag_ukf'])
    error_pred = channels['error_pred']

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

        error_gyro_std = np.std(channels['error_w_true'], axis=0)

        if SHOW_EKF_PLOT:
            fig = plt.figure()
            plt.title("Absolute Angular velocity estimation error")
            plt.plot(np.abs(channels['error_w_true']), '.-', alpha=0.7)
            plt.ylabel("Angular velocity error [rad/s]")
            plt.xlabel("Steps [s]")
            plt.grid()
            plt.legend([
                f'x (std: {error_gyro_std[0]:.3f})',
                f'y (std: {error_gyro_std[1]:.3f})',
                f'z (std: {error_gyro_std[2]:.3f})'
            ])
            plt.xticks(rotation=15)
            plt.ticklabel_format(useOffset=False)
            plt.tight_layout()
            plt.yscale('log')
            fig.savefig(PROJECT_FOLDER + "results/" + "abs_error_gyro_est.png")

    # q_train = q_est[:int(0.1 * len(q_est))]
    # theta_r = np.arccos(q_train[:, 3]) * 2
    # quat_vec = q_train[:, :3] / np.sin(theta_r * 0.5)


    error_mag_std = np.std(error_mag, axis=0)

    color_lists = ['blue', 'orange', 'green']
    name_lists = ['x', 'y', 'z']

    fov = 48 * np.deg2rad(1) / 2
    view_of_moon = np.cos(fov)
    moon_sc_b = [Quaternions(ekf_q).frame_conv(vec_) for ekf_q, vec_ in zip(channels['q_est'], channels['moon_sc_i'])]
    moon_sc_b = np.asarray(moon_sc_b) / np.linalg.norm(moon_sc_b, axis=1).reshape(-1, 1)


    if SHOW_EKF_PLOT or True:
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

    # ==================================================================================================================
    # MONITOR
    monitor = Monitor(channels, PROJECT_FOLDER + "results/", video_dataset=data_video_list)
    monitor.set_position('sat_pos_i')
    monitor.set_quaternion('q_est')
    monitor.set_sideral('sideral')
    monitor.set_time("timestamp")
    df_basic = monitor.get_basic_dataframe()
    df_basic.to_csv(PROJECT_FOLDER + "results/" + "basic_data_to_view3d.csv")

    monitor.add_vector('sun_sc_i', color='yellow')
    monitor.add_vector('mag_i', color='orange')
    monitor.add_vector('moon_sc_i', color='white')

    monitor.plot_all()

    if SHOW_BASIC_PLOT or True:
        monitor.plot('mjd', 'mag_ukf', xname="MJD", yname="Magnetic Field [mG]", title="ukf_mag_res")
        sensors.plot_key(['mag_x'], name='mag_x', color=['blue'], label=['x [mG]'], show=True)
        sensors.plot_key(['mag_y'], name='mag_y', color=['orange'], label=['y [mG]'], show=True)
        sensors.plot_key(['mag_z'], name='mag_z', color=['green'], label=['z [mG]'], show=True)
        sensors.plot_key(['sun3'], name='sun3', color=['blue'], label=['-x [mA]'], show=True)
        sensors.plot_key(['sun2'], name='sun2', color=['orange'], label=['-y [mA]'], show=True)
        sensors.plot_key(['sun4', 'sun3'], name='sun4', color=['blue', 'green'], label=['-z [mA]', '-x [mA]'], show=True)

    if SIMULATION:
        monitor.plot(x_dataset='mjd', y_dataset='error_q_true', xname="MJD", yname="Quaternion Error [deg]", log_scale=False)
        monitor.plot(x_dataset='mjd', y_dataset='error_w_true', xname="MJD", yname="Ang. Velocity Error [rad/s]", log_scale=False)

    if GET_VECTOR_FROM_PICTURE:
        monitor.plot_video_performance()

    # fig_mag, axes_mag = plt.subplots(nrows=3, ncols=1, sharex=True)
    # fig_mag.suptitle("Magnetometer estimation @ BodyFrame")
    # for i in range(3):
    #     axes_mag[i].step(channels['mjd'], np.array(channels['mag_est'])[:, i], '.-', color=color_lists[i], alpha=0.3)
    #     axes_mag[i].grid()
    #     axes_mag[i].set_ylabel(f"{name_lists[i]} [mG]")
    # axes_mag[2].set_xlabel("MJD")
    # plt.tight_layout()
    # fig_mag.savefig(PROJECT_FOLDER + "results/" + "magnetometer_estimation.png")
    #
    # fig, axes = plt.subplots(nrows=3, ncols=1, sharex=True)
    # fig.suptitle("Coarse sun sensor estimation @ BodyFrame")
    # for i in range(3):
    #     axes[i].step(channels['mjd'], np.array(channels['css_est'])[:, i], '.-', color=color_lists[i], alpha=0.3)
    #     axes[i].grid()
    #     axes[i].set_ylabel(f"{name_lists[i]} [mA]")
    # axes[2].set_xlabel("MJD")
    # plt.tight_layout()
    # fig.savefig(PROJECT_FOLDER + "results/" + "coarse_sun_sensor_estimation.png")
    plt.show()
    #plt.close("all")
