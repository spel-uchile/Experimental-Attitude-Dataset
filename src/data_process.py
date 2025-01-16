"""
Created by Elias Obreque
Date: 04-09-2023
email: els.obrq@gmail.com
"""
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram
from scipy.optimize import fsolve
from sklearn.cluster import AgglomerativeClustering

from src.dynamics.Quaternion import Quaternions
from tools.mathtools import *
from tools.two_step_mag_calibration import two_step
import pandas as pd
import numpy as np
import datetime
import functools
import json
import os
from src.kalman_filter.ekf_omega import EKFOmega
from scipy.spatial import ConvexHull
from matplotlib.patches import Rectangle
from sklearn.metrics import mean_squared_error
from sklearn.cluster import KMeans

ts2022 = 1640995200 / 86400  # day
jd2022 = 2459580.50000
_MJD_1858 = 2400000.5


class RealData:

    def __init__(self, folder, file_directory):
        # Real data
        self.std_rn_w = np.deg2rad(0.2) # 1e-3  # gyro noise standard deviation [rad/s]
        self.std_rw_w = 1e-4  # gyro random walk standard deviation [rad/s*s^0.5]
        self.std_rn_mag = 2.8  # magnetometer noise standard deviation [mG]
        self.I_nr_std_cos = np.deg2rad(1.8) # max cosine error [deg]
        self.I_max = 930 # Max expected value [mA]
        self.bias_true = None
        self.time_format = '%Y-%m-%d %H:%M:%S'
        self.step = 1
        self.folder_path = folder
        self.file_name = file_directory
        self.MAX_SAMPLES = None
        if os.path.exists(self.folder_path + self.file_name):
            self.data = pd.read_excel(folder + file_directory)
            self.data.dropna(inplace=True)
            self.data['jd'] = [timestamp_to_julian(float(ts)) for ts in self.data['timestamp'].values]
            self.data['mjd'] = self.data['jd'] - _MJD_1858
            self.data[['acc_x', 'acc_y', 'acc_z']] *= np.deg2rad(1)
            self.data['||mag||'] = np.linalg.norm(self.data[['mag_x', 'mag_y', 'mag_z']], axis=1)
            self.data.sort_values(by=['timestamp'], inplace=True)
            self.set_geometric_mag_bias()
            self.data.dropna(inplace=True)
            self.data.reset_index(inplace=True)
        else:
            self.data = pd.DataFrame()#columns=['timestamp', 'acc_x', 'acc_y', 'acc_z', 'mag_x', 'mag_y', 'mag_z',
            #       'sun3', 'sun2', 'sun4', 'jd', 'mjd', '||mag||'])
        self.start_time = 0.
        self.end_time = 0.
        self.sc_inertia = np.array([38478.678, 38528.678, 6873.717, 0, 0, 0]) * 1e-6
        self.data_video = {}

    def set_geometric_mag_bias(self):
        # points_xy = self.data[['mag_x', 'mag_y']].values
        # points_xz = self.data[['mag_x', 'mag_z']].values
        # points_yz = self.data[['mag_y', 'mag_z']].values
        #
        # points_xy = points_xy[points_xy[:, 1] > 0]
        # points_xz = points_xz[points_xz[:, 1] > 0]
        # points_yz = points_yz[points_yz[:, 1] > 0]
        #
        # hull_xy = ConvexHull(points_xy)
        # hull_xz = ConvexHull(points_xz)
        # hull_yz = ConvexHull(points_yz)
        #
        # hull_points_xy = points_xy[hull_xy.vertices]
        # hull_points_xz = points_xz[hull_xz.vertices]
        # hull_points_yz = points_yz[hull_yz.vertices]
        #
        # contour_center_xy = np.mean(hull_points_xy, axis=0)
        # contour_center_xz = np.mean(hull_points_xz, axis=0)
        # contour_center_yz = np.mean(hull_points_yz, axis=0)
        self.show_mag_geometry("Raw Mag Data")

    def show_mag_geometry(self, title: str =None):
        fig, axes = plt.subplots(1, 3, figsize=(12, 4.1))
        fig.suptitle(title) if title is not None else None
        center_mean = self.data[['mag_x', 'mag_y', 'mag_z']].mean()
        axes[0].set_title(f"({np.round(center_mean[0], 2)}, {np.round(center_mean[2], 2)})")
        axes[1].set_title(f"({np.round(center_mean[1], 2)}, {np.round(center_mean[2], 2)})")
        axes[2].set_title(f"({np.round(center_mean[0], 2)}, {np.round(center_mean[1], 2)})")

        axes[0].plot(self.data['mag_x'], self.data['mag_z'], '.')
        axes[0].plot(center_mean[0], center_mean[2], 'r*')
        # axes[0].plot(contour_center_xz[0], contour_center_xz[1], 'y*')
        # axes[0].plot(hull_points_xz[:, 0], hull_points_xz[:, 1], 'r-', label='Contorno exterior')
        axes[0].grid()
        axes[0].set_ylabel('Mag z [mG]')
        axes[0].set_xlabel('Mag x [mG]')
        axes[0].set_box_aspect(1)

        axes[1].plot(self.data['mag_y'], self.data['mag_z'], '.')
        axes[1].plot(center_mean[1], center_mean[2], 'r*')
        axes[1].grid()
        # axes[1].plot(contour_center_yz[0], contour_center_yz[1], 'y*')
        # axes[1].plot(hull_points_yz[:, 0], hull_points_yz[:, 1], 'r-', label='Contorno exterior')
        axes[1].set_ylabel('Mag y [mG]')
        axes[1].set_xlabel('Mag z [mG]')
        axes[1].set_box_aspect(1)
        axes[2].plot(self.data['mag_x'], self.data['mag_y'], '.')
        axes[2].plot(center_mean[0], center_mean[1], 'r*')
        axes[2].grid()
        # axes[2].plot(contour_center_xy[0], contour_center_xy[1], 'y*')
        # axes[2].plot(hull_points_xy[:, 0], hull_points_xy[:, 1], 'r-', label='Contorno exterior')
        axes[2].set_ylabel('Mag x [mG]')
        axes[2].set_xlabel('Mag y [mG]')
        axes[2].set_box_aspect(1)
        plt.tight_layout()
        name_ = self.folder_path + "results/"+ f'{title.replace(" ", "_").lower()}_geo.jpg'
        fig.savefig(name_)
        plt.close()

    def set_inertia(self, inertia_):
        self.sc_inertia = np.diag(inertia_[:3])
        self.sc_inertia[0][1] = inertia_[3]
        self.sc_inertia[1][0] = inertia_[3]
        self.sc_inertia[0][2] = inertia_[4]
        self.sc_inertia[2][0] = inertia_[4]
        self.sc_inertia[1][2] = inertia_[5]
        self.sc_inertia[2][1] = inertia_[5]

    def create_datetime_from_timestamp(self, time_format=None):
        if "timestamp" in self.data.columns:
            self.time_format = time_format if time_format is not None else "%Y-%m-%d %H:%M:%S"
            self.data['DateTime'] = [datetime.datetime.fromtimestamp(ts, tz=datetime.timezone.utc).strftime(self.time_format)
                                     for ts in self.data['timestamp']]

    def set_gyro_bias(self, gx, gy, gz, unit='deg'):
        if unit == 'deg':
            gx = gx * np.pi / 180
            gy = gy * np.pi / 180
            gz = gz * np.pi / 180

        self.data[['acc_x', 'acc_y', 'acc_z']] += np.array([gx, gy, gz])

    def plot_main_data(self, sim_flag=False):
        self.plot_key(['mag_x', 'mag_y', 'mag_z', '||mag||'], color=['blue', 'orange', 'green', 'black'],
                      name="mag_sensor_mg", title="Raw Mag Sensor [mG]",
                      label=['x [mG]', 'y [mG]', 'z [mG]', '||mag||'], drawstyle=['steps-post'] * 4, marker=['.'] * 4)
        self.plot_key(['acc_x', 'acc_y', 'acc_z'], color=['blue', 'orange', 'green'],
                      name="gyro_sensor_rps", title="Raw Gyro Sensor [rad/s]",
                      label=['x [rps]', 'y [rps]', 'z [rps]'], drawstyle=['steps-post'] * 4, marker=['.'] * 4)
        self.plot_key(['sun3'], color=['blue'], label=['-x [mA]'], name='css3', title="Intensity -x [mA]",
                      drawstyle=['steps-post'], marker=['.'] * 3)
        self.plot_key(['sun2'], color=['orange'], label=['-y [mA]'], name='css2', title="Intensity -y [mA]",
                      drawstyle=['steps-post'], marker=['.'])
        self.plot_key(['sun4'], color=['green'], label=['-z [mA]'], name='css4', title="Intensity -z [mA]",
                      drawstyle=['steps-post'], marker=['.'])
        if sim_flag:
            self.plot_key(['mag_x_t', 'mag_y_t', 'mag_z_t', '||mag_t||'], color=['blue', 'orange', 'green', 'black'],
                          name="true_mag_mg_sim", title="True Mag Sensor [mG]",
                          label=['x [mG]', 'y [mG]', 'z [mG]', '||mag||'], marker=['.'] * 4)
            self.plot_key(['w_x', 'w_y', 'w_z'], color=['blue', 'orange', 'green'],
                          name="true_omega_rps_sim", title="True Angular velocity [rad/s]",
                          label=['x [rps]', 'y [rps]', 'z [rps]'], marker=['.'] * 4)
            self.plot_key(['bias_x', 'bias_y', 'bias_z'], color=['blue', 'orange', 'green'],
                          name="true_bias_rps_sim", title="True bias - Angular velocity [rad/s]",
                          label=['x [rps]', 'y [rps]', 'z [rps]'], marker=['.'] * 4)
            self.plot_key(['q_i2b_x', 'q_i2b_y', 'q_i2b_z', 'q_i2b_r'], color=['blue', 'orange', 'green', 'black'],
                          name="quaternion_i2b_sim", title="Quaternion i2b [-]",
                          label=['qx', 'qy', 'qz', 'qs'], marker=['.'] * 4)
            self.plot_key(['sun3', 'sun3_t'], color=['blue', 'red'], label=['-x [mA]', '-x True [mA]'], name='css3_sim',
                          title="True Intensity -x [mA]", marker=['.'] * 4)
            self.plot_key(['sun2', 'sun2_t'], color=['orange', 'red'], label=['-y [mA]', '-y True [mA]'], name='css2_sim',
                          title="True Intensity -y [mA]", marker=['.'] * 4)
            self.plot_key(['sun4', 'sun4_t'], color=['green', 'red'], label=['-z [mA]', '-z True [mA]'], name='css4_sim',
                          title="True Intensity -z [mA]", marker=['.'] * 4)
            self.plot_key(['is_dark'], color=['black'], label=['dark flag'], name='is_dark_sim',
                          title="Dark zone", marker=['.'] * 1)

        plt.show()


    def plot_mag_error(self, channels, sub_name):
        error_mag_ts = np.linalg.norm(channels['mag_i'], axis=1) - np.linalg.norm(
            self.data[['mag_x', 'mag_y', 'mag_z']],
            axis=1)

        mse_ts = mean_squared_error(np.linalg.norm(channels['mag_i'], axis=1),
                                    np.linalg.norm(self.data[['mag_x', 'mag_y', 'mag_z']], axis=1))
        fig = plt.figure()
        plt.title(f"Magnitude Error {sub_name} TWO STEP Calibration")
        plt.ylabel('Error [mG]')
        plt.plot(self.data['mjd'], error_mag_ts, label='RMSE: {:2f} [mG]'.format(np.sqrt(mse_ts)),
                 drawstyle='steps-post', marker='.', alpha=0.3)
        plt.legend()
        plt.xlabel("Modified Julian Date")
        plt.xticks(rotation=15)
        plt.ticklabel_format(useOffset=False)
        plt.tight_layout()
        plt.grid()
        fig.savefig(self.folder_path + f"results/two_step_mag_{sub_name}" + '.jpg')
        plt.close(fig)

    def plot_key(self, to_plot: list, name='', title: str = None, show: bool = False, **kwargs):
        for key, value in kwargs.items():
            print("%s == %s" % (key, value))
        sample_n = self.MAX_SAMPLES if self.MAX_SAMPLES is not None else len(self.data['mjd'])
        fig = plt.figure()
        plt.grid()
        plt.title(title) if title is not None else None
        for i, elem in enumerate(to_plot):
            dict_temp = {}
            for key, value in kwargs.items():
                dict_temp[key] = value[i]
            plt.plot(self.data['mjd'][:sample_n], self.data[elem][:sample_n], **dict_temp, alpha=0.3)
        plt.xlabel('Modified Julian Date')
        plt.legend()
        plt.xticks(rotation=15)
        plt.ticklabel_format(useOffset=False)
        plt.tight_layout()
        fig.savefig(self.folder_path + "results/" + name + ".jpg")
        plt.show() if show else plt.close(fig)

    def search_nearly_tle(self, sat_name: str):
        jd_init = self.data['jd'].values[0]
        jd_end = self.data['jd'].values[-1]
        if sat_name.lower() == "suchai-3":
            tle_file = open("data/sat000052191.txt")
        elif sat_name.lower() == "plantsat":
            tle_file = open("data/sat000052188.txt")
        elif sat_name.lower() == "suchai-2":
            tle_file = open("data/sat000052192.txt")
        else:
            print(" ERROR selecting Satellite name")
            exit()
        tle_info = tle_file.readlines()
        tle_1 = [tle_info_ for tle_info_ in tle_info if tle_info_[0] == '1']
        tle_2 = [tle_info_ for tle_info_ in tle_info if tle_info_[0] == '2']
        tle_epoch = [tle_info_.split(" ")[4] for tle_info_ in tle_1]
        tle_jd = np.array([tle_epoch_to_julian(tle_epoch_) for tle_epoch_ in tle_epoch])
        idx_ = np.argmin(np.abs(tle_jd - jd_init))
        line1 = tle_1[idx_]
        line2 = tle_2[idx_]
        print(line1, line2)
        return line1, line2

    def scale_mag(self, scale_):
        self.data[['mag_x', 'mag_y', 'mag_z']] *= scale_

    def calibrate_gyro(self):
        print("Calibrating Gyroscope ...")
        sigma_omega2 = self.std_rn_w ** 2 # 0.3 * np.deg2rad(1)
        R = np.eye(3) * sigma_omega2
        P = np.eye(6) * 1.0
        Q = np.eye(6) * self.std_rw_w
        ekf_omega = EKFOmega(self.sc_inertia, R, Q, P)
        ekf_omega.set_first_state(np.concatenate((self.data[['acc_x', 'acc_y', 'acc_z']].values[0], np.zeros(3))))
        ekf_omega_hist = {'new_omega': [self.data[['acc_x', 'acc_y', 'acc_z']].values[0]],
                          'bias': [np.zeros(3)]}
        for i, gyro_k in enumerate(self.data[['acc_x', 'acc_y', 'acc_z']].values[1:]):
            print(f" - Gyro Progress - {i} / {len(self.data['mjd'])}")
            ekf_omega.update(1.0, gyro_k)
            ekf_omega_hist['new_omega'].append(ekf_omega.state[:3])
            ekf_omega_hist['bias'].append(ekf_omega.state[3:])

        self.data[['acc_x', 'acc_y', 'acc_z']] -= ekf_omega.state[3:]
        ekf_omega.plot_cov(self.data['mjd'], self.folder_path + "results/")
        fig, axes = plt.subplots(2, 1, sharex=True)
        axes[0].plot(self.data['mjd'], ekf_omega_hist['new_omega'])
        axes[0].set_xlabel('Modified Julian Date')
        axes[0].set_ylabel('Calibrated Angular velocity')
        axes[0].grid(True)
        axes[1].plot(self.data['mjd'], ekf_omega_hist['bias'])
        axes[1].set_xlabel('Modified Julian Date')
        axes[1].set_ylabel('Bias')
        axes[1].grid(True)
        plt.xticks(rotation=15)
        plt.ticklabel_format(useOffset=False)
        plt.tight_layout()
        fig.savefig(self.folder_path + "results/" + "online_omega_ekf_calibration.jpg")
        plt.close()
        return ekf_omega.historical[-1]

    def calibrate_mag(self, scale: np.array = None, bias: np.array = None, mag_i: np.array = None,
                      by_file=False, force=False):
        if scale is not None and bias is not None:
            self.data[['mag_x', 'mag_y', 'mag_z']] = np.matmul(self.data[['mag_x', 'mag_y', 'mag_z']],
                                                               (np.eye(3) + scale)) - bias
        elif mag_i is not None:
            # if no exist file_name
            file_name = self.folder_path + "calibration_" + self.file_name.split('.')[0] + ".json"
            if os.path.exists(file_name) and not force:
                with open(self.folder_path + "calibration_" + self.file_name.split('.')[0] + ".json") as data_file:
                    data_loaded = json.load(data_file)
                d = np.asarray(data_loaded['D'])
                scale = np.diag(d[:3])
                scale[0, 1] = d[3]
                scale[1, 0] = d[3]
                scale[0, 2] = d[4]
                scale[2, 0] = d[4]
                scale[1, 2] = d[5]
                scale[2, 1] = d[5]

                bias = np.asarray(data_loaded['bias'])
                self.data[['mag_x', 'mag_y', 'mag_z']] = np.matmul(self.data[['mag_x', 'mag_y', 'mag_z']],
                                                                   (np.eye(3) + scale)) - bias
            else:
                # factor_ = np.mean(np.linalg.norm(mag_i, axis=1)) / np.mean(np.linalg.norm(self.data[['mag_x', 'mag_y', 'mag_z']].values, axis=1))
                x_non_sol, sig3_non, x_lin_sol, sig3_lin = two_step(
                    self.data[['mag_x', 'mag_y', 'mag_z']].values[:len(mag_i)], np.asarray(mag_i))

                data = {'D': list(x_non_sol[3:]), 'bias': list(x_non_sol[:3])}
                with open(file_name, 'w') as f:
                    json.dump(data, f)
                    f.close()
                print(x_non_sol, sig3_non, x_lin_sol, sig3_lin)
                bias = x_non_sol[:3]
                scale = get_full_D(x_non_sol[3:])
                self.data[['mag_x', 'mag_y', 'mag_z']] = np.matmul(self.data[['mag_x', 'mag_y', 'mag_z']],
                                                                   (np.eye(3) + scale)) - bias
        self.data['||mag||'] = np.linalg.norm(self.data[['mag_x', 'mag_y', 'mag_z']], axis=1)

    def set_window_time(self, start_str=None, stop_str=None, format_time=None, dt=1, sim_flag=False):
        if sim_flag:
            init = datetime.datetime.strptime(start_str, format_time)
            stop = datetime.datetime.strptime(stop_str, format_time)
            init = init.replace(tzinfo=datetime.timezone.utc).timestamp()
            stop = stop.replace(tzinfo=datetime.timezone.utc).timestamp()
            self.start_time = timestamp_to_julian(init)
            self.end_time = timestamp_to_julian(stop)
            self.data['timestamp'] = np.arange(init, stop + dt, dt)
            self.data['jd'] = [timestamp_to_julian(float(ts)) for ts in self.data['timestamp'].values]
            self.data['mjd'] = self.data['jd'] - _MJD_1858
            self.step = dt

        if start_str is None and stop_str is None and format_time is None:
            self.plot_dendrogram_time()
            n_c = int(input("Set the number of cluster to find window time: "))
            model = self.get_windows_time(n_c=n_c)
            selected_cluster = int(input("Set the number of cluster to set window time: "))
            temp = np.argwhere(model.labels_ == selected_cluster)
            self.data = self.data.iloc[list(temp.T)[0]]
            self.start_time = self.data['jd'].values[0]
            self.end_time = self.data['jd'].values[-1]
        else:
            init = datetime.datetime.strptime(start_str, format_time)
            stop = datetime.datetime.strptime(stop_str, format_time)
            init = init.replace(tzinfo=datetime.timezone.utc).timestamp()
            stop = stop.replace(tzinfo=datetime.timezone.utc).timestamp()
            temp = np.argwhere((init <= self.data['timestamp']) & (self.data['timestamp'] <= stop))
            self.data = self.data.iloc[list(temp.T)[0]]
            self.start_time = self.data['jd'].values[0]
            self.end_time = timestamp_to_julian(stop)
        self.create_datetime_from_timestamp()

    def estimate_inertia_matrix(self, guess=None):
        if guess is None:
            guess = np.array([0.03, 0.03, 0.03, 0, 0, 0])
        omega = self.data[['acc_x', 'acc_y', 'acc_z']].values
        dt_ = np.diff(self.data['timestamp'].values)
        acc = np.diff(omega, axis=0) / np.atleast_2d(dt_).T

        def f_c(inertia_):
            inertia_ = get_full_D(inertia_)
            error_ = [inertia_.reshape(3, 3) @ a_ + np.cross(w_, inertia_.reshape(3, 3) @ a_) for w_, a_ in zip(omega[:-1], acc)]
            vec_3 = np.mean(error_, axis=0)
            return np.array([*vec_3, *vec_3])

        sol = fsolve(f_c, guess)
        return sol

    def plot_dendrogram_time(self, timestamp_distance: float = 3600):
        """

        :param timestamp_distance: minimum distance from time to find cluster
        :return: number of cluster
        """

        # setting distance_threshold=0 ensures we compute the full tree.
        model = AgglomerativeClustering(distance_threshold=timestamp_distance, n_clusters=None)

        model = model.fit(self.data['timestamp'].values.reshape(-1, 1))
        plt.title("Hierarchical Clustering Dendrogram")
        # plot the top three levels of the dendrogram
        plot_dendrogram(model, truncate_mode="level", p=5)
        plt.xlabel("Number of points in node (or index of point if no parenthesis).")
        plt.show()

    def get_windows_time(self, n_c = 2):
        model = KMeans(n_clusters=n_c)
        model.fit(self.data['timestamp'].values.reshape(-1, 1))
        for i in range(n_c):
            temp = self.data['DateTime'].values[np.argwhere(model.labels_ == i)]
            print(f"Cluster {i}: Start: {temp[0]} - Stop: {temp[-1]}")
        return model

    def plot_video_data(self, data_video, name_, VIDEO_FOLDER):
        self.data_video[name_] = data_video

        fig = plt.figure()
        plt.title("Angular position - rotation (3, 1, 2) - BF @ LVLH")
        plt.ylabel("Angle position - rotation [deg]")
        plt.xlabel("MJD")
        plt.plot(data_video['MJD'], np.rad2deg(data_video['pitch']), '.', label=r'$\psi$')
        plt.plot(data_video['MJD'], np.rad2deg(data_video['roll']), '.', label=r'$\theta$')
        plt.legend()
        plt.grid()
        fig.savefig(VIDEO_FOLDER + "results/" + "pitch_roll_lvlh.png")

        pitch_ = np.unwrap(data_video['pitch'])
        roll_ = np.unwrap(data_video['roll'])

        d_pitch = np.gradient(pitch_, data_video['MJD'])
        d_roll = np.gradient(roll_, data_video['MJD'])

        fig = plt.figure()
        plt.title("Angular velocity - rotation (3, 1, 2) - BF @ LVLH")
        plt.ylabel("Angle velocity [deg/s]")
        plt.xlabel("MJD")
        plt.plot(data_video['MJD'], d_pitch, '.', label=r'$\dot{\psi}$')
        plt.plot(data_video['MJD'], d_roll, '.', label=r'$\dot{\theta}$')
        plt.legend()
        plt.grid()
        fig.savefig(VIDEO_FOLDER + "results/" + "dot_pitch_roll_lvlh.png")
        plt.close("all")

    def plot_windows(self, VIDEO_FOLDER):
        fig, ax = plt.subplots()
        # add rectangle to plot
        x0 = min(self.data["mjd"])
        # dt_imu = max(self.data["mjd"]) - min(self.data["mjd"])
        # print(max(self.data["mjd"]) - min(self.data["mjd"]))
        ax.hlines(0.1, x0, max(self.data["mjd"]), color='blue', lw=3, label='IMU')
        color = ['orange', 'red', 'green']
        i = 0
        for key, item in self.data_video.items():
            dt_video =  max(item["MJD"]) - min(item["MJD"])
            print((min(item["MJD"]) - x0) * 86400/ 60)
            ax.scatter(min(item["MJD"]), 0.5, color=color[i], label=key, marker='x', s=100)
            # ax.hlines(0.5, min(item["MJD"]), dt_video + min(item["MJD"]), color=color[i], lw=10, label=key)
            i += 1
        plt.grid()
        plt.ylim([0, 1])
        plt.xlabel("MJD")
        plt.legend()
        fig.savefig(VIDEO_FOLDER + "results/" + "wind_data.png")

    def create_sim_data(self, channels):
        q_i2b = [Quaternions(q_) for q_ in channels['q_i2b']]
        w_b = channels['w_b']
        sun_sc_i = channels['sun_sc_i']
        mag_i = channels['mag_i']
        # MAG MODEL
        mag_b_true = np.array([q_.frame_conv(m_) for q_, m_ in zip(q_i2b, mag_i)])
        b_true_ = np.array([112, -275, -290])
        d_true = np.array([[1.5, 0.00, 0.0], [0.00, 1.1, 0.0], [0.0, 0.0, 2.5]]) * 0.01
        eta_noise = np.random.normal(0, self.std_rn_mag, size=(len(mag_b_true), 3))
        mag_sensor_ = [np.linalg.inv(np.eye(3) + d_true) @ (mag_true_ + eta_ + b_true_)
                       for eta_, mag_true_ in zip(eta_noise, mag_b_true)]
        mag_sensor_ = np.array(mag_sensor_)
        data_ = {}
        data_['mag_x'] = mag_sensor_[:, 0]
        data_['mag_y'] = mag_sensor_[:, 1]
        data_['mag_z'] = mag_sensor_[:, 2]
        data_['||mag||'] = np.linalg.norm(mag_sensor_, axis=1)
        data_['mag_x_t'] = mag_b_true[:, 0]
        data_['mag_y_t'] = mag_b_true[:, 1]
        data_['mag_z_t'] = mag_b_true[:, 2]
        data_['||mag_t||'] = np.linalg.norm(mag_b_true, axis=1)

        # SUN MODEL
        sun_sc_b = np.array([q_.frame_conv(m_) for q_, m_ in zip(q_i2b, sun_sc_i)])
        unit_sun_sc_b = sun_sc_b / np.linalg.norm(sun_sc_b, axis=1).reshape(-1, 1)
        minus_x = np.array([-1, 0, 0])
        minus_y = np.array([0, -1, 0])
        minus_z = np.array([0, 0, -1])

        cos_theta_x = np.array([minus_x @ unit_sun_sc_b_ for unit_sun_sc_b_ in unit_sun_sc_b])
        cos_theta_y = np.array([minus_y @ unit_sun_sc_b_ for unit_sun_sc_b_ in unit_sun_sc_b])
        cos_theta_z = np.array([minus_z @ unit_sun_sc_b_ for unit_sun_sc_b_ in unit_sun_sc_b])

        cos_theta_x_ns = np.cos(np.arccos(cos_theta_x) + np.random.normal(0, self.I_nr_std_cos, len(cos_theta_z)))
        cos_theta_y_ns = np.cos(np.arccos(cos_theta_y) + np.random.normal(0, self.I_nr_std_cos, len(cos_theta_z)))
        cos_theta_z_ns = np.cos(np.arccos(cos_theta_z) + np.random.normal(0, self.I_nr_std_cos, len(cos_theta_z)))

        cos_theta_x[np.array(channels['is_dark'], dtype=bool)] = 0
        cos_theta_y[np.array(channels['is_dark'], dtype=bool)] = 0
        cos_theta_z[np.array(channels['is_dark'], dtype=bool)] = 0
        cos_theta_x_ns[np.array(channels['is_dark'], dtype=bool)] = 0
        cos_theta_y_ns[np.array(channels['is_dark'], dtype=bool)] = 0
        cos_theta_z_ns[np.array(channels['is_dark'], dtype=bool)] = 0

        # cos_theta_x[cos_theta_x < 0] = 0
        # cos_theta_y[cos_theta_y < 0] = 0
        # cos_theta_z[cos_theta_z < 0] = 0
        cos_theta_x_ns[cos_theta_x_ns < 0] = 0
        cos_theta_y_ns[cos_theta_y_ns < 0] = 0
        cos_theta_z_ns[cos_theta_z_ns < 0] = 0

        i_minus_x = self.I_max * cos_theta_x_ns
        i_minus_y = self.I_max * cos_theta_y_ns
        i_minus_z = self.I_max * cos_theta_z_ns
        data_['sun3'] = np.abs(i_minus_x)
        data_['sun2'] = np.abs(i_minus_y)
        data_['sun4'] = np.abs(i_minus_z)
        data_['sun3_t'] = self.I_max * cos_theta_x
        data_['sun2_t'] = self.I_max * cos_theta_y
        data_['sun4_t'] = self.I_max * cos_theta_z
        # GYRO MODEL
        dt = self.step
        b_true_gyro = -np.array([0.07, 0.01, -0.04])
        d_true_gyro = np.array([[1.5, 0.00, 0.0], [0.00, 1.1, 0.0], [0.0, 0.0, 2.5]]) * 0.001
        gyro_matrix_noise = 0.5 * (self.std_rn_w ** 2 / dt + self.std_rw_w ** 2 * dt / 12) ** 0.5
        bias_true = [b_true_gyro]
        omega_measure = [w_b[0] + b_true_gyro + gyro_matrix_noise * np.random.normal(0, np.array( [1,  1, 1]))]
        for i in range(1, len(w_b)):
            current_bias_true = bias_true[-1] + self.std_rw_w * dt ** 0.5 * np.random.normal(0, np.array([1, 1, 1]))
            diff_bias = 0.5 * (current_bias_true + bias_true[-1])
            current_omega = w_b[i] + diff_bias + gyro_matrix_noise * np.random.normal(0, np.array( [1,  1, 1]))
            bias_true.append(current_bias_true)
            omega_measure.append(current_omega)
        omega_measure = np.array(omega_measure)
        bias_true = np.array(bias_true)
        data_['acc_x'] = omega_measure[:, 0] * np.rad2deg(1)
        data_['acc_y'] = omega_measure[:, 1] * np.rad2deg(1)
        data_['acc_z'] = omega_measure[:, 2] * np.rad2deg(1)
        data_['bias_x'] = bias_true[:, 0]
        data_['bias_y'] = bias_true[:, 1]
        data_['bias_z'] = bias_true[:, 2]
        data_['q_i2b_x'] = channels['q_i2b'][:, 0]
        data_['q_i2b_y'] = channels['q_i2b'][:, 1]
        data_['q_i2b_z'] = channels['q_i2b'][:, 2]
        data_['q_i2b_r'] = channels['q_i2b'][:, 3]
        data_['w_x'] = w_b[:, 0]
        data_['w_y'] = w_b[:, 1]
        data_['w_z'] = w_b[:, 2]
        data_['is_dark'] = channels['is_dark']
        data_['timestamp'] = self.data['timestamp']
        data_['jd'] = self.data['jd']
        data_['mjd'] = self.data['mjd']

        self.data = pd.DataFrame(data_)
        self.data.reset_index()
        self.data.to_excel(self.folder_path + self.file_name)
        self.data['acc_x'] *= np.deg2rad(1)
        self.data['acc_y'] *= np.deg2rad(1)
        self.data['acc_z'] *= np.deg2rad(1)

def module_exception_log(method, *args, **kwargs):
    """ Catch and log exceptions to do not crash the GUI"""
    @functools.wraps(method, *args, **kwargs)
    def wrapper(self, *args, **kwargs):
        try:
            method(self, *args, **kwargs)
        except Exception as e:
            print("Error: {}".format(str(e)))
    return wrapper


def get_full_D(d_vector):
    d_ = np.zeros((3, 3))
    d_[0, 0] = d_vector[0]
    d_[1, 1] = d_vector[1]
    d_[2, 2] = d_vector[2]

    d_[0, 1] = d_vector[3]
    d_[0, 2] = d_vector[4]
    d_[1, 0] = d_vector[3]
    d_[2, 0] = d_vector[4]

    d_[1, 2] = d_vector[5]
    d_[2, 1] = d_vector[5]
    return d_


def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)


if __name__ == '__main__':
    PROJECT_FOLDER = '../data/20230904/'
    OBC_DATA = 'gyros-S3-040923.xlsx'
    TIME_FORMAT = "%Y/%m/%d %H:%M:%S"
    sensors = RealData(PROJECT_FOLDER, OBC_DATA)
    sensors.create_datetime_from_timestamp(TIME_FORMAT)
    sensors.plot_key(['mag_x', 'mag_y', 'mag_z'], color=('blue', 'orange', 'green'),
                     label=('x [mG]', 'y [mG]', 'z [mG]'))
    sensors.plot_key(['acc_x', 'acc_y', 'acc_z'], color=('blue', 'orange', 'green'),
                     label=('x [deg/s]', 'y [deg/s]', 'z [deg/s]'))
    sensors.plot_key(['mag_x'], color=['blue'], label=['x [mG]'])
    sensors.plot_key(['mag_y'], color=['orange'], label=['y [mG]'])
    sensors.plot_key(['mag_z'], color=['green'], label=['z [mG]'])
    sensors.plot_key(['acc_x'], color=['blue'], label=['x [deg/s]'])
    sensors.plot_key(['acc_y'], color=['orange'], label=['y [deg/s]'])
    sensors.plot_key(['acc_z'], color=['green'], label=['z [deg/s]'])
    sensors.plot_key(['sun3'], color=['blue'], label=['x [mA]'])
    sensors.plot_key(['sun2'], color=['orange'], label=['y [mA]'])
    sensors.plot_key(['sun4'], color=['green'], label=['z [mA]'])
    sensors.plot_key(['sun3', 'sun2', 'sun4'], show=True, color=('blue', 'orange', 'green'),
                     label=('-x [mA]', '-y [mA]', '-z [mA]'))



