"""
Created by Elias Obreque
Date: 04-09-2023
email: els.obrq@gmail.com
"""

import os
import json
import pickle

import pandas as pd
import numpy as np
import datetime
import functools
import cv2
from future.backports.http.cookiejar import debug
from tqdm import tqdm
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from matplotlib.colorbar import make_axes_gridspec
from scipy.cluster.hierarchy import dendrogram
from scipy.optimize import fsolve, minimize
from sklearn.metrics import mean_squared_error
from sklearn.cluster import KMeans, AgglomerativeClustering
from scipy.spatial.transform import Rotation, Slerp

from src.kalman_filter.ekf_omega import EKFOmega
from src.dynamics.quaternion import Quaternions
from tools.mathtools import *
from tools.two_step_mag_calibration import two_step
from tools.camera_sensor import CamSensor


ts2022 = 1640995200 / 86400  # day
jd2022 = 2459580.50000
_MJD_1858 = 2400000.5
SIM_VIDEO_DURATION = 20 # sec
ROT_CAM2BODY = Rotation.from_euler('zx', [180, -90], degrees=True).inv().as_matrix()
fourcc = cv2.VideoWriter_fourcc(*'mp4v')


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

        axes[0].plot(self.data['mag_x'], self.data['mag_z'], '.',  alpha=0.7)
        axes[0].plot(center_mean[0], center_mean[2], 'r*')
        # axes[0].plot(contour_center_xz[0], contour_center_xz[1], 'y*')
        # axes[0].plot(hull_points_xz[:, 0], hull_points_xz[:, 1], 'r-', label='Contorno exterior')
        axes[0].grid()
        axes[0].set_ylabel('Mag z [mG]')
        axes[0].set_xlabel('Mag x [mG]')
        axes[0].set_box_aspect(1)

        axes[1].plot(self.data['mag_y'], self.data['mag_z'], '.',  alpha=0.7)
        axes[1].plot(center_mean[1], center_mean[2], 'r*')
        axes[1].grid()
        # axes[1].plot(contour_center_yz[0], contour_center_yz[1], 'y*')
        # axes[1].plot(hull_points_yz[:, 0], hull_points_yz[:, 1], 'r-', label='Contorno exterior')
        axes[1].set_ylabel('Mag y [mG]')
        axes[1].set_xlabel('Mag z [mG]')
        axes[1].set_box_aspect(1)
        axes[2].plot(self.data['mag_x'], self.data['mag_y'], '.', alpha=0.7)
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
        if "acc_x" in self.data.columns:
            omega = self.data[['acc_x', 'acc_y', 'acc_z']].values
            h_ = angular_momentum(self.sc_inertia, omega)
            self.data['h_x'] = h_[:, 0]
            self.data['h_y'] = h_[:, 1]
            self.data['h_z'] = h_[:, 2]
            self.data['h_norm'] = np.linalg.norm(h_, axis=1)
            self.data['energy'] = kinetic_energy(self.sc_inertia, omega)

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
                      name="mag_sensor_mg", title="Raw Magnetic Sensor", y_name="Magnetic field [mG]",
                      label=['x', 'y', 'z', r'$||\cdot||$'], drawstyle=['steps-post'] * 4, marker=['.'] * 4, unit="[mG]")

        self.plot_key(['acc_x', 'acc_y', 'acc_z'], color=['blue', 'orange', 'green'], y_name="Angular velocity [rad/s]",
                      name="gyro_sensor_rps", title="Raw Gyro Sensor",
                      label=['x', 'y', 'z'], drawstyle=['steps-post'] * 4, marker=['.'] * 4, unit="[rad/s]")

        self.plot_key(['sun3', 'sun2', 'sun4'], color=['blue', 'orange', 'green'], y_name="Current [mA]",
                      name="css_current", title="CSS Intensity",
                      label=['-x', '-y', '-z'], drawstyle=['steps-post'] * 4, marker=['.'] * 4, unit="[mA]")

        self.plot_key(['sun3'], color=['blue'], label=['-x'], name='css3', title="CSS Face -x",
                      drawstyle=['steps-post'], marker=['.'] * 3, y_name="Intensity [mA]", unit="[mA]")

        self.plot_key(['sun2'], color=['orange'], label=['-y'], name='css2', title="CSS Face -y",
                      drawstyle=['steps-post'], marker=['.'], y_name="Intensity [mA]", unit="[mA]")

        self.plot_key(['sun4'], color=['green'], label=['-z'], name='css4', title="CSS Face -z",
                      drawstyle=['steps-post'], marker=['.'], y_name="Intensity [mA]", unit="[mA]")

        self.plot_key(['energy'], color=['blue'], label=[None], name='energy_b', drawstyle=['steps-post'],
                      marker=['.'], y_name='Energy [J]', unit='[J]', title="Rotational Energy")

        self.plot_key(['h_x', 'h_y', 'h_z', 'h_norm'], color=['blue', 'orange', 'green', 'black'],
                      y_name="Angular momentum [kg m^2/s]",
                      name="angular_momentum", title="Angular Momentum",
                      label=['x', 'y', 'z', r'$||\cdot||$'], drawstyle=['steps-post'] * 4, marker=['.'] * 4, unit="[kg m^2/s]")
        if sim_flag:
            self.plot_key(['mag_x_t', 'mag_y_t', 'mag_z_t', '||mag_t||'], color=['blue', 'orange', 'green', 'black'],
                          name="true_mag_mg_sim", title="True Magnetic Sensor", y_name="Magnetic field [mG]",
                          label=['x', 'y', 'z', r'$||\cdot||$'], marker=['.'] * 4, unit="[mG]")

            self.plot_key(['w_x', 'w_y', 'w_z'], color=['blue', 'orange', 'green'],
                          name="true_omega_rps_sim", title="True Angular velocity", y_name="Angular velocity [rad/s]",
                          label=['x', 'y', 'z'], marker=['.'] * 4, unit="[rad/s]")

            self.plot_key(['bias_x', 'bias_y', 'bias_z'], color=['blue', 'orange', 'green'],
                          name="true_bias_rps_sim", title="True Gyro bias", y_name="Bias [rad/s]",
                          label=['x', 'y', 'z'], marker=['.'] * 4, unit="[rad/s]")

            self.plot_key(['q_i2b_x', 'q_i2b_y', 'q_i2b_z', 'q_i2b_r'], color=['blue', 'orange', 'green', 'black'],
                          name="quaternion_i2b_sim", title="Quaternion from Inertial to Body Frame", y_name="Quaternion values [-]",
                          label=['qx', 'qy', 'qz', 'qs'], marker=['.'] * 4, unit="[-]")

            self.plot_key(['sun3', 'sun3_t'], color=['blue', 'red'], label=['-x', '-x True'], name='css3_sim',
                          title="True CSS Face -x", marker=['.'] * 4, y_name="Intensity [mA]", unit="[mA]")

            self.plot_key(['sun2', 'sun2_t'], color=['orange', 'red'], label=['-y', '-y True'], name='css2_sim',
                          title="True CSS Face -y", marker=['.'] * 4, y_name="Intensity [mA]", unit="[mA]")

            self.plot_key(['sun4', 'sun4_t'], color=['green', 'red'], label=['-z', '-z True'], name='css4_sim',
                          title="True CSS Face -z", marker=['.'] * 4, y_name="Intensity [mA]", unit="[mA]")
        plt.close()

    def plot_mag_error(self, channels_mag, sub_name):
        error_mag_ts = np.linalg.norm(channels_mag, axis=1) - np.linalg.norm(
            self.data[['mag_x', 'mag_y', 'mag_z']],
            axis=1)

        mse_ts = mean_squared_error(np.linalg.norm(channels_mag, axis=1),
                                    np.linalg.norm(self.data[['mag_x', 'mag_y', 'mag_z']], axis=1))
        fig = plt.figure()
        plt.title(f"Magnitude Error {sub_name} TWO STEP Calibration")
        plt.ylabel('Error [mG]')
        plt.plot(self.data['mjd'], error_mag_ts, label='RMSE: {:2f} [mG]'.format(np.sqrt(mse_ts)),
                 drawstyle='steps-post', marker='.', alpha=0.7)
        plt.legend()
        plt.xlabel("Modified Julian Date")
        plt.xticks(rotation=15)
        plt.ticklabel_format(useOffset=False)
        plt.tight_layout()
        plt.grid()
        fig.savefig(self.folder_path + f"results/two_step_mag_{sub_name}" + '.jpg')
        plt.close(fig)

    def plot_key(self, to_plot: list, y_name='', name='', title: str = None, show: bool = False, unit: str="", **kwargs):
        # for key, value in kwargs.items():
        #     print("%s == %s" % (key, value))
        sample_n = self.MAX_SAMPLES if self.MAX_SAMPLES is not None else len(self.data['mjd'])
        fig = plt.figure(figsize=(8.5, 5))
        plt.grid()
        plt.title(title) if title is not None else None
        for i, elem in enumerate(to_plot):
            dict_temp = {}
            for key, value in kwargs.items():
                if value is not None:
                    dict_temp[key] = value[i]
            plt.plot(self.data['mjd'][:sample_n], self.data[elem][:sample_n], **dict_temp, alpha=0.7)
        plt.xlabel('Modified Julian Date')
        plt.ylabel(y_name)
        plt.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
        plt.xticks(rotation=15)
        plt.ticklabel_format(useOffset=False)

        handles, labels = plt.gca().get_legend_handles_labels()
        plt.tight_layout()
        fig.legend(handles, labels, loc='center left', bbox_to_anchor=(0.85, 0.5), frameon=True)
        plt.subplots_adjust(right=0.84)
        fig.savefig(self.folder_path + "results/" + name + ".png", dpi=300)
        plt.show() if show else plt.close(fig)

        if len(to_plot) > 1:
            fig_ind, axes_ind = plt.subplots(len(to_plot), 1, figsize=(8.5, 5), sharex=True)
            fig_ind.suptitle(title + " " + unit) if title is not None else None
            for i, elem in enumerate(to_plot):
                axes_ind[i].grid()
                dict_temp = {}
                for key, value in kwargs.items():
                    dict_temp[key] = value[i]
                axes_ind[i].plot(self.data['mjd'][:sample_n], self.data[elem][:sample_n], **dict_temp, alpha=0.7)
                axes_ind[i].set_ylabel(labels[i])

            axes_ind[-1].set_xlabel('Modified Julian Date')
            plt.xticks(rotation=15)
            plt.ticklabel_format(useOffset=False)
            plt.tight_layout()
            fig_ind.savefig(self.folder_path + "results/ind_" + name + ".png", dpi=300)
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

    def estimate_inertia_matrix_old(self, guess=None):
        if guess is None:
            guess = np.array([0.03, 0.03, 0.03, 0, 0, 0])
        omega = self.data[['acc_x', 'acc_y', 'acc_z']].values
        # dt_ = np.diff(self.data['timestamp'].values)
        acc = np.gradient(omega, self.data['timestamp'].values, axis=0)

        def f_c(inertia_):
            inertia_ = get_full_D(inertia_)
            error_ = [inertia_.reshape(3, 3) @ a_ + np.cross(w_, inertia_.reshape(3, 3) @ a_) for w_, a_ in zip(omega[:-1], acc)]
            vec_3 = np.mean(error_, axis=0)
            return np.array([*vec_3, *vec_3])

        sol = fsolve(f_c, guess)
        return sol

    def estimate_inertia_matrix(self, guess=None):
        if guess is None:
            guess = np.array([0.03, 0.03, 0.03, 0, 0, 0])

        omega = self.data[['acc_x', 'acc_y', 'acc_z']].values
        acc = np.gradient(omega, self.data['timestamp'].values, axis=0)

        def loss(I_params, omega):
            I = get_full_D(I_params)

            T = kinetic_energy(I, omega)
            H = angular_momentum(I, omega)

            # error = variación de energía y módulo de momento angular
            dT = np.var(T) / np.sum(I)
            dH = np.var(np.linalg.norm(H, axis=1)) / np.sum(I)
            print("VAR: ", dT, dH)
            return dT + dH - np.sum(I) * 1e-6

        res = minimize(loss, guess, args=(omega,), bounds=[(1e-9, 0.04)] * 6)

        I_est_energy = res.x
        I = get_full_D(I_est_energy)
        T = kinetic_energy(I, omega)
        H = angular_momentum(I, omega)
        h_norm = np.linalg.norm(H, axis=1)
        return I_est_energy

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

        d_pitch = np.gradient(pitch_, data_video['MJD'] * 86400)
        d_roll = np.gradient(roll_, data_video['MJD'] * 86400)

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

        if os.path.exists(self.folder_path + "/" + f"{name_}_synth_vec.pkl"):
            with open(self.folder_path + "/" + f"{name_}_synth_vec.pkl", 'rb') as fp:
                channels_video = pickle.load(fp)

            fig = plt.figure()
            plt.title("Earth center diection - BF")
            plt.ylabel("Unit vector")
            plt.xlabel("MJD")
            plt.plot(np.array(data_video['MJD'] - data_video['MJD'][0]) * 86400, data_video[['e_b_x', 'e_b_y', 'e_b_z']], '.', label=r'Determination')
            plt.plot((channels_video['MJD'] - channels_video['MJD'][0]) * 86400, channels_video['earth'], label='True')
            plt.legend()
            plt.grid()
            fig.savefig(VIDEO_FOLDER + "results/" + "vec_earth_b_comp.png")
            plt.show()

            fig = plt.figure()
            plt.title("Sun center direction - BF")
            plt.ylabel("Unit vector")
            plt.xlabel("MJD")
            plt.plot(np.array(data_video['MJD'] - data_video['MJD'][0]) * 86400,
                     data_video[['s_b_x', 's_b_y', 's_b_z']], '.', label=r'Determination')
            plt.plot((channels_video['MJD'] - channels_video['MJD'][0]) * 86400, channels_video['sun'], label='True')
            plt.legend()
            plt.grid()
            fig.savefig(VIDEO_FOLDER + "results/" + "vec_sun_b_comp.png")
            plt.show()

    def plot_windows(self, VIDEO_FOLDER):
        fig, ax = plt.subplots()
        # add rectangle to plot
        x0 = min(self.data["mjd"])
        # dt_imu = max(self.data["mjd"]) - min(self.data["mjd"])
        # print(max(self.data["mjd"]) - min(self.data["mjd"]))
        ax.hlines(0.1, x0, max(self.data["mjd"]), color='blue', lw=3, label='IMU', linestyles ="dotted")
        color = ['orange', 'red', 'green']
        i = 0
        for key, item in self.data_video.items():
            dt_video =  max(item["MJD"]) - min(item["MJD"])
            print((min(item["MJD"]) - x0) * 86400/ 60)
            plt.vlines(min(item["MJD"]), ymin=0.0, ymax=0.2, color=color[i], label=key, lw=1)
            # ax.hlines(0.5, min(item["MJD"]), dt_video + min(item["MJD"]), color=color[i], lw=10, label=key)
            i += 1
        plt.grid()
        plt.ylim([0, 0.2])
        plt.xlabel("MJD")
        plt.legend()
        fig.savefig(VIDEO_FOLDER + "results/" + "wind_data.png")
        plt.close()

    def plot_gt_full_videos(self, folder_name, channels, channels_video_list):
        fig, ax = plt.subplots()
        lon = channels['lonlat'][:, 0]
        lat = channels['lonlat'][:, 1]

        fig = plt.figure(figsize=(12, 8))
        ax = plt.axes(projection=ccrs.PlateCarree())
        ax.stock_img()
        gl = ax.gridlines(draw_labels=True, linewidth=1, color='gray', linestyle='--')
        gl.top_labels = False
        gl.right_labels = False
        gl.xformatter = LongitudeFormatter()
        gl.yformatter = LatitudeFormatter()

        ax.add_feature(cfeature.COASTLINE, linewidth=1)
        ax.add_feature(cfeature.BORDERS, linewidth=1)
        # ax.plot(lon, lat, linestyle='-', color='r', transform=ccrs.Geodetic())
        plt.title('Groundtrack', pad=20, fontsize=12, color='black')
        ax.plot(lon, lat, color='blue', transform=ccrs.Geodetic())

        for key, item in channels_video_list.items():
            lon = item["lonlat"][:, 0]
            lat = item["lonlat"][:, 1]
            ax.scatter(lon, lat, color='red', s=10, transform=ccrs.Geodetic())

        plt.tight_layout()
        fig.savefig(folder_name, dpi=300)
        plt.close(fig)


    def create_sim_data(self, channels):
        q_i2b = [Quaternions(q_) for q_ in channels['q_i2b']]
        # test
        # q_i2b = [Quaternions(np.array([0, 0, 0, 1.0])) for q_ in channels['q_i2b']]
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
        h_ = angular_momentum(self.sc_inertia, w_b)
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
        data_['energy'] = kinetic_energy(self.sc_inertia, w_b)
        data_['h_x'] = h_[:, 0]
        data_['h_y'] = h_[:, 1]
        data_['h_z'] = h_[:, 2]
        data_['h_norm'] = np.linalg.norm(h_, axis=1)

        self.data = pd.DataFrame(data_)
        self.data.reset_index()
        self.data.to_excel(self.folder_path + self.file_name)
        self.data['acc_x'] *= np.deg2rad(1)
        self.data['acc_y'] *= np.deg2rad(1)
        self.data['acc_z'] *= np.deg2rad(1)

    def create_sim_video(self, channels: dict, video_names: list[str], video_last_frame_date: list[str], fps: int) -> None:
        # camera and frames
        timestamp_list = [datetime.datetime.strptime(video_last_frame, '%Y/%m/%d %H:%M:%S').replace(tzinfo=datetime.timezone.utc).timestamp() for video_last_frame in video_last_frame_date]
        tim_sec_list = timestamp_to_julian(np.array(timestamp_list)) - _MJD_1858

        cameras = [CamSensor(r_c2b=ROT_CAM2BODY, add_filter=False, debug=False, target_resolution=(640, 480)) for _ in range(len(tim_sec_list))]

        earth_pos = - np.array(channels['sat_pos_i'])
        earth_vel = np.array(channels['sat_vel_i'])
        sun_pos_sc = np.array(channels['sun_sc_i'])

        q_i2b = self.data[['q_i2b_x', 'q_i2b_y', 'q_i2b_z', 'q_i2b_r']]
        time_mjd = self.data['mjd']

        # interpolation
        qs = np.array(q_i2b)
        mjd = time_mjd.to_numpy()

        t_sec = (mjd - mjd[0]) * 86400.0  # seconds since first sample
        key_rots = Rotation.from_quat(qs)
        slerp = Slerp(t_sec, key_rots)  # one object covers the entire span

        dt = 1.0 / fps
        tim_sec_list -= mjd[0]
        tim_sec_list *= 86400
        tim_sec_list -= SIM_VIDEO_DURATION

        for i, cam in enumerate(cameras):
            if not os.path.exists(self.folder_path + f"{video_names[i].split('.')[0]}.avi"):
                t_new = np.arange(tim_sec_list[i], tim_sec_list[i] + SIM_VIDEO_DURATION, dt)
                interp_rots = slerp(t_new)  # vectorised evaluation
                q_new = interp_rots.as_quat()  # still (x,y,z,w)
                earth_b_vec = np.zeros((len(q_new), 3))
                sun_b_vec = np.zeros((len(q_new), 3))

                plt.figure()
                plt.title("Quaternion interpolation")
                plt.plot(t_new, q_new, label='Interpolation')
                mask = np.logical_and(t_sec >= tim_sec_list[i], t_sec <= tim_sec_list[i] + SIM_VIDEO_DURATION)
                plt.plot(t_sec[mask], qs[mask], 'o', label='Quaternion')
                plt.grid()
                plt.legend()
                plt.show()

                video_salida = cv2.VideoWriter(self.folder_path + f"{video_names[i].split('.')[0]}.avi", fourcc,
                                               fps, (640, 480)) # frame_width, frame_height
                idx = np.argmin(np.abs(t_sec - tim_sec_list[i]))
                earth_pos_i = earth_pos[idx]
                earth_vel_i = earth_vel[idx]
                sun_pos_sc_i = sun_pos_sc[idx]

                j = 0
                for q_ in tqdm(q_new, desc="Creating video ...", total=len(q_new)):
                    earth_pos_b = Quaternions(q_).frame_conv(earth_pos_i)
                    sun_pos_sc_b = Quaternions(q_).frame_conv(sun_pos_sc_i)

                    earth_b_vec[j] = earth_pos_b / np.linalg.norm(earth_pos_b)
                    sun_b_vec[j] = sun_pos_sc_b / np.linalg.norm(sun_pos_sc_b)

                    cam.compute_picture(q_, earth_pos_b, earth_vel_i, sun_pos_sc_b)
                    frame = (cam.current_imagen * 255).astype(np.uint8)

                    frame = np.repeat(frame[:, :, np.newaxis], 3, axis=2)
                    video_salida.write(frame)
                    j += 1
                video_salida.release()
                data_ref_video = {'MJD': t_new/86400 + mjd[0], 'earth': earth_b_vec, 'sun': sun_b_vec}
                with open(self.folder_path + f"{video_names[i].split('.')[0]}_synth_vec.pkl", 'wb') as vec_file:
                    pickle.dump(data_ref_video, vec_file)


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
    d_[0, 0] = np.abs(d_vector[0])
    d_[1, 1] = np.abs(d_vector[1])
    d_[2, 2] = np.abs(d_vector[2])

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


def kinetic_energy(I, omega):
    # I: [Ixx, Iyy, Izz] (solo momentos principales)
    return 0.5 * np.einsum('ij,ij->i', omega @ I, omega)

def angular_momentum(I, omega):
    return omega @ I  # Nx3 matriz


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



