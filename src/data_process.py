"""
Created by Elias Obreque
Date: 04-09-2023
email: els.obrq@gmail.com
"""
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram
from sklearn.cluster import AgglomerativeClustering
from tools.mathtools import *
from tools.two_step_mag_calibration import two_step
import pandas as pd
import numpy as np
import datetime
import json
import os
from sklearn.cluster import KMeans

ts2022 = 1640995200 / 86400  # day
jd2022 = 2459580.50000


class RealData:
    def __init__(self, folder, file_directory):
        # Real data
        self.folder_path = folder
        self.file_name = file_directory
        self.data = pd.read_excel(folder + file_directory)
        self.data.sort_values(by=['timestamp'], inplace=True)
        self.data.dropna(inplace=True)
        self.data.reset_index(inplace=True)
        self.data['jd'] = [timestamp_to_julian(float(ts)) for ts in self.data['timestamp'].values]
        self.data[['acc_x', 'acc_y', 'acc_z']] *= np.deg2rad(1)

    def create_datetime_from_timestamp(self, time_format):
        self.data['DateTime'] = [datetime.datetime.fromtimestamp(ts, tz=datetime.timezone.utc).strftime(time_format)
                                 for ts in self.data['timestamp']]

    def plot_key(self, to_plot: list, show: bool = False, **kwargs):
        for key, value in kwargs.items():
            print("%s == %s" % (key, value))
        plt.figure()
        plt.grid()
        for i, elem in enumerate(to_plot):
            dict_temp = {}
            for key, value in kwargs.items():
                dict_temp[key] = value[i]
            plt.plot(self.data['jd'], self.data[elem], **dict_temp)
        plt.xlabel('Julian Date')
        plt.legend()
        plt.show() if show else None

    def search_nearly_tle(self):
        jd_init = self.data['jd'].values[0]
        jd_end = self.data['jd'].values[-1]
        tle_file = open("data/sat000052191.txt")
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
                scale = self.get_full_D(x_non_sol[3:])
                self.data[['mag_x', 'mag_y', 'mag_z']] = np.matmul(self.data[['mag_x', 'mag_y', 'mag_z']],
                                                                   (np.eye(3) + scale)) - bias

    def set_window_time(self, start_str=None, stop_str=None, format_time=None):
        if start_str is None and stop_str is None and format_time is None:
            self.plot_dendrogram_time()
            n_c = int(input("Set the number of cluster to find window time: "))
            model = self.get_windows_time(n_c=n_c)
            selected_cluster = int(input("Set the number of cluster to set window time: "))
            temp = np.argwhere(model.labels_ == selected_cluster)
            self.data = self.data.iloc[list(temp.T)[0]]
        else:
            init = datetime.datetime.strptime(start_str, format_time)
            stop = datetime.datetime.strptime(stop_str, format_time)
            init = init.replace(tzinfo=datetime.timezone.utc).timestamp()
            stop = stop.replace(tzinfo=datetime.timezone.utc).timestamp()
            temp = np.argwhere((init <= self.data['timestamp']) & (self.data['timestamp'] <= stop))
            self.data = self.data.iloc[list(temp.T)[0]]

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

    @staticmethod
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



