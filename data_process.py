"""
Created by Elias Obreque
Date: 04-09-2023
email: els.obrq@gmail.com
"""
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram
from sklearn.cluster import AgglomerativeClustering
from sklearn.datasets import load_iris
import pandas as pd
import numpy as np
import datetime
from sklearn.cluster import KMeans

ts2022 = 1640995200 / 86400  # day
jd2022 = 2459580.50000


class RealData:
    def __init__(self, file_directory):
        # Real data
        self.sensor_data = pd.read_excel(file_directory)
        self.sensor_data.sort_values(by=['timestamp'], inplace=True)
        self.sensor_data.dropna(inplace=True)
        self.sensor_data['jd'] = self.sensor_data['timestamp'].values / 86400 - ts2022 + jd2022

    def create_datetime_from_timestamp(self, time_format):
        self.sensor_data['DateTime'] = [datetime.datetime.fromtimestamp(ts).strftime(time_format)
                                        for ts in self.sensor_data['timestamp']]

    def plot_key(self, to_plot: list, show: bool = False):
        self.sensor_data[to_plot].plot()
        plt.show() if show else None

    def calibrate_mag(self, scale: np.array = None, bias: np.array = None):
        self.sensor_data[['mag_x', 'mag_y', 'mag_z']] = np.matmul(self.sensor_data[['mag_x', 'mag_y', 'mag_z']],
                                                                  (np.eye(3) + scale)) - bias

    def set_window_time(self, start_str=None, stop_str=None, format_time=None):
        if start_str is None and stop_str is None and format_time is None:
            self.plot_dendrogram_time()
            n_c = int(input("Set the number of cluster to find window time: "))
            model = self.get_windows_time(n_c=n_c)
            selected_cluster = int(input("Set the number of cluster to set window time: "))
            temp = np.argwhere(model.labels_ == selected_cluster)
            self.sensor_data = self.sensor_data.iloc[list(temp.T)[0]]
        else:
            init = datetime.datetime.strptime(start_str, format_time).timestamp()
            stop = datetime.datetime.strptime(stop_str, format_time).timestamp()
            temp = np.argwhere((init <= self.sensor_data['timestamp']) & (self.sensor_data['timestamp'] <= stop))
            self.sensor_data = self.sensor_data.iloc[list(temp.T)[0]]

    def plot_dendrogram_time(self, timestamp_distance: float = 3600):
        """

        :param timestamp_distance: minimum distance from time to find cluster
        :return: number of cluster
        """

        # setting distance_threshold=0 ensures we compute the full tree.
        model = AgglomerativeClustering(distance_threshold=timestamp_distance, n_clusters=None)

        model = model.fit(self.sensor_data['timestamp'].values.reshape(-1, 1))
        plt.title("Hierarchical Clustering Dendrogram")
        # plot the top three levels of the dendrogram
        plot_dendrogram(model, truncate_mode="level", p=5)
        plt.xlabel("Number of points in node (or index of point if no parenthesis).")
        plt.show()

    def get_windows_time(self, n_c = 2):
        model = KMeans(n_clusters=n_c)
        model.fit(self.sensor_data['timestamp'].values.reshape(-1, 1))
        for i in range(n_c):
            temp = self.sensor_data['DateTime'].values[np.argwhere(model.labels_ == i)]
            print(f"Cluster {i}: Start: {temp[0]} - Stop: {temp[-1]}")
        return model

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
    pass
