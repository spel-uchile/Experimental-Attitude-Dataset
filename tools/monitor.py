"""
Created by Elias Obreque
els.obrq@gmail.com
Date: 04-12-2022
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyvista as pv
from pyvista import examples


class Monitor:
    def __init__(self, dataset):
        self.dataset = dataset
        self.fft_dataset = {}

    def add_data(self, new_data: dict):
        self.dataset = {**self.dataset, **new_data}

    def plot(self, x_dataset, y_dataset, xname=None, yname=None, title=None, step=True, scale=1, fft=False):
        if fft:
            dataset = self.fft_dataset
        else:
            dataset = self.dataset
        fig = plt.figure()
        plt.title(title)
        plt.ylabel(yname)
        plt.xlabel(xname)
        plt.grid()
        if type(x_dataset) == str:
            x = dataset[x_dataset]
            y = dataset[y_dataset]
            if step and fft is False:
                plt.step(x, y * scale, label=y_dataset)
            elif fft:
                plt.stem(x, y * scale, label=y_dataset)
            else:
                plt.plot(x, y * scale, label=y_dataset)
        else:
            color = ['b', 'r']
            i = 0
            for xset, yset in zip(x_dataset, y_dataset):
                x = dataset[xset]
                y = dataset[yset]

                if step and fft is False:
                    plt.step(x, y * scale, label=yset)
                elif fft:
                    plt.stem(x, y * scale, color[i], label=yset)
                else:
                    plt.plot(x, y * scale,  'o-', label=yset)
                i += 1
        plt.legend()
        plt.draw()
        return fig

    def fft(self, yset, fstep_list):
        for elem, fstep in zip(yset, fstep_list):
            y = self.dataset[elem]
            y -= np.mean(y)
            trans_fourier = np.fft.fft(y) / len(y)
            frequency = np.linspace(0, (len(trans_fourier) - 1) * fstep, len(trans_fourier)) / len(y)
            f_plot = frequency[0:int(len(trans_fourier) / 2) + 1]
            w_plot = 2 * np.abs(trans_fourier)[0:int(len(trans_fourier) / 2) + 1]
            self.fft_dataset[elem + '_amp'] = np.abs(w_plot)
            self.fft_dataset[elem + '_freq'] = f_plot

    @staticmethod
    def show_monitor():
        plt.show()


class Monitor3d:
    LVLH_VECTOR = [pv.Arrow(np.zeros(3), direction=np.array([1, 0, 0]), scale=1),
                   pv.Arrow(np.zeros(3), direction=np.array([0, 1, 0]), scale=1),
                   pv.Arrow(np.zeros(3), direction=np.array([0, 0, 1]), scale=1)]
    plotter3d = pv.Plotter()
    earth3d = examples.load_globe()
    earth3d.points /= 1000000

    def __init__(self, dataset_info):
        self.dataset_info = dataset_info

        # earth3d = pv.Sphere(radius=re, phi_resolution=180, theta_resolution=360)



        plotter3d.add_mesh(earth3d)  # , color='#0C007C')
        plotter3d.add_axes()
        quaternion_t0 = pyquat([1, 0, 0, 0])
        k_matrix = quaternion_t0.transformation_matrix
        plotter3d.add_points(np.array([elem for elem in dataset_info['sc_pos'].values]), render_points_as_spheres=True,
                             point_size=10)
        for i in range(len(dataset_info['earth_pos_lvlh'])):
            pos_i = np.zeros(3)

            LVLH_VECTOR[0] = pv.Arrow(pos_i, direction=dataset_info['earth_pos_b'][i], scale=1)
            earth_b_arrows[-1].transform(k_matrix.dot(10 * np.identity(4) * np.array([10, 10, 10, 10])), inplace=True)
            pos_i = dataset_info['sc_pos'][i]

            earth_b_arrows[-1].translate(pos_i, inplace=True)

            plotter3d.add_mesh(earth_b_arrows[-1], color='yellow')

        plotter3d.show()

