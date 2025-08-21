"""
Created by Elias Obreque
els.obrq@gmail.com
Date: 04-12-2022
"""
import time
import matplotlib.pyplot as plt
import numpy as np

RAD2DEG = 180 / np.pi
DEG2RAD = 1 / RAD2DEG


class Monitor:
    def __init__(self, dataset, folder_save, video_dataset: dict=None):
        self.dataset = dataset
        self.fft_dataset = {}
        self.datetime = self.dataset['DateTime']
        self.position = None
        self.q_i2b = None
        self.sideral = None
        self.folder_save = folder_save
        self.vectors = {}
        self.video_dataset = video_dataset

    def add_data(self, new_data: dict):
        self.dataset = {**self.dataset, **new_data}

    def set_position(self, name):
        self.position = self.dataset[name]

    def set_quaternion(self, name):
        self.q_i2b = self.dataset[name]

    def set_sideral(self, name):
        self.sideral = self.dataset['sideral']

    def add_vector(self, name, color='white'):
        self.vectors[name] = {'data': self.dataset[name],
                              'color': color}

    def plot(self, x_dataset, y_dataset, xname=None, yname=None, title=None, step=False, scale=1.0, fft=False, ls='-',
             legend_list=None, log_scale=False):
        if fft:
            dataset = self.fft_dataset
        else:
            dataset = self.dataset
        fig = plt.figure()
        plt.title(title if title is not None else y_dataset)
        plt.ylabel(yname)
        plt.xlabel(xname)
        plt.grid()
        if type(x_dataset) == str:
            x = dataset[x_dataset]
            y = np.array(dataset[y_dataset])
            if step and fft is False:
                plt.step(x, y * scale, alpha=0.7)
            elif fft:
                plt.stem(x, y * scale, alpha=0.7)
            else:
                plt.plot(x, y * scale, ls=ls, alpha=0.7)
        else:
            color = ['b', 'r']
            i = 0
            for xset, yset in zip(x_dataset, y_dataset):
                x = dataset[xset]
                y = dataset[yset]

                if step and fft is False:
                    plt.step(x, y * scale, label=yset, alpha=0.7)
                elif fft:
                    plt.stem(x, y * scale, color[i], label=yset)
                else:
                    plt.plot(x, y * scale, 'o-', label=yset, lw=0.7)
                i += 1

        if y_dataset == 'p_cov' or log_scale:
            plt.yscale('log')
        if legend_list is not None:
            plt.legend(legend_list)
        plt.draw()
        names_to_save = x_dataset + "_" + y_dataset
        plt.xticks(rotation=15)
        # plt.ticklabel_format(useOffset=False)
        plt.tight_layout()
        fig.savefig(self.folder_save + names_to_save + '.png')
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
        plt.show(block=False)

    def plot_video_performance(self):
        fig_picture, axes = plt.subplots(nrows=2, ncols=1, sharex=True)
        axes[0].grid()
        axes[0].set_xlabel("MJD")
        axes[1].grid()
        axes[1].set_xlabel("MJD")
        axes[0].set_ylabel("Roll [deg]")
        axes[1].set_ylabel("Pitch [deg]")
        axes[0].plot(self.dataset['mjd'], np.array(self.dataset['ypr_lvlh2b'])[:, 2] * RAD2DEG, label="EKF")
        axes[1].plot(self.dataset['mjd'], np.array(self.dataset['ypr_lvlh2b'])[:, 1] * RAD2DEG, label="EKF")
        for key, data in self.video_dataset.items():
            axes[0].plot(data['MJD'], data['roll'] * RAD2DEG, 'o', label="Video {}".format(key))
            axes[1].plot(data['MJD'], data['pitch'] * RAD2DEG, 'o', label="Video {}".format(key))
            # axes[0].set_xlim(np.min(data['MJD']), np.max(data['MJD']))
            # axes[1].set_xlim(np.min(data['MJD']), np.max(data['MJD']))
        axes[0].legend()
        axes[1].legend()
        fig_picture.savefig(self.folder_save + f"ypr_estimation_lvlh.png")

        E = np.asarray(self.dataset['earth_b_est'], float)
        if E.dtype == object: E = np.stack(E)
        En = np.linalg.norm(E, axis=1, keepdims=True)
        earth_unit_est = np.divide(E, En, out=np.zeros_like(E), where=En > 0)

        S = np.asarray(self.dataset['sun_b_est'], float)
        if S.dtype == object: S = np.stack(S)
        Sn = np.linalg.norm(S, axis=1, keepdims=True)
        sun_unit_est = np.divide(S, Sn, out=np.zeros_like(S), where=Sn > 0)

        for key, data in self.video_dataset.items():
            fig_earth_cam, axes_earth_cam = plt.subplots(figsize=(8.5, 5), nrows=1, ncols=1)
            axes_earth_cam.grid()
            axes_earth_cam.set_title("MEKF and Camera Earth vector estimation - BF")
            axes_earth_cam.set_xlabel("MJD")
            axes_earth_cam.set_ylabel("Unit vector [-]")

            h_cx, = axes_earth_cam.plot(data['MJD'], data['e_b_x'], 'o', label=r'CAM: $e_{x,c}$')
            h_cy, = axes_earth_cam.plot(data['MJD'], data['e_b_y'], 'o', label=r'CAM: $e_{y,c}$')
            h_cz, = axes_earth_cam.plot(data['MJD'], data['e_b_z'], 'o', label=r'CAM: $e_{z,c}$')

            h_tx, = axes_earth_cam.plot(self.dataset['mjd'], earth_unit_est[:, 0],
                                        label=r'MEKF: $e_{x,c}$')
            h_ty, = axes_earth_cam.plot(self.dataset['mjd'], earth_unit_est[:, 1],
                                        label=r'MEKF: $e_{y,c}$')
            h_tz, = axes_earth_cam.plot(self.dataset['mjd'], earth_unit_est[:, 2],
                                        label=r'MEKF: $e_{z,c}$')

            h_tx.set_color(h_cx.get_color())
            h_ty.set_color(h_cy.get_color())
            h_tz.set_color(h_cz.get_color())

            axes_earth_cam.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), borderaxespad=0.)
            fig_earth_cam.subplots_adjust(right=0.8)
            axes_earth_cam.set_xlim(data['MJD'].min() - 1 / 86400, data['MJD'].max() + 1 / 86400)
            plt.xticks(rotation=15)
            plt.ticklabel_format(useOffset=False)
            plt.tight_layout()
            fig_earth_cam.savefig(self.folder_save + f"{key} - cam_mekf_earth_estimation_b.png")

        for key, data in self.video_dataset.items():
            fig_sun_cam, axes_sun_cam = plt.subplots(figsize=(8.5, 5), nrows=1, ncols=1)
            axes_sun_cam.grid()
            axes_sun_cam.set_title(f"MEKF and Camera Sun vector estimation - BF")
            axes_sun_cam.set_xlabel("MJD")
            axes_sun_cam.set_ylabel("Unit vector [-]")

            h_cx, = axes_sun_cam.plot(data['MJD'], data['s_b_x'], 'o', label=r'CAM: $s_{x,c}$')
            h_cy, = axes_sun_cam.plot(data['MJD'], data['s_b_y'], 'o', label=r'CAM: $s_{y,c}$')
            h_cz, = axes_sun_cam.plot(data['MJD'], data['s_b_z'], 'o', label=r'CAM: $s_{z,c}$')

            h_tx, = axes_sun_cam.plot(self.dataset['mjd'], sun_unit_est[:, 0], label=r'MEKF: $s_{x,c}$')
            h_ty, = axes_sun_cam.plot(self.dataset['mjd'], sun_unit_est[:, 1], label=r'MEKF: $s_{y,c}$')
            h_tz, = axes_sun_cam.plot(self.dataset['mjd'], sun_unit_est[:, 2], label=r'MEKF: $s_{z,c}$')

            h_tx.set_color(h_cx.get_color())
            h_ty.set_color(h_cy.get_color())
            h_tz.set_color(h_cz.get_color())

            axes_sun_cam.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), borderaxespad=0.)
            fig_sun_cam.subplots_adjust(right=0.8)
            axes_sun_cam.set_xlim(np.min(data['MJD']) - 1 / 86400, np.max(data['MJD']) + 1 / 86400)
            plt.xticks(rotation=15)
            plt.ticklabel_format(useOffset=False)
            plt.tight_layout()
            fig_sun_cam.savefig(self.folder_save + f"{key} - cam_mekf_sun_estimation_b.png")