"""
Created by Elias Obreque
els.obrq@gmail.com
Date: 04-12-2022
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyvista as pv
from pyquaternion import Quaternion as pyquat
from pyvista import examples
import pyvista


class Monitor:
    def __init__(self, dataset):
        self.dataset = dataset
        self.fft_dataset = {}
        self.position = None
        self.q_i2b = None
        self.sideral = None
        self.vectors = {}

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

    def plot(self, x_dataset, y_dataset, xname=None, yname=None, title=None, step=False, scale=1, fft=False):
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
                    plt.plot(x, y * scale, 'o-', label=yset)
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

    def plot3d(self):
        monitor_3d = Monitor3d(self.position, self.q_i2b, self.sideral)
        monitor_3d.add_vectors(self.vectors)
        monitor_3d.plotter3d.show()

    @staticmethod
    def show_monitor():
        plt.show()


class Monitor3d:
    AUX_VECTOR = [pv.Arrow(np.zeros(3), direction=np.array([1, 0, 0]), scale=1),
                  pv.Arrow(np.zeros(3), direction=np.array([0, 1, 0]), scale=1),
                  pv.Arrow(np.zeros(3), direction=np.array([0, 0, 1]), scale=1)]
    plotter3d = pv.Plotter()
    vectors = {}

    def __init__(self, position, q_i2b, sideral):
        self.last_index_ = 0
        self.last_pos = np.zeros(3)
        self.last_q_i2b = pyquat(np.array([1, 0, 0, 0]))
        self.last_sideral = 0
        self.earth_sideral = np.asarray(sideral)
        self.sat_pos = np.asarray(position)
        new_q = np.asarray(q_i2b)
        self.sat_q_i2b = np.concatenate([new_q[:, 3].reshape(-1, 1), new_q[:, :3]], axis=1)

        # earth3d = pv.Sphere(radius=re, phi_resolution=180, theta_resolution=360)
        self.earth3d, texture = load_earth(radius=6378.137)

        self.sat_model = pv.PolyData("tools/cad/basic3U.stl")
        self.plotter3d.add_mesh(self.earth3d, texture=texture)  # , color='#0C007C')
        self.plotter3d.add_mesh(self.sat_model, color='white')
        self.plotter3d.add_points(np.array([elem for elem in self.sat_pos]), render_points_as_spheres=True,
                                  point_size=3, color='black')
        self.plotter3d.add_axes()
        self.add_eci_frame()

        # reset earth
        self.earth3d.rotate_z(180, inplace=True)

        self.update(0)

        self.plotter3d.add_slider_widget(self.update, [0, len(self.sat_pos) - 1], value=0, title='Step')
        # self.btn = self.plotter3d.add_checkbox_button_widget(self.forward, value=True)

    def forward(self, flag):
        self.update(np.min([self.last_index_ + 1, len(self.sat_pos) - 1]))

    def add_vectors(self, vectors_list):
        for key, item in vectors_list.items():
            data, color = item['data'], item['color']
            if 'mag' in key:
                relative_to_sc = data[0]
            else:
                relative_to_sc = data[0] - self.sat_pos[0]

            relative_to_sc /= np.linalg.norm(relative_to_sc)
            self.vectors[key] = {}
            self.vectors[key]['data'] = data
            self.vectors[key]['model'] = pv.Arrow(self.sat_pos[0], direction=relative_to_sc,
                                                  scale=200)
            self.plotter3d.add_mesh(self.vectors[key]['model'], color=color, reset_camera=False)

    def update(self, index_):
        index_ = int(index_)
        # position
        sc_pos_i = self.sat_pos[index_]
        relative_pos = sc_pos_i - self.last_pos
        self.sat_model.translate(relative_pos, inplace=True)
        # orientation
        quaternion_tn = pyquat(self.sat_q_i2b[index_]).unit
        inv_quaternion = self.last_q_i2b.inverse
        d_quaternion = quaternion_tn * inv_quaternion
        self.sat_model.rotate_vector(vector=tuple(d_quaternion.vector),
                                     angle=d_quaternion.angle * np.rad2deg(1),
                                     point=sc_pos_i, inplace=True)
        # earth
        sideral = self.earth_sideral[index_]
        self.earth3d.rotate_z((sideral - self.last_sideral) * np.rad2deg(1), inplace=True)
        # vectors
        for key, arrow in self.vectors.items():
            if 'mag' in key:
                vect_pos = arrow['data'][index_]
                last_vect_pos = arrow['data'][self.last_index_]
            else:
                vect_pos = arrow['data'][index_] - sc_pos_i
                last_vect_pos = arrow['data'][self.last_index_] - self.last_pos

            en = np.linalg.norm(vect_pos)
            ln = np.linalg.norm(last_vect_pos)
            rot_vec = np.cross(last_vect_pos, vect_pos) / (en * ln)
            if np.linalg.norm(rot_vec) > 1e-9:
                rot_vec /= np.linalg.norm(rot_vec)
            else:
                rot_vec = np.zeros(3)
            ang_rot = np.arccos(np.dot(last_vect_pos, vect_pos) / (en * ln))
            arrow['model'].translate(relative_pos, inplace=True)
            arrow['model'].rotate_vector(vector=rot_vec, angle=np.rad2deg(ang_rot), point=sc_pos_i, inplace=True)

        self.last_sideral = sideral
        self.last_pos = sc_pos_i
        self.last_q_i2b = quaternion_tn
        self.last_index_ = index_

    def add_eci_frame(self):
        scale = 1e-3
        self.plotter3d.add_lines(np.array([[0, 0, 0], [1e7 * scale, 0, 0]]), color=[255, 0, 0], width=2,
                                 label='X-axis')
        self.plotter3d.add_lines(np.array([[0, 0, 0], [0, 1e7 * scale, 0]]), color=[0, 255, 0], width=2,
                                 label='Y-axis')
        self.plotter3d.add_lines(np.array([[0, 0, 0], [0, 0, 1e7 * scale]]), color=[0, 0, 255], width=2,
                                 label='Z-axis')


def load_earth(radius=1.0, lat_resolution=100, lon_resolution=200):
    """Load the planet Earth as a textured sphere.

    Parameters
    ----------
    radius : float, default: 1.0
        Sphere radius.

    lat_resolution : int, default: 50
        Set the number of points in the latitude direction.

    lon_resolution : int, default: 100
        Set the number of points in the longitude direction.

    Returns
    -------
    pyvista.PolyData
        Earth dataset with texture.

    Examples
    --------
    """
    sphere = _sphere_with_texture_map(
        radius=radius, lat_resolution=lat_resolution, lon_resolution=lon_resolution
    )
    sphere.translate(-np.array(sphere.center), inplace=True)

    f = pyvista.read_texture("tools/img/2k_earth_daymap.jpg")
    return sphere, f


def _sphere_with_texture_map(radius=1.0, lat_resolution=50, lon_resolution=100):
    """Sphere with texture coordinates.

    Parameters
    ----------
    radius : float, default: 1.0
        Sphere radius.

    lat_resolution : int, default: 100
        Set the number of points in the latitude direction.

    lon_resolution : int, default: 100
        Set the number of points in the longitude direction.

    Returns
    -------
    pyvista.PolyData
        Sphere mesh with texture coordinates.

    """
    # https://github.com/pyvista/pyvista/pull/2994#issuecomment-1200520035
    theta, phi = np.mgrid[0: np.pi: lat_resolution * 1j, 0: 2 * np.pi: lon_resolution * 1j]
    x = radius * np.sin(theta) * np.cos(phi)
    y = radius * np.sin(theta) * np.sin(phi)
    z = radius * np.cos(theta)
    sphere = pyvista.StructuredGrid(x, y, z)
    texture_coords = np.empty((sphere.n_points, 2))
    texture_coords[:, 0] = phi.ravel('F') / phi.max()
    texture_coords[:, 1] = theta[::-1, :].ravel('F') / theta.max()
    sphere.active_t_coords = texture_coords
    return sphere.extract_surface(pass_pointid=False, pass_cellid=False)
