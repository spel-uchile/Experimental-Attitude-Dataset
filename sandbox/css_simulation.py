"""
Created by Elias Obreque
Date: 03-12-2023
email: els.obrq@gmail.com
"""
import numpy as np
import matplotlib.pyplot as plt
import pyvista as pv


def fibonacci_sphere(samples=10000):

    points = []
    phi = np.pi * (np.sqrt(5.) - 1.)  # golden angle in radians

    for i in range(samples):
        y = 1 - (i / float(samples - 1)) * 2  # y goes from 1 to -1
        radius = np.sqrt(1 - y * y)  # radius at y

        theta = phi * i  # golden angle increment

        x = np.cos(theta) * radius
        z = np.sin(theta) * radius

        points.append((x, y, z))

    return points

def circular_orbit(points):
    theta = np.linspace(0, 2 * np.pi, points)
    phi = np.linspace(0, np.pi, points)
    theta, phi = np.meshgrid(theta, phi)
    r = 1
    x = np.sin(phi) * np.cos(theta) * r
    y = np.sin(phi) * np.sin(theta) * r
    z = np.cos(phi) * r

    x_full = np.concatenate((x.flatten(), np.concatenate((y.flatten(), z.flatten()))))
    y_full = np.concatenate((y.flatten(), np.concatenate((z.flatten(), x.flatten()))))
    z_full = np.concatenate((z.flatten(), np.concatenate((x.flatten(), y.flatten()))))
    return np.array([x_full, y_full, z_full]).T


if __name__ == '__main__':
    # get circular orbit position from time array
    time_array = np.linspace(0, 36 * 12, 2000)
    position = fibonacci_sphere()
    # plot position 3d in pyvista
    plotter = pv.Plotter()
    # add sphere of radius one
    plotter.add_mesh(pv.Sphere(radius=1), color='blue', opacity=0.2)
    # add square of large 0.2
    plotter.add_mesh(pv.Box((-0.2, .2, -.2, .2, -.2, .2)), color='gray', opacity=0.7)
    # vector x, y, z
    plotter.add_mesh(pv.Arrow(start=(0, 0, 0), direction=(1, 0, 0)), color='red')
    plotter.add_mesh(pv.Arrow(start=(0, 0, 0), direction=(0, 1, 0)), color='green')
    plotter.add_mesh(pv.Arrow(start=(0, 0, 0), direction=(0, 0, 1)), color='blue')
    plotter.add_mesh(pv.PolyData(position), color='orange', render_points_as_spheres=True)
    plotter.show()

    # fig = plt.figure()
    # ax = fig.add_subplot(projection='3d')
    # ax.plot(position[0], position[1], position[2], 'o-')
    # plt.show()