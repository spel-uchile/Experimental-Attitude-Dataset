"""
Created by Elias Obreque
els.obrq@gmail.com
Date: 22-11-2022
"""

import sqlite3
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from tools.mathtools import timestamp_to_julian
from scipy.signal import savgol_filter
from sklearn.metrics import r2_score
import matplotlib as mpl

mpl.rcParams['font.size'] = 12


def rungeonestep(func, x, t, dt, inertia):
    k1 = func(x, t, inertia)
    xk2 = x + (dt / 2.0) * k1

    k2 = func(xk2, (t + dt / 2.0), inertia)
    xk3 = x + (dt / 2.0) * k2

    k3 = func(xk3, (t + dt / 2.0), inertia)
    xk4 = x + dt * k3

    k4 = func(xk4, (t + dt), inertia)
    next_x = x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    return next_x


def rotation_dynamics(w, t, inertia):
    a, b, c = inertia[0], inertia[1], inertia[2]
    w_dot = np.zeros(3)
    w_dot[0] = a * w[1] * w[2]
    w_dot[1] = b * w[0] * w[2]
    w_dot[2] = c * w[1] * w[0]
    return w_dot


def estimation_gyro_z(wx, wy, init, niter, wz=255):
    niter = niter if len(wx) >= niter else len(wx) - 1
    i = init
    dt = 0.1
    period = 1
    historical_mse = []
    i1 = 38478.678
    i2 = 38910.883
    i3 = 6873.717
    a = (i2 - i3) / i1
    b = (i3 - i1) / i2
    c = (i1 - i2) / i3
    print(a, b, c)
    alpha = .1
    historical_wz = [np.array([wx[i], wy[i], wz])]
    while i < niter:
        w = np.array([wx[i], wy[i], wz])
        w_aux = np.array([a * wy[i], b * wx[i]])
        error_mse = 0
        w_k = np.array([wx[i + 1], wy[i + 1], wz])
        next_w = np.array([0, 0, 0])
        for j in range(int(period / dt)):
            next_w = rungeonestep(rotation_dynamics, w, 0, dt, [a, b, c])
            w = next_w

        historical_wz.append(next_w)
        error_mse = np.sqrt(np.mean((w_k[:2] - next_w[:2]) ** 2))
        error = w_k[:2] - next_w[:2]
        update = period * error.dot(w_aux)
        wz = wz + alpha * update
        # w = np.array([wx[i], wy[i], wz])

        # next_w = rungeonestep(rotation_dynamics, w, 0, dt, [a, b, c])
        historical_mse.append(error_mse)
        i += 1
        # wz = next_w[2]
        print(i, error_mse, wz)
    return historical_wz, historical_mse


def get_data_frame(name, init=None, end=None):
    cnx = sqlite3.connect(name)

    df = pd.read_sql_query("SELECT * FROM dat_ads_data_3", cnx)
    print(df.keys())
    data = df[['sat_index', 'timestamp', 'acc_x', 'acc_y', 'acc_z', 'mag_x', 'mag_y', 'mag_z',
               #'sun1', 'sun2', 'sun3'
               ]].drop_duplicates()
    data = data.sort_values(by=['sat_index'])
    if init is not None:
        idx_init = np.where(data['timestamp'] == init)[0][0]
    else:
        idx_init = 0
    if end is not None:
        idx_end = np.where(data['timestamp'] == end)[0][0]
    else:
        idx_end = -1
    data = data[idx_init: idx_end]
    return data


def get_w_by_fft(omega, dt=1, plot=False):
    wx = np.fft.fft(omega[:, 0] * np.deg2rad(1) - np.mean(omega[:, 0] * np.deg2rad(1)))
    wy = np.fft.fft(omega[:, 1] * np.deg2rad(1) - np.mean(omega[:, 1] * np.deg2rad(1)))
    print(np.mean(omega[:, 1]))
    wx /= len(wx)
    wy /= len(wy)
    fstep = 1
    frequency = np.linspace(0, (len(wx) - 1), len(wx)) * fstep / len(wx)

    f = frequency[0:int(len(wx) / 2)]
    wx_plot = 2 * np.abs(wx)[0:int(len(wx) / 2)]
    wx_plot[0] = wx_plot[0] / 2
    wy_plot = 2 * np.abs(wy)[0:int(len(wx) / 2)]
    wy_plot[0] = wy_plot[0] / 2
    xi = 0.5 * (f[np.argmax(np.abs(wx_plot))] + f[np.argmax(np.abs(wy_plot))])
    center = np.abs(wx_plot).dot(f) / np.sum(np.abs(wx_plot))
    xi = 0.8 * xi + 0.2 * center
    time_list = np.arange(0, len(omega[:, 0]))
    time_sim = np.linspace(0, max(time_list), 200)

    phi = np.arctan2(omega[0, 0], omega[0, 1])
    w2 = np.mean(np.linalg.norm(omega[:, :2], axis=1))
    if np.sign(w2 * np.sin(2 * np.pi * xi * dt + phi)) != np.sign(omega[1, 0]):
        xi *= -1

    i1 = 38478.678
    i2 = 38910.883
    it = 0.5 * (i1 + i2)
    i3 = 6873.717
    eta = xi * it / (it - i3) * 2 * np.pi
    print("Angular velocity: ", eta * np.rad2deg(1), it / (it - i3))
    if plot:
        plt.figure()
        plt.title("X")
        plt.plot(time_list, omega[:, 0] - np.mean(omega[:, 0]), 'o-', label='Data')
        plt.plot(time_sim, w2 * np.sin(2 * np.pi * xi * time_sim + phi), '+-', label='Estimation')
        plt.legend()
        plt.grid()

        plt.figure()
        plt.title("Y")
        plt.plot(time_list, omega[:, 1] - np.mean(omega[:, 1]), 'o-', label='Data')
        plt.plot(time_sim, w2 * np.cos(2 * np.pi * xi * time_sim + phi), '+-', label='Estimation')
        plt.grid()
        plt.legend()

        plt.figure()
        plt.ylabel("Amplitude [deg]")
        plt.xlabel("Frequency [Hz]")
        plt.stem(f, wx_plot * np.rad2deg(1), 'b')
        plt.stem(f, wy_plot * np.rad2deg(1), 'g')
        plt.show()
    return eta


if __name__ == '__main__':
    # conn = sqlite3.connect("suchai.10.db")
    #1663085088
    #1663084999
    _MJD_1858 = 2400000.5

    data = get_data_frame('../data/suchai.10.db', None, None)
    dt = 1  # sec
    date = datetime.fromtimestamp(data['timestamp'].values[0])
    print(date)
    bias = np.array([-3.846, 0.1717, -0.6937])
    sat_idx = data['timestamp'].values
    omega = (data[['acc_x', 'acc_y', 'acc_z']].values) * np.deg2rad(1) # - bias
    mag = data[['mag_x', 'mag_y', 'mag_z']].values
    # sun = data[['sun1', 'sun2', 'sun3']].values
    # compute DFT with optimized FFT
    # eta = get_w_by_fft(omega, plot=True)
    #
    # plt.figure()
    # plt.title("W transversal")
    # plt.plot(np.sqrt(omega[:, 1] ** 2 + omega[:, 0] ** 2), 'o-')
    # plt.grid()
    #
    # plt.show()
    # wz_est, error_mse = estimation_gyro_z(omega[:, 0] * np.deg2rad(1), omega[:, 1] * np.deg2rad(1), 0, 500, wz=np.mean(eta))
    #
    # plt.figure()
    # [plt.plot(np.array(wz_est)[:, i] * np.rad2deg(1), '-+', label=str(i + 1)) for i in range(3)]
    # [plt.plot(omega[0:-1][:, i], label=str(i + 1)) for i in range(3)]
    # plt.grid()
    # plt.legend()
    # plt.show()
    #
    # plt.figure()
    # plt.ylabel("RMSE [deg/sec]")
    # plt.plot(np.array(error_mse) * np.rad2deg(1))
    # plt.grid()
    #

    labels = ['x', 'y', 'z']
    colors = ['tab:blue', 'tab:orange', 'tab:green']

    mjd_time = np.array([timestamp_to_julian(t_) - _MJD_1858 for t_ in sat_idx])
    print(f"Full days: {mjd_time[-1] - mjd_time[0]}")
    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(12, 6))
    fig.suptitle("Total measurement of angular velocity")
    for i in range(3):
        axes[i].grid(True)
        axes[i].set_ylabel(f"{labels[i]} [rad/s]")
        axes[i].plot(mjd_time, omega[:, i], lw=1.2, color=colors[i])
    axes[-1].set_xlabel("Time [MJD]")
    plt.xticks(rotation=15)
    plt.tight_layout()

    mjd_cut = mjd_time[mjd_time < 59880]
    omega_cut = omega[mjd_time < 59880]

    # Check for space
    time_diffs = np.diff(mjd_cut)
    gap_idx = np.where(time_diffs > 1)[0]
    split_indices = gap_idx + 1

    # Split blocks
    mjd_blocks = np.split(mjd_cut, split_indices)
    omega_blocks = np.split(omega_cut, split_indices)

    # Paso 2: graficar cada bloque como un tramo continuo
    fig, axes = plt.subplots(3, 1, figsize=(12, 6), sharex=True)


    offset = 0
    xticks = []
    xticklabels = []

    for mjd_block, omega_block in zip(mjd_blocks, omega_blocks):
        length = len(mjd_block)
        if length < 2:
            continue

        new_x = np.arange(offset, offset + length)

        for i in range(3):  # x, y, z
            axes[i].plot(new_x, omega_block[:, i], color=colors[i])

        mid = new_x[len(new_x) // 2]
        start = int(mjd_block[0])
        end = int(mjd_block[-1])
        xticks.append(mid)
        xticklabels.append(f"{start}–{end}")

        offset += length + 10

    for i in range(3):
        axes[i].set_ylabel(f'{labels[i]} [rad/s]')
        axes[i].grid(True)

    axes[-1].set_xticks(xticks)
    axes[-1].set_xticklabels(xticklabels, rotation=45)
    axes[-1].set_xlabel("Time block [MDJ]")
    plt.suptitle("Angular velocity")
    plt.tight_layout()
    plt.show()

    # plt.figure()
    # plt.title("Magnetometer")
    # mag_filter = np.array([savgol_filter(mag[:, i], 5, 2) for i in range(3)])
    # plt.plot(sat_idx, mag_filter.T, '-')
    # plt.plot(sat_idx, mag, '+')
    # plt.grid()
    #
    # plt.figure()
    # plt.title("Magnetometer MAG")
    # plt.plot(sat_idx, np.linalg.norm(mag, axis=1), 'o-')
    # plt.grid()

    # fig, ax = plt.subplots(1, 3)
    # ax[0].plot(sat_idx, sun[:, 0], '-')
    # ax[1].plot(sat_idx, sun[:, 1], '-')
    # ax[2].plot(sat_idx, sun[:, 2], '-')
    # plt.grid()

    plt.show()




