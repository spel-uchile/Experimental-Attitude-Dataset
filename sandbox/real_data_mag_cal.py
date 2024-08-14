"""
Created by Elias Obreque
Date: 08-07-2024
email: els.obrq@gmail.com
"""
from src.kalman_filter.ekf_mag_calibration import MagUKF, MagEKF
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import serial

if __name__ == '__main__':
    b_est = np.array([1, 1, 1]) * 100
    D_est = np.array([1.1, 1.2, 1.3, 0.001, 0.002, 0.003]) * 1e-7
    #    D_est[3:] *= 1e-5
    P_est = np.diag([*b_est, *D_est])  # (uT)^2
    # P_est = np.identity(9) * 1e-2

    std_measure = 0.5  # uT
    step = 1  # s
    tend = 5400 * 1

    time_array = np.arange(0, tend, step)

    ct = step
    hist_x_est = np.zeros((9, len(time_array))).T
    k = 0
    hist_x_est[0][3] = 0.0
    P_k = P_est

    ekf_cal = MagEKF()
    ekf_cal.pCov = P_est.copy()
    ukf = MagUKF(alpha=1, beta=2)
    ukf.pCov_x = P_est.copy()
    ukf.pCov_zero = P_est.copy()

    new_sensor = []
    new_sensor_ukf = []
    new_sensor_pso = []
    cov_sensor = std_measure ** 2

    for k in range(len(time_array)):
        # update state
        ekf_cal.save()
        ukf.save()

        ekf_cal.run(mag_sensor[k], mag_true[k], cov_sensor)
        ukf.run(mag_sensor[k], mag_true[k], cov_sensor, 150, 100)
        # pso_est.optimize(mag_sensor[k], mag_true[k], clip=False)

        bias_, D_scale = ekf_cal.get_calibration()
        bias_ukf, D_scale_ukf = ukf.get_calibration()
        # bias_pso, D_scale_pso = pso_est.get_calibration()

        new_sensor.append((np.eye(3) + D_scale) @ mag_sensor[k] - bias_)
        new_sensor_ukf.append((np.eye(3) + D_scale_ukf) @ mag_sensor[k] - bias_ukf)
        # new_sensor_pso.append((np.eye(3) + D_scale_pso) @ mag_sensor[k] - bias_pso)
        # print("RMS error:",
        #       np.dot(mag_true[k], new_sensor[-1]) / np.linalg.norm(mag_true[k]) / np.linalg.norm(new_sensor[-1]))

    ekf_cal.plot(np.linalg.norm(np.asarray(new_sensor), axis=1), np.linalg.norm(mag_true, axis=1))
    ukf.plot(np.linalg.norm(np.asarray(new_sensor_ukf), axis=1), np.linalg.norm(mag_true, axis=1))

    # D_ekf_vector = ekf_cal.historical['scale'][-1]
    # b_ekf = ekf_cal.historical['bias'][-1]
    # D_ukf_vector = ukf.historical['scale'][-1]
    # b_ukf = ukf.historical['bias'][-1]
    # print(b_ekf, D_ekf_vector)
    # print(b_ukf, D_ukf_vector)
    # D_ekf = np.array([[D_ekf_vector[0], D_ekf_vector[3], D_ekf_vector[4]],
    #                   [D_ekf_vector[3], D_ekf_vector[1], D_ekf_vector[5]],
    #                   [D_ekf_vector[4], D_ekf_vector[5], D_ekf_vector[2]]])
    # D_ukf = np.array([[D_ukf_vector[0], D_ukf_vector[3], D_ukf_vector[4]],
    #                   [D_ukf_vector[3], D_ukf_vector[1], D_ukf_vector[5]],
    #                   [D_ukf_vector[4], D_ukf_vector[5], D_ukf_vector[2]]])

    mag_ekf = np.asarray(new_sensor)  # (np.eye(3) + D_ekf).dot(mag_sensor.T).T - b_ekf.reshape(-1, 1).T
    mag_ukf = np.asarray(new_sensor_ukf)  # (np.eye(3) + D_ukf).dot(mag_sensor.T).T - b_ukf.reshape(-1, 1).T

    dic_ekf_x = pd.DataFrame({'Error': (mag_ekf - mag_true)[:, 0], 'Filters': ['EKF'] * len(mag_ekf),
                              'Axis': ['x'] * len(mag_ekf)})
    dic_ekf_y = pd.DataFrame({'Error': (mag_ekf - mag_true)[:, 1], 'Filters': ['EKF'] * len(mag_ekf),
                              'Axis': ['y'] * len(mag_ekf)})
    dic_ekf_z = pd.DataFrame({'Error': (mag_ekf - mag_true)[:, 2], 'Filters': ['EKF'] * len(mag_ekf),
                              'Axis': ['z'] * len(mag_ekf)})

    dic_ukf_x = pd.DataFrame({'Error': (mag_ukf - mag_true)[:, 0], 'Filters': ['UKF'] * len(mag_ukf),
                              'Axis': ['x'] * len(mag_ukf)})
    dic_ukf_y = pd.DataFrame({'Error': (mag_ukf - mag_true)[:, 1], 'Filters': ['UKF'] * len(mag_ukf),
                              'Axis': ['y'] * len(mag_ukf)})
    dic_ukf_z = pd.DataFrame({'Error': (mag_ukf - mag_true)[:, 2], 'Filters': ['UKF'] * len(mag_ukf),
                              'Axis': ['z'] * len(mag_ukf)})

    #
    result = pd.concat([dic_ekf_x, dic_ekf_y, dic_ekf_z, dic_ukf_x, dic_ukf_y, dic_ukf_z], ignore_index=True)

    fig, ax = plt.subplots(3, 1, sharex=True)
    bias, sigma_up, sigma_low = ekf_cal.get_bias()
    ax[0].set_title("Bias - EKF")
    for i in range(3):
        ax[i].grid()
        ax[i].plot(time_array, bias[:, i], label='Estimated bias')
        ax[i].plot(time_array, sigma_up[:, i], color='red', linestyle='--', label=r'$+3\sigma$')
        ax[i].plot(time_array, sigma_low[:, i], color='red', linestyle='--', label=r'$-3\sigma$')
        ax[i].plot(time_array, b_true[:, i], linestyle="--", color='black', label='True')
        if i == 0:
            ax[i].legend(loc='upper center', bbox_to_anchor=(0.5, 1.05),
                         fancybox=True, shadow=True, ncol=4)
    ax[i].set_xlabel("Time")

    plt.figure()
    plt.title("D - EKF")
    plt.grid()
    plt.plot(time_array, ekf_cal.historical['scale'])
    plt.xlabel("Time")
    plt.hlines(D_true, 0, tend, linestyle="--", color="black")

    fig_est_ekf, ax_est_ekf = plt.subplots(3, 1, sharex=True)
    fig_est_ekf.suptitle("Calibration - EKF")
    ax_est_ekf[0].set_ylabel("x")
    ax_est_ekf[1].set_ylabel("y")
    ax_est_ekf[2].set_ylabel("z")
    for i in range(3):
        ax_est_ekf[i].grid()
        ax_est_ekf[i].plot(time_array, mag_true[:, i], color='black', lw=0.7, label='true')
        ax_est_ekf[i].plot(time_array, mag_ekf[:, i], color='red', lw=0.7, label='est')
        ax_est_ekf[i].legend()
    ax_est_ekf[i].set_xlabel("Time")

    # UKF
    fig, ax = plt.subplots(3, 1, sharex=True)
    bias, sigma_up, sigma_low = ukf.get_bias()
    ax[0].set_title("Bias - UKF")
    for i in range(3):
        ax[i].grid()
        ax[i].plot(time_array, bias[:, i], label='Estimated bias')
        ax[i].plot(time_array, sigma_up[:, i], color='red', linestyle='--', label=r'$+3\sigma$')
        ax[i].plot(time_array, sigma_low[:, i], color='red', linestyle='--', label=r'$-3\sigma$')
        ax[i].plot(time_array, b_true[:, i], linestyle="--", color='black', label='True')
        if i == 0:
            ax[i].legend(loc='upper center', bbox_to_anchor=(0.5, 1.05),
                         fancybox=True, shadow=True, ncol=4)
    ax[i].set_xlabel("Time")

    plt.figure()
    plt.title("D - UKF")
    plt.grid()
    plt.plot(time_array, ukf.historical['scale'])
    plt.xlabel("Time")
    plt.hlines(D_true, 0, tend, linestyle="--", color="black")

    fig_est_ukf, ax_est_ukf = plt.subplots(3, 1, sharex=True)
    fig_est_ukf.suptitle("Calibration - UKF")
    ax_est_ukf[0].set_ylabel("x")
    ax_est_ukf[1].set_ylabel("y")
    ax_est_ukf[2].set_ylabel("z")
    for i in range(3):
        ax_est_ukf[i].grid()
        ax_est_ukf[i].plot(time_array, mag_true[:, i], color='black', lw=0.7, label='true')
        ax_est_ukf[i].plot(time_array, mag_ukf[:, i], color='red', lw=0.7, label='est')
        ax_est_ukf[i].legend()
    ax_est_ukf[i].set_xlabel("Time")
    plt.legend()

    plt.figure()
    sns.set(style="whitegrid")
    sns.violinplot(data=result, y="Error", x="Filters", hue="Axis",
                   inner_kws=dict(box_width=4, whis_width=2))
    plt.show()
