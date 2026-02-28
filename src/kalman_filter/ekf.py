"""
Created by Elias Obreque
els.obrq@gmail.com
Date: 21-01-2023
"""
import numpy as np

from src.dynamics.quaternion import Quaternions
from sklearn.metrics import mean_squared_error
from tools.mathtools import *

rev_day = 15.23166528
w_orbit = np.array([0, - 2 * np.pi * rev_day / 86400, 0])


class EKF:
    def __init__(self, inertia, R, Q, P):
        # self.inertia = inertia
        self.init_state_save = False
        # inv_inertia = np.linalg.inv(inertia)
        # self.inv_inertia = inv_inertia
        self.kf_R = R
        self.kf_R_a = {'mag': R, 'css': R}
        self.kf_Q = Q
        self.dim_rows = len(R)
        self.dim_cols = len(R[0])
        self.sigma_omega = 0
        self.sigma_bias = 0
        self.kf_S = np.zeros([self.dim_rows, self.dim_cols])
        self.kf_K = np.zeros([self.dim_rows, self.dim_cols])
        self.s_hist = {'mag': [], 'css': []}
        self.S_hat_i = np.zeros([self.dim_rows, self.dim_cols])
        self.covariance_P = P
        self.internal_cov_P = self.covariance_P.copy()
        self.state = np.zeros(len(self.covariance_P))
        self.internal_state = self.state.copy()
        self.current_measure = np.zeros(len(self.covariance_P))

    def set_first_state(self, new_state):
        self.state = new_state

    def update(self, step, current_measure=None, reference=None):
        new_x_k, new_P_k = self.get_prediction(self.state, self.covariance_P, step)
        if current_measure is not None:
            self.current_measure = current_measure
            new_z_k = self.get_observer_prediction(new_x_k, reference)
            self.update_covariance_matrix(new_x_k, new_P_k)
            self.get_kalman_gain(new_x_k, new_P_k)
            new_x = self.update_state(new_x_k, current_measure, new_z_k)
            new_P = self.update_covariance_P_matrix(new_x_k, new_P_k)
            self.state = new_x
            self.covariance_P = new_P
        else:
            self.state = new_x_k
            self.covariance_P = new_P_k

    def static_update(self, guess_r, guess_qs, guess_qb, step, vector_b, vector_i):
        self.kf_R = guess_r * np.eye(len(vector_i))
        self.sigma_omega = guess_qs
        self.sigma_bias = guess_qb

        new_x_k, new_p_k = self.get_prediction(self.state, self.covariance_P, step)
        vector_b = vector_b / np.linalg.norm(vector_b)
        new_z_k = self.get_observer_prediction(new_x_k, vector_i, save=False)
        print("residual angle error (deg): {}".format(np.rad2deg(1) * np.arccos(vector_b @ new_z_k)))
        H = self.attitude_observer_model(new_x_k, vector_i / np.linalg.norm(vector_i))
        self.update_covariance_matrix(H, new_p_k)
        self.get_kalman_gain(H, self.internal_cov_P)
        new_x_k = self.update_state(new_x_k, vector_b, new_z_k)
        new_p_k = self.update_covariance_P_matrix(H, new_p_k)
        return new_p_k

    def propagate(self, step):
        self.internal_state, self.internal_cov_P = self.get_prediction(self.state, self.covariance_P, step)

    def inject_vector(self, vector_b, vector_i, gain=None, sensor='mag', sigma2=None):
        if sigma2 is not None:
            self.kf_R = sigma2 * np.eye(len(vector_i))
            self.kf_R_a[sensor] = sigma2 * np.eye(len(vector_i))
        new_z_k = self.get_observer_prediction(self.internal_state, vector_i, sensor_type=sensor)
        H = self.attitude_observer_model(self.internal_state, vector_i)
        if gain is not None:
            H[:3, :3] = gain @ H[:3, :3] / np.linalg.norm(vector_i)
        # print("residual MRSE error (deg): {}".format(mean_squared_error(new_z_k, vector_b)))
        # else:
            # print("residual angle error (deg): {}".format(
            #    np.rad2deg(1) * np.arccos(vector_b / np.linalg.norm(vector_b) @ new_z_k / np.linalg.norm(new_z_k))))

        r_sensor = self.kf_R #_a[sensor]
        s_k = self.update_covariance_matrix(H, self.internal_cov_P, r_sensor)

        beta_i = 0.002
        alpha_i = 0.00005
        nu =  vector_b - new_z_k
        self.S_hat_i = (1 - beta_i) * self.S_hat_i + beta_i * (nu @ nu.T)

        #self.kf_R_a[sensor] = (1 - alpha_i) * self.kf_R_a[sensor]  + alpha_i * (self.S_hat_i  - H @ self.internal_cov_P @ H.T)

        self.s_hist[sensor] = s_k
        self.get_kalman_gain(H, self.internal_cov_P)
        self.internal_state = self.update_state(self.internal_state, vector_b, new_z_k, H)
        self.internal_cov_P = self.update_covariance_P_matrix(H, self.internal_cov_P, r_sensor)
        return new_z_k

    def inject_vector_6(self, vector1_b, vector1_i, vector2_b, vector2_i, sigma1, sigma2):
        self.kf_R = sigma2 * np.eye(6)
        new_z_k1 = self.get_observer_prediction(self.internal_state, vector1_i)
        new_z_k2 = self.get_observer_prediction(self.internal_state, vector2_i)
        print("residual error 1: {}".format(np.rad2deg(1) * np.arccos(vector1_b @ new_z_k1)))
        print("residual error 2: {}".format(np.rad2deg(1) * np.arccos(vector2_b @ new_z_k2)))
        H1 = self.attitude_observer_model(self.internal_state, vector1_i)
        H2 = self.attitude_observer_model(self.internal_state, vector2_i)
        H = np.zeros((6, 6))
        H[:3, :3] = H1[:3, :3]
        H[3:, :3] = H2[:3, :3]
        vector_b = np.array([*vector1_b, *vector2_b])
        new_z_k = np.array([*new_z_k1, *new_z_k2])
        self.update_covariance_matrix(H, self.internal_cov_P)
        self.get_kalman_gain(H, self.internal_cov_P)
        self.internal_state = self.update_state(self.internal_state, vector_b, new_z_k)
        self.internal_cov_P = self.update_covariance_P_matrix(H, self.internal_cov_P)

    def get_state(self):
        return self.state

    def reset_state(self):
        self.state = self.internal_state
        self.covariance_P = self.internal_cov_P

    def get_internal_state(self):
        return self.internal_state

    def optimize_R_Q(self, vector_b, vector_i, step):
        r = np.random.uniform(0, 1.0, 10)
        qs = np.random.uniform(0, 1.0, 10)
        qb = np.random.uniform(0, 1.0, 10)
        p_k_rq = []
        for r_, q_s, q_b in zip(r, qs, qb):
            p_k_rq.append(np.max(self.static_update(r_, q_s, q_b, step, vector_b, vector_i)))

        print(p_k_rq)
        # save index from r and q with minimum p_k_rq
        index_min = np.argmin(p_k_rq)
        self.kf_R = r[index_min]
        self.sigma_omega = qs[index_min]
        self.sigma_bias = qb[index_min]

    def get_prediction(self, x_est, P_est, step, u_ctrl=np.zeros(3)):
        new_x_k = self.attitude_discrete_model(x_est, u_ctrl, step)
        f_1 = self.attitude_jacobian_model(x_est, step)
        p1 = f_1.dot(P_est).dot(f_1.T)
        l_1 = self.noise_jacobian_model(x_est, step)
        p2 = l_1.dot(self.kf_Q).dot(l_1.T)
        new_P_k = p1 + p2
        return new_x_k, new_P_k

    def get_observer_prediction(self, new_x_k, reference_vector, save=True, sensor_type='mag'):
        z_k = self.attitude_observer_model(new_x_k, reference_vector) @ reference_vector
        return z_k

    def update_covariance_matrix(self, H, new_P_k, kf_R):
        if np.any(np.isnan(H)):
            print("H - NAN")
        self.kf_S = H @ (new_P_k @ H.T) + kf_R
        return self.kf_S

    def get_kalman_gain(self, H, new_P_k):
        if len(self.kf_S) > 1:
            try:
                s_inv = np.linalg.inv(self.kf_S)
            except Exception as error:
                print("{}: {}".format(error, self.kf_S))
                s_inv = np.zeros_like(self.kf_S)
            self.kf_K = new_P_k @ H.T @ s_inv
        else:
            s_inv = 1/self.kf_S
            self.kf_K = new_P_k.dot(H.T) * s_inv

    def update_state(self, new_x_k, z_k_medido, z_from_observer, H_):
        new_x = new_x_k + self.kf_K.dot(z_k_medido - z_from_observer + H_ @ new_x_k)
        #new_x[:3] = new_x[:3] / np.linalg.norm(new_x[:3])
        return new_x

    def update_covariance_P_matrix(self, H, new_P_k, kf_r):
        # full solution to minimizing p+ = (I - KH) @ p- @ (I - KH).T + K @ R @ K.T,
        # if K is optimized, then p+ = (I -KH) @ p-,
        # but it is not recommended when numerical instabilities are presents
        I_nn = np.eye(len(self.state))
        new_P = (I_nn - self.kf_K @ H) @ new_P_k#   @ (I_nn - self.kf_K @ H).T + self.kf_K @ self.kf_R @ self.kf_K.T
        return new_P

    def attitude_observer_model(self, new_x, vector_i) -> np.ndarray:
        pass

    def attitude_discrete_model(self, x, torque_b, dt) -> np.ndarray:
        pass

    def attitude_jacobian_model(self, x, dt) -> np.ndarray:
        pass

    def noise_jacobian_model(self, x, dt) -> np.ndarray:
        pass


if __name__ == "__main__":
    import pandas as pd
    import matplotlib.pyplot as plt
    from src.kalman_filter.ekf_multy import MEKF

    NOISE = True

    inertia = np.array([[38478.678, 0, 0], [0, 38528.678, 0], [0, 0, 6873.717]]) * 1e-6
    model_data = pd.read_csv("spacecraft_52191U_dataset.csv")
    time_model = model_data['time[sec]'].values - model_data['time[sec]'].values[0]
    mag_i_model = model_data[['mag_x_i[nT]', 'mag_y_i[nT]', 'mag_z_i[nT]']].values
    mag_b_model = model_data[['mag_x_b[nT]', 'mag_y_b[nT]', 'mag_z_b[nT]']].values
    gyro_b_model = model_data[['omega_t_b(X)[rad/s]', 'omega_t_b(Y)[rad/s]', 'omega_t_b(Z)[rad/s]']].values
    q_i2b_model = model_data[['q_t_i2b(0)[-]', 'q_t_i2b(1)[-]', 'q_t_i2b(2)[-]', 'q_t_i2b(3)[-]']].values
    sun_i_model = model_data[['Sun_pos_from_sc_i(X)[m]', 'Sun_pos_from_sc_i(Y)[m]', 'Sun_pos_from_sc_i(Z)[m]']].values
    sun_b_model = model_data[['Sun_pos_from_sc_b(X)[m]', 'Sun_pos_from_sc_b(Y)[m]', 'Sun_pos_from_sc_b(Z)[m]']].values

    mrp_i2b_model = np.array([get_mrp_from_q(q_) for q_ in q_i2b_model])
    mrp_i2b_mag = np.linalg.norm(mrp_i2b_model, axis=1)
    mag_from_mrp = np.array([dcm_from_mrp(p_) @ mag_i_ for p_, mag_i_ in zip(mrp_i2b_model, mag_i_model)])
    mag_from_q = np.array([Quaternions(q_).frame_conv(mag_i_) for q_, mag_i_ in zip(q_i2b_model, mag_i_model)])

    error_mrp = mean_squared_error(mag_b_model, mag_from_mrp)
    error_quat = mean_squared_error(mag_b_model, mag_from_q)

    print(error_mrp, error_quat)

    if NOISE:
        mag_b_sensor = mag_b_model + np.random.normal(0, 0.01 * np.mean(
            np.linalg.norm(mag_i_model, axis=1)) * np.ones_like(mag_b_model))
        sun_b_sensor = sun_b_model + np.random.normal(0, 0.01 * np.mean(
            np.linalg.norm(sun_b_model, axis=1)) * np.ones_like(sun_b_model))
        gyro_b_sensor = gyro_b_model + np.random.normal(0, 0.03 * np.pi / 180 * np.ones_like(gyro_b_model))
        # gyro_b_sensor += (np.array([15, -15, 10]) * np.deg2rad(1))
        sd_k_mag = .06 ** 2
        sd_k_sun = .06 ** 2
        sd_q = 0.03 * np.deg2rad(1)
        sd_p = 1.0
        sd_u = 0.001
        sd_v = 0.001
    else:
        mag_b_sensor = mag_b_model.copy()
        gyro_b_sensor = gyro_b_model.copy()
        sun_b_sensor = sun_b_model.copy()
        sd_k_mag = 0.01 ** 2
        sd_k_sun = 0.01 ** 2
        sd_q = 0.03 * np.deg2rad(1)
        sd_p = 0.5
        sd_u = 0.001
        sd_v = 0.001

    dt = 0.1
    ct = 0
    hist_state = []
    k = 1
    P_k = np.diag(np.array([0.5, 0.5, 0.5, 0.01, 0.01, 0.01]))
    # P_k = np.random.normal(0, sd_p, size=(6, 6))
    tend = time_model[-1]

    ekf_mrp = MEKF(inertia, R=np.eye(3) * 0.0, Q=np.eye(6) * sd_q, P=P_k)
    ekf_mrp.set_first_state(np.zeros(6))
    ekf_mrp.set_gyro_measure(gyro_b_sensor[0])
    ekf_mrp.set_quat(q_i2b_model[0] + np.random.normal(0, 0.2, size=4) * float(NOISE), save=True)
    ekf_mrp.sigma_bias = 0.0 if not NOISE else sd_u
    ekf_mrp.sigma_omega = 0.0 if not NOISE else sd_v

    hist_cov_P = [P_k.flatten()]
    hist_state.append(ekf_mrp.get_state())
    count_omega = 1
    measure_cycle_omega = 0.1 / dt
    count_idx_omega = 1
    fixed_vector = np.array([Quaternions(q_).frame_conv(np.array([1, 0, 0])) for q_ in q_i2b_model])
    while ct <= tend:
        print("time: {}".format(ct))
        # EKF prediction
        ekf_mrp.propagate(dt)
        if count_omega % measure_cycle_omega == 0:
            # mag injection
            ekf_mrp.inject_vector(mag_b_sensor[k] / np.linalg.norm(mag_b_sensor[k]),
                                  mag_i_model[k] / np.linalg.norm(mag_i_model[k]),
                                  sigma2=sd_k_mag)
            # # sun injection
            ekf_mrp.inject_vector(sun_b_sensor[k] / np.linalg.norm(sun_b_sensor[k]),
                                  sun_i_model[k] / np.linalg.norm(sun_i_model[k]),
                                  sigma2=sd_k_sun)
            # fixed
            # ekf_mrp.inject_vector_6(mag_b_sensor[k] / np.linalg.norm(mag_b_sensor[k]),
            #                         mag_i_model[k] / np.linalg.norm(mag_i_model[k]),
            #                         sun_b_model[k] / np.linalg.norm(sun_b_model[k]),
            #                         sun_i_model[k] / np.linalg.norm(sun_i_model[k]),
            #                         sigma1=sd_k_mag,
            #                         sigma2=sd_k_sun)
            # ekf_mrp.inject_vector(fixed_vector[k] / np.linalg.norm(fixed_vector[k]),
            #                       np.array([1, 0, 0]),
            #                       sigma2=sd_k_sun)
            ekf_mrp.reset_state()

            count_omega = 1
            ekf_mrp.set_gyro_measure(gyro_b_sensor[k])
            # ekf_mrp.set_quat(q_i2b_model[k])
            k += 1
        count_omega += 1
        ct += dt
        ct = np.round(ct, 4)
        hist_state.append(ekf_mrp.get_state())
        hist_cov_P.append(ekf_mrp.covariance_P.flatten())

    # bias_est = np.array([elem[3:] for elem in hist_state])
    # sigma_mrp = np.array([elem[:3] for elem in hist_state])
    bias_est = np.array(ekf_mrp.historical['b'])
    quat_est = np.array(ekf_mrp.historical['q'])

    sigma_mrp = np.array([get_mrp_from_q(q_) for q_ in quat_est])

    est_mag_b = [dcm_from_mrp(p_) @ mag_i_ for p_, mag_i_ in zip(sigma_mrp,
                                                                         mag_i_model)]
    est_mag_b = np.array(est_mag_b)
    est_sun_b = [dcm_from_mrp(p_) @ sun_i_ for p_, sun_i_ in zip(sigma_mrp,
                                                                         sun_i_model)]
    est_sun_b = np.array(est_sun_b)

    plt.figure()
    plt.plot(time_model, hist_cov_P)
    plt.grid()

    error_quat = np.array([(Quaternions(q_t) * Quaternions(Quaternions(q_e).conjugate()))()
                          for q_t, q_e in zip(q_i2b_model, quat_est)])
    error_theta = 2 * np.arccos(error_quat[:, 3])
    error_theta[error_theta > np.pi] = 2 * np.pi - error_theta[error_theta > np.pi]

    plt.figure()
    plt.plot(time_model, mrp_i2b_mag - np.linalg.norm(sigma_mrp, axis=1), label='error')
    plt.legend()
    plt.grid()

    plt.figure()
    plt.plot(time_model, error_theta * np.rad2deg(1), '.', label='error theta')
    plt.legend()
    plt.grid()

    plt.figure()
    plt.plot(time_model, bias_est, label=['bx', 'by', 'bz'])
    plt.legend()

    plt.figure()
    plt.plot(est_mag_b, ls='--', label='est')
    plt.plot(mag_b_sensor[:len(sigma_mrp)], label='sensor')
    plt.legend()

    plt.figure()
    plt.plot(est_sun_b, ls='--', label='sun est')
    plt.plot(sun_b_model[:len(sigma_mrp)], label='sun')
    plt.legend()

    fig_, ax = plt.subplots(3, 1)
    ax[0].plot(time_model, mrp_i2b_model[:, 0], label=['sx'])
    ax[0].plot(time_model, sigma_mrp[:, 0], label=['kf sx'])
    ax[1].plot(time_model, mrp_i2b_model[:, 1], label=['sy'])
    ax[1].plot(time_model, sigma_mrp[:, 1], label=['kf sy'])
    ax[2].plot(time_model, mrp_i2b_model[:, 2], label=['sz'])
    ax[2].plot(time_model, sigma_mrp[:, 2], label=['kf sz'])
    plt.legend()

    fig_q, axq = plt.subplots(4, 1)
    axq[0].plot(time_model, q_i2b_model[:, 0], label=['qx'])
    axq[0].plot(time_model, quat_est[:, 0], label=['kf qx'])
    axq[1].plot(time_model, q_i2b_model[:, 1], label=['qy'])
    axq[1].plot(time_model, quat_est[:, 1], label=['kf qy'])
    axq[2].plot(time_model, q_i2b_model[:, 2], label=['qz'])
    axq[2].plot(time_model, quat_est[:, 2], label=['kf qz'])
    axq[3].plot(time_model, q_i2b_model[:, 3], label=['qs'])
    axq[3].plot(time_model, quat_est[:, 3], label=['kf qs'])
    plt.legend()

    plt.figure()
    plt.plot(time_model, mrp_i2b_mag, ls='dotted', label='mrp true')
    plt.plot(time_model, np.linalg.norm(sigma_mrp, axis=1), label='mrp est')
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(time_model, mrp_i2b_model, label=['sx', 'sy', 'sz'], lw=0.7)
    plt.plot(time_model, mrp_i2b_mag, label='Magnitude', lw=1.2, color='red')
    plt.legend()

    plt.figure()
    plt.plot(time_model, gyro_b_model, label=['omega_x_i', 'omega_y_i', 'omega_z_i'])
    plt.plot(time_model, gyro_b_sensor, label=['gyro_x_i', 'gyro_y_i', 'gyro_z_i'])
    plt.legend()

    plt.figure()
    plt.plot(time_model, mag_i_model, label=['mag_x_i', 'mag_y_i', 'mag_z_i'])
    plt.plot(time_model, mag_b_model, label=['mag_x_b', 'mag_y_b', 'mag_z_b'])
    plt.legend()

    plt.figure()
    plt.plot(time_model, mag_b_model, label=['mag_x_b', 'mag_y_b', 'mag_z_b'])
    plt.plot(time_model, mag_b_sensor, label=['mag_x_b_s', 'mag_y_b_s', 'mag_z_b_s'])
    plt.legend()
    plt.show()