"""
Created by Elias Obreque
Date: 10-09-2023
email: els.obrq@gmail.com
"""
# UKF algorithm for nonlinear state estimation

import numpy as np
from filterpy.kalman import UnscentedKalmanFilter as UKF_
from filterpy.kalman import unscented_transform, MerweScaledSigmaPoints
from ..dynamics.Quaternion import Quaternions


class UKF(UKF_):
    def __init__(self, cov_p=None, dt=0.1, x0=None, dim_x=2, dim_z=1, alpha=0.3, beta=2, kappa=0.1):
        sigmas = MerweScaledSigmaPoints(dim_x, alpha, beta, kappa)
        super().__init__(dim_x=dim_x, dim_z=dim_z, fx=self.f_cv, hx=self.h_cv, dt=dt, points=sigmas)
        self.x = x0
        self.R = None
        self.dt = dt
        self.covariance_P = cov_p
        self.current_measure = np.zeros(3)
        self.omega_state = np.zeros(3)
        self.reference_vector = np.zeros(3)
        self.current_quaternion = np.zeros(4)
        self.current_bias = np.zeros(3)
        self.sigma_omega = 0
        self.sigma_bias = 0
        self.sensor_type = 'mag'
        self.historical = {'q_est': [], 'b_est': [np.zeros(3)], 'mag_est': [], 'omega_est': [],
                           'p_cov': [self.covariance_P.flatten()], 'css_est': []}

    def inject_vector(self, body_vec_, mag_ref_, sigma2=5000, sensor='mag'):
        self.sensor_type = sensor
        self.update(body_vec_)

    def f_cv(self):
        pass

    def h_cv(self, reference_vector):
        new_z_k = Quaternions(self.current_quaternion).frame_conv(reference_vector)
        if self.sensor_type == 'mag':
            new_z_k = new_z_k  # / np.linalg.norm(new_z_k)
        elif self.sensor_type == 'css':
            new_z_k = - 930 * np.eye(3) @ new_z_k / np.linalg.norm(new_z_k)
            new_z_k[new_z_k < 0] = 0
        return new_z_k

    def set_Q(self, sigma_omega, sigma_bias):
        step = self.dt
        self.Q = np.zeros((6, 6))
        self.Q[:3, :3] = np.identity(3) * (sigma_omega ** 2 * step + 1 / 3 * sigma_bias ** 2 * step ** 3)
        self.Q[3:, 3:] = np.identity(3) * sigma_bias ** 2 * step
        self.Q[:3, 3:] = - np.identity(3) * 0.5 * sigma_bias ** 2 * step ** 2
        self.Q[3:, :3] = - np.identity(3) * 0.5 * sigma_omega ** 2 * step ** 2

    def set_quat(self, value, save=False):
        value = value / np.linalg.norm(value)
        self.current_quaternion = value
        if save:
            self.historical['q_est'].append(self.current_quaternion)

    def save_vector(self, name=None, vector=None):
        self.historical[name].append(vector)

    def set_gyro_measure(self, value):
        self.omega_state = value
        print(self.omega_state - self.current_bias)
        self.historical['omega_est'].append(self.omega_state - self.current_bias)

    def get_observer_prediction(self, new_x_k, reference_vector, save=True, sensor_type='mag'):
        new_z_k = Quaternions(self.current_quaternion).frame_conv(reference_vector)
        if save:
            if sensor_type == 'mag':
                new_z_k = new_z_k # / np.linalg.norm(new_z_k)
            elif sensor_type == 'css':
                new_z_k = - 930 * np.eye(3) @ new_z_k / np.linalg.norm(new_z_k)
                new_z_k[new_z_k < 0] = 0
        return new_z_k


if __name__ == '__main__':
    pass
