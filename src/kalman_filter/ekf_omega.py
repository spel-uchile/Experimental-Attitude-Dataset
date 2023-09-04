"""
Created by Elias Obreque
Date: 04-09-2023
email: els.obrq@gmail.com
"""
from ekf import EKF
import numpy as np


class EKFOmega(EKF):
    def __init__(self, inertia, R, Q, P):
        EKF.__init__(self, inertia, R, Q, P)

    def attitude_discrete_model(self, x, torque_b, dt):
        new_x = np.zeros(len(x))
        x_omega_b = x
        s_omega = self.skewsymmetricmatrix(x_omega_b)
        h_total_b = self.inertia.dot(x_omega_b)
        new_x[:3] = x_omega_b - dt * self.inv_inertia.dot(s_omega.dot(h_total_b) - torque_b)
        return new_x[:3]

    def attitude_jacobian_model(self, x, dt):
        x_omega_b = x
        df1_dx = np.identity(3) - dt * self.inv_inertia.dot(
            self.skewsymmetricmatrix(x_omega_b).dot(self.inertia) - self.skewsymmetricmatrix(self.inertia.dot(x_omega_b)))
        return df1_dx

    def noise_jacobian_model(self, x, dt):
        return np.identity(3) * dt

    def attitude_observer_model(self, new_x, vector_i):
        H = np.zeros((2, 3))
        H[0, 0] = 1
        H[1, 1] = 1
        return H

    @staticmethod
    def skewsymmetricmatrix(x_omega_b):
        S_omega = np.zeros((3, 3))
        S_omega[1, 0] = x_omega_b[2]
        S_omega[2, 0] = -x_omega_b[1]

        S_omega[0, 1] = -x_omega_b[2]
        S_omega[0, 2] = x_omega_b[1]

        S_omega[2, 1] = x_omega_b[0]
        S_omega[1, 2] = -x_omega_b[0]
        return S_omega

    @staticmethod
    def omega4kinematics(x_omega_b):
        Omega = np.zeros((4,4))
        Omega[1, 0] = -x_omega_b[2]
        Omega[2, 0] = x_omega_b[1]
        Omega[3, 0] = -x_omega_b[0]

        Omega[0, 1] = x_omega_b[2]
        Omega[0, 2] = -x_omega_b[1]
        Omega[0, 3] = x_omega_b[0]

        Omega[1, 2] = x_omega_b[0]
        Omega[1, 3] = x_omega_b[1]

        Omega[2, 1] = -x_omega_b[0]
        Omega[2, 3] = x_omega_b[2]

        Omega[3, 1] = -x_omega_b[1]
        Omega[3, 2] = -x_omega_b[2]
        return Omega


if __name__ == '__main__':
    pass
