"""
Created by Elias Obreque
Date: 04-09-2023
email: els.obrq@gmail.com
"""
from src.kalman_filter.ekf import EKF
import numpy as np
from src.dynamics.dynamics_kinematics import calc_omega_b
import matplotlib.pyplot as plt


class EKFOmega(EKF):
    def __init__(self, inertia, R, Q, P):
        EKF.__init__(self, inertia, R, Q, P)
        self.inertia = inertia
        self.inv_inertia = np.linalg.inv(self.inertia)
        self.observation_matrix = np.concatenate((np.eye(3), np.eye(3))).T
        self.historical = []
        self.historical.append(np.diag(self.covariance_P.copy()))

    def update(self, step, current_measure=None, reference=None):
        new_x_k, new_P_k = self.get_prediction(self.state, self.covariance_P, step)
        self.current_measure = current_measure
        new_z_k = self.observation_matrix @ new_x_k
        self.update_covariance_matrix(self.observation_matrix, new_P_k)
        self.get_kalman_gain(self.observation_matrix, new_P_k)
        new_x = self.update_state(new_x_k, current_measure, new_z_k, 0 * self.observation_matrix)
        new_P = self.update_covariance_P_matrix(self.observation_matrix, new_P_k)
        self.state = new_x
        self.covariance_P = new_P
        self.historical.append(np.diag(self.covariance_P.copy()))

    def update_state(self, new_x_k, z_k_medido, z_from_observer, H_):
        return new_x_k + self.kf_K.dot(z_k_medido - z_from_observer + H_ @ new_x_k)

    def attitude_discrete_model(self, x, torque_b, dt):
        new_x = np.zeros(len(x))
        x_omega_b = x[:3]
        s_omega = self.skewsymmetricmatrix(x_omega_b)
        h_total_b = self.inertia.dot(x_omega_b)
        # new_x[:3] = x_omega_b - dt * self.inv_inertia.dot(s_omega.dot(h_total_b) - torque_b)
        new_x[:3] = calc_omega_b(x_omega_b, dt, self.inertia)
        new_x[3:] = x[3:]
        return new_x

    def attitude_jacobian_model(self, x, dt):
        x_omega_b = x[:3]
        f_dx = np.zeros_like(self.covariance_P)
        f_dx[:3, :3] = np.identity(3) - dt * self.inv_inertia.dot(
            self.skewsymmetricmatrix(x_omega_b).dot(self.inertia) - self.skewsymmetricmatrix(self.inertia.dot(x_omega_b)))
        f_dx[3:, 3:] = np.eye(3)
        return f_dx

    def noise_jacobian_model(self, x, dt):
        return np.identity(6) * dt

    def attitude_observer_model(self, new_x, vector_i):
        H = np.zeros((2, 3))
        H[0, 0] = 1
        H[1, 1] = 1
        return H

    def plot_cov(self, time_, folder_save):
        fig = plt.figure()
        plt.title("Covariance P - EKF")
        plt.plot(time_, np.array(self.historical))
        plt.grid()
        plt.legend([r"$\omega_x$", "$\omega_y$", "$\omega_z$", "$b_x$", "$b_y$", "$b_z$"])
        plt.xlabel("Modified Julian Date")
        plt.xticks(rotation=15)
        plt.ticklabel_format(useOffset=False)
        plt.yscale('log')
        plt.tight_layout()
        fig.savefig(folder_save + 'ekf_omega_covariance.jpg')
        plt.close(fig)

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


if __name__ == '__main__':
    pass
