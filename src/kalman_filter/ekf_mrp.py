"""
Created by Elias Obreque
Date: 10-09-2023
email: els.obrq@gmail.com
"""

from ..dynamics.Quaternion import Quaternions
from sklearn.metrics import mean_squared_error
from tools.mathtools import *
from .ekf import EKF


# TODO: need refactoring
class MRP_EKF(EKF):
    # https://ntrs.nasa.gov/api/citations/19960035754/downloads/19960035754.pdf
    """
    modified Rogriguez parameters and gyro bias correction
    p = MRP
    b = bias

        def update(self, step, current_measure=None):
            new_x_k, new_P_k = self.get_prediction(self.state, self.covariance_P, step)

            if current_measure is not None:
                self.current_measure = current_measure
                new_z_k = self.get_observer_prediction(new_x_k)
                self.update_covariance_matrix(new_P_k)
                self.get_kalman_gain(new_x_k, new_P_k)
                new_x = self.update_state(new_x_k, current_measure, new_z_k)
                new_P = self.update_covariance_P_matrix(new_P_k)
                self.state = new_x
                self.covariance_P = new_P
            else:
                self.state = new_x_k
                self.covariance_P = new_P_k
            return
    """

    def __init__(self, inertia, R, Q, P):
        super().__init__(inertia, R, Q, P)
        self.current_measure = np.zeros(3)
        self.omega_state = np.zeros(3)
        self.reference_vector = np.zeros(3)
        self.current_quaternion = np.zeros(4)
        self.current_bias = np.zeros(3)
        self.sigma_v = 0
        self.sigma_u = 0
        self.historical = {'q_est': [], 'b': [np.zeros(3)], 'mag_est': []}

    def add_reference_vector(self, vector):
        self.reference_vector = vector

    def set_gyro_measure(self, value):
        self.omega_state = value

    def set_quat(self, value, save=False):
        value = value / np.linalg.norm(value)
        self.current_quaternion = value
        if save:
            self.historical['q_est'].append(self.current_quaternion)

    def get_prediction(self, x_est, P_est, step, u_ctrl=np.zeros(3), measure=None):
        # new_x_k = self.attitude_discrete_mrp(x_est, u_ctrl, step)
        # quat
        omega = self.omega_state - self.current_bias
        self.current_quaternion = self.attitude_discrete_q(self.current_quaternion, omega, step)
        new_x_k = np.zeros(6)
        new_P_k = self.propagate_cov_P_sim(step, omega)
        return new_x_k, new_P_k

    def propagate_cov_P_sim(self, step, omega):
        f_x = np.zeros((6, 6))
        f_x[:3, :3] = -skew(omega)
        f_x[:3, 3:] = -np.identity(3)

        self.kf_Q[:3, :3] = np.identity(3) * (self.sigma_v ** 2 * step + 1/3 * self.sigma_u ** 2 * step ** 3)
        self.kf_Q[3:, 3:] = np.identity(3) * self.sigma_u ** 2 * step
        phi = (np.eye(6) + f_x) * step
        new_p_k = phi.dot(self.covariance_P).dot(phi.T) + self.kf_Q
        return new_p_k

    def propagate_cov_P(self, step, omega):
        F_x = np.zeros((6, 6))
        F_x[:3, :3] = self.get_discrete_theta(step, omega)
        F_x[:3, 3:] = self.get_discrete_psi(step, omega)
        F_x[3:, 3:] = np.identity(3)

        u_x = skew(omega)
        self.kf_Q[:3, :3] = np.identity(3) * (self.sigma_v ** 2 * step + 1/3 * self.sigma_u ** 2 * step ** 3)
        self.kf_Q[3:, :3] = - np.identity(3) * 0.5 * self.sigma_u ** 2 * step ** 2
        self.kf_Q[:3, 3:] = - np.identity(3) * 0.5 * self.sigma_u ** 2 * step ** 2
        self.kf_Q[3:, 3:] = np.identity(3) * self.sigma_u ** 2 * step
        new_p_k = F_x @ self.covariance_P @ F_x.T + F_x @ self.kf_Q @ F_x.T
        return new_p_k

    def propagate_cov_P_dep(self, x_est, step, p_est_):
        f_ = self.attitude_jacobian_model(x_est, step)
        p1 = f_.dot(p_est_.dot(f_.T))
        g_ = self.noise_jacobian_model(x_est, step)
        p2 = g_.dot(self.kf_Q).dot(g_.T)
        p_ = p1 + p2
        new_P_k_ = p_ * step
        return new_P_k_

    def get_discrete_theta(self, dt, omega):
        rot = np.linalg.norm(omega * dt)
        mag = rot/dt
        if rot != 0:
            u_x = skewsymmetricmatrix(omega)
            u_x2 = u_x @ u_x
            theta = np.identity(3) - u_x * dt + 0.5 * u_x2 * dt**2
        else:
            theta = np.identity(3)
        return theta

    def get_discrete_psi(self, dt, omega):
        rot = np.linalg.norm(omega * dt)
        mag = rot / dt
        if rot != 0:
            omega_x = skewsymmetricmatrix(omega)
            omega_x2 = omega_x @ omega_x
            psi = - np.identity(3) * dt + 0.5 * omega_x * dt**2 - 1/6 * omega_x2 * dt**3
        else:
            psi = - np.identity(3) * dt
        return psi

    def attitude_discrete_mrp(self, x_est, u_ctrl, step):
        new_est = x_est + runge_kutta_4(mrp_dot, np.array([*x_est[:3], *(self.omega_state - x_est[3:6])]), step)
        if np.any(np.isnan(new_est)):
            print("nan")
        new_est[3:] = x_est[3:6]
        if np.linalg.norm(new_est[:3]) > 1:
            new_est[:3] = get_shadow_set_mrp(new_est[:3])
        return new_est

    def attitude_jacobian_model(self, x_est, step):
        f_ = np.zeros((6, 6))
        p = x_est[:3]
        gyro_measure = self.omega_state
        omega = gyro_measure - self.current_bias
        f_p = 0.5 * (np.multiply(p, omega.reshape(-1, 1)) - np.multiply(omega, p.reshape(-1, 1)) -
                     skew(omega) + omega.dot(p) * np.eye(3))
        f_b = - 0.5 * Bmatrix_mrp(x_est[:3])
        f_[:3, :3] = f_p
        f_[:3, 3:] = f_b
        return f_

    def noise_jacobian_model(self, x_est, step):
        g_ = np.zeros((6, 6))
        g_[:3, :3] = - 0.25 * Bmatrix_mrp(x_est[:3])
        g_[3:, 3:] = np.eye(3)
        return g_

    def get_observer_prediction(self, new_x_k, reference_vector):
        p_k = new_x_k[:3]
        new_z_k = dcm_from_mrp(p_k) @ reference_vector
        self.historical['mag_est'].append(new_z_k)
        return new_z_k

    def attitude_observer_model(self, new_x, vector_i):
        p_ = new_x[:3]
        H = np.zeros((3, 6))
        temp1 = 4 / (1 + np.linalg.norm(p_))**2
        temp2 = skew(dcm_from_mrp(p_) @ vector_i)
        temp3 = (1 + np.linalg.norm(p_)**2) * np.eye(3) - 2 * skew(p_) + 2 * np.outer(p_, p_)
        L = temp1 * temp2 @ temp3
        H[:3, :3] = L
        return H

    def update_state(self, new_x_k, z_k_medido, z_from_observer):
        new_x = np.zeros(6)
        error = z_k_medido - z_from_observer
        correction = self.kf_K @ error
        if np.any(np.isnan(correction)):
            print("correction: {}".format(correction))
        new_x[:3] = add_mrp(correction[:3], new_x_k[:3])
        new_x[3:] = new_x_k[3:] + correction[3:]
        if np.linalg.norm(new_x[:3]) > 1:
            new_x[:3] = get_shadow_set_mrp(new_x[:3])
        return new_x

    def reset_state(self):
        dot_error = self.internal_state[:3] @ self.internal_state[:3]
        if dot_error < 1:
            error_q = Quaternions(np.array([*self.internal_state[:3] * 0.5,
                                            np.sqrt(1 - dot_error)]))
        else:
            error_q = Quaternions(np.array([*self.internal_state[:3] * 0.5, 1]) / np.sqrt(1 + dot_error))
        error_q.normalize()
        # diff = error_q * Quaternions(self.current_quaternion)
        current_quaternion = error_q * Quaternions(self.current_quaternion)
        current_quaternion.normalize()
        self.current_quaternion = current_quaternion()
        # self.current_bias += self.internal_state[3:]
        self.covariance_P = self.internal_cov_P
        self.state = np.zeros(6)
        self.internal_state = np.zeros(6)
        self.historical['q_est'].append(self.current_quaternion)
        self.historical['b'].append(self.current_bias)


if __name__ == '__main__':
    pass
