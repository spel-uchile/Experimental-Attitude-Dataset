"""
Created by Elias Obreque
Date: 10-09-2023
email: els.obrq@gmail.com
"""
from ..dynamics.Quaternion import Quaternions
from sklearn.metrics import mean_squared_error
from tools.mathtools import *
from .ekf import EKF


class MEKF(EKF):
    # https://ntrs.nasa.gov/api/citations/19960035754/downloads/19960035754.pdf
    """
    """

    def __init__(self, inertia, R, Q, P):
        super().__init__(inertia, R, Q, P)
        self.current_measure = np.zeros(3)
        self.omega_state = np.zeros(3)
        self.reference_vector = np.zeros(3)
        self.current_quaternion = np.zeros(4)
        self.current_bias = np.zeros(3)
        self.sigma_omega = 0
        self.sigma_bias = 0
        self.historical = {'q_est': [], 'b_est': [np.zeros(3)], 'mag_est': [], 'omega_est': [],
                           'p_cov': [self.covariance_P.flatten()]}

    def add_reference_vector(self, vector):
        self.reference_vector = vector

    def set_gyro_measure(self, value):
        self.omega_state = value
        print(self.omega_state - self.current_bias)
        self.historical['omega_est'].append(self.omega_state - self.current_bias)

    def set_quat(self, value, save=False):
        value = value / np.linalg.norm(value)
        self.current_quaternion = value
        if save:
            self.historical['q_est'].append(self.current_quaternion)

    def get_prediction(self, x_est, P_est, step, u_ctrl=np.zeros(3), measure=None):
        omega = self.omega_state - self.current_bias
        self.current_quaternion = self.attitude_discrete(self.current_quaternion, omega, step)
        new_x_k = np.zeros(6)
        new_P_k = self.propagate_cov_P_sim(step, omega)
        return new_x_k, new_P_k

    def propagate_cov_P_sim(self, step, omega):
        f_x = np.zeros((6, 6))
        f_x[:3, :3] = -skew(omega)
        f_x[:3, 3:] = -np.identity(3)

        self.kf_Q[:3, :3] = np.identity(3) * (self.sigma_omega ** 2 * step + 1 / 3 * self.sigma_bias ** 2 * step ** 3)
        self.kf_Q[3:, 3:] = np.identity(3) * self.sigma_bias ** 2 * step
        self.kf_Q[:3, 3:] = - np.identity(3) * 0.5 * self.sigma_bias ** 2 * step ** 2
        self.kf_Q[3:, :3] = - np.identity(3) * 0.5 * self.sigma_bias ** 2 * step ** 2

        phi = (np.eye(6) + f_x + 0.5 * f_x @ f_x * step) * step
        new_p_k = phi.dot(self.covariance_P).dot(phi.T) + self.kf_Q
        return new_p_k

    def propagate_cov_P(self, step, omega):
        F_x = np.zeros((6, 6))
        F_x[:3, :3] = self.get_discrete_theta(step, omega)
        F_x[:3, 3:] = self.get_discrete_psi(step, omega)
        F_x[3:, 3:] = np.identity(3)

        self.kf_Q[:3, :3] = np.identity(3) * (self.sigma_omega ** 2 * step + 1 / 3 * self.sigma_bias ** 2 * step ** 3)
        self.kf_Q[3:, :3] = - np.identity(3) * 0.5 * self.sigma_bias ** 2 * step ** 2
        self.kf_Q[:3, 3:] = - np.identity(3) * 0.5 * self.sigma_bias ** 2 * step ** 2
        self.kf_Q[3:, 3:] = np.identity(3) * self.sigma_bias ** 2 * step
        new_p_k = F_x @ self.covariance_P @ F_x.T + F_x @ self.kf_Q @ F_x.T
        return new_p_k

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

    def attitude_discrete(self, current_quaternion, omega, step):
        def q_dot(x):
            x_quaternion_i2b = x
            omega4 = omega4kinematics(omega)
            q_dot = 0.5 * omega4 @ x_quaternion_i2b
            return q_dot

        new_q = current_quaternion + runge_kutta_4(q_dot, current_quaternion, step)
        new_q /= np.linalg.norm(new_q)
        return new_q

    def get_observer_prediction(self, new_x_k, reference_vector):
        new_z_k = Quaternions(self.current_quaternion).frame_conv(reference_vector)
        self.historical['mag_est'].append(new_z_k)
        new_z_k = new_z_k / np.linalg.norm(new_z_k)
        return new_z_k

    def attitude_observer_model(self, new_x, vector_i):
        H = np.zeros((3, 6))
        H[:3, :3] = skew(Quaternions(self.current_quaternion).frame_conv(vector_i))
        return H

    def update_state(self, new_x_k, z_k_medido, z_from_observer):
        error = z_k_medido - z_from_observer
        correction = self.kf_K @ error
        if np.any(np.isnan(correction)):
            print("correction: {}".format(correction))
        new_x = new_x_k + correction
        return new_x

    def reset_state(self):
        dot_error = self.internal_state[:3] @ self.internal_state[:3]
        # if dot_error < 1:
        #     error_q = Quaternions(np.array([*self.internal_state[:3] * 0.5,
        #                                     np.sqrt(1 - dot_error)]))
        # else:
        error_q = Quaternions(np.array([*self.internal_state[:3] * 0.5, 1]) / np.sqrt(1 + dot_error))
        error_q.normalize()
        # diff = error_q * Quaternions(self.current_quaternion)
        current_quaternion = Quaternions(self.current_quaternion) * error_q
        current_quaternion.normalize()
        self.current_quaternion = current_quaternion()
        self.current_bias += self.internal_state[3:]
        self.covariance_P = self.internal_cov_P
        self.state = np.zeros(6)
        self.internal_state = np.zeros(6)
        self.historical['q_est'].append(self.current_quaternion)
        self.historical['b_est'].append(self.current_bias.copy())
        self.historical['p_cov'].append(self.covariance_P.flatten())


if __name__ == '__main__':
    pass
