"""
Created by Elias Obreque
Date: 10-09-2023
email: els.obrq@gmail.com
"""
import numpy as np

from ..dynamics.Quaternion import Quaternions
from sklearn.metrics import mean_squared_error
from tools.tools import *
from .ekf import EKF


class MEKF_FULL(EKF):
    # https://arc.aiaa.org/doi/abs/10.2514/2.4834?journalCode=jgcd
    """
    """

    def __init__(self, inertia, R, Q, P):
        super().__init__(inertia, R, Q, P)
        self.current_measure = np.zeros(3)
        self.omega_m = np.zeros(3)
        self.current_quaternion = np.zeros(4)

        self.theta_error = np.zeros(3)
        self.current_bias = np.zeros(3)
        self.scale = np.zeros(3)
        self.k_l = np.zeros(3)
        self.k_u = np.zeros(3)
        self.sigma2_elements = np.ones(5) * 100

        self.historical = {'q_est': [], 'b': [np.zeros(3)], 'mag_est': [], 'omega_est': [],
                           'scale': [np.zeros(3)], 'ku': [np.zeros(3)], 'kl': [np.zeros(3)]}

    def set_gyro_measure(self, value):
        self.omega_m = value
        self.historical['omega_est'].append(self.get_calibrate_omega())

    def get_calibrate_omega(self):
        return (np.eye(3) - self.get_scale()) @ (self.omega_m - self.current_bias)

    def set_quat(self, value, save=False):
        value = value / np.linalg.norm(value)
        self.current_quaternion = value
        if save:
            self.historical['q_est'].append(self.current_quaternion)

    def get_prediction(self, x_est, P_est, step, u_ctrl=np.zeros(3), measure=None):
        omega = self.get_calibrate_omega()
        self.current_quaternion = self.attitude_discrete(self.current_quaternion, omega, step)
        new_x_k = np.zeros(15)
        new_P_k = self.propagate_cov_P_sim(step, omega)
        return new_x_k, new_P_k

    def propagate_cov_P_sim(self, step, omega):
        f_x = np.zeros((15, 15))
        f_x[:3, :3] = -skew(omega)
        f_x[:3, 3:6] = -(np.identity(3) - self.get_scale())
        f_x[:3, 6:9] = -np.diag(self.omega_m - self.current_bias)
        f_x[:3, 9:12] = -self.get_u()
        f_x[:3, 12:] = -self.get_l()

        g_x = np.eye(15)
        g_x[:3, :3] = -2 * np.eye(3) + self.get_scale()

        q_ = np.eye(15) * np.kron(self.sigma2_elements, np.ones(3))

        phi = (np.eye(15) + f_x + 0.5 * f_x @ f_x * step) * step

        kf_q = g_x @ q_ @ g_x.T * step
        new_p_k = phi.dot(self.covariance_P).dot(phi.T) + kf_q
        return new_p_k

    def get_u(self):
        temp = self.omega_m - self.current_bias
        u_ = np.zeros((3, 3))
        u_[0, 0] = temp[1]
        u_[0, 1] = temp[2]
        u_[1, 2] = temp[2]
        return u_

    def get_l(self):
        temp = self.omega_m - self.current_bias
        l_ = np.zeros((3, 3))
        l_[1, 0] = temp[0]
        l_[2, 1] = temp[0]
        l_[2, 2] = temp[1]
        return l_

    def get_scale(self):
        s = np.diag(self.scale)
        s[0, 1] = self.k_u[0]
        s[0, 2] = self.k_u[1]
        s[1, 2] = self.k_u[2]

        s[1, 0] = self.k_l[0]
        s[2, 0] = self.k_l[1]
        s[2, 1] = self.k_l[2]
        return s

    def propagate_cov_P(self, step, omega):
        F_x = np.zeros((6, 6))
        F_x[:3, :3] = self.get_discrete_theta(step, omega)
        F_x[:3, 3:] = self.get_discrete_psi(step, omega)
        F_x[3:, 3:] = np.identity(3)

        self.kf_Q[:3, :3] = np.identity(3) * (self.sigma_v ** 2 * step + 1/3 * self.sigma_u ** 2 * step ** 3)
        self.kf_Q[3:, :3] = - np.identity(3) * 0.5 * self.sigma_u ** 2 * step ** 2
        self.kf_Q[:3, 3:] = - np.identity(3) * 0.5 * self.sigma_u ** 2 * step ** 2
        self.kf_Q[3:, 3:] = np.identity(3) * self.sigma_u ** 2 * step
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
        H = np.zeros((3, 15))
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
        # theta error
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
        # bias error
        self.current_bias += self.internal_state[3:6]
        # scale error
        self.scale += self.internal_state[6:9]
        # kup
        self.k_u += self.internal_state[9:12]
        # klp
        self.k_l += self.internal_state[12:]
        self.covariance_P = self.internal_cov_P
        self.state = np.zeros(15)
        self.internal_state = np.zeros(15)

        self.historical['q_est'].append(self.current_quaternion)
        self.historical['b'].append(self.current_bias)
        self.historical['scale'].append(self.scale)
        self.historical['ku'].append(self.k_u)
        self.historical['kl'].append(self.k_l)


if __name__ == '__main__':
    import numpy as np
    from src.dynamics.dynamics_kinematics import calc_quaternion
    from src.dynamics.Quaternion import Quaternions

    tend = 90 * 60
    dt = 1
    R = np.eye(3) * 36  # arc sec2
    s_true = np.diag([1500, 1000, 1500]) # ppm
    s_true[0, 1] = 1000
    s_true[0, 2] = 1500
    s_true[1, 2] = 2000

    s_true[1, 0] = 500
    s_true[2, 0] = 1000
    s_true[2, 1] = 1500

    s_true *= 1e-6

    bias_true = np.ones(3) * 0.1 * np.deg2rad(1)
    sigma_bias = np.sqrt(10) * 1e-10
    sigma_omega = np.sqrt(10) * 1e-7
    sigma_scale = 0
    sigma_ku = 0
    sigma_kl = 0


    def get_error_quat(q_m, q_p):
        q_m_c = Quaternions(q_m)
        q_p_c = Quaternions(q_p)
        q_temp = q_m_c * Quaternions(q_p_c.conjugate())
        theta_error = 2 * q_temp()[:3] / q_temp()[3]
        return theta_error

    def omega_true(t_):  # deg/sec - rad/sec
        return np.array([np.sin(0.01 * t_),
                         np.sin(0.0085 * t_),
                         np.cos(0.0085 * t_)]) * 0.1 * np.deg2rad(1)

    def gyro_model(t_, old_bias):
        new_bias = old_bias + sigma_bias * dt ** 0.5 * np.random.normal(0, 1, size=3)
        ot = omega_true(t_)
        return (((np.eye(3) + s_true) @ ot + 0.5 * (new_bias + old_bias) +
                (sigma_omega ** 2 / dt + 1/12 * sigma_bias**2 * dt) ** 0.5 * np.random.normal(0, 1, size=3)),
                new_bias)


    P = np.ones((15, 15))
    P[:3, :3] = (6 / 3600) ** 2 * np.eye(3)
    P[3:6, 3:6] = (0.2 / 3600) ** 2 * np.eye(3)
    P[6:9, 6:9] = (0.002 / 3) ** 2 * np.eye(3)
    P[9:12, 9:12] = (0.002 / 3) ** 2 * np.eye(3)
    P[12:, 12:] = (0.002 / 3) ** 2 * np.eye(3)

    q0 = np.sqrt(2) / 2 * np.array([1, 0, 0, 1])
    ekf_full_cal = MEKF_FULL(None, R, Q=np.zeros(15), P=P)
    bias_true, gyro_t = gyro_model(0, bias_true)
    ekf_full_cal.set_gyro_measure(gyro_model(0, bias_true))
    ekf_full_cal.set_quat(q0, save=True)

    historical = {'q_true': [q0], 'omega_true': [omega_true(0)]}
    ct = 0
    while ct <= tend:
        new_q_true = calc_quaternion(q0, omega_true(ct), dt)
        ekf_full_cal.propagate(dt)

        ekf_full_cal.inject_vector()

        ct += dt

        q0 = new_q_true
        historical['q_true'].append(q0)
        historical['omega_true'].append(omega_true(ct))





