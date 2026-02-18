"""
Created by Elias Obreque
Date: 15/11/2025
email: els.obrq@gmail.com
"""
from matplotlib import pyplot as plt

from tools.mathtools import *


class MEKF_OBC:
    """

    https://ntrs.nasa.gov/api/citations/19960035754/downloads/19960035754.pdf
    https://arxiv.org/pdf/1711.02508
    https://arxiv.org/pdf/2110.13666
    https://mars.cs.umn.edu/tr/reports/Trawny05b.pdf
    """

    def __init__(self, cov_state):
        self.sigma_u_bias = 0
        self.kf_R_a = {'mag': None, 'css': None}
        self.dim_rows = 3
        self.dim_cols = 3
        self.kf_Phi = np.zeros((6, 6))
        self.kf_Q_cov = np.zeros((6, 6))
        self.kf_R = np.zeros((3, 3))
        self.kf_S = np.zeros([self.dim_rows, self.dim_cols])
        self.kf_K = np.zeros([self.dim_rows, self.dim_cols])
        self.s_hist = {'mag': [], 'css': []}
        self.S_hat_i = np.zeros([self.dim_rows, self.dim_cols])
        self.covariance_P = cov_state
        self.internal_cov_P = self.covariance_P.copy()
        self.state = np.zeros(len(self.covariance_P))
        self.internal_state = self.state.copy()

        self.current_measure = np.zeros(3)
        self.omega_state = np.zeros(3)
        self.reference_vector = np.zeros(3)
        self.current_quaternion = np.zeros(4)
        self.current_bias = np.zeros(3)
        self.sigma_v_omega = 0
        self.sigma_u_bias = 0
        self.tau = 1000
        self.gamma = 0

        self.bias_model = None #"GM"
        self.historical = {'mjd_ekf':[], 'q_est': [], 'b_est': [np.zeros(3)], 'mag_ref_est': [], 'omega_est': [],
                           'p_cov': [np.diag(self.covariance_P)], 'css_est': [], 'sun_b_est': [], 'earth_b_est': []}

    def propagate(self, step):
        self.internal_state, self.internal_cov_P = self.get_prediction(self.state, self.covariance_P, step)

    def inject_vector(self, vector_b, vector_i, gain=None, sensor='mag', sigma2=None):
        if sigma2 is not None:
            self.kf_R = sigma2 * np.eye(len(vector_i))

        new_z_k = self.get_observer_prediction(self.internal_state, vector_i, sensor_type=sensor)
        H = self.attitude_observer_model(self.internal_state, vector_i)
        if gain is not None:
            H[:3, :3] = gain @ H[:3, :3] / np.linalg.norm(vector_i)
        # print("residual MRSE error (deg): {}".format(mean_squared_error(new_z_k, vector_b)))
        # else:
            # print("residual angle error (deg): {}".format(
            #    np.rad2deg(1) * np.arccos(vector_b / np.linalg.norm(vector_b) @ new_z_k / np.linalg.norm(new_z_k))))

        r_sensor = self.kf_R
        s_k = self.update_covariance_matrix(H, self.internal_cov_P, r_sensor)
        self.s_hist[sensor] = s_k
        self.get_kalman_gain(H, self.internal_cov_P)
        self.internal_state = self.update_state(self.internal_state, vector_b, new_z_k, H)
        self.internal_cov_P = self.update_covariance_P_matrix(H, self.internal_cov_P, r_sensor)
        return new_z_k

    def update_covariance_matrix(self, H, new_P_k, kf_R):
        if np.any(np.isnan(H)):
            print("H - NAN")
        self.kf_S = H @ (new_P_k @ H.T) + kf_R
        return self.kf_S

    def get_kalman_gain(self, H, new_P_k):
        if len(self.kf_S) > 1:
            np.fill_diagonal(self.kf_S, self.kf_S.diagonal() + 1e-12)
            try:
                s_inv = np.linalg.inv(self.kf_S)
            except Exception as error:
                print("{}: {}".format(error, self.kf_S))
                s_inv = np.zeros_like(self.kf_S)
            self.kf_K = new_P_k @ H.T @ s_inv
        else:
            s_inv = 1/self.kf_S
            self.kf_K = new_P_k.dot(H.T) * s_inv

    def update_covariance_P_matrix(self, H, new_P_k, kf_r):
        # full solution to minimizing p+ = (I - KH) @ p- @ (I - KH).T + K @ R @ K.T,
        # if K is optimized, then p+ = (I -KH) @ p-,
        # but it is not recommended when numerical instabilities are presents
        I_nn = np.eye(len(self.state))
        new_P = (I_nn - self.kf_K @ H) @ new_P_k  @ (I_nn - self.kf_K @ H).T + self.kf_K @ self.kf_R @ self.kf_K.T
        return new_P

    def add_reference_vector(self, vector):
        self.reference_vector = vector

    def set_gyro_measure(self, value):
        self.omega_state = value
        # print("Current bias", self.current_bias, "Gyro", self.omega_state, "Omega est", self.omega_state - self.current_bias)
        self.historical['omega_est'].append(self.omega_state - self.current_bias)

    def save_time(self, value):
        self.historical['mjd_ekf'].append(value)

    def set_quat(self, value, save=False):
        value = value / np.linalg.norm(value)
        self.current_quaternion = value
        if save:
            self.historical['q_est'].append(self.current_quaternion)

    def get_prediction(self, x_est, p_est, step, u_ctrl=np.zeros(3), measure=None):
        omega = self.omega_state - self.current_bias
        self.current_quaternion = self.attitude_discrete(self.current_quaternion, omega, step)
        new_x_k = np.zeros(6)
        new_p_k = self.propagate_cov_P_full(step, omega)
        return new_x_k, new_p_k

    def propagate_cov_P_sim(self, step, omega):
        f_x = np.zeros((6, 6))
        f_x[:3, :3] = -skew(omega)
        f_x[:3, 3:] = -np.identity(3)

        self.kf_Q_cov[:3, :3] = np.identity(3) * (self.sigma_v_omega ** 2 * step + 1/3 * self.sigma_u_bias ** 2 * step ** 3)
        self.kf_Q_cov[3:, 3:] = np.identity(3) * self.sigma_u_bias ** 2 * step
        phi = (np.eye(6) + f_x) * step
        new_p_k = phi.dot(self.covariance_P).dot(phi.T) + self.kf_Q_cov
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

    def propagate_cov_P(self, step, omega):
        F_x = np.zeros((6, 6))
        F_x[:3, :3] = self.get_discrete_theta(step, omega)
        F_x[:3, 3:] = self.get_discrete_psi(step, omega)
        F_x[3:, 3:] = np.identity(3)

        u_x = skew(omega)
        self.kf_Q_cov[:3, :3] = np.identity(3) * (self.sigma_v_omega ** 2 * step + 1/3 * self.sigma_u_bias ** 2 * step ** 3)
        self.kf_Q_cov[3:, :3] = - np.identity(3) * 0.5 * self.sigma_u_bias ** 2 * step ** 2
        self.kf_Q_cov[:3, 3:] = - np.identity(3) * 0.5 * self.sigma_u_bias ** 2 * step ** 2
        self.kf_Q_cov[3:, 3:] = np.identity(3) * self.sigma_u_bias ** 2 * step
        new_p_k = F_x @ self.covariance_P @ F_x.T + F_x @ self.kf_Q_cov @ F_x.T
        return new_p_k

    def propagate_cov_P_full(self, step, omega):
        # Phi computation
        phi_tt = rodrigues_exp(omega, step)
        phi_tb = left_jacobian_SO3(omega, step) # 3x3
        phi_bb = np.eye(3)
        self.kf_Phi = np.block([[phi_tt, phi_tb], [np.zeros((3, 3)), phi_bb]])

        om_skew = skew(omega)
        w_norm = np.linalg.norm(omega)
        d_theta = w_norm * step  # ||ω||*dt
        om_2 = (om_skew @ om_skew)

        if w_norm < 1e-8:
            self.kf_Q_cov[:3, :3] = np.identity(3) * (self.sigma_v_omega ** 2 * step + 1 / 3 * self.sigma_u_bias ** 2 * step ** 3)
            self.kf_Q_cov[3:, 3:] = np.identity(3) * self.sigma_u_bias ** 2 * step
            self.kf_Q_cov[:3, 3:] = - np.identity(3) * 0.5 * self.sigma_u_bias ** 2 * step ** 2
            self.kf_Q_cov[3:, :3] = - np.identity(3) * 0.5 * self.sigma_u_bias ** 2 * step ** 2

        else:
            q_tt = np.identity(3) * self.sigma_v_omega ** 2 * step + self.sigma_u_bias ** 2 * (
                    1/3 * np.identity(3) * step ** 3
                    - om_2 * (2 * d_theta - 2 * np.sin(d_theta) - 1/3 * d_theta**3) / w_norm ** 5)

            q_tb = self.sigma_u_bias ** 2 * (- np.identity(3) * 0.5 * step ** 2
                                             + om_skew * (d_theta - np.sin(d_theta)) / w_norm ** 3
                                             - om_2 * (0.5 * d_theta ** 2 + np.cos(d_theta) - 1) / w_norm ** 4
                                             )
            q_bt = q_tb.T
            q_bb = np.identity(3) * self.sigma_u_bias ** 2 * step

            self.kf_Q_cov = np.block([[q_tt, q_tb], [q_bt, q_bb]])

        return self.kf_Phi @ self.covariance_P @ self.kf_Phi.T + self.kf_Q_cov

    def attitude_discrete_(self, current_quaternion, omega, step):
        def q_dot(x):
            x_quaternion_i2b = x
            omega4 = omega4kinematics(omega)
            q_dot = 0.5 * omega4 @ x_quaternion_i2b
            return q_dot

        new_q = current_quaternion + runge_kutta_4(q_dot, current_quaternion, step)
        new_q /= np.linalg.norm(new_q)
        return new_q

    def attitude_discrete(self, current_quaternion, omega, step):
        """
        Closed-form integration for [x, y, z, w] convention.
        """
        # 1. Calculate magnitude of rotation
        omega_norm = np.linalg.norm(omega)
        theta = omega_norm * step

        # 2. Singularity check (if satellite is still)
        if theta < 1e-8:
            return current_quaternion / np.linalg.norm(current_quaternion)

        # 3. Construct Delta Quaternion [x, y, z, w]
        # The vector part (x,y,z) gets the sine terms
        v_delta = (omega / omega_norm) * np.sin(theta / 2.0)
        # The real part (w) gets the cosine term and goes at the END
        w_delta = np.cos(theta / 2.0)

        delta_q = np.array([v_delta[0], v_delta[1], v_delta[2], w_delta])

        # 4. Update: New_Q = Current_Q * Delta_Q
        new_q = self.quaternion_multiply(current_quaternion, delta_q)

        # 5. Normalize
        return new_q / np.linalg.norm(new_q)

    def quaternion_multiply(self, q1, q2):
        """
        Quaternion multiplication for [x, y, z, w] convention.
        """
        x1, y1, z1, w1 = q1
        x2, y2, z2, w2 = q2

        # The math is the same, but the output order is swapped to place W last.
        w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
        z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

        # Return in [x, y, z, w] order
        return np.array([x, y, z, w])

    def get_observer_prediction(self, new_x_k, reference_vector, save=True, sensor_type='mag'):
        new_z_k = Quaternions(self.current_quaternion).frame_conv(reference_vector)
        if save:
            if sensor_type == 'mag':
                new_z_k = new_z_k # / np.linalg.norm(new_z_k)
            elif sensor_type == 'css':
                # multiplication for the negative vectors in structure
                new_z_k = - 930 * np.eye(3) @ new_z_k / np.linalg.norm(new_z_k)
                # new_z_k[0] *= -1
                # new_z_k[2] *= -1

                new_z_k[new_z_k < 0] = 0
        return new_z_k

    def save_vector(self, name=None, vector=None):
        self.historical[name].append(vector)

    def get_calibrate_omega(self):
        return self.omega_state - self.current_bias

    def attitude_observer_model(self, new_x, vector_i):
        H = np.zeros((3, 6))
        H[:3, :3] = skew(Quaternions(self.current_quaternion).frame_conv(vector_i))
        return H

    def update_state(self, new_x_k, z_k_medido, z_from_observer, H_):
        error = z_k_medido - z_from_observer
        correction = self.kf_K @ error#  + self.kf_K @ H_ @ new_x_k
        if np.any(np.isnan(correction)):
            print("correction: {}".format(correction))
        new_x = new_x_k + correction
        return new_x

    def reset_state(self):
        d_theta = self.internal_state[:3]
        d_quat = 0.5 * d_theta
        dot_error = d_quat @ d_quat

        temp = 1 / np.sqrt(1 + dot_error)
        if dot_error > 1:
            error_q = Quaternions(np.array([*d_quat, 1]) * temp)
        else:
            error_q = Quaternions(np.array([*d_quat, np.sqrt(1 - dot_error)]))
        #
        # ================================================================
        # dot_error = self.internal_state[:3] @ self.internal_state[:3]
        # if dot_error < 1:
        #     error_q = Quaternions(np.array([*self.internal_state[:3] * 0.5,
        #                                     np.sqrt(1 - dot_error)]))
        # else:
        #     error_q = Quaternions(np.array([*self.internal_state[:3] * 0.5, 1]) / np.sqrt(1 + dot_error))

        error_q.normalize()
        # diff = error_q * Quaternions(self.current_quaternion)
        # current_quaternion = error_q * Quaternions(self.current_quaternion)

        current_quaternion =  Quaternions(self.current_quaternion) * error_q
        current_quaternion.normalize()
        self.current_quaternion = current_quaternion()
        self.current_bias += self.internal_state[3:]
        self.covariance_P = (self.internal_cov_P + self.internal_cov_P.T) / 2.0
        self.state = np.zeros(6)
        self.internal_state = np.zeros(6)
        self.historical['q_est'].append(self.current_quaternion)
        self.historical['b_est'].append(self.current_bias.copy())
        self.historical['p_cov'].append(np.diag(self.covariance_P))

    def plot_cov(self, folder_save=None):
        fig = plt.figure()
        plt.title("Covariance P - EKF")
        plt.plot(self.historical['mjd_ekf'], np.array(self.historical['p_cov']))
        plt.grid()
        plt.legend([r"$\delta\theta_x$", r"$\delta \theta_y$", r"$\delta\theta_z$", "$b_x$", "$b_y$", "$b_z$"])
        plt.xlabel("Modified Julian Date")
        plt.xticks(rotation=15)
        plt.ticklabel_format(useOffset=False)
        plt.yscale('log')
        plt.tight_layout()
        if folder_save is not None:
            fig.savefig(folder_save + 'ekf_omega_covariance.jpg')


if __name__ == '__main__':
    pass
