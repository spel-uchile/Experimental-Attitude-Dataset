"""
Created by Elias Obreque
Date: 10-09-2023
email: els.obrq@gmail.com
"""
# VARIABLE
# B_k: measure, R_k: Reference, x_state = [bT, DT]
import numpy as np
import matplotlib.pyplot as plt


class MagEKF():
    def __init__(self):
        self.x_est = np.zeros(9)
        b_est_ = np.zeros(3)
        d_est_ = np.zeros(6)
        self.x_est = np.array([*b_est_, *d_est_])

        b_est_std = np.ones(3) * 50
        d_est_std = np.ones(6) * 1e-2
        self.pCov = np.diag([*b_est_std, *d_est_std])  # (uT)^2
        self.historical = {'bias': [], 'scale': [], 'error': [], 'P': []}

    def update_state(self, mag_measure, mag_reference, cov_sensor_):
        self.historical['bias'].append(self.x_est[:3])
        self.historical['scale'].append(self.x_est[3:])
        self.historical['P'].append(np.diag(self.pCov))
        current_x = self.x_est
        p_k = self.pCov

        error_measure = self.y_k_error(mag_measure, mag_reference)
        error_model = self.error_measurement_model(mag_measure, current_x)
        error_ = (error_measure - error_model) * 1e-6
        self.historical['error'].append(error_)
        print("Error: ", error_)
        jH = self.sensitivity_matrix_H(current_x, mag_measure)
        sigma_2_ = self.sigma_2(current_x, cov_sensor_, mag_measure)
        print(sigma_2_)
        kGain = self.update_error_K(p_k, jH, sigma_2_)
        self.x_est = self.update_error_x(current_x, kGain, error_, jH)
        self.pCov = self.update_error_P(p_k, jH, kGain)

    def get_calibration(self):
        bias_ = self.x_est[:3]
        d_ = get_full_D(self.x_est[3:])
        return bias_, d_

    def plot(self, new_sensor):
        fig, axes = plt.subplots(3, 1)
        axes[0].grid()
        axes[0].set_title('Bias')
        axes[0].plot(self.historical['bias'])
        axes[1].grid()
        axes[1].set_title('D scale')
        axes[1].plot(self.historical['scale'])
        axes[2].grid()
        axes[2].set_title('Error')
        axes[2].plot(self.historical['error'])

        plt.figure()
        plt.title("Covariance P")
        plt.plot(self.historical['P'])

        plt.figure()
        plt.title("New sensor")
        plt.plot(new_sensor)
        plt.show()

    @staticmethod
    def E_true(d):
        e_matrix = 2 * d + d @ d
        return np.array(
            [e_matrix[0, 0], e_matrix[1, 1], e_matrix[2, 2], e_matrix[0, 1], e_matrix[0, 2], e_matrix[1, 2]])

    @staticmethod
    def M_ed(D_):
        m_ed = np.zeros((6, 6))
        m_ed[0, 0] = 2 * D_[0, 0]
        m_ed[1, 1] = 2 * D_[1, 1]
        m_ed[2, 2] = 2 * D_[2, 2]

        m_ed[3, 3] = D_[0, 0] + D_[1, 1]
        m_ed[4, 4] = D_[0, 0] + D_[2, 2]
        m_ed[5, 5] = D_[1, 1] + D_[2, 2]

        m_ed[0, 3] = 2 * D_[0, 1]
        m_ed[0, 4] = 2 * D_[0, 2]

        m_ed[1, 3] = 2 * D_[0, 1]
        m_ed[1, 5] = 2 * D_[1, 2]

        m_ed[2, 4] = 2 * D_[0, 2]
        m_ed[2, 5] = 2 * D_[1, 2]

        m_ed[3, 0] = D_[0, 1]
        m_ed[3, 1] = D_[0, 1]
        m_ed[3, 4] = D_[1, 2]
        m_ed[3, 5] = D_[0, 2]

        m_ed[4, 0] = D_[0, 2]
        m_ed[4, 2] = D_[0, 2]
        m_ed[4, 3] = D_[1, 2]
        m_ed[4, 5] = D_[0, 2]

        m_ed[5, 1] = D_[1, 2]
        m_ed[5, 2] = D_[1, 2]
        m_ed[5, 3] = D_[0, 2]
        m_ed[5, 4] = D_[0, 1]
        return m_ed + 2 * np.eye(6)

    @staticmethod
    def c_true(D, b):
        return (np.eye(3) + D).dot(b)

    @staticmethod
    def S_k(B_k):
        return np.array([B_k[0] ** 2,
                         B_k[1] ** 2,
                         B_k[2] ** 2,
                         2 * B_k[0] * B_k[1],
                         2 * B_k[0] * B_k[2],
                         2 * B_k[1] * B_k[2]])

    @staticmethod
    def y_k_error(b_sensor, b_model):
        return np.linalg.norm(b_sensor) ** 2 - np.linalg.norm(b_model) ** 2

    def error_measurement_model(self, mag_sensor_, x_est):
        b_est_ = x_est[:3]
        d_vector = x_est[3:]
        D_ = get_full_D(d_vector)
        # temp1 = -self.S_k(mag_sensor_).T @ self.E_true(D_)
        # temp2 = 2 * mag_sensor_ @ ((np.eye(3) + D_) @ b_est_)
        # temp3 = - np.linalg.norm(b_est_) ** 2
        # h_k_ = temp1 + temp2 + temp3
        h_k_ = -mag_sensor_ @ (2 * D_ + D_ @ D_) @ mag_sensor_ + 2 * mag_sensor_.T @ (np.eye(3) + D_) @ b_est_ - np.linalg.norm(b_est_) ** 2
        return h_k_

    @staticmethod
    def sigma_2(x_est_, r_cov_sensor, B_k):
        cov_sensor = r_cov_sensor * np.eye(3)
        b_ = x_est_[:3]
        D_vector = x_est_[3:]
        D_ = get_full_D(D_vector)
        A_ = (np.eye(3) + D_) @ B_k - b_
        sigma_temp = 4 * A_.T @ (cov_sensor @ A_) + 2 * np.trace(cov_sensor ** 2)
        return sigma_temp

    @staticmethod
    def update_error_x(x_k, K_k, error_, jh):
        x_k1 = x_k + K_k * (error_)
        return x_k1

    @staticmethod
    def propagation_x(x_k):
        # b and D is constant
        return x_k

    @staticmethod
    def update_error_P(p_k_, h_k_, k_k_):
        I_nn = np.eye(9)
        new_P = (I_nn - np.multiply(k_k_.reshape(-1, 1), h_k_)) @ p_k_
        return  new_P

    @staticmethod
    def update_error_K(P_k, H_k, sigma_2_):
        temp_matrix = H_k @ (P_k @ H_k) + sigma_2_
        inv_matrix = 1 / temp_matrix if np.abs(temp_matrix) > 1e-12 else 0.0
        return P_k @ H_k.T * inv_matrix

    def sensitivity_matrix_H(self, x_est, B_k):
        b_est_ = x_est[:3]
        d_est_ = x_est[3:]
        d_c = get_full_D(d_est_)
        H_k = np.zeros(9)
        H_k[:3] = 2 * B_k.T @ (np.eye(3) + d_c) - 2 * b_est_
        J = np.array([B_k[0] * b_est_[0],
                      B_k[1] * b_est_[1],
                      B_k[2] * b_est_[2],
                      B_k[0] * b_est_[1] + B_k[1] * b_est_[0],
                      B_k[0] * b_est_[2] + B_k[2] * b_est_[0],
                      B_k[1] * b_est_[2] + B_k[2] * b_est_[1]])
        H_k[3:] = -self.S_k(B_k) @ self.M_ed(d_c) + 2 * J
        return H_k


def get_full_D(d_vector):
    d_ = np.zeros((3, 3))
    d_[0, 0] = d_vector[0]
    d_[1, 1] = d_vector[1]
    d_[2, 2] = d_vector[2]

    d_[0, 1] = d_vector[3]
    d_[0, 2] = d_vector[4]
    d_[1, 0] = d_vector[3]
    d_[2, 0] = d_vector[4]

    d_[1, 2] = d_vector[5]
    d_[2, 1] = d_vector[5]
    return d_


if __name__ == '__main__':
    import scipy.io
    import matplotlib.pyplot as plt

    b_est = np.array([1, 1, 1]) * 1e-3
    D_est = np.ones(6) * 1e-4
    D_est[3:] *= 1e-5
    P_est = np.diag([*b_est, *D_est]) * 10 # (uT)^2

    std_measure = 0.01  # uT
    step = 10  # s
    tend = 28801
    time = np.arange(0, tend, step)

    trmm_data = scipy.io.loadmat('../../tools/trmm_data.mat')
    mag_true = trmm_data['mag_i'] / 10

    b_true = np.array([-10, 20, .50])
    D_true = np.array([[0.5, 0.01, 0.01], [0.01, 0.5, 0.01], [0.01, 0.01, 0.5]])

    mag_sensor = [np.linalg.inv(np.eye(3) + D_true) @ (mag_true_ + np.random.normal(0, std_measure) + b_true)
                  for mag_true_ in mag_true]
    mag_sensor = np.array(mag_sensor)

    plt.figure()
    plt.title("Mag values")
    plt.plot(time, mag_true)
    plt.plot(time, mag_sensor, '.')

    # Calibration

    ct = step
    hist_x_est = np.zeros((9, len(time))).T
    k = 0
    hist_x_est[0][3] = 0.0
    P_k = P_est

    ekf_cal = MagEKF()
    ekf_cal.pCov = P_est
    new_sensor = []
    cov_sensor = std_measure ** 2
    for k in range(len(time)):
        # update state
        ekf_cal.update_state(mag_sensor[k], mag_true[k], cov_sensor)
        bias_, D_scale = ekf_cal.get_calibration()
        new_sensor.append((np.eye(3) + D_scale) @ (mag_sensor[k] - bias_))
        print(mag_true[k], new_sensor[-1])


    ekf_cal.plot(np.asarray(new_sensor) - mag_true)

    D_ekf_vector = hist_x_est[-1]
    b_ekf = hist_x_est[-1][:3]
    print(b_ekf, D_ekf_vector)
    D_ekf = np.array([[D_ekf_vector[0], D_ekf_vector[3], D_ekf_vector[4]],
                      [D_ekf_vector[3], D_ekf_vector[1], D_ekf_vector[5]],
                      [D_ekf_vector[4], D_ekf_vector[5], D_ekf_vector[2]]])

    mag_ekf = (np.eye(3) + D_ekf).dot(mag_sensor.T).T - b_ekf.reshape(-1, 1).T

    plt.figure()
    plt.title("Bias")
    plt.plot(time, hist_x_est[:, :3])

    plt.figure()
    plt.title("D")
    plt.plot(time, hist_x_est[:, 3:])

    plt.figure()
    plt.title("Estimation")
    plt.plot(time, mag_ekf)
    plt.plot(time, new_sensor)

    plt.show()
