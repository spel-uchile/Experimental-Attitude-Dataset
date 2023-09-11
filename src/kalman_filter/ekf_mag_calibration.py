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

        b_est_std = np.ones(3) * 10
        d_est_std = np.ones(6) * 1
        self.x_est = np.array([*b_est_, *d_est_])
        self.pCov = np.diag([*b_est_std, *d_est_std])  # (uT)^2
        self.historical = {'bias': [], 'scale': [], 'error': [], 'P': []}

    def update_state(self, mag_measure, mag_reference):
        self.historical['bias'].append(self.x_est[:3])
        self.historical['scale'].append(self.x_est[3:])
        self.historical['P'].append(self.pCov.flatten())
        current_x = self.x_est
        p_k = self.pCov

        error_measure = self.y_k_error(mag_measure, mag_reference)
        error_model = self.error_measurement_model(mag_measure, current_x)
        error_ = error_measure - error_model
        self.historical['error'].append(error_)
        print("Error: ", error_)
        jH = self.sensitivity_matrix_H(current_x, mag_measure)
        sigma_2_ = self.sigma_2(current_x, 100 * np.eye(3), mag_measure)
        kGain = self.update_error_K(p_k, jH, sigma_2_)
        self.x_est = self.update_error_x(current_x, kGain, error_)
        self.pCov = self.update_error_P(p_k, jH, kGain)

    def get_calibration(self):
        bias_ = self.x_est[:3]
        d_ = get_full_D(self.x_est[3:])
        return bias_, d_

    def plot(self, new_sensor):
        fig, axes = plt.subplots(1, 3)
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
        plt.plot(self.historical['P'])

        plt.figure()
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
        m_ed[4, 5] = D_[0, 1]

        m_ed[5, 1] = D_[1, 2]
        m_ed[5, 2] = D_[1, 2]
        m_ed[5, 3] = D_[0, 2]
        m_ed[5, 4] = D_[0, 1]
        return 2 * np.eye(6) + m_ed

    @staticmethod
    def c_true(D, b):
        return (np.eye(3) + D).dot(b)

    @staticmethod
    def S_k(B_k):
        return np.array([*B_k ** 2, 2 * B_k[0] * B_k[1], 2 * B_k[0] * B_k[2], 2 * B_k[1] * B_k[2]])

    @staticmethod
    def y_k_error(b_sensor, b_model):
        return np.linalg.norm(b_sensor) ** 2 - np.linalg.norm(b_model) ** 2

    def error_measurement_model(self, mag_sensor_, x_est):
        b_est_ = x_est[:3]
        d_vector = x_est[3:]
        D_ = get_full_D(d_vector)
        temp1 = -self.S_k(mag_sensor_).T @ self.E_true(D_)
        temp2 = 2 * mag_sensor_ @ ((np.eye(3) + D_) @ b_est_)
        # temp2 = 2 * mag_sensor_ @ b_est_
        temp3 = - np.linalg.norm(b_est_) ** 2
        h_k_ = temp1 + temp2 + temp3
        return h_k_

    @staticmethod
    def sigma_2(x_est_, cov_sensor, B_k):
        D_vector = x_est_[3:]
        b_ = x_est_[:3]
        D_ = get_full_D(D_vector)
        A_ = (np.eye(3) + D_) @ B_k - b_
        sigma_temp = 4 * A_.T @ (cov_sensor @ A_) + 2 * np.trace(cov_sensor) ** 2
        return sigma_temp

    @staticmethod
    def update_error_x(x_k, K_k, error_):
        x_k1 = x_k + K_k * error_
        return x_k1

    @staticmethod
    def propagation_x(x_k):
        # b and D is constant
        return x_k

    @staticmethod
    def update_error_P(p_k_, h_k_, k_k_):
        return (np.eye(9) - np.outer(k_k_, h_k_)) @ p_k_

    @staticmethod
    def update_error_K(P_k, H_k, sigma_2_):
        temp_matrix = H_k @ (P_k @ H_k) + sigma_2_
        inv_matrix = 1 / temp_matrix
        return P_k @ (H_k.T * inv_matrix)

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

    b_est = np.array([1.0, 1.0, 1.0])
    D_est = np.ones(6) * 0.0001
    P_est = np.diag([*b_est, *D_est])  # (uT)^2
    print(P_est)

    std_measure = 0.03  # uT
    step = 10  # s
    tend = 28801
    time = np.arange(0, tend, step)

    trmm_data = scipy.io.loadmat('../../tools/trmm_data.mat')
    mag_true = trmm_data['mag_i'] / 10

    b_true = np.array([.5, .3, .6]) * 10
    sigm = 0.05
    D_true = np.array([[0.05, 0.05, 0.05], [0.05, 0.1, 0.05], [0.05, 0.05, 0.05]])

    mag_sensor = np.linalg.inv(np.eye(3) + D_true).dot(
        mag_true.T + np.random.normal(0, std_measure, size=mag_true.shape).T + b_true.reshape(-1, 1)).T

    plt.figure()
    plt.plot(time, mag_true)
    plt.plot(time, mag_sensor, '.')

    # Calibration

    ct = step
    hist_x_est = np.zeros((9, len(time))).T
    k = 0
    hist_x_est[0][3] = 0.0
    P_k = P_est
    covariance_sensor = np.eye(3) * std_measure ** 2

    while ct <= tend:
        # update state
        x_est_k = hist_x_est[k]
        x_est_k1 = propagation_x(x_est_k)
        H_k = sensitivity_matrix_H(x_est_k, mag_sensor[k + 1])
        sigma_k = sigma_2(x_est_k1, covariance_sensor, mag_sensor[k + 1])
        K_k = update_error_K(P_k, H_k, sigma_k)
        # Update state
        z_measure = np.linalg.norm(mag_sensor[k + 1]) ** 2 - np.linalg.norm(mag_true[k]) ** 2
        h_k = measurement_model(mag_sensor[k + 1], x_est_k1[3:], x_est_k1[:3])
        error = z_measure - h_k
        x_k1 = x_est_k1 + 0.1 * K_k * error
        print("Error: {} - norm: {}".format(error, np.linalg.norm(x_k1)))
        P_k = np.matmul(np.eye(9) - np.outer(K_k, H_k), P_k)
        # time
        k += 1
        hist_x_est[k] = x_k1
        ct += step

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
    plt.plot(time, mag_true)

    plt.show()
