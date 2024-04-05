"""
Created by Elias Obreque
Date: 10-09-2023
email: els.obrq@gmail.com
"""
# VARIABLE
# B_k: measure, R_k: Reference, x_state = [bT, DT]
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import seaborn as sns
import pandas as pd
from tools.pso import PSOMagCalibration
from scipy.linalg import cholesky


def pso_cost(x_est, mag_sensor_, mag_ref):
    hk = error_measurement_model(mag_sensor_, x_est)
    zk = np.linalg.norm(mag_sensor_) ** 2 - np.linalg.norm(mag_ref) ** 2
    return (zk - hk) ** 2


def error_measurement_model(mag_sensor_, x_est):
    b_est_ = x_est[:3]
    d_vector = x_est[3:]
    D_ = get_full_D(d_vector)
    # temp1 = -self.S_k(mag_sensor_).T @ self.E_true(D_)
    # temp2 = 2 * mag_sensor_ @ ((np.eye(3) + D_) @ b_est_)
    # temp3 = - np.linalg.norm(b_est_) ** 2
    # h_k_ = temp1 + temp2 + temp3
    h_k_ = -mag_sensor_ @ (2 * D_ + D_ @ D_) @ mag_sensor_ + 2 * mag_sensor_.T @ (
            np.eye(3) + D_) @ b_est_ - np.linalg.norm(b_est_) ** 2
    return h_k_


def sigma_2(x_est_, r_cov_sensor, B_k):
    cov_sensor = r_cov_sensor * np.eye(3)
    b_ = x_est_[:3]
    D_vector = x_est_[3:]
    D_ = get_full_D(D_vector)
    A_ = (np.eye(3) + D_) @ B_k - b_
    sigma_temp = 4 * A_.T @ (cov_sensor @ A_) + 2 * np.trace(cov_sensor ** 2)
    return sigma_temp


class MagUKF():
    # Unscented Filter Formulation
    def __init__(self, alpha=0.1, beta=2):
        self.x_est = np.zeros(9)

        b_est_std = np.ones(3) * 50
        d_est_std = np.ones(6) * 1e-2
        sensor_noise = .1
        self.dim_x = 9
        self.dim_v = 0
        self.dim_n = 0
        self.full_dim = self.dim_x + self.dim_v + self.dim_n
        self.pCov_x = np.diag([*b_est_std, *d_est_std])  # state covariance
        self.flag = False
        self.pCov_zero = self.pCov_x.copy() * 0.5
        self.fisher_matrix = np.diag([*b_est_std, *d_est_std])
        # self.pCov_v = np.diag(np.zeros(0))  # process covariance
        # self.pCov_n = np.diag(np.zeros(0))  # measurement covariance
        # fill matrix with diagonal
        # self.pCov = np.zeros((self.full_dim, self.full_dim))

        # self.pCov[:self.dim_x, :self.dim_x] = self.pCov_x
        # self.pCov[self.dim_x:self.dim_x + self.dim_v, self.dim_x:self.dim_x + self.dim_v] = self.pCov_v
        # self.pCov[self.dim_x + self.dim_v:self.full_dim, self.dim_x + self.dim_v:self.full_dim] = self.pCov_n
        self.historical = {'bias': [], 'scale': [], 'error': [], 'P': [], 'eigP': []}

        self.alpha = alpha
        self.beta = beta
        self.n = self.x_est.size
        self.kappa = 3.0 - self.n
        self.lamda = self.alpha ** 2 * (self.n + self.kappa) - self.n
        self.gamma = np.sqrt(self.n + self.lamda)
        self.weights_mean, self.weights_cov = self.get_weights()

    def sigma_points(self, x_est, method='SQUARE-ROOT'):
        # Sigma Points
        sigma_points = np.zeros((2 * self.n + 1, self.n))
        # sigma_points_sigma2 = np.zeros((2 * self.n, self.n))
        sigma_points[0] = x_est
        if method == 'SQUARE-ROOT':
            uu_ = cholesky((self.n + self.lamda) * self.pCov_x)
        else:
            uu_ = self.gamma * np.sqrt(self.pCov_x)
        for i in range(self.n):
            sigma_points[i + 1] = x_est + np.real(uu_[i])
            sigma_points[i + 1 + self.n] = x_est - np.real(uu_[i])
            # sigma_points_sigma2[i] = x_est + 2 * np.real(uu_[i])
            # sigma_points_sigma2[i + self.n] = x_est - 2 * np.real(uu_[i])
        return sigma_points# , sigma_points_sigma2

    def save(self):
        current_x = self.x_est.copy()
        self.historical['bias'].append(current_x[:3])
        self.historical['scale'].append(current_x[3:])
        self.historical['P'].append(np.diag(self.pCov_x.copy()))
        self.historical['eigP'].append(np.sqrt(np.linalg.eigvals(self.pCov_x)))

    def run(self, sensor_, reference_, cov_sensor_, error_up=None, error_down=None):
        # if error_up is not None and error_down is not None:
        #     if len(self.historical['error']) > 0:
        #         if abs(self.historical['error'][-1]) > error_up and not self.flag:
        #             self.pCov_x = self.pCov_zero
        #             self.flag = True
        #         else:#if abs(self.historical['error'][-1]) <= error_down and self.flag:
        #             self.flag = False

        current_x = self.x_est.copy()

        self.lamda = self.alpha ** 2 * (self.n + self.kappa) - self.n
        self.gamma = np.sqrt(self.n + self.lamda)
        self.weights_mean, self.weights_cov = self.get_weights()

        # sigma points
        sigma_points = self.sigma_points(current_x)
        # update
        # zero
        x_a = [wi * xi for wi, xi in zip(self.weights_mean, sigma_points)]
        # x_a_sigma2 = [wi * xi for wi, xi in zip(self.weights_mean[1:], sigma_points_sigma2)]

        x_k = np.sum(np.array(x_a), axis=0) # + np.sum(np.array(x_a_sigma2), axis=0)
        p_cov_x = self.get_state_covariance(sigma_points)

        # sensor model
        error_measure = np.linalg.norm(sensor_) ** 2 - np.linalg.norm(reference_) ** 2
        error_model_mean, error_y_direct = self.get_observation(sigma_points, sensor_)
        error_ = (error_measure - error_model_mean)
        print("ukf Error: ", error_)
        sig2 = sigma_2(x_k, cov_sensor_, sensor_)

        p_xz = self.get_cross_correlation_matrix(sigma_points, x_k, error_model_mean, error_y_direct)
        p_zz = self.get_output_covariance(sigma_points, x_k, error_model_mean, error_y_direct)
        p_zz_inv = np.linalg.inv(p_zz + sig2)
        # correction
        kk_ = np.dot(p_xz, p_zz_inv)
        # self.corr = kk_ @ np.atleast_1d(error_)
        self.x_est = x_k + kk_ @ np.atleast_1d(error_)
        # self.x_est[3:] = np.maximum(self.x_est[3:], 0)
        self.pCov_x = p_cov_x - kk_ @ (p_zz + sig2) @ kk_.T
        if len(self.historical['error']) > 0 and error_up is not None and error_down is not None:
            if abs(error_) > error_up and not self.flag:
                # self.pCov_x[:3, :3] = self.pCov_zero[:3, :3]
                self.pCov_x = self.pCov_zero

                self.flag = True
                print("After:", error_)
            elif abs(error_) <= error_down and self.flag:
                self.flag = False
                print("low:", error_)
        # if len(self.historical['error']) > 1:
        #     self.alpha *= np.abs(error_ / self.historical['error'][-1]) ** 2
        #     self.alpha = min(max(self.alpha, 1), 40)
        #     print(self.alpha)
        self.historical['error'].append(error_)
        # fill matrix with diagonal
        # self.pCov[:self.dim_x, :self.dim_x] = self.pCov_x

    def get_state_covariance(self, sigma_points, sigma_points_2=None):
        px = np.zeros((self.dim_x, self.dim_x))
        for i in range(len(sigma_points)):
            x_ = sigma_points[i][:self.dim_x] - self.x_est[:self.dim_x]
            px += self.weights_cov[i] * np.outer(x_, x_)
        if sigma_points_2 is not None:
            for i in range(len(sigma_points_2)):
                x_ = sigma_points_2[i][:self.dim_x] - self.x_est[:self.dim_x]
                px += self.weights_cov[i] * np.outer(x_, x_)
        return px

    def get_weights(self):
        # Weights
        weights_mean = np.zeros(2 * self.n + 1)
        weights_cov = np.zeros(2 * self.n + 1)
        weights_mean[0] = self.lamda / (self.n + self.lamda)
        weights_cov[0] = self.lamda / (self.n + self.lamda) + (1 - self.alpha ** 2 + self.beta)
        for i in range(2 * self.n):
            weights_mean[i + 1] = 1 / (2 * (self.n + self.lamda))
            weights_cov[i + 1] = 1 / (2 * (self.n + self.lamda))
        return weights_mean, weights_cov

    def get_observation(self, sigma_points, mag_sensor_, sigma_points_2=None):
        mean_y = 0
        mean_y_direct = []
        for i in range(2 * self.n + 1):
            mean_y_direct.append(error_measurement_model(mag_sensor_, sigma_points[i][:self.dim_x]))
            mean_y += self.weights_mean[i] * mean_y_direct[-1]
        if sigma_points_2 is not None:
            for i in range(2 * self.n):
                mean_y_direct.append(error_measurement_model(mag_sensor_, sigma_points_2[i][:self.dim_x]))
                mean_y += self.weights_mean[i] * mean_y_direct[-1]
        return mean_y, np.atleast_2d(mean_y_direct).T

    def get_cross_correlation_matrix(self, sigma_points, x_k_, z_k, sigma_z, sigma_points_sigma2=None):
        pxz = np.zeros((self.dim_x, 1))
        for i in range(2 * self.n + 1):
            dx = sigma_points[i][:self.dim_x] - x_k_
            dz = sigma_z[i] - z_k
            pxz += self.weights_cov[i] * np.outer(dx, dz)
        if sigma_points_sigma2 is not None:
            for i in range(2 * self.n):
                dx = sigma_points_sigma2[i][:self.dim_x] - x_k_
                dz = sigma_z[2 * self.n + 1 + i] - z_k
                pxz += self.weights_cov[i] * np.outer(dx, dz)
        return pxz

    def get_bias(self):
        bias_ = np.array(self.historical['bias'])
        sigma_ = np.array(self.historical['eigP'])[:, :3]
        return bias_, bias_ + sigma_ * 3, bias_ - sigma_ * 3

    def get_output_covariance(self, sigma_points, x_k_, z_k, sigma_z, sigma_points_sigma2=None):
        kmax, n = sigma_z.shape
        pzz = np.zeros((n, n))
        for i in range(kmax):
            dz = sigma_z[i] - z_k
            pzz += self.weights_cov[i] * np.outer(dz, dz)
        return pzz

    def get_calibration(self):
        bias_ = self.x_est[:3]
        d_ = get_full_D(self.x_est[3:9])
        return bias_, d_

    def plot(self, new_sensor_error, y_true):
        val = np.linalg.eigvals(self.pCov_x)
        fig, axes = plt.subplots(3, 1)
        fig.suptitle('UKF')
        axes[0].grid()
        axes[0].set_title('Bias')
        axes[0].plot(self.historical['bias'])
        axes[1].grid()
        axes[1].set_title('D scale')
        axes[1].plot(self.historical['scale'])
        axes[2].set_title('Error - UKF')
        axes[2].plot(self.historical['error'])
        axes[2].grid()

        plt.figure()
        plt.title("Covariance P - UKF")
        plt.grid()
        plt.plot(self.historical['P'])
        plt.xlabel("Step")
        # bias and D legend
        plt.legend(["bx", "by", "bz", "Dxx", "Dyy", "Dzz", "Dxy", "Dxz", "Dyz"])

        mse_ukf = mean_squared_error(y_true, new_sensor_error)
        plt.figure()
        plt.title("Magnitude Sensor error - UKF")
        plt.grid()
        plt.xlabel("Step")
        plt.plot(y_true - new_sensor_error, label='RMSE: {:2f}'.format(np.sqrt(mse_ukf)))
        plt.legend()


class MagEKF():
    def __init__(self):
        self.x_est = np.zeros(9)
        b_est_std = np.ones(3) * 50
        d_est_std = np.ones(6) * 1e-2
        self.pCov = np.diag([*b_est_std, *d_est_std])  # (uT)^2
        self.historical = {'bias': [], 'scale': [], 'error': [], 'P': [], 'eigP': []}

    def save(self):
        current_x = self.x_est.copy()
        self.historical['bias'].append(current_x[:3])
        self.historical['scale'].append(current_x[3:])
        self.historical['P'].append(np.diag(self.pCov.copy()))
        self.historical['eigP'].append(np.sqrt(np.linalg.eigvals(self.pCov)))

    def run(self, mag_measure, mag_reference, cov_sensor_):
        current_x = self.x_est.copy()
        p_k = self.pCov

        error_measure = self.y_k_error(mag_measure, mag_reference)
        error_model = error_measurement_model(mag_measure, current_x)
        error_ = (error_measure - error_model)
        self.historical['error'].append(error_)
        print("ekf Error: ", error_)
        jH = self.sensitivity_matrix_H(current_x, mag_measure)
        sigma_2_ = sigma_2(current_x, cov_sensor_, mag_measure)
        # print("new sigma", sigma_2_)
        kGain = self.update_error_K(p_k, jH, sigma_2_)
        self.x_est = self.update_error_x(current_x, kGain, error_, jH)
        # self.x_est[3:] = np.maximum(self.x_est[3:], 0)
        self.pCov = self.update_error_P(p_k, jH, kGain)
        # if np.abs(error_) > 10 * np.abs(self.historical['error'][-1]):
        #     self.pCov *= 1.5
        #     print("update_pcov")

    def get_calibration(self):
        bias_ = self.x_est[:3]
        d_ = get_full_D(self.x_est[3:])
        return bias_, d_

    def plot(self, new_sensor_error, y_true):
        fig, axes = plt.subplots(3, 1)
        fig.suptitle('EKF')
        axes[0].grid()
        axes[0].set_title('Bias')
        axes[0].plot(self.historical['bias'])
        axes[0].plot()
        axes[1].grid()
        axes[1].set_title('D scale')
        axes[1].plot(self.historical['scale'])
        axes[2].grid()
        axes[2].set_title('Error')
        axes[2].plot(self.historical['error'])

        plt.figure()
        plt.title("Covariance P - EKF")
        plt.grid()
        plt.xlabel("Step")
        plt.plot(self.historical['P'])
        # bias and D legend
        plt.legend(["bx", "by", "bz", "Dxx", "Dyy", "Dzz", "Dxy", "Dxz", "Dyz"])

        mse_ukf = mean_squared_error(y_true, new_sensor_error)
        plt.figure()
        plt.title("Magnitude Sensor error - EKF")
        plt.xlabel("Step")
        plt.grid()
        plt.plot(y_true - new_sensor_error, label='RMSE: {:2f}'.format(np.sqrt(mse_ukf)))
        plt.legend()

    def get_bias(self):
        bias_ = np.array(self.historical['bias'])
        sigma_ = np.array(self.historical['eigP'])[:, :3]
        return bias_, bias_ + sigma_ * 3, bias_ - sigma_ * 3

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
        return new_P

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


def create_example(b_true_, D_true_, std_measure_, time_array_):
    mag_true_ = np.array([10 * np.sin(time_array_ * 0.001), -5 * np.cos(time_array_ * 0.025),
                         20 * np.sin(time_array_ * 0.0025) * np.cos(time_array_ * 0.01)]).T
    mag_true_ = np.multiply(mag_true_, 2 - time_array_.reshape(-1, 1) / tend)
    eta_noise = np.random.normal(0, std_measure_, size=(len(mag_true_), 3))
    mag_sensor_ = [np.linalg.inv(np.eye(3) + D_true_) @ (mag_true_ + eta_ + b_)
                   for eta_, mag_true_, b_ in zip(eta_noise, mag_true_, b_true_)]
    return mag_true_, np.array(mag_sensor_)


if __name__ == '__main__':
    import scipy.io
    import matplotlib.pyplot as plt

    np.random.seed(42)

    b_est = np.array([1, 1, 1]) * 100
    D_est = np.array([1.1, 1.2, 1.3, 0.001, 0.002, 0.003]) * 1e-7
    #    D_est[3:] *= 1e-5
    P_est = np.diag([*b_est, *D_est]) # (uT)^2
    # P_est = np.identity(9) * 1e-2

    std_measure = 0.5  # uT
    step = 1  # s
    tend = 5400 * 1

    time_array = np.arange(0, tend, step)

    # trmm_data = scipy.io.loadmat('../../tools/trmm_data.mat')

    # b_true = [np.array([-50, 25, 10])] * len(time_array)
    b_true = [np.array([-5.0, 2.5, 1.]) +
              10 * np.array([-5.0, 2.5, 1.]) * np.min([(t_ - tend * 0.3) / 1000, 1]) if t_ >= tend * 0.3 else np.array([-5.0, 2.5, 1.]) for
              t_ in time_array]
    b_true = np.array(b_true)
    D_true = np.array([[1.5, 0.00, 0.0], [0.00, 1.1, 0.0], [0.0, 0.0, 2.5]]) * 0.01

    mag_true, mag_sensor = create_example(b_true, D_true, std_measure, time_array)

    plt.figure()
    plt.title("True Signal")
    plt.plot(mag_true)
    plt.ylabel("Magnitude")
    plt.xlabel("Time")
    plt.grid()
    plt.legend(["x", "y", "z"])

    plt.figure()
    plt.title("Signal with bias and scale")
    plt.plot(mag_sensor)
    plt.ylabel("Magnitude")
    plt.xlabel("Time")
    plt.grid()
    plt.legend(["x", "y", "z"])
    # Calibration

    ct = step
    hist_x_est = np.zeros((9, len(time_array))).T
    k = 0
    hist_x_est[0][3] = 0.0
    P_k = P_est

    ekf_cal = MagEKF()
    ekf_cal.pCov = P_est.copy()
    ukf = MagUKF(alpha=1, beta=2)
    ukf.pCov_x = P_est.copy()
    ukf.pCov_zero = P_est.copy()
    pso_est = PSOMagCalibration(pso_cost, n_particles=50)
    pso_est.initialize([[-50, 50],
                        [-50, 50],
                        [-50, 50],
                        [-10, 10], [-10, 10], [-10, 10],
                        [-1, 1], [-1, 1], [-1, 1]])
    new_sensor = []
    new_sensor_ukf = []
    new_sensor_pso = []
    cov_sensor = std_measure ** 2

    for k in range(len(time_array)):
        # update state
        ekf_cal.save()
        ukf.save()

        ekf_cal.run(mag_sensor[k], mag_true[k], cov_sensor)
        ukf.run(mag_sensor[k], mag_true[k], cov_sensor, 150, 100)
        # pso_est.optimize(mag_sensor[k], mag_true[k], clip=False)

        bias_, D_scale = ekf_cal.get_calibration()
        bias_ukf, D_scale_ukf = ukf.get_calibration()
        # bias_pso, D_scale_pso = pso_est.get_calibration()

        new_sensor.append((np.eye(3) + D_scale) @ mag_sensor[k] - bias_)
        new_sensor_ukf.append((np.eye(3) + D_scale_ukf) @ mag_sensor[k] - bias_ukf)
        # new_sensor_pso.append((np.eye(3) + D_scale_pso) @ mag_sensor[k] - bias_pso)
        # print("RMS error:",
        #       np.dot(mag_true[k], new_sensor[-1]) / np.linalg.norm(mag_true[k]) / np.linalg.norm(new_sensor[-1]))

    ekf_cal.plot(np.linalg.norm(np.asarray(new_sensor), axis=1), np.linalg.norm(mag_true, axis=1))
    ukf.plot(np.linalg.norm(np.asarray(new_sensor_ukf), axis=1), np.linalg.norm(mag_true, axis=1))

    # D_ekf_vector = ekf_cal.historical['scale'][-1]
    # b_ekf = ekf_cal.historical['bias'][-1]
    # D_ukf_vector = ukf.historical['scale'][-1]
    # b_ukf = ukf.historical['bias'][-1]
    # print(b_ekf, D_ekf_vector)
    # print(b_ukf, D_ukf_vector)
    # D_ekf = np.array([[D_ekf_vector[0], D_ekf_vector[3], D_ekf_vector[4]],
    #                   [D_ekf_vector[3], D_ekf_vector[1], D_ekf_vector[5]],
    #                   [D_ekf_vector[4], D_ekf_vector[5], D_ekf_vector[2]]])
    # D_ukf = np.array([[D_ukf_vector[0], D_ukf_vector[3], D_ukf_vector[4]],
    #                   [D_ukf_vector[3], D_ukf_vector[1], D_ukf_vector[5]],
    #                   [D_ukf_vector[4], D_ukf_vector[5], D_ukf_vector[2]]])

    mag_ekf = np.asarray(new_sensor)    # (np.eye(3) + D_ekf).dot(mag_sensor.T).T - b_ekf.reshape(-1, 1).T
    mag_ukf = np.asarray(new_sensor_ukf)    # (np.eye(3) + D_ukf).dot(mag_sensor.T).T - b_ukf.reshape(-1, 1).T

    dic_ekf_x = pd.DataFrame({'Error': (mag_ekf - mag_true)[:, 0], 'Filters': ['EKF'] * len(mag_ekf),
                              'Axis': ['x'] * len(mag_ekf)})
    dic_ekf_y = pd.DataFrame({'Error': (mag_ekf - mag_true)[:, 1], 'Filters': ['EKF'] * len(mag_ekf),
                              'Axis': ['y'] * len(mag_ekf)})
    dic_ekf_z = pd.DataFrame({'Error': (mag_ekf - mag_true)[:, 2], 'Filters': ['EKF'] * len(mag_ekf),
                              'Axis': ['z'] * len(mag_ekf)})

    dic_ukf_x = pd.DataFrame({'Error': (mag_ukf - mag_true)[:, 0], 'Filters': ['UKF'] * len(mag_ukf),
                              'Axis': ['x'] * len(mag_ukf)})
    dic_ukf_y = pd.DataFrame({'Error': (mag_ukf - mag_true)[:, 1], 'Filters': ['UKF'] * len(mag_ukf),
                              'Axis': ['y'] * len(mag_ukf)})
    dic_ukf_z = pd.DataFrame({'Error': (mag_ukf - mag_true)[:, 2], 'Filters': ['UKF'] * len(mag_ukf),
                              'Axis': ['z'] * len(mag_ukf)})

    #
    result = pd.concat([dic_ekf_x, dic_ekf_y, dic_ekf_z, dic_ukf_x, dic_ukf_y, dic_ukf_z], ignore_index=True)

    fig, ax = plt.subplots(3, 1, sharex=True)
    bias, sigma_up, sigma_low = ekf_cal.get_bias()
    ax[0].set_title("Bias - EKF")
    for i in range(3):
        ax[i].grid()
        ax[i].plot(time_array, bias[:, i], label='Estimated bias')
        ax[i].plot(time_array, sigma_up[:, i], color='red', linestyle='--', label=r'$+3\sigma$')
        ax[i].plot(time_array, sigma_low[:, i], color='red', linestyle='--', label=r'$-3\sigma$')
        ax[i].plot(time_array, b_true[:, i], linestyle="--", color='black', label='True')
        if i==0:
            ax[i].legend(loc='upper center', bbox_to_anchor=(0.5, 1.05),
                         fancybox=True, shadow=True, ncol=4)
    ax[i].set_xlabel("Time")

    plt.figure()
    plt.title("D - EKF")
    plt.grid()
    plt.plot(time_array, ekf_cal.historical['scale'])
    plt.xlabel("Time")
    plt.hlines(D_true, 0, tend, linestyle="--", color="black")

    fig_est_ekf, ax_est_ekf = plt.subplots(3, 1, sharex=True)
    fig_est_ekf.suptitle("Calibration - EKF")
    ax_est_ekf[0].set_ylabel("x")
    ax_est_ekf[1].set_ylabel("y")
    ax_est_ekf[2].set_ylabel("z")
    for i in range(3):
        ax_est_ekf[i].grid()
        ax_est_ekf[i].plot(time_array, mag_true[:, i], color='black', lw=0.7, label='true')
        ax_est_ekf[i].plot(time_array, mag_ekf[:, i], color='red', lw=0.7, label='est')
        ax_est_ekf[i].legend()
    ax_est_ekf[i].set_xlabel("Time")

    # UKF
    fig, ax = plt.subplots(3, 1, sharex=True)
    bias, sigma_up, sigma_low = ukf.get_bias()
    ax[0].set_title("Bias - UKF")
    for i in range(3):
        ax[i].grid()
        ax[i].plot(time_array, bias[:, i], label='Estimated bias')
        ax[i].plot(time_array, sigma_up[:, i], color='red', linestyle='--', label=r'$+3\sigma$')
        ax[i].plot(time_array, sigma_low[:, i], color='red', linestyle='--', label=r'$-3\sigma$')
        ax[i].plot(time_array, b_true[:, i], linestyle="--", color='black', label='True')
        if i==0:
            ax[i].legend(loc='upper center', bbox_to_anchor=(0.5, 1.05),
                         fancybox=True, shadow=True, ncol=4)
    ax[i].set_xlabel("Time")

    plt.figure()
    plt.title("D - UKF")
    plt.grid()
    plt.plot(time_array, ukf.historical['scale'])
    plt.xlabel("Time")
    plt.hlines(D_true, 0, tend, linestyle="--", color="black")

    fig_est_ukf, ax_est_ukf = plt.subplots(3, 1, sharex=True)
    fig_est_ukf.suptitle("Calibration - UKF")
    ax_est_ukf[0].set_ylabel("x")
    ax_est_ukf[1].set_ylabel("y")
    ax_est_ukf[2].set_ylabel("z")
    for i in range(3):
        ax_est_ukf[i].grid()
        ax_est_ukf[i].plot(time_array, mag_true[:, i], color='black', lw=0.7, label='true')
        ax_est_ukf[i].plot(time_array, mag_ukf[:, i], color='red', lw=0.7, label='est')
        ax_est_ukf[i].legend()
    ax_est_ukf[i].set_xlabel("Time")
    plt.legend()

    plt.figure()
    sns.set(style="whitegrid")
    sns.violinplot(data=result, y="Error", x="Filters", hue="Axis",
                   inner_kws=dict(box_width=4, whis_width=2))
    plt.show()
