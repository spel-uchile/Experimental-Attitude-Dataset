"""
Created by Elias Obreque
Date: 22-04-2024
email: els.obrq@gmail.com
"""
import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
from src.dynamics.dynamics_kinematics import calc_omega_b, calc_quaternion
from tools.pso import PSOStandard
from src.kalman_filter.ekf_mag_calibration import MagUKF
INERTIA = np.array([38478.678, 38528.678, 6873.717, 0, 0, 0]) * 1e-6


class InertiaUKF(MagUKF):
    def __init__(self, dt_):
        self.dt = dt_
        MagUKF.__init__(self, alpha=1, beta=2, x_dim=6)
        P_est = np.diag([1e4, 1e4, 1e4, 1e3, 1e3, 1e3])
        self.pCov_x = P_est.copy()
        self.pCov_zero = P_est.copy()
        self.last_omega = np.zeros(3)

    def get_observation(self, sigma_points, mag_sensor_, sigma_points_2=None):
        mean_y = 0
        mean_y_direct = []
        for i in range(2 * self.n + 1):
            mean_y_direct.append(calc_omega_b(self.last_omega, self.dt, get_full_D(sigma_points[i][:self.dim_x])))
            mean_y += self.weights_mean[i] * mean_y_direct[-1]
        return mean_y, np.atleast_2d(mean_y_direct)

    def get_cross_correlation_matrix(self, sigma_points, x_k_, z_k, sigma_z, sigma_points_sigma2=None):
        pxz = np.zeros((self.dim_x, 3))
        for i in range(2 * self.n + 1):
            dx = sigma_points[i][:self.dim_x] - x_k_
            dz = sigma_z[i] - z_k
            pxz += self.weights_cov[i] * np.outer(dx, dz)
        return pxz

    def get_state_covariance_(self, sigma_points):
        px = np.zeros((self.dim_x, self.dim_x))
        for i in range(len(sigma_points)):
            x_ = sigma_points[i][:self.dim_x] - self.x_est[:self.dim_x]
            px += self.weights_cov[i] * np.outer(x_, x_)
        return px

    def get_output_covariance_(self, z_k, sigma_z):
        kmax, n = sigma_z.shape
        pzz = np.zeros((n, n))
        for i in range(kmax):
            dz = sigma_z[i] - z_k
            pzz += self.weights_cov[i] * np.outer(dz, dz)
        return pzz

    def run(self, sensor_, reference_, cov_sensor_, error_up=None, error_down=None):
        current_x = self.x_est.copy()

        self.lamda = self.alpha ** 2 * (self.n + self.kappa) - self.n
        self.gamma = np.sqrt(self.n + self.lamda)
        self.weights_mean, self.weights_cov = self.get_weights()

        # sigma points
        sigma_points = self.sigma_points(current_x)
        # update
        # zero
        x_a = [wi * xi for wi, xi in zip(self.weights_mean, sigma_points)]

        x_k = np.sum(np.array(x_a), axis=0)
        p_cov_x = self.get_state_covariance_(sigma_points)

        # sensor model
        y_k, y_sigma = self.get_observation(sigma_points, sensor_)
        error_ = (sensor_ - y_k)
        print("ukf Error: ", error_)
        sig2 = cov_sensor_

        p_xz = self.get_cross_correlation_matrix(sigma_points, x_k, y_k, y_sigma)
        p_zz = self.get_output_covariance_(y_k, y_sigma) + sig2
        p_zz_inv = np.linalg.inv(p_zz)
        # correction
        kk_ = np.dot(p_xz, p_zz_inv)
        self.x_est = x_k + kk_ @ np.atleast_1d(error_)
        self.pCov_x = p_cov_x - kk_ @ p_zz @ kk_.T
        if len(self.historical['error']) > 0 and error_up is not None and error_down is not None:
            if abs(error_) > error_up and not self.flag:
                # self.pCov_x[:3, :3] = self.pCov_zero[:3, :3]
                self.pCov_x = self.pCov_zero
                self.flag = True
                print("After:", error_)
            elif abs(error_) <= error_down and self.flag:
                self.flag = False
                print("low:", error_)
        self.historical['error'].append(error_)


def estimate_inertia_matrix_ukf(w_true_, time_list_):
    dt_ = np.mean(np.diff(time_list_))
    ukf = InertiaUKF(dt_)
    ukf.x_est = np.array([30000.0, 30000.0, 5000.0, 0, 0, 0]) * 1e-6
    cov_sensor = 0.1 * np.deg2rad(1)

    ukf.last_omega = w_true_[0] + np.random.normal(scale=cov_sensor, size=3)
    for w_ in w_true_[1:]:
        w_ref = calc_omega_b(ukf.last_omega, dt_, get_full_D(ukf.x_est))
        w_sensor = w_ + np.random.normal(scale=cov_sensor, size=3)
        ukf.run(w_sensor, w_ref, np.eye(3) * cov_sensor ** 2)
        ukf.last_omega = w_sensor
    return ukf.x_est


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


def create_data(dt_, t_end, inertia_=None, w0: np.array = None, plot_=True):
    if inertia_ is not None:
        inertia = inertia_
    else:
        inertia = INERTIA
    if w0 is None:
        w0 = np.array([-1, -30, 0.5]) * np.deg2rad(1)
    w_array_ = [w0]
    for i in range(int(t_end/dt_)):
        w_array_.append(calc_omega_b(w0, dt_, get_full_D(inertia)))
        w0 = w_array_[-1]
    w_array_ = np.array(w_array_)
    h_b = np.array([get_full_D(inertia) @ w_ for w_ in w_array_])
    kinetic_energy = np.array([0.5 * np.dot(h_, w_) for w_, h_ in zip(w_array_, h_b)])
    inv_i = np.linalg.inv(get_full_D(inertia))
    acc_ = np.array([inv_i @ (- np.cross(w_, h_)) for w_, h_ in zip(w_array_, h_b)])
    acc_b_diff = np.diff(w_array_, axis=0) / dt_

    if not plot_:
        return w_array_, np.arange(0, len(w_array_), 1) * dt_, h_b, acc_
    plt.figure()
    plt.title(r'$\omega_b$')
    plt.plot(w_array_)
    plt.legend(['x', 'y', 'z'])
    plt.grid()

    plt.figure()
    plt.title(r'$T$')
    plt.plot(kinetic_energy)
    plt.legend(['x', 'y', 'z'])
    plt.grid()

    plt.figure()
    plt.title(r'$h_b$')
    plt.plot(h_b)
    plt.plot(np.linalg.norm(h_b, axis=1))
    plt.legend(['x', 'y', 'z'])
    plt.grid()

    plt.figure()
    plt.title(r'$\dot{\omega_b}$')
    plt.plot(acc_)
    plt.plot(acc_b_diff)
    plt.legend(['x', 'y', 'z', 'x_p', 'y_p', 'z_p'])
    plt.grid()

    return w_array_, np.arange(0, len(w_array_), 1) * dt_, h_b, acc_


def f_c(inertia_, omega, t_list):
    dt_ = np.diff(t_list)
    acc = np.diff(omega, axis=0) / np.atleast_2d(dt_).T

    inertia_ = get_full_D(inertia_)
    error_ = [inertia_ @ a_ + np.cross(w_, inertia_ @ w_) for w_, a_ in zip(omega[:-1], acc)]
    error_ = np.array(error_) ** 2
    vec_3 = np.mean(error_, axis=0)
    return np.array([*vec_3, vec_3[0] * vec_3[1], vec_3[0] * vec_3[2], vec_3[1] * vec_3[2]])


def f_c_v2(inertia_, omega, t_list):
    dt_ = np.diff(t_list)
    w_array_, t_array_, h_b_array_, acc_b_array_ = create_data(np.mean(dt_), inertia_=np.array([*inertia_, 0, 0, 0]), w0=omega[0],
                                                               t_end=t_list[-1], plot_=False)
    error_v3 = omega - w_array_
    error_ = np.max(np.linalg.norm(error_v3, axis=1))

    # inertia_ = get_full_D(inertia_)
    # h_b_ = [inertia_ @ w_ for w_ in omega]

    return np.mean(error_v3, axis=0)


def f_c_pso(args):
    inertia_ = args[0]
    omega = args[1]
    t_list = args[2]
    dt_ = np.diff(t_list)
    acc = np.diff(omega, axis=0) / np.atleast_2d(dt_).T

    # inertia_ = get_full_D(inertia_) * 1e-6
    inertia_ = np.diag(inertia_) * 1e-6
    h_b_ = [inertia_ @ w_ for w_ in omega[:-1]]
    error_ = [a_ + np.linalg.inv(inertia_) @ (np.cross(w_, h_)) for w_, h_, a_ in zip(omega[:-1], h_b_, acc)]
    error_2 = np.array(error_)
    vec_3 = np.mean(error_2, axis=1)
    h_norm_ = np.linalg.norm(h_b_, axis=1)
    # print(np.sqrt(np.sum(vec_3)), np.std(np.linalg.norm(h_b_, axis=1)) * 1e5)
    return np.abs(np.sum(vec_3)) + np.max(np.abs(h_norm_ - np.mean(h_norm_))) * 1e5, error_


def f_c_pso_v2(args):
    inertia_ = args[0] * 1e-6
    omega = args[1]
    t_list = args[2]
    dt_ = np.diff(t_list)
    acc = np.diff(omega, axis=0) / np.atleast_2d(dt_).T


    w_array_, t_array_, h_b_array_, acc_b_array_ = create_data(np.mean(dt_), inertia_=inertia_, w0=omega[0],
                                                               t_end=t_list[-1])
    inertia_ = get_full_D(inertia_)
    h_b_ = [inertia_ @ w_ for w_ in omega]
    h_norm_ = np.linalg.norm(h_b_, axis=1)
    error_v3 = omega - w_array_
    error_ = np.max(np.linalg.norm(error_v3, axis=1)) + np.max(np.abs(1 - h_norm_/np.mean(h_norm_)))
    return error_, error_


def estimate_inertia_matrix(omega, t_list, guess=None):
    if guess is None:
        guess = np.array([0.03, 0.03, 0.03, 0, 0, 0])

    dt_ = np.diff(t_list)
    acc = np.diff(omega, axis=0) / np.atleast_2d(dt_).T
    sol_ = fsolve(f_c, guess, args=(omega, t_list), full_output=True)
    return sol_[0]


def estimate_inertia_matrix_pso(omega, t_list, guess=None):
    dt_ = np.diff(t_list)
    acc = np.diff(omega, axis=0) / np.atleast_2d(dt_).T

    sol_pso = PSOStandard(f_c_pso_v2, n_steps=50, n_particles=20)
    sol_pso.initialize([[35000, 40000],
                        [35000, 40000],
                        [6000, 7000],
                        [-0, 0],
                        [-0, 0],
                        [-0, 0]
                        ])
    res_ = sol_pso.optimize(args=(omega, t_list))
    return np.array([*res_[0], 0, 0, 0]) * 1e-6


def get_matrix_A(d_w_, w_):
    return np.array([[d_w_[0], -w_[2] * w_[1], w_[2] * w_[1]],
                      [w_[0] * w_[2], d_w_[1], -w_[0] * w_[2]],
                      [-w_[0] * w_[1], w_[0] * w_[1], d_w_[2]]])


def estimate_inertia_matrix_lineal(omega, t_list, guess=None):
    # Inicializar matrices para el sistema de ecuaciones
    A = np.zeros((3 * len(t_list), 6))
    b = np.zeros((3 * len(t_list)))

    dt_ = np.diff(t_list)
    acc = np.diff(omega, axis=0) / np.atleast_2d(dt_).T
    # Construir el sistema de ecuaciones
    b = np.zeros((3, 1))
    error_ = np.zeros((3, len(t_list) - 1))
    for i in range(len(omega) - 1):
        A = get_matrix_A(acc[i], omega[i])
        error_[:, i] = A @ np.array([38478.678, 38528.678, 6873.717]).T


    # Resolver el sistema de ecuaciones para obtener los coeficientes de la matriz de inercia
    x = np.linalg.lstsq(A, b, rcond=None)[0]
    return np.array([*x.T[0], 0, 0, 0]) * 1e-6


if __name__ == '__main__':
    dt = 0.1
    w_array_true, t_array_true, h_b_array_true, acc_b_array_true = create_data(dt, t_end=1000)

    # estimation of inertia
    # sol = estimate_inertia_matrix_lineal(w_array_true, t_array_true)
    sol = estimate_inertia_matrix_pso(w_array_true, t_array_true)
    # sol = estimate_inertia_matrix_ukf(w_array_true, t_array_true)
    # sol = fsolve(f_c_v2, np.array([40000, 40000, 7000]), args=(w_array_true, t_array_true))

    print(np.array(sol) * 1e6)
    w_array, t_array, h_b_array, acc_b_array = create_data(dt, t_end=1000, inertia_=np.array(sol))

    error_w = w_array_true - w_array
    error_hb = h_b_array_true - h_b_array
    error_acc = acc_b_array_true - acc_b_array

    plt.figure()
    plt.title(r'$\delta \omega_b$ [deg/s]')
    plt.plot(error_w * np.rad2deg(1))
    plt.grid()
    plt.legend(['x', 'y', 'z'])

    plt.figure()
    plt.title(r'$\delta h_b$')
    plt.plot(error_hb)
    plt.grid()
    plt.legend(['x', 'y', 'z'])

    plt.figure()
    plt.title(r'$\delta \dot{\omega_b}$')
    plt.plot(error_acc)
    plt.grid()
    plt.legend(['x', 'y', 'z'])

    plt.show()
