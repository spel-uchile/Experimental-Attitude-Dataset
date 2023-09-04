"""
Created by Elias Obreque
Date: 04-09-2023
email: els.obrq@gmail.com
"""
import numpy as np


def runge_kutta_4(function, x, dt):
    k1 = function(x)
    xk2 = x + (dt / 2.0) * k1

    k2 = function(xk2)
    xk3 = x + (dt / 2.0) * k2

    k3 = function(xk3)
    xk4 = x + dt * k3

    k4 = function(xk4)

    next_x = (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    return next_x


def jday(year, mon, day, hr, minute, sec):
    jd0 = 367.0 * year - 7.0 * (year + ((mon + 9.0) // 12.0)) * 0.25 // 1.0 + 275.0 * mon // 9.0 + day + 1721013.5
    utc = ((sec / 60.0 + minute) / 60.0 + hr)  # utc in hours#
    return jd0 + utc / 24.


def skewsymmetricmatrix(x_omega_b):
    S_omega = np.zeros((3, 3))
    S_omega[1, 0] = x_omega_b[2]
    S_omega[2, 0] = -x_omega_b[1]

    S_omega[0, 1] = -x_omega_b[2]
    S_omega[0, 2] = x_omega_b[1]

    S_omega[2, 1] = x_omega_b[0]
    S_omega[1, 2] = -x_omega_b[0]
    return S_omega


def skew(x):
    return np.array([[0, -x[2], x[1]],
                     [x[2], 0, -x[0]],
                     [-x[1], x[0], 0]])


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


def get_mrp_from_q(q):
    p = q[:3]/(1 + q[3])
    if np.linalg.norm(p) > 1:
        p = -p / np.linalg.norm(p) ** 2
    return p


def add_mrp(sigma_left, sigma_right):
    snorm_l = 1 - np.linalg.norm(sigma_left) ** 2
    snorm_r = 1 - np.linalg.norm(sigma_right) ** 2
    new_sigma = snorm_r * sigma_left + snorm_l * sigma_right - 2 * np.cross(sigma_left, sigma_right)
    new_sigma /= (1 + np.linalg.norm(sigma_left) ** 2 * np.linalg.norm(sigma_right) ** 2 - 2 * sigma_left.dot(sigma_right))
    return new_sigma


def dcm_from_mrp(sigma):
    """
    N -> B
    """
    sigma2 = np.linalg.norm(sigma) ** 2
    temp = 1 / (1 + sigma2) ** 2
    c = 8 * skew(sigma).dot(skew(sigma)) - 4 * (1 - sigma2) * skew(sigma)
    c *= temp
    c += np.eye(3)
    return c


def get_shadow_set_mrp(sigma):
    sigma_ = -sigma / np.linalg.norm(sigma) ** 2
    return sigma_