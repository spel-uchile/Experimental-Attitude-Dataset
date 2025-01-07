"""
Created by Elias Obreque
Date: 13-05-2023
email: els.obrq@gmail.com
"""

import numpy as np


def crp_from_dcm(dcm):
    q = np.zeros(3)
    eta = np.sqrt(np.trace(dcm) + 1)
    q[0] = dcm[1, 2] - dcm[2, 1]
    q[1] = dcm[2, 0] - dcm[0, 2]
    q[2] = dcm[0, 1] - dcm[1, 0]
    q /= eta**2
    return q


def skew(x):
    return np.array([[0, -x[2], x[1]],
                     [x[2], 0, -x[0]],
                     [-x[1], x[0], 0]])


def dcm_from_crp(q: np.ndarray):
    temp = 1/(1 + q.dot(q))
    c = (1 - q.dot(q)) * np.eye(3)
    c += 2 * q.reshape(-1, 1) * q
    c -= 2 * skew(q)
    c *= temp
    return c


def add_crp(q_left, q_right):
    q = q_left + q_right  - np.cross(q_left, q_right)
    q /= (1 - q_left.dot(q_right))
    return q


def Bmatrix_crp(q):
    b_matrix = np.zeros((3, 3))
    for i in range(3):
        b_matrix[i, i] = 1 + q[i] ** 2
    b_matrix[0, 1] = q[0] * q[1] - q[2]
    b_matrix[0, 2] = q[0] * q[2] + q[1]

    b_matrix[1, 0] = q[1] * q[0] + q[2]
    b_matrix[1, 2] = q[1] * q[2] - q[0]

    b_matrix[2, 0] = q[2] * q[0] - q[1]
    b_matrix[2, 1] = q[2] * q[1] + q[0]
    return b_matrix


def get_shadow_set_mrp(sigma):
    sigma_ = -sigma / np.linalg.norm(sigma) ** 2
    return sigma_


def dcm_from_mrp(sigma):
    sigma2 = np.linalg.norm(sigma) ** 2
    temp = 1/(1 + sigma2) ** 2
    c = 8 * skew(sigma).dot(skew(sigma)) - 4 * (1 - sigma2) * skew(sigma)
    c *= temp
    c += np.eye(3)
    return c


def mrp_from_dcm(dcm):
    q = np.zeros(3)
    eta = np.sqrt(np.trace(dcm) + 1)
    q[0] = dcm[1, 2] - dcm[2, 1]
    q[1] = dcm[2, 0] - dcm[0, 2]
    q[2] = dcm[0, 1] - dcm[1, 0]
    q /= (eta * (eta + 2))
    return q


def add_mrp(sigma_left, sigma_right):
    snorm_l = 1 - np.linalg.norm(sigma_left) ** 2
    snorm_r = 1 - np.linalg.norm(sigma_right) ** 2
    new_sigma = snorm_r * sigma_left + snorm_l * sigma_right - 2 * np.cross(sigma_left, sigma_right)
    new_sigma /= (1 + np.linalg.norm(sigma_left) ** 2 * np.linalg.norm(sigma_right) ** 2 - 2 * sigma_left.dot(sigma_right))
    return new_sigma


def Bmatrix_mrp(sigma):
    b_matrix = np.zeros((3, 3))
    sigma2 = np.linalg.norm(sigma) ** 2
    for i in range(3):
        b_matrix[i, i] = (1 - sigma2 + 2 * sigma[i] ** 2) * 0.5

    b_matrix[0, 1] = sigma[0] * sigma[1] - sigma[2]
    b_matrix[0, 2] = sigma[0] * sigma[2] + sigma[1]

    b_matrix[1, 0] = sigma[1] * sigma[0] + sigma[2]
    b_matrix[1, 2] = sigma[1] * sigma[2] - sigma[0]

    b_matrix[2, 0] = sigma[2] * sigma[0] - sigma[1]
    b_matrix[2, 1] = sigma[2] * sigma[1] + sigma[0]
    return b_matrix * 2


def omega_from_mpr(sigma_, d_sigma_):
    B_t = Bmatrix_mrp(sigma_).T
    omega_ = 4 / ((1 + np.linalg.norm(sigma_) ** 2) ** 2) * B_t @ d_sigma_
    return omega_


def d_omega_from_mpr(sigma_, d_sigma_, dd_sigma_):
    d_omega_ = np.zeros(3)
    temp = 4 / ((1 + np.linalg.norm(sigma_) ** 2) ** 2)
    d_temp = -16 / ((1 + np.linalg.norm(sigma_) ** 2) ** 3) * np.dot(d_sigma_, sigma_)
    b_ = Bmatrix_mrp(sigma_)
    d_b_ = -2 * np.dot(d_sigma_, sigma_) * np.identity(3) + 2 * skew(d_sigma_) + \
           np.multiply(sigma_, d_sigma_.reshape(-1, 1)) + np.multiply(d_sigma_, sigma_.reshape(-1, 1))

    d_omega_ += temp * b_.T @ dd_sigma_
    d_omega_ += d_temp * b_.T @ d_sigma_
    d_omega_ += temp * d_b_.T @ d_sigma_
    return d_omega_


if __name__ == '__main__':
    q = np.array([0.1, 0.2, 0.3])
    # print(dcm_from_crp(q))

    dcm_ = np.array([[0.333333, -0.666667, 0.666667],
                     [0.871795, 0.487179, 0.0512821],
                     [-0.358974, 0.564103, 0.74359]])

    print(crp_from_dcm(dcm_))

    q_fn = np.array([0.1, 0.2, 0.3])
    q_bn = np.array([-0.3, 0.3, 0.1])

    q_bf = add_crp(q_bn, -q_fn)
    print(*q_bf)

    # integration
    # q0 = np.array([0.4, 0.2, -0.1])
    # ct=0
    # dt = 0.01
    # tend = 42
    #
    # while ct <= tend:
    #     print(np.linalg.norm(q0))
    #     omega_b = np.array([np.sin(0.1 * ct), 0.01, np.cos(0.1 * ct)]) * 3 * np.deg2rad(1)
    #     qdot = 0.5 * Bmatrix(q0).dot(omega_b)
    #     q_new = q0 + dt * qdot
    #     q0 = q_new
    #     ct += dt

    sigma = np.array([0.1, 0.2, 0.3])
    print(get_shadow_set_mrp(sigma))
    print(dcm_from_mrp(sigma))

    dcm_ = np.array([[0.763314, -0.568047, -0.307692],
                     [0.0946746, -0.372781, 0.923077],
                     [-0.639053, -0.733728, -0.230769]]).T
    print(mrp_from_dcm(dcm_))

    sigma_bn = np.array([0.1, 0.2, 0.3])
    sigma_rb = np.array([-0.1, 0.3, 0.1])

    sigma_rn = add_mrp(sigma_rb, sigma_bn)
    print(sigma_rn)

    sigma_rn = np.array([0.5, 0.3, 0.1])

    sigma_br = add_mrp(sigma_bn, -sigma_rn)
    print(sigma_br)

    # integration
    q0 = np.array([0.4, 0.2, -0.1])
    ct = 0
    dt = 0.01
    tend = 42

    while ct <= tend:
        print(np.linalg.norm(q0))
        omega_b = np.array([np.sin(0.1 * ct), 0.01, np.cos(0.1 * ct)]) * 20 * np.deg2rad(1)
        qdot = 0.5 * Bmatrix_mrp(q0).dot(omega_b)
        q_new = q0 + dt * qdot
        q0 = q_new
        if np.linalg.norm(q0) >1:
            q0 = get_shadow_set_mrp(q0)
        ct += dt