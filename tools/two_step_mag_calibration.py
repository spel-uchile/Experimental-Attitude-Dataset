"""
Created by Elias Obreque
Date: 29-05-2023
email: els.obrq@gmail.com
"""
import io

import numpy as np
import scipy.io
import matplotlib.pyplot as plt


def two_step(bm_, ar_):
    # Monte Carlo Runs
    num_mc = 5000
    m = len(ar_)
    i100 = 0
    x_lin = np.zeros((num_mc, 9))
    x_non = np.zeros((num_mc, 9))
    for jjj in range(1, num_mc+1):
        # Display When Every 100th Point is Reached
        if i100 == 100:
            print(f"      Monte Carlo has reached point {jjj-1}")
            i100 = 0
        i100 += 1

        # Dot Product for Attitude Independent Measurement
        ym = bm_[:, 0] ** 2 + bm_[:, 1] ** 2 + bm_[:, 2] ** 2 - (ar_[:, 0] ** 2 + ar_[:, 1] ** 2 + ar_[:, 2] ** 2)

        # TWOSTEP
        # bbtrue = (np.eye(3) + d) @ ctrue
        # etrue = 2 * d + d @ d
        xc = np.zeros(9)
        xe = [xc]

        norm_dx = 100
        i = 1

        while norm_dx > 1e-8:
            # E
            emat = np.array([[xc[3], xc[6], xc[7]], [xc[6], xc[4], xc[8]], [xc[7], xc[8], xc[5]]])
            # 6.143a
            ee = np.kron(xc[:3], np.ones(m).reshape(-1, 1)) @ np.linalg.inv(np.eye(3) + emat).T
            # 6.126b (S_k)
            kk = np.column_stack((bm_[:, 0] ** 2, bm_[:, 1] ** 2, bm_[:, 2] ** 2,
                                  2 * bm_[:, 0] * bm_[:, 1], 2 * bm_[:, 0] * bm_[:, 2], 2 * bm_[:, 1] * bm_[:, 2]))
            # sensitivity matrix
            h = np.column_stack(
                (2 * (bm_ - ee), -kk[:, 0] + ee[:, 0] ** 2, -kk[:, 1] + ee[:, 1] ** 2, -kk[:, 2] + ee[:, 2] ** 2,
                 -kk[:, 3] + 2 * ee[:, 0] * ee[:, 1], -kk[:, 4] + 2 * ee[:, 0] * ee[:, 2],
                 -kk[:, 5] + 2 * ee[:, 1] * ee[:, 2]))
            # 6.147 (measurement model)
            ye = 2 * (bm_[:, 0] * xc[0] + bm_[:, 1] * xc[1] + bm_[:, 2] * xc[2]) \
                 - xc[3] * kk[:, 0] - xc[4] * kk[:, 1] - xc[5] * kk[:, 2] - xc[6] * kk[:, 3] - xc[7] * kk[:, 4] - xc[
                     8] * kk[:, 5] \
                 - xc[0] * ee[:, 0] - xc[1] * ee[:, 1] - xc[2] * ee[:, 2]
            dy = ym - ye
            dx = np.linalg.inv(h.T @ h) @ h.T @ dy
            norm_dx = np.linalg.norm(dx)
            if i%100 == 0:
                print(norm_dx)
            xc = xc + dx
            xe.append(xc)
            i += 1

        # Linear Batch Solution Using Centering (Approximate Solution)
        ybar = 0
        llbar = np.zeros((9, 1))

        for i in range(m):
            ybar += 1 / m * ym[i]
            llbar = llbar + 1 / m * np.array([*2 * bm_[i, :].T, *-kk[i, :]]).reshape(-1, 1)

        ytilde = ym - ybar
        ll = np.column_stack((2 * bm_, -kk))
        lltilde = ll - np.kron(llbar.T, np.ones((m, 1)))

        res = 0
        info = np.zeros((9, 9))
        for i in range(m):
            res += ytilde[i] * lltilde[i, :]
            info = info + np.outer(lltilde[i, :], lltilde[i, :])

        xe_lin = np.linalg.inv(info) @ res

        # Show TWOSTEP and Linear Solutions
        evec = xc[3:9]
        emat = np.array([[evec[0], evec[3], evec[4]], [evec[3], evec[1], evec[5]], [evec[4], evec[5], evec[2]]])
        ss, uu = np.linalg.eig(emat)
        ww = np.diag([-1 + np.sqrt(1 + ss[0]), -1 + np.sqrt(1 + ss[1]), -1 + np.sqrt(1 + ss[2])])
        dd = uu @ ww @ uu.T
        bb = np.linalg.inv(np.eye(3) + dd) @ xc[:3]
        xcc = np.array([bb[0], bb[1], bb[2], dd[0, 0], dd[1, 1], dd[2, 2], dd[0, 1], dd[0, 2], dd[1, 2]])

        evec = xe_lin[3:9]
        emat = np.array([[evec[0], evec[3], evec[4]], [evec[3], evec[1], evec[5]], [evec[4], evec[5], evec[2]]])
        ss, uu = np.linalg.eig(emat)
        ww = np.diag([-1 + np.sqrt(1 + ss[0]), -1 + np.sqrt(1 + ss[1]), -1 + np.sqrt(1 + ss[2])])
        dd = uu @ ww @ uu.T
        bb = np.linalg.inv(np.eye(3) + dd) @ xe_lin[:3]
        xee_lin = np.array([bb[0], bb[1], bb[2], dd[0, 0], dd[1, 1], dd[2, 2], dd[0, 1], dd[0, 2], dd[1, 2]])

        x_non[jjj - 1, :] = xcc
        x_lin[jjj - 1, :] = xee_lin

    x_non_sol = np.mean(x_non, axis=0)
    sig3_non = np.std(x_non, axis=0) * 3

    x_lin_sol = np.mean(x_lin, axis=0)
    sig3_lin = np.std(x_lin, axis=0) * 3

    plt.clf()
    plt.plot(x_lin[:, 0:3])
    plt.xlabel('Run Number')
    plt.ylabel('Bias Estimates')
    plt.show()

    plt.clf()
    plt.plot(x_non[:, 0:3])
    plt.xlabel('Run Number')
    plt.ylabel('Bias Estimates')
    plt.show()

    plt.clf()
    plt.plot(x_non[:, 3:])
    plt.xlabel('Run Number')
    plt.ylabel('D Estimates')
    plt.show()
    return x_non_sol, sig3_non, x_lin_sol, sig3_lin


if __name__ == '__main__':

    dt = 10
    trmm_data = scipy.io.loadmat('trmm_data.mat')
    mag_i = trmm_data['mag_i']

    t = np.arange(0, 28801, dt)
    m = len(t)
    ar = mag_i / 10
    ctrue = np.array([.5, .3, .6]) * 10
    sigm = 0.05
    d = np.array([[0.05, 0.05, 0.05], [0.05, 0.1, 0.05], [0.05, 0.05, 0.05]])

    # Measurements
    bm = (ar + np.kron(ctrue, np.ones((m, 1))) + sigm * np.random.randn(m, 3)) @ np.linalg.inv(np.eye(3) + d).T
    x_non_sol, sig3_non, x_lin_sol, sig3_lin = two_step(bm, ar)
    print(x_non_sol, sig3_non)
    print(x_lin_sol, sig3_lin)