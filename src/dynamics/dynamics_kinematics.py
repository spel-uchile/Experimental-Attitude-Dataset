"""
Created by Elias Obreque
Date: 04-09-2023
email: els.obrq@gmail.com
"""
from sgp4.api import Satrec
from sgp4.api import WGS84
from tools.tools import *
import numpy as np

RAD2DEG = 180 / np.pi
DEG2RAD = 1 / RAD2DEG
au = 149597870.691  # km
twopi = 2.0 * np.pi
radius_earth = 6378.137  # km
earth_flat = 1.0 / 298.257223563
earth_e2 = earth_flat * (2 - earth_flat)
geod_tolerance = 1e-10  # rad

inertia = np.array([[38478.678, 0, 0], [0, 38528.678, 0], [0, 0, 6873.717]]) * 1e-6
inv_inertia = np.linalg.inv(inertia)


def calc_sun_pos_i(jd):
    # all in degree
    n = jd - 2451545.0
    l = (280.459 + 0.98564736 * n) % 360.0
    m = (357.529 + 0.98560023 * n) % 360.0
    m *= DEG2RAD
    lam = (l + 1.915 * np.sin(m) + 0.0200 * np.sin(2 * m)) % 360.0
    lam *= DEG2RAD
    e = 23.439 - 3.56e-7 * n
    e *= DEG2RAD

    r_sun = (1.00014 - 0.01671 * np.cos(m) - 0.000140 * np.cos(2 * m)) * au
    u_v = np.array([np.cos(lam), np.cos(e) * np.sin(lam), np.sin(lam) * np.sin(e)])
    return r_sun * u_v


def calc_sat_pos_i(l1: list, l2: list, jd: float):
    satellite = Satrec.twoline2rv(l1, l2, WGS84)
    _, pos, vel = satellite.sgp4(int(jd), jd % 1)
    return pos, vel


def calc_geod_lat_lon_alt(pos, c_jd):
    current_sideral = gstime(c_jd)
    r = np.sqrt(pos[0] ** 2 + pos[1] ** 2)

    long = fmod2(np.arctan2(pos[1], pos[0]) - current_sideral)
    lat = np.arctan2(pos[2], r)

    flag_iteration = True

    while flag_iteration:
        phi = lat
        c = 1 / np.sqrt(1 - earth_e2 * np.sin(phi) * np.sin(phi))
        lat = np.arctan2(pos[2] + radius_earth * c
                         * earth_e2 * np.sin(phi), r)
        if (np.abs(lat - phi)) <= geod_tolerance:
            flag_iteration = False

    alt = r / np.cos(lat) - radius_earth * c  # *metros
    if lat > np.pi / 2:
        lat -= twopi
    return lat, long, alt, current_sideral


def calc_quaternion(q0, omega, dt):
    new_q = q0 + runge_kutta_4(dquaternion, q0, dt, omega)
    return new_q


def calc_omega_b(omega0, dt):
    new_omega = omega0 + runge_kutta_4(domega, omega0, dt)
    return new_omega


def domega(x_omega_b):
    sk = skewsymmetricmatrix(x_omega_b)
    h_total_b = inertia.dot(x_omega_b)
    w_dot = - inv_inertia @ (sk @ h_total_b)
    return w_dot


def dquaternion(x_quaternion_i2b, x_omega_b):
    ok = omega4kinematics(x_omega_b)
    q_dot = 0.5 * ok @ x_quaternion_i2b
    return q_dot


if __name__ == '__main__':
    pass
