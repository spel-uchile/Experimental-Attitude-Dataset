"""
Created by Elias Obreque
Date: 04-09-2023
email: els.obrq@gmail.com
"""
import numpy as np
import datetime
from src.dynamics.quaternion import Quaternions


twopi = 2.0 * np.pi
deg2rad = np.pi / 180.0
# Constant: JD en el momento 1970-01-01 00:00:00 UTC
JD_AT_UNIX_EPOCH = 2440587.5
# 2451544.5 es el JD para 2000-01-01 00:00 UTC
JD_2000 = 2451544.5


def runge_kutta_4(function, x, dt, *args):
    k1 = function(x, *args)
    xk2 = x + (dt / 2.0) * k1

    k2 = function(xk2, *args)
    xk3 = x + (dt / 2.0) * k2

    k3 = function(xk3, *args)
    xk4 = x + dt * k3

    k4 = function(xk4, *args)

    next_x = (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    return next_x


def jday(year, mon, day, hr, minute, sec):
    jd0 = 367.0 * year - 7.0 * (year + ((mon + 9.0) // 12.0)) * 0.25 // 1.0 + 275.0 * mon // 9.0 + day + 1721013.5
    utc = ((sec / 60.0 + minute) / 60.0 + hr)  # utc in hours#
    return jd0 + utc / 24.


def gstime(jdut1):
    tut1 = (jdut1 - 2451545.0) / 36525.0
    temp = -6.2e-6 * tut1 * tut1 * tut1 + 0.093104 * tut1 * tut1 + \
           (876600.0 * 3600 + 8640184.812866) * tut1 + 67310.54841  # sec
    temp = (temp * deg2rad / 240.0) % twopi  # 360/86400 = 1/240, to deg, to rad

    #  ------------------------ check quadrants ---------------------
    if temp < 0.0:
        temp += twopi

    return temp


def fmod2(x):
    if x > np.pi:
        x -= twopi
    elif x < -np.pi:
        x += twopi
    else:
        x = x
    return x


def julian_to_datetime(julian_date):
    base_date = datetime.datetime(2000, 1, 1, tzinfo=datetime.timezone.utc)  # Julian date 2451544.5 corresponds to this datetime
    days_elapsed = julian_date - JD_2000
    delta = datetime.timedelta(days=days_elapsed)
    converted_datetime = base_date + delta
    return converted_datetime.replace(tzinfo=datetime.timezone.utc)


def jd_to_decyear_old(jd):
    # --------------- find year and days of the year ---------------
    temp = jd - 2415019.5
    tu = temp / 365.25
    year = 1900 + np.floor(tu)
    leapyrs = np.floor((year - 1901) * 0.25)

    # optional nudge by 8.64x10-7 sec to get even outputs
    days = temp - ((year - 1900) * 365.0 + leapyrs) + 0.00000000001

    # ------------ check for case of beginning of a year -----------
    if days < 1.0:
        year = year - 1
        leapyrs = np.floor((year - 1901) * 0.25)
        days = temp - ((year - 1900) * 365.0 + leapyrs)

    decyear = year + days / 365.25
    return decyear

def jd_to_decyear(julian_date_):
    # jd to  datetime UTC
    dt = julian_to_datetime(julian_date_)

    # year
    year = dt.year

    # years N and N+1
    start_of_year = datetime.datetime(year, 1, 1, tzinfo=datetime.timezone.utc)
    start_of_next = datetime.datetime(year + 1, 1, 1, tzinfo=datetime.timezone.utc)

    # elapsed
    elapsed = (dt - start_of_year).total_seconds()

    # total secons
    total_seconds = (start_of_next - start_of_year).total_seconds()

    # decimal years
    return year + elapsed / total_seconds


def timestamp_to_julian(timestamp):
    julian_date = JD_AT_UNIX_EPOCH + timestamp / 86400
    return julian_date


def tle_epoch_to_julian(tle_epoch):
    # Extrae el año y el día del año del valor de época
    year = int(tle_epoch[:2])
    day_of_year = int(tle_epoch[2:5])
    fraction_of_day = float("0" + tle_epoch[5:])
    # Calcula la fecha juliana
    base_date = jday(2000 + year, 1, 0,  0, 0, 0)
    julian_date = base_date + day_of_year + fraction_of_day
    return julian_date


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


# MRP
def get_shadow_set_mrp(sigma):
    sigma_ = -sigma / np.linalg.norm(sigma) ** 2
    return sigma_


def mrp_dot(x):
    sigma_ = x[:3]
    omega_ = x[3:]
    return np.array([*0.25 * Bmatrix_mrp(sigma_).dot(omega_), *np.zeros(3)])


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


def shadow_zone(sc_pos_from_center_i, sun_pos_from_center_i, center_object_radius=6378.135):
    point_prodcut = np.dot(sun_pos_from_center_i, sc_pos_from_center_i)
    r_sun = np.linalg.norm(sun_pos_from_center_i)
    r_sc = np.linalg.norm(sc_pos_from_center_i)
    theta = np.arccos(point_prodcut/(r_sc * r_sun))
    theta_sun = np.arccos(center_object_radius / r_sun)
    theta_sc = np.arccos(center_object_radius / r_sc)
    if theta_sc + theta_sun < theta:
        # Shadow
        is_dark = True
    else:
        is_dark = False
    return is_dark


def inertial_to_lvlh(vec, r_vec, v_vec):
    r = np.asarray(r_vec, dtype=float)
    v = np.asarray(v_vec, dtype=float)
    vec = np.asarray(vec, dtype=float)

    # Vector angular momentum
    h = np.cross(r, v)

    # base RSW = [R, S, W]
    R_hat = r / np.linalg.norm(r)  # radial
    W_hat = h / np.linalg.norm(h)  # cross-track (orbit normal)
    S_hat = np.cross(W_hat, R_hat)  # along-track

    # Projection
    lvlh_vec = np.array([
        np.dot(vec, R_hat),  #  radial
        np.dot(vec, S_hat),  # along-track
        np.dot(vec, W_hat)  #  cross-track
    ])
    return lvlh_vec

def trace_method(matrix):
    """
    This code uses a modification of the algorithm described in:
    https://d3cw3dd2w32x2b.cloudfront.net/wp-content/uploads/2015/01/matrix-to-quat.pdf
    which is itself based on the method described here:
    http://www.euclideanspace.com/maths/geometry/rotations/conversions/matrixToQuaternion/

    Altered to work with the column vector convention instead of row vectors
    q = [i, j, k, 0]
    """
    m = matrix.conj().transpose()  # This method assumes row-vector and postmultiplication of that vector
    if m[2, 2] < 0:
        if m[0, 0] > m[1, 1]:
            t = 1 + m[0, 0] - m[1, 1] - m[2, 2]
            q = [t, m[0, 1] + m[1, 0], m[2, 0] + m[0, 2], m[1, 2] - m[2, 1]]
        else:
            t = 1 - m[0, 0] + m[1, 1] - m[2, 2]
            q = [m[0, 1] + m[1, 0], t, m[1, 2] + m[2, 1], m[2, 0] - m[0, 2]]
    else:
        if m[0, 0] < -m[1, 1]:
            t = 1 - m[0, 0] - m[1, 1] + m[2, 2]
            q = [m[2, 0] + m[0, 2], m[1, 2] + m[2, 1], t, m[0, 1] - m[1, 0]]
        else:
            t = 1 + m[0, 0] + m[1, 1] + m[2, 2]
            q = [m[1, 2] - m[2, 1], m[2, 0] - m[0, 2], m[0, 1] - m[1, 0], t]

    q = np.array(q).astype('float64')
    q *= 0.5 / np.sqrt(t)
    q[0:3] *= -1
    return q

def get_lvlh2b(sat_pos_, sat_vel_, q_i2b_):
    matrix_i2lvlh = compute_i2lvlh(sat_pos_, sat_vel_)
    q_ = trace_method(matrix_i2lvlh)
    current_quaternion_lvlh2b = Quaternions(q_i2b_) * Quaternions(Quaternions(q_).conjugate())
    return current_quaternion_lvlh2b(), current_quaternion_lvlh2b.toypr()

def compute_i2lvlh(pos_i: np.ndarray, vel_i: np.ndarray):
    r_norm = np.linalg.norm(pos_i)
    if r_norm <= 1e-12:
        return np.eye(3)

    lvlh_z = -pos_i / r_norm
    lvlh_y = -np.cross(pos_i, vel_i)
    lvlh_y_norm = np.linalg.norm(lvlh_y)
    if lvlh_y_norm <= 1e-12:
        return np.eye(3)
    lvlh_y /= lvlh_y_norm
    lvlh_x = np.cross(lvlh_y, lvlh_z)
    # matrix_i2lvlh = np.empty((3, 3))
    return np.vstack((lvlh_x, lvlh_y, lvlh_z))


if __name__ == '__main__':
    print(julian_to_datetime(JD_2000))

    print(jd_to_decyear(JD_2000 + 230), " - -", jd_to_decyear_old(JD_2000 + 230))