"""
Created by Elias Obreque
Date: 04-09-2023
email: els.obrq@gmail.com
"""
import pandas as pd
from sgp4.api import Satrec
from sgp4.api import WGS84
from Quaternion import Quaternions
import numpy as np

re = 6378.137  # km
RAD2DEG = 180 / np.pi
DEG2RAD = 1 / RAD2DEG
au = 149597870.691  # km
twopi = 2.0 * np.pi
radius_earth = 6378.137  # km
earth_flat = 1.0 / 298.257223563
earth_e2 = earth_flat * (2 - earth_flat)
geod_tolerance = 1e-10  # rad


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


if __name__ == '__main__':
    pass
