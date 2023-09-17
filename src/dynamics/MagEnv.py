
import numpy as np
from .igrf.IGRF import calculate_igrf
from tools.mathtools import jd_to_decyear


RAD2DEG = 180/np.pi
DEG2RAD = 1/RAD2DEG


class MagEnv(object):
    def __init__(self):
        self.Mag_i = np.zeros(3)
        self.Mag_b = np.zeros(3)
        self.Mag_e = np.zeros(3)

    def update(self, lat, lon, alt, decyear, sideral, q_i2b):
        self.calc_mag(decyear, sideral, lat, lon, alt)

    def calc_mag(self, julian_date, sideral, lat, lon, alt):
        """
         itype = 1 if geodetic(spheroid)
         itype = 2 if geocentric(sphere)
         alt   = height in km above sea level if itype = 1
               = distance from centre of Earth in km if itype = 2 (>3485 km)
        """
        decyear = jd_to_decyear(julian_date)
        x, y, z, f, gccolat = calculate_igrf(0, decyear, alt, lat, lon, itype=1)
        mag_local = [x, y, z]

        self.mag_NED_to_ECI(mag_local, gccolat, lon, sideral)
        return self.Mag_i, np.array(mag_local)

    def add_mag_noise(self):
        return

    def mag_NED_to_ECI(self, mag_0, theta, lonrad, gmst):
        mag_local_0y = rotationY(mag_0, np.pi - theta)
        mag_local_yz = rotationZ(mag_local_0y, -lonrad)
        self.Mag_e = mag_local_yz
        self.Mag_i = rotationZ(mag_local_yz, -gmst)

    def get_mag_b(self):
        return self.Mag_b

    def get_mag_i(self):
        return self.Mag_i


def rotationY(bfr, theta):
    temp = np.zeros(3)
    temp[0] = np.cos(theta)*bfr[0] - np.sin(theta)*bfr[2]
    temp[1] = bfr[1]
    temp[2] = np.sin(theta)*bfr[0] + np.cos(theta)*bfr[2]
    return temp


def rotationZ(bfr, theta):
    temp = np.zeros(3)
    temp[0] = np.cos(theta)*bfr[0] + np.sin(theta)*bfr[1]
    temp[1] = -np.sin(theta)*bfr[0] + np.cos(theta)*bfr[1]
    temp[2] = bfr[2]
    return temp