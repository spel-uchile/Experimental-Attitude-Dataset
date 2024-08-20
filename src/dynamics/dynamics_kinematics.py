"""
Created by Elias Obreque
Date: 04-09-2023
email: els.obrq@gmail.com
"""
import numpy as np
from sgp4.api import Satrec
from sgp4.api import WGS84
from sgp4.api import SGP4_ERRORS
from tools.mathtools import *
from astropy.time import Time
from astropy.coordinates import solar_system_ephemeris, EarthLocation, get_body
from astropy.coordinates import TEME, GCRS, ITRS, CartesianDifferential, CartesianRepresentation
from astropy import units as u
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter

from src.dynamics.MagEnv import MagEnv, rotationY, rotationZ

solar_system_ephemeris.set('de430')

RAD2DEG = 180 / np.pi
DEG2RAD = 1 / RAD2DEG
au = 149597870.691  # km
twopi = 2.0 * np.pi
radius_earth = 6378.137  # km
earth_flat = 1.0 / 298.257223563
earth_e2 = earth_flat * (2 - earth_flat)
geod_tolerance = 1e-10  # rad

INERTIA = np.array([[38478.678, 0, 0], [0, 38528.678, 0], [0, 0, 6873.717]]) * 1e-6
INV_INERTIA = np.linalg.inv(INERTIA)
loc = EarthLocation(0, 0, 0)


class Dynamics(object):
    def __init__(self, time_array, line1, line2):
        self.time_array = time_array
        self.l1 = line1
        self.l2 = line2
        self.channels = {}
        self.mag_model = MagEnv()

    def get_dynamics(self):
        n = len(self.time_array)
        sun_pos_gcrs = np.zeros((n, 3))
        moon_pos_gcrs = np.zeros((n, 3))
        sat_pos_gcrs = np.zeros((n, 3))
        sat_vel_gcrs = np.zeros((n, 3))
        sat_lon = np.zeros(n)
        sat_lat = np.zeros(n)
        sat_alt = np.zeros(n)
        mag_eci = np.zeros((n, 3))
        mag_ecef = np.zeros((n, 3))
        mag_ned = np.zeros((n, 3))
        sideral = np.zeros(n)
        sun_sc_i = np.zeros((n, 3))
        moon_sc_i = np.zeros((n, 3))
        for i, t_ in enumerate(self.time_array):
            time_ = Time(t_, format='jd', scale='utc')
            sun_pos_gcrs[i] = calc_sun_pos_i(time_)
            moon_pos_gcrs[i] = calc_moon_pos_i(time_)
            sc_pos, sc_vel, sc_lon, sc_lat, sc_alt = calc_sat_pos_i(self.l1, self.l2, time_)
            sat_pos_gcrs[i] = sc_pos
            sat_vel_gcrs[i] = sc_vel
            sat_lon[i] = sc_lon
            sat_lat[i] = sc_lat
            sat_alt[i] = sc_alt
            sideral[i] = gstime(time_.ut1.value)
            mag_eci[i], mag_ecef[i], mag_ned[i] = self.mag_model.calc_mag(t_, sideral[i], sc_lat * DEG2RAD, sc_lon * DEG2RAD, sc_alt)
            sun_sc_i[i] = sun_pos_gcrs[i] - sc_pos
            moon_sc_i[i] = moon_pos_gcrs[i] - sc_pos
            print("  - {}/{}".format(i, n))

        self.channels = {'full_time': self.time_array,
                         'sim_time': [0],
                         'sat_pos_i': sat_pos_gcrs,
                         'lonlat': np.array([sat_lon, sat_lat]).T,
                         'sat_vel_i': sat_vel_gcrs,
                         'q_i2b_pred': [],
                         'omega_b_pred': [],
                         'time_pred': [],
                         'mag_i': mag_eci * 0.01,  # nT to mG,
                         'mag_ecef': mag_ecef * 0.01,  # nT to mG,
                         'mag_ned': mag_ned * 0.01,
                         'sun_i': sun_pos_gcrs,
                         'moon_i': moon_pos_gcrs,
                         'sun_sc_i': sun_sc_i,
                         'moon_sc_i': moon_sc_i,
                         'sideral': sideral}
        return self.channels

    def load_data(self, data_):
        self.channels = dict(data_)

    def calc_mag(self):
        mag_ned = self.channels['mag_ned']
        lon = self.channels['lonlat'][:, 0]
        lat = self.channels['lonlat'][:, 1]
        sideral = self.channels['sideral']
        mag_local_0y = [rotationY(vec_, np.pi/2 + lat_ * DEG2RAD) for vec_, lat_ in zip(mag_ned, lat)]
        mag_local_yz = [rotationZ(mag_, -lon_ * DEG2RAD) for mag_, lon_ in zip(mag_local_0y, lon)]
        self.channels['mag_ecef'] = np.array(mag_local_yz)
        self.channels['mag_i'] = np.array([rotationZ(mag_, -sideral_) for mag_, sideral_ in zip(mag_local_yz, sideral)])

    def plot_gt(self):
        lon = self.channels['lonlat'][:, 0]
        lat = self.channels['lonlat'][:, 1]

        plt.figure(figsize=(12, 8))
        ax = plt.axes(projection=ccrs.PlateCarree())
        ax.stock_img()
        gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', linestyle='--')
        gl.top_labels = False
        gl.right_labels = False
        gl.xformatter = LongitudeFormatter()
        gl.yformatter = LatitudeFormatter()

        ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
        ax.add_feature(cfeature.BORDERS, linewidth=0.5)
        # ax.plot(lon, lat, linestyle='-', color='r', transform=ccrs.Geodetic())
        plt.title('Groundtrack', pad=20, fontsize=12, color='black')
        ax.scatter(lon, lat, color='blue', s=10, transform=ccrs.Geodetic())
        plt.tight_layout()
        plt.show(block=False)

    def plot_mag(self):
        mag_x = self.channels['mag_i'][:, 0]
        mag_y = self.channels['mag_i'][:, 1]
        mag_z = self.channels['mag_i'][:, 2]
        time_ = self.channels['full_time']
        mag_norm = np.linalg.norm(self.channels['mag_i'], axis=1)

        plt.figure()
        plt.title('Magnetic Field - ECI [mG]', pad=20, fontsize=12, color='black')
        plt.plot(time_, mag_x, label='x')
        plt.plot(time_, mag_y, label='y')
        plt.plot(time_, mag_z, label='z')
        plt.plot(time_, mag_norm, color='black', label='mag norm')
        plt.legend()
        plt.tight_layout()
        plt.grid()

        mag_x = self.channels['mag_ecef'][:, 0]
        mag_y = self.channels['mag_ecef'][:, 1]
        mag_z = self.channels['mag_ecef'][:, 2]
        time_ = self.channels['full_time']
        mag_norm = np.linalg.norm(self.channels['mag_ecef'], axis=1)

        plt.figure()
        plt.title('Magnetic Field - ECEF [mG]', pad=20, fontsize=12, color='black')
        plt.plot(time_, mag_x, label='x')
        plt.plot(time_, mag_y, label='y')
        plt.plot(time_, mag_z, label='z')
        plt.plot(time_, mag_norm, color='black', label='mag norm')
        plt.legend()
        plt.tight_layout()
        plt.grid()

        mag_x = self.channels['mag_ned'][:, 0]
        mag_y = self.channels['mag_ned'][:, 1]
        mag_z = self.channels['mag_ned'][:, 2]
        time_ = self.channels['full_time']
        mag_norm = np.linalg.norm(self.channels['mag_ned'], axis=1)
        plt.figure()
        plt.title('Magnetic Field - NED [mG]', pad=20, fontsize=12, color='black')
        plt.plot(time_, mag_x, label='x')
        plt.plot(time_, mag_y, label='y')
        plt.plot(time_, mag_z, label='z')
        plt.plot(time_, mag_norm, color='black', label='mag norm')
        plt.legend()
        plt.tight_layout()
        plt.grid()
        plt.show()


def calc_moon_pos_i(t: Time):
    # time = Time(DATE)
    moon_body = get_body('moon', t, loc)
    return np.asarray(moon_body.cartesian.xyz) / 1000  # km GCRS


def calc_sun_pos_i(t: Time):
    # all in degree
    # n = jd - 2451545.0
    # l = (280.459 + 0.98564736 * n) % 360.0
    # m = (357.529 + 0.98560023 * n) % 360.0
    # m *= DEG2RAD
    # lam = (l + 1.915 * np.sin(m) + 0.0200 * np.sin(2 * m)) % 360.0
    # lam *= DEG2RAD
    # e = 23.439 - 3.56e-7 * n
    # e *= DEG2RAD
    #
    # r_sun = (1.00014 - 0.01671 * np.cos(m) - 0.000140 * np.cos(2 * m)) * au
    # u_v = np.array([np.cos(lam), np.cos(e) * np.sin(lam), np.sin(lam) * np.sin(e)])
    sun_body = get_body('sun', t, loc)
    sun_pos = np.asarray(sun_body.cartesian.xyz) / 1000  # km GCRS
    return sun_pos


def calc_sat_pos_i(l1: list, l2: list, t: Time):
    satellite = Satrec.twoline2rv(l1, l2, WGS84)
    error_code, teme_p, teme_v = satellite.sgp4(t.jd1, t.jd2)  # in km and km/s
    if error_code != 0:
        raise RuntimeError(SGP4_ERRORS[error_code])
    teme_p = CartesianRepresentation(teme_p * u.km)
    teme_v = CartesianDifferential(teme_v * u.km / u.s)
    teme = TEME(teme_p.with_differentials(teme_v), obstime=t)
    gcrs_geo = teme.transform_to(GCRS(obstime=t))  # Geocentric Celestial Reference System - J2000
    gcrs_geo_p = gcrs_geo.cartesian.without_differentials()
    gcrs_geo_v = gcrs_geo.cartesian.differentials['s']

    itrs_geo = teme.transform_to(ITRS(obstime=t))
    location = itrs_geo.earth_location
    lon, lat, height = location.geodetic.lon.value, location.geodetic.lat.value, location.geodetic.height.value
    return gcrs_geo_p.xyz.value, gcrs_geo_v.d_xyz.value, lon, lat, height


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
    new_q /= np.linalg.norm(new_q)
    return new_q


def calc_omega_b(omega0, dt, inertia_=None):
    args_ = ()
    if inertia_ is not None:
        inertia_ = inertia_
        inv_inertia_ = np.linalg.inv(inertia_)
    else:
        inertia_ = INERTIA
        inv_inertia_ = INV_INERTIA
    new_omega = omega0 + runge_kutta_4(domega, omega0, dt, inertia_, inv_inertia_)
    return new_omega


def domega(x_omega_b, *args):
    if len(args) > 0:
        inertia_ = args[0]
        inv_inertia_ = args[1]
    else:
        inertia_ = INERTIA
        inv_inertia_ = INV_INERTIA
    sk = skewsymmetricmatrix(x_omega_b)
    h_total_b = inertia_.dot(x_omega_b)
    w_dot = - inv_inertia_ @ (sk @ h_total_b)
    return w_dot


def dquaternion(x_quaternion_i2b, x_omega_b):
    ok = omega4kinematics(x_omega_b)
    q_dot = 0.5 * ok @ x_quaternion_i2b
    return q_dot


if __name__ == '__main__':
    pass
