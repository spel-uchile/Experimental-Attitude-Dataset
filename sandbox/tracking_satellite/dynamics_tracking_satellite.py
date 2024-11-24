"""
Created by Elias Obreque
Date: 04-09-2023
email: els.obrq@gmail.com
"""
import pickle
import numpy as np
import pandas as pd
from sgp4.api import Satrec
from sgp4.api import WGS84
from sgp4.api import SGP4_ERRORS
from astropy.time import Time
from astropy.coordinates import solar_system_ephemeris, EarthLocation, get_body, AltAz
from astropy.coordinates import TEME, GCRS, ITRS, CartesianDifferential, CartesianRepresentation
from astropy import units as u
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter

from src.dynamics.MagEnv import MagEnv, rotationY, rotationZ

solar_system_ephemeris.set('de430')
_MJD_1858 = 2400000.5
RAD2DEG = 180 / np.pi
DEG2RAD = 1 / RAD2DEG
au = 149597870.691  # km
twopi = 2.0 * np.pi
radius_earth = 6378.137  # km
earth_flat = 1.0 / 298.257223563
earth_e2 = earth_flat * (2 - earth_flat)
geod_tolerance = 1e-10  # rad
loc = EarthLocation(0, 0, 0)


def jday(year, mon, day, hr, minute, sec):
    jd0 = 367.0 * year - 7.0 * (year + ((mon + 9.0) // 12.0)) * 0.25 // 1.0 + 275.0 * mon // 9.0 + day + 1721013.5
    utc = ((sec / 60.0 + minute) / 60.0 + hr)  # utc in hours#
    return jd0 + utc / 24.


def tle_epoch_to_julian(tle_epoch):
    # Extrae el año y el día del año del valor de época
    year = int(tle_epoch[:2])
    day_of_year = int(tle_epoch[2:5])
    fraction_of_day = float("0" + tle_epoch[5:])
    # Calcula la fecha juliana
    base_date = jday(2000 + year, 1, 0,  0, 0, 0)
    julian_date = base_date + day_of_year + fraction_of_day
    return julian_date


def gstime(jdut1):
    tut1 = (jdut1 - 2451545.0) / 36525.0
    temp = -6.2e-6 * tut1 * tut1 * tut1 + 0.093104 * tut1 * tut1 + \
           (876600.0 * 3600 + 8640184.812866) * tut1 + 67310.54841  # sec
    temp = (temp * DEG2RAD / 240.0) % twopi  # 360/86400 = 1/240, to deg, to rad
    #  ------------------------ check quadrants ---------------------
    if temp < 0.0:
        temp += twopi
    return temp



class Dynamics(object):
    def __init__(self, time_array, satellite_norad_path, sat_name, gs=None):
        if gs is None:
            gs = {}
        self.sat_name = sat_name
        self.time_array = time_array
        self.channels = {}
        self.mag_model = MagEnv()
        self.ground_station = gs
        self.current_info_gs = None
        tle_file = open(satellite_norad_path)
        tle_info = tle_file.readlines()
        tle_info = {"l1": [tle_info_ for tle_info_ in tle_info if tle_info_[0] == '1'],
                    "l2": [tle_info_ for tle_info_ in tle_info if tle_info_[0] == '2']}
        tle_info["epoch"] = [tle_info_.split(" ")[4] for tle_info_ in tle_info["l1"]]
        tle_info["jd"] = np.array([tle_epoch_to_julian(tle_epoch_) for tle_epoch_ in tle_info["epoch"]])
        self.tle_info = pd.DataFrame(tle_info)
        self.search_all_nearly_tle()

    def search_all_nearly_tle(self):
        jd_init = self.time_array[0]
        jd_end = self.time_array[-1]
        tle_info = self.tle_info[np.logical_and(jd_init <= self.tle_info["jd"].values, self.tle_info["jd"].values <= jd_end)]
        if len(tle_info) == 0:
            idx_ini = np.argmin(np.abs(self.tle_info["jd"].values - jd_init))
            idx_end = np.argmin(np.abs(self.tle_info["jd"].values - jd_end))
            tle_info = self.tle_info.loc[[idx_ini, idx_end]]
            self.tle_info = tle_info
        self.tle_info.reset_index(drop=True, inplace=True)
        self.tle_info.drop_duplicates(inplace=True)

    def get_nearly_tle(self, current_jd):
        idx_ = np.argmin(np.abs(self.tle_info["jd"].values - current_jd))
        return self.tle_info["l1"][idx_], self.tle_info["l2"][idx_]

    def save_data(self):
        # save channels in pickle
        """
        to open in matlab 
        >> fid = py.open('data.pickle','rb');
        >> data = py.pickle.load(fid)
        >> y = double(data{"y"})
        """

        with open("./" + self.sat_name + ".pickle", "wb") as pickle_file:
            pickle.dump(self.channels, pickle_file)

    def run(self):
        n = len(self.time_array)
        sun_pos_gcrs = np.zeros((n, 3))
        moon_pos_gcrs = np.zeros((n, 3))
        sat_pos_gcrs = np.zeros((n, 3))
        sat_vel_gcrs = np.zeros((n, 3))
        sat_pos_itrs = np.zeros((n, 3))
        sat_vel_itrs = np.zeros((n, 3))
        sat_lon = np.zeros(n)
        sat_lat = np.zeros(n)
        sat_alt = np.zeros(n)
        sideral = np.zeros(n)
        sun_sc_i = np.zeros((n, 3))
        moon_sc_i = np.zeros((n, 3))
        ground_station_relative_ecef = {name_gs: np.zeros((n, 3)) for name_gs in list(self.ground_station.keys())}
        ground_station_relative_razel = {name_gs: np.zeros((n, 3)) for name_gs in list(self.ground_station.keys())}
        for i, t_ in enumerate(self.time_array):
            time_ = Time(t_, format='jd', scale='utc')
            sun_pos_gcrs[i] = calc_sun_pos_i(time_)
            moon_pos_gcrs[i] = calc_moon_pos_i(time_)
            l1_, l2_ = self.get_nearly_tle(time_.jd)
            sc_pos, sc_vel, sc_lon, sc_lat, sc_alt, sc_pos_ecef, sc_vel_ecef = self.calc_sat_pos_i(l1_, l2_, time_)
            sat_pos_gcrs[i] = sc_pos
            sat_vel_gcrs[i] = sc_vel
            sat_pos_itrs[i] = sc_pos_ecef
            sat_vel_itrs[i] = sc_vel_ecef
            sat_lon[i] = sc_lon
            sat_lat[i] = sc_lat
            sat_alt[i] = sc_alt
            if self.current_info_gs is not None:
                for name_gs, item in self.current_info_gs.items():
                    ground_station_relative_ecef[name_gs][i] = item[0]
                    ground_station_relative_razel[name_gs][i] = item[1]
            sideral[i] = gstime(time_.ut1.value)
            sun_sc_i[i] = sun_pos_gcrs[i] - sc_pos
            moon_sc_i[i] = moon_pos_gcrs[i] - sc_pos
            print("  - {}/{}".format(i, n))

        self.channels = {'time': self.time_array,
                         'sat_pos_eci': sat_pos_gcrs,
                         'sat_vel_eci': sat_vel_gcrs,
                         'sat_pos_ecef': sat_pos_itrs,
                         'sat_vel_ecef': sat_vel_itrs,
                         'lonlat': np.array([sat_lon, sat_lat]).T,
                         'q_i2b_pred': [],
                         'omega_b_pred': [],
                         'time_pred': [],
                         'sun_i': sun_pos_gcrs,
                         'moon_i': moon_pos_gcrs,
                         'sun_sc_i': sun_sc_i,
                         'moon_sc_i': moon_sc_i,
                         'sideral': sideral,
                         "ground_station_relative_ecef": ground_station_relative_ecef,
                         "ground_station_relative_razel": ground_station_relative_razel
                         }

    def calc_sat_pos_i(self, l1: list, l2: list, t: Time):
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
        itrs_geo_p = itrs_geo.cartesian.without_differentials()
        itrs_geo_v = itrs_geo.cartesian.differentials['s']
        location = itrs_geo.earth_location
        lon, lat, height = location.geodetic.lon.value, location.geodetic.lat.value, location.geodetic.height.value

        if self.ground_station is not None:
            self.current_info_gs = {}
            for key, item in self.ground_station.items():
                # lon (deg), lat (deg), height (m)
                gs_place = EarthLocation.from_geodetic(item[0], item[1], item[2])
                topo_itrs_repr = itrs_geo.cartesian.without_differentials() - gs_place.get_itrs(t).cartesian
                itrs_topo = ITRS(topo_itrs_repr, obstime=t, location=gs_place)
                aa = itrs_topo.transform_to(AltAz(obstime=t, location=gs_place))
                rel_pos_ecef = itrs_topo.cartesian.without_differentials().xyz.value
                razel_rel = np.array([aa.distance.value, aa.az.value, aa.alt.value]) # km, deg, deg
                self.current_info_gs[key] = [rel_pos_ecef, razel_rel]
        return gcrs_geo_p.xyz.value, gcrs_geo_v.d_xyz.value, lon, lat, height, itrs_geo_p.xyz.value, itrs_geo_v.d_xyz.value

    def load_data(self, data_):
        self.channels = dict(data_)

    def plot_gt(self, folder_save="./"):
        lon = self.channels['lonlat'][:, 0]
        lat = self.channels['lonlat'][:, 1]

        fig = plt.figure(figsize=(12, 8))
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
        for key, item in self.ground_station.items():
            ax.scatter(item[0], item[1], color='red', s=10, transform=ccrs.Geodetic())
            ax.text(item[0], item[1], key, transform=ccrs.Geodetic())
        plt.tight_layout()
        fig.savefig(folder_save + self.sat_name + '-groundtrack.jpg')
        plt.close(fig)

    def plot_gs_razel(self, folder_save="./"):
        for key, item in self.ground_station.items():
            fig, ax = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
            for i in range(3):
                ax[i].plot(self.channels["time"] - _MJD_1858, self.channels["ground_station_relative_razel"][key][:, i])
                ax[i].grid()

            ax[0].set_ylabel('Range [km]')
            ax[1].set_ylabel('Azimuth [deg]')
            ax[2].set_ylabel('Elevation [deg]')
            ax[2].set_xlabel('Time [MJD]')
            plt.tight_layout()
            fig.savefig(folder_save + self.sat_name + '-' + key + '-azel.jpg')
            plt.close(fig)



def calc_moon_pos_i(t: Time):
    # time = Time(DATE)
    moon_body = get_body('moon', t, loc)
    return np.asarray(moon_body.cartesian.xyz) / 1000  # km GCRS


def calc_sun_pos_i(t: Time):
    sun_body = get_body('sun', t, loc)
    sun_pos = np.asarray(sun_body.cartesian.xyz) / 1000  # km GCRS
    return sun_pos



if __name__ == '__main__':
    # dd-mm-yyyy HH:MM:SS
    start_date = [1, 6, 2023, 00, 00, 00]
    end_date = [1, 6, 2023, 5, 00, 00]
    dt = 10 # sec
    start_julian_date = jday(start_date[2], start_date[1], start_date[0],
                             start_date[3], start_date[4], start_date[5])
    stop_julian_date = jday(end_date[2], end_date[1], end_date[0],
                            end_date[3], end_date[4], end_date[5])
    time_vector = np.arange(start_julian_date, stop_julian_date + dt / 86400, dt / 86400)

    satellite_spel = {'SUCHAI': "42788", "SUCHAI-2": "52192", "SUCHAI-3": "52191", "PlantSat": "52188"}

    satellite_path = "./sat000052192.txt"
    satellite_name = "SUCHAI-2"

    # lon - lat - alt (deg, deg, m)
    ground_station = {"Santiago": [-70.673676, -33.447487, 514],
                      "Punta Arenas": [-70.87988233, -53.13695513, 34],
                      "La Serena - Tololo": [-70.8, -30.166667, 2200]}

    dynamic_orbital = Dynamics(time_vector, satellite_path, satellite_name, ground_station)
    dynamic_orbital.run()
    dynamic_orbital.save_data()
    dynamic_orbital.plot_gt()
    dynamic_orbital.plot_gs_razel()

