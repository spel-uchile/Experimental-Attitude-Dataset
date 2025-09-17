"""
Created by Elias Obreque
Date: 25/08/2025
email: els.obrq@gmail.com
"""

import numpy as np
from tqdm import tqdm
from sklearn.metrics import mean_squared_error

from src.kalman_filter.ekf_multy import MEKF
from src.kalman_filter.ekf_mag_calibration import MagUKF
from src.dynamics.quaternion import Quaternions
from src.data_process import RealData
from tools.mathtools import get_lvlh2b
from src.dynamics.dynamics_kinematics import calc_quaternion, calc_omega_b, shadow_zone


def run_main_estimation(sc_inertia, channels, sensors: RealData, mag_i_on_obc, max_samples=None,
                        online_mag_calibration=False, pred_step_sec=0, sigma_bias=1e-4, sigma_omega=0.02,
                        mag_sig=2.8, css_sig=1.5, verbose=True):
    prediction_dict = {'q_i2b_pred': [],
                       'omega_b_pred': [],
                       'time_pred': [],
                       'mjd_pred': [],}
    aux_data = {'q_lvlh2b': [],
                'ypr_lvlh2b': [],
                'earth_b_lvlh': []}

    dt_obc = sensors.get_dt()
    omega_b = sensors.data[['acc_x', 'acc_y', 'acc_z']].values[0]
    q_i2b_0 = Quaternions.get_from_two_v(mag_i_on_obc[0], sensors.data[['mag_x', 'mag_y', 'mag_z']].values[0])()
    mag_b = Quaternions(q_i2b_0).frame_conv(mag_i_on_obc[0])
    moon_b = Quaternions(q_i2b_0).frame_conv(channels['moon_sc_i'][0])

    # theta, bias
    P = np.diag([1.0, 1.0, 1.0, 1.0, 1, 1]) * 1e0
    ekf_model = MEKF(sc_inertia, P=P, Q=np.zeros((6, 6)), R=np.zeros((3, 3)))
    ekf_model.sigma_bias = sigma_bias  # gyro noise standard deviation [rad/s]
    ekf_model.sigma_omega = np.deg2rad(sigma_omega)  # gyro random walk standard deviation [rad/s*s^0.5]
    ekf_model.current_bias = np.array([0.0, 0.0, 0])

    D_est = np.zeros(6) + 1e-9
    b_est = np.zeros(3) + 50
    ukf = MagUKF(b_est, D_est, alpha=0.2)

    # q_i2b = Quaternions.get_from_two_v(channels['mag_i'][0], sensors.data[['mag_x', 'mag_y', 'mag_z']].values[0])()
    q_i2b = np.array([0, 0, 0, 1])
    # Save initial data
    ekf_model.set_quat(q_i2b, save=True)
    ekf_model.set_gyro_measure(omega_b)
    ekf_model.save_time(channels['mjd'][0])
    ekf_model.save_vector(name='mag_est', vector=sensors.data[['mag_x', 'mag_y', 'mag_z']].values[0])
    ekf_model.save_vector(name='css_est', vector=sensors.data[['sun3', 'sun2', 'sun4']].values[0])
    ekf_model.save_vector(name='sun_b_est', vector=Quaternions(q_i2b).frame_conv(channels['sun_sc_i'][0]))
    ekf_model.save_vector(name='earth_b_est', vector=Quaternions(q_i2b).frame_conv(-channels['sat_pos_i'][0]))

    # sensors_idx = 1
    moon_sc_b = [moon_b]
    t0 = channels['full_time'][0]
    ekf_sensors_step = 0.1
    flag_css = False

    q_lvlhl_, ypr_lvlh_ = get_lvlh2b(channels['sat_pos_i'][0], channels['sat_vel_i'][0], q_i2b)
    aux_data['q_lvlh2b'].append(q_lvlhl_)
    aux_data['ypr_lvlh2b'].append(ypr_lvlh_)
    aux_data['earth_b_lvlh'].append(np.array([0, 0, 1]))

    rng = enumerate(channels['full_time'][1:max_samples])
    if verbose:
        rng = tqdm(rng, total=max_samples - 1, desc="Main loop Estimation")

    for ch_idx, t_jd in rng:
        ch_idx += 1
        mag_ref_ = channels['mag_i'][ch_idx]
        sun_sc_i_ = channels['sun_sc_i'][ch_idx]
        sat_pos_i_ = channels['sat_pos_i'][ch_idx]
        sat_vel_i_ = channels['sat_vel_i'][ch_idx]
        sun_pos_i_ = channels['sun_i'][ch_idx]
        moon_pos_i_ = channels['moon_sc_i'][ch_idx]
        mag_body_vec_ = sensors.data[['mag_x', 'mag_y', 'mag_z']].values[ch_idx]
        css_3_ = sensors.data[['sun3', 'sun2', 'sun4']].values[ch_idx]

        omega_gyro_ = sensors.data[['acc_x', 'acc_y', 'acc_z']].values[ch_idx]
        mjd_ = channels['mjd'][ch_idx]
        t_sec: float = np.round((t_jd - channels['full_time'][1]) * 86400, 3)

        # ======================================================================================================== #
        # PREDICTION with base time (Orbit and OBC @ 1 sec) in a big Windows

        omega_b_pred = ekf_model.get_calibrate_omega()
        q_i2b_pred = ekf_model.current_quaternion

        rng_pred = range(int(pred_step_sec / 0.1))
        # if verbose:
        #     rng_pred =
        for _ in tqdm(rng_pred, total=int(pred_step_sec / 0.1), desc=f"Prediction calculation - {omega_b_pred}"):
            q_i2b_pred = calc_quaternion(q_i2b_pred, omega_b_pred, 0.1)
            sat_pos_b_ = Quaternions(q_i2b_pred).frame_conv(sat_pos_i_) * 1e3
            omega_b_pred = calc_omega_b(omega_b_pred, 0.1, inertia_=sensors.sc_inertia, rb=None)
        t_pred = t_sec + pred_step_sec

        prediction_dict['time_pred'].append(t_pred)
        prediction_dict['mjd_pred'].append(mjd_ + pred_step_sec / 86400)
        prediction_dict['q_i2b_pred'].append(q_i2b_pred)
        prediction_dict['omega_b_pred'].append(omega_b_pred)

        # # integration
        ekf_model.propagate(dt_obc)

        # mag
        if not online_mag_calibration:
            mag_body_vec_ = ukf.calibrate([mag_ref_], [mag_body_vec_], mag_sig=mag_sig)[0]

        mag_est = ekf_model.inject_vector(mag_body_vec_, mag_ref_, sigma2=mag_sig ** 2, sensor='mag')

        # css
        css_est = np.zeros(3)
        is_dark = shadow_zone(sat_pos_i_, sun_pos_i_)
        error_mag = np.linalg.norm(mag_est - mag_body_vec_)
        flag_css = True

        if not is_dark and False:  # and (error_mag < 200 or flag_css): # mG
            css_3_[css_3_ < 300] = 0.0
            gains = -sensors.I_max * np.eye(3)
            # gains[0, 0] *= -1
            # gains[2, 2] *= -1

            css_est = ekf_model.inject_vector(css_3_, sun_sc_i_, gain=gains, sigma2=css_sig ** 2, sensor='css')
            flag_css = True
            if error_mag > 500:
                flag_css = False

        e_b_est = Quaternions(ekf_model.current_quaternion).frame_conv(-sat_pos_i_)

        ekf_model.save_vector(name='css_est', vector=css_est)
        ekf_model.save_vector(name='mag_est', vector=mag_est)
        ekf_model.save_vector(name='sun_b_est', vector=Quaternions(ekf_model.current_quaternion).frame_conv(sun_sc_i_))
        ekf_model.save_vector(name='earth_b_est', vector=e_b_est)
        ekf_model.reset_state()
        moon_sc_b.append(Quaternions(ekf_model.current_quaternion).frame_conv(moon_pos_i_))
        ekf_model.set_gyro_measure(omega_gyro_)

        q_lvlh2b, ypr_lvlh2b = get_lvlh2b(sat_pos_i_, sat_vel_i_, ekf_model.current_quaternion)

        e_b_est /= np.linalg.norm(e_b_est)
        aux_data['q_lvlh2b'].append(q_lvlh2b)
        aux_data['ypr_lvlh2b'].append(ypr_lvlh2b)
        aux_data['earth_b_lvlh'].append(Quaternions(q_lvlh2b).conjugate_class().frame_conv(e_b_est))


    error_pred = [(Quaternions(Quaternions(q_p).conjugate()) * Quaternions(q_kf)).get_angle(error_flag=True)
                  for q_p, q_kf in zip(prediction_dict['q_i2b_pred'][:-pred_step_sec],
                                       ekf_model.historical['q_est'][pred_step_sec + 1:])]

    prediction_dict['error_pred'] = error_pred
    rmse_prediction = mean_squared_error(prediction_dict['q_i2b_pred'][:-pred_step_sec], ekf_model.historical['q_est'][pred_step_sec+1:])
    return {**prediction_dict, **ekf_model.historical, **aux_data}, rmse_prediction

if __name__ == '__main__':
    pass
