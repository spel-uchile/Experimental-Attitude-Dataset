"""
Created by Elias Obreque
Date: 16-03-2024
email: els.obrq@gmail.com
"""
import numpy as np
from src.dynamics.MagEnv import MagEnv


if __name__ == '__main__':
    # time
    timestamp_data = 1709910531
    # omega
    bias_w = np.array([-3.846, 0.1717, -0.6937])
    mean_velocity_measure = np.array([-0.968407, 18.056318, -0.549451])
    omega = mean_velocity_measure - bias_w
    # mag field
    mag_model = MagEnv()
    mag_model.calc_mag(c_j, s_, lat_, lon_, alt_)
    mag_measure = np.array([784.615417,	-520.769226,	-386.153839])
