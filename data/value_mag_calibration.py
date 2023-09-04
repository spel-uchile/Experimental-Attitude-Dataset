"""
Created by Elias Obreque
Date: 04-09-2023
email: els.obrq@gmail.com
"""

import numpy as np


# SUCHAI3
def get_s3_mag_cal():
    D = np.array([[123.57714418, 2.16237359, -5.42143837],
                  [2.16237359, 105.6190004, 0.72444676],
                  [-5.42143837, 0.72444676, 110.89890511]])
    bias = np.array([124241.06014311, -28749.71305113, -48690.65466832])
    return D, bias


# PLANTSAT
""
