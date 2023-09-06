"""
Created by Elias Obreque
Date: 04-09-2023
email: els.obrq@gmail.com
"""

import numpy as np
import json


# SUCHAI3
def get_s3_mag_cal():
    D = np.array([[-0.01745035420910312, 2.16237359, -5.42143837],
                  [2.16237359, 0.0002888274805405843, 0.72444676],
                  [-5.42143837, 0.72444676, 110.89890511]])
    bias = np.array([695.88767607, 1162.50052713, 10495.62982439])
    return D, bias


def get_s3_mag_cal_past():
    D = np.array([[123.57714418, 2.16237359, -5.42143837],
                  [2.16237359, 105.6190004, 0.72444676],
                  [-5.42143837, 0.72444676, 110.89890511]])
    bias = np.array([124241.06014311, -28749.71305113, -48690.65466832])
    return D, bias
# PLANTSAT
""