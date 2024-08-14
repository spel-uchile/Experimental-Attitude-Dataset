"""
Created by Elias Obreque
Date: 09-05-2024
email: els.obrq@gmail.com
"""
import math
import time
import numpy as np

first_pi = math.pi + 0.1

n_check = 10000000

t0 = time.time()
save_error = []
for i in range(n_check):
    rest = math.pi - first_pi + 0.1
    save_error.append(rest)

print(np.mean(save_error), np.std(save_error))
t1 = time.time()
print(t1 - t0)

t0 = time.time()
save_error = [math.pi - first_pi + 0.1 for i in range(n_check)]
print(np.mean(save_error), np.std(save_error))
t1 = time.time()
print(t1 - t0)


