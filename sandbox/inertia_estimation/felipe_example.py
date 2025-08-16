"""
Created by Felipe Diaz
Date: 05/08/2025
email: els.obrq@gmail.com
"""

import os
import sys
from datetime import datetime
from icm20948 import ICM20948
from subprocess import PIPE, run

CMD_OK           =  1 # Command executed successfully
CMD_ERROR        =  0 # Command not executed as expected
CMD_SYNTAX_ERROR = -1 # Command parameters syntax error

def measure_n_save_imu_data(imu, n):
    samples = []
    samples.append(f"Time,ax,ay,az,gx,gy,gz\n")
    counter = 0
    while (counter < n):
        #x, y, z = imu.read_magnetometer_data()
        t = datetime.utcnow().timestamp()
        ax, ay, az, gx, gy, gz = imu.read_accelerometer_gyro_data()
        samples.append(f"{t},{ax:5.5f},{ay:5.5f},{az:5.5f},{gx:5.5f},{gy:5.5f},{gz:5.5f}\n") #,{x:05.5f},{y:05.5f},{z:05.5f}\n")
        counter += 1
    t0 = datetime.utcnow()
    YYYYMMDD = f"{t0.year:04d}{t0.month:02d}{t0.day:02d}"
    HHMMSS = f"{t0.hour:02d}{t0.minute:02d}{t0.second:02d}"
    user = os.environ["USER"]
    filename = f"/home/{user}/Documents/IMU/imu_{YYYYMMDD}_{HHMMSS}.csv"
    with open(filename, 'a') as log_file:
        for sample in samples:
            log_file.write(sample)
    run(['zstd', '-9', '-T0', '--rm', filename], stdout=PIPE)

def main():
    imu = ICM20948()
    imu.set_accelerometer_sample_rate(4500)
    imu.set_accelerometer_full_scale(scale=8)
    imu.set_gyro_sample_rate(9000)
    user = os.environ["USER"]
    imu_folder = f"/home/{user}/Documents/IMU"
    if not os.path.exists(imu_folder):
        os.makedirs(imu_folder)
    if (len(sys.argv) < 2):
        sys.exit("Missing argument: Number of samples")
        return CMD_ERROR
    measure_n_save_imu_data(imu, int(sys.argv[1]))
    return CMD_OK

if __name__ == "__main__":
    main()

