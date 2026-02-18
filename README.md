# Attitude Estimation by real data

To use this tool you must open and execute the file "main.py".

## Requirements
```shell
# Create virtual environment
python -m venv .ps_venv

# Activate virtual environment
# Unix
source .ps_venv/bin/activate

# Windows
ps_venv\Scripts\activate.bat

# Intall requirements
pip install -r requirements.txt
```

## Videos Release

| Path                        | Title           | Datetime                      | size [kB] | Duration [s] | FPS | Command                                            | status |
|-----------------------------|-----------------|-------------------------------|-----------|--------------|-----|----------------------------------------------------|--------|
| atittude_spel/data/20230824 | attitude-1.mp4  | 2023-08-24 15:44:52.230000000 | 122       | 5            | 30  | libcamera-vid --width 320 --height 180 -n -t 5000  | ok     |
| atittude_spel/data/20230824 | attitude-2.mp4  | 2023-08-24 15:45:11.640000000 | 42        | 5            | 30  | libcamera-vid --width 160 --height 90 -n -t 5000   | ok     |
| atittude_spel/data/20230824 | attitude-3.mp4  | 2023-08-24 17:18:51.240000000 | 89        | 5            | 30  | libcamera-vid --width 320 --height 180 -n -t 5000  | ok     |
| atittude_spel/data/20230824 | attitude-4.mp4  | 2023-08-24 17:19:08.470000000 | 39        | 5            | 30  | libcamera-vid --width 160 --height 90 -n -t 5000   | ok     |
| atittude_spel/data/20230904 | attitude-5.mp4  | 2023-09-04 15:48:51.380000000 | 266       | 5            | 30  | libcamera-vid --width 320 --height 180 -n -t 5000  | lost   |
| atittude_spel/data/20230904 | attitude-6.mp4  | 2023-09-04 15:49:04.700000000 | 247       | 10           | 30  | libcamera-vid --width 160 --height 90 -n -t 10000  | ok     |
| atittude_spel/data/20230904 | attitude-7.mp4  | 2023-09-04 17:21:21.850000000 | 852       | 5            | 30  | libcamera-vid --width 320 --height 180 -n -t 5000  | ok     |
|                             | attitude-8.mp4  |                               |           | 10           | 30  | libcamera-vid --width 160 --height 90 -n -t 10000  | fail   |
| atittude_spel/data/20230904 | attitude-9.mp4  | 2023-09-04 17:22:40.280000000 | 387       | 10           | 30  | libcamera-vid --width 160 --height 90 -n -t 10000  | ok     |
| atittude_spel/data/20231113 | attitude-10.mp5 | 2023-11-13 15:49:05.240000000 | 137       |              |     |                                                    |        |
| atittude_spel/data/20231113 | attitude-11.mp4 | 2023-11-13 15:50:53.970000000 | 165       |              |     |                                                    |        |
| atittude_spel/data/20231113 | attitude-12.mp4 | 2023-11-13 15:53:19.480000000 | 53        |              |     |                                                    |        |
| atittude_spel/data/20231113 | attitude-13.mp4 | 2023-11-13 15:55:28.320000000 | 127       |              |     |                                                    |        |
| atittude_spel/data/20231211 | attitude-14.mp4 | 2023-12-11 15:20:11.690000000 | 157       | 10           | 30  | libcamera-vid --width 160 --height 90 -n -t 10000  | ok     |
| atittude_spel/data/20231211 | attitude-15.mp4 | 2023-12-11 15:20:49.500000000 | 64        | 5            | 30  | libcamera-vid --width 320 --height 180 -n -t 5000  | ok     |
| atittude_spel/data/20231211 | attitude-16.mp5 | 2023-12-11 15:22:15.880000000 | 68        | 10           | 30  | libcamera-vid --width 160 --height 90 -n -t 10000  | ok     |
| atittude_spel/data/20231211 | attitude-17.mp4 | 2023-12-11 15:24:53.000000000 | 253       | 60           | 30  | libcamera-vid --width 160 --height 90 -n -t 60000  | ok     |


***Contact***

- els.obrq@gmail.com
