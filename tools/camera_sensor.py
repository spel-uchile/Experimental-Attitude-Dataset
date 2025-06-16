"""
Created by Elias Obreque
els.obrq@gmail.com
Date: 15-09-2022

reference
https://link.springer.com/article/10.1007/s12567-022-00461-0#Equ6
"""
import os
from PIL import Image
import numpy as np
from src.dynamics.quaternion import Quaternions
from tools.conv_im_to_vec import get_file_info, get_earth_pointing
from tools.get_point_vector_from_picture import get_vector_v2
from tools.mathtools import inertial_to_lvlh

radius_earth = 6378.137  # km
radius_sun = 696340 # km
RAD2DEG = 180 / np.pi
DEG2RAD = 1 / RAD2DEG


class CamSensor:
    sensor_width_h = 3.68e-3  # m
    sensor_width_v = 2.76e-3  # m
    resolution_h = 320#3280
    resolution_v = 240#2464
    sensor_pixel_h = sensor_width_h / resolution_h
    sensor_pixel_v = sensor_width_v / resolution_v
    fov_h = 62.2  # deg
    fov_v = 48.8  # deg
    focal_length = 0.00304 # m

    # sensor_width = 2.76 * 1e-3 # m

    def __init__(self, q_c2b=None, r_c2b=None, debug=False, add_filter=False, target_resolution=102):
        self.normal_vec_c = np.array([1.0, 0.0, 0.0])
        self.q_b2c = None
        self.debug_test = debug
        self.r_b2c = None
        if q_c2b is not None:
            self.q_b2c = Quaternions(q_c2b.conjugate())
            self.q_b2c.normalize()
            self.normal_vec_b = q_c2b.frame_conv(self.normal_vec_c)
        else:
            self.r_b2c = r_c2b.T
            self.normal_vec_b = r_c2b @ self.normal_vec_c


        self.current_imagen = np.zeros((self.resolution_v, self.resolution_h), dtype=float)
        self.half_ver = np.deg2rad(self.fov_v) / 2
        self.half_hor = np.deg2rad(self.fov_h) / 2

        self.target_resolution = target_resolution
        self.add_filter_flag = add_filter
        temph = self.focal_length * np.tan(self.half_hor)
        tempv = self.focal_length * np.tan(self.half_ver)

        px = np.linspace(-temph, temph, self.resolution_h)
        py = np.linspace(-tempv, tempv, self.resolution_v)

        self.px_full, self.py_full = np.meshgrid(px, py)

        self.historical_imagen = []
        self.condition_ = None

    def compute_picture(self, current_q_lvlh2b: np.ndarray, earth_pos_i_: np.ndarray, sc_vel_i_: np.ndarray, sun_pos_sc_i_: np.ndarray):
        lvlh_z = earth_pos_i_ / np.linalg.norm(earth_pos_i_)
        lvlh_y = np.cross(earth_pos_i_, sc_vel_i_) / np.linalg.norm(np.cross(earth_pos_i_, sc_vel_i_) )
        lvlh_x = np.cross(lvlh_y, lvlh_z)
        matrix_i2lvlh = np.array([lvlh_x, lvlh_y, lvlh_z])

        earth_pos_lvlh_ = matrix_i2lvlh @ earth_pos_i_
        sun_pos_lvlh_ = matrix_i2lvlh @ sun_pos_sc_i_
        self.measure(current_q_lvlh2b, earth_pos_lvlh_, sun_pos_lvlh_)

    def compute_picture_from_vectors(self, earth_c_v, sun_c_v):
        self.current_imagen *= 0
        if np.linalg.norm(earth_c_v) > 0:
            self.add_body_in_picture(earth_c_v, radius_earth)
        if np.linalg.norm(sun_c_v) > 0:
            self.add_body_in_picture(sun_c_v, radius_sun * 15)

        if self.add_filter_flag:
            self.add_filter()

    def measure(self, q_lvlh2b, earth_pos_lvlh_, sun_pos_lvlh_):
        q_lvlh2b = Quaternions(q_lvlh2b)
        q_lvlh2b.normalize()

        earth_pos_b_ = q_lvlh2b.frame_conv(earth_pos_lvlh_)
        sun_pos_b_ = q_lvlh2b.frame_conv(sun_pos_lvlh_)

        if self.q_b2c is not None:
            earth_pos_c = self.q_b2c.frame_conv(earth_pos_b_)
            sun_pos_c = self.q_b2c.frame_conv(sun_pos_b_)
        else:
            earth_pos_c = self.r_b2c @ earth_pos_b_
            sun_pos_c = self.r_b2c @ sun_pos_b_

        self.current_imagen = np.zeros((self.resolution_v, self.resolution_h), dtype=float)
        if np.linalg.norm(earth_pos_c) > 100:
            self.add_body_in_picture(earth_pos_c, radius_earth)
        if np.linalg.norm(sun_pos_c) > 100:
            self.add_body_in_picture(sun_pos_c, radius_sun * 15)

        if self.add_filter_flag:
            self.add_filter()

        if self.debug_test:
            self.compare_simulation()

    def add_filter(self):
        self.current_imagen = self.crop_to_square(self.current_imagen)
        self.current_imagen = self.resize_gray_array(self.current_imagen, self.target_resolution, "nearest")

    @staticmethod
    def resize_gray_array(
            square_array: np.ndarray,
            new_size: int,
            interpolation: str = 'bilinear'
    ) -> np.ndarray:
        """
        Resize a square 2D grayscale NumPy array to (new_size, new_size) using PIL.

        - square_array: 2D NumPy array of shape (S, S) (must be square).
        - new_size: integer, desired output dimension (new_size x new_size).
        - interpolation: one of 'nearest', 'bilinear', 'bicubic', 'lanczos'. Default 'bilinear'.

        Returns a 2D NumPy array of shape (new_size, new_size), dtype=uint8 if input was uint8,
        or float if input was float (values clipped to [0,1] or [0,255] accordingly).
        """
        # Map string to PIL interpolation mode
        interp_map = {
            'nearest': Image.NEAREST,
            'bilinear': Image.BILINEAR,
            'bicubic': Image.BICUBIC,
            'lanczos': Image.LANCZOS
        }
        if interpolation not in interp_map:
            raise ValueError(f"Unsupported interpolation '{interpolation}'. "
                             f"Choose from {list(interp_map.keys())}.")

        pil_mode = interp_map[interpolation]

        # Convert NumPy array to PIL Image.
        # Determine how to handle dtype:
        #  - If input is float (assuming [0,1] range), scale to [0,255] uint8 for PIL, then back to float.
        #  - If input is uint8, use directly.
        orig_dtype = square_array.dtype
        if np.issubdtype(orig_dtype, np.floating):
            # Clip values to [0,1], then scale
            array_clipped = np.clip(square_array, 0.0, 1.0)
            pil_in = Image.fromarray((array_clipped * 255.0).round().astype(np.uint8))
        else:
            # Assume integer format (e.g. uint8). Clamp to [0,255] just in case.
            arr_uint8 = np.clip(square_array, 0, 255).astype(np.uint8)
            pil_in = Image.fromarray(arr_uint8)

        # Resize with PIL
        pil_resized = pil_in.resize((new_size, new_size), resample=pil_mode)

        # Convert back to NumPy
        arr_resized = np.array(pil_resized)

        # If original was floating, convert back to float in [0,1]
        if np.issubdtype(orig_dtype, np.floating):
            arr_resized = arr_resized.astype(np.float32) / 255

        return arr_resized

    @staticmethod
    def crop_to_square(gray_array: np.ndarray) -> np.ndarray:
        """
        Crop a 2D grayscale array to a centered square, whose side is the
        smaller of the two dimensions.

        - gray_array: 2D NumPy array of shape (H, W), dtype can be uint8, float, etc.
        - Returns a 2D array of shape (min(H, W), min(H, W)), same dtype as input.
        """
        # Get original dimensions
        height, width = gray_array.shape

        # Determine new side length (the smaller dimension)
        new_side = min(height, width)

        # Compute offsets to center the crop
        top = (height - new_side) // 2
        left = (width - new_side) // 2

        # Perform the crop
        cropped = gray_array[top: top + new_side, left: left + new_side]
        return cropped

    def compare_simulation(self):
        earth_pos_c = np.array([-1000, radius_earth + 500, 3000])
        altitude_earth = np.linalg.norm(earth_pos_c) - radius_earth
        sun_pos_c = np.array([0.067, -0.285, 0.92]) * 149597870.70
        # sun_pos_c = np.array([0, 0, 1]) * 149597870.70

        altitude_sun = np.linalg.norm(sun_pos_c) - radius_sun
        self.current_imagen = np.zeros((self.resolution_v, self.resolution_h), dtype=float)
        if np.linalg.norm(earth_pos_c) > 0:
            self.add_body_in_picture(earth_pos_c, radius_earth)
        if np.linalg.norm(sun_pos_c) > 0:
            self.add_body_in_picture(sun_pos_c, radius_sun * 15)

        if self.add_filter_flag:
            self.add_filter()

        plt.title("Test Earth and sun")
        plt.imshow(self.current_imagen, cmap=plt.cm.gray, vmin=0, vmax=1)

        gray_uint8 = (self.current_imagen * 255).astype(np.uint8)

        # Now stack three identical channels to form an H × W × 3 array (uint8).
        rgb_uint8 = np.stack((gray_uint8,) * 3, axis=-1)  # shape = (H, W, 3), dtype=uint8

        # --------------------------------------------------------------
        # Step 3: Create a PIL Image from the RGB array and save it
        # --------------------------------------------------------------
        # Mode "RGB" expects a 3D uint8 array
        img = Image.fromarray(rgb_uint8, mode='RGB')
        # Save to disk (e.g., PNG or JPEG)
        img.save('grayscale_as_rgb.png')
        edge_, img_cv2_, p_, r_, center_point_c = get_vector_v2('grayscale_as_rgb.png', altitude_earth,
                                                                altitude_sun, self.sensor_pixel_v, self.sensor_pixel_v,
                                                                self.focal_length)

        earth_error = np.arccos(earth_pos_c / np.linalg.norm(earth_pos_c) @ center_point_c['Earth_c']) * np.rad2deg(1)
        sun_error = np.arccos(sun_pos_c / np.linalg.norm(sun_pos_c) @ center_point_c['Sun_c']) * np.rad2deg(1)

        fig_, ax = plt.subplots(1, 2, figsize=(10, 7))
        # fig_list.append(fig_)
        fig_.suptitle("Angular error between vectors - "
                      f"Earth: {np.round(earth_error, 4)} [deg] - Sun: {np.round(sun_error, 4)} [deg]")
        ax[0].set_title(f"Simulated picture and vectors"
                        f" \nTrue earth: {np.round(earth_pos_c / np.linalg.norm(earth_pos_c), 4)}"
                        f" \nTrue Sun: {np.round(sun_pos_c / np.linalg.norm(sun_pos_c), 4)}")
        ax[0].imshow(self.current_imagen, cmap=plt.cm.gray, vmin=0, vmax=1)
        ax[0].hlines(self.current_imagen.shape[0] * 0.5, 0, self.current_imagen.shape[1] - 0.5)
        ax[0].vlines(self.current_imagen.shape[1] * 0.5, 0, self.current_imagen.shape[0] - 0.5)
        # ax[0].scatter(self.current_imagen.shape[0] * 0.5, self.current_imagen.shape[1] * 0.5, marker='x',
        #               color='red')
        ax[0].grid()

        ax[1].set_title("Get vector in simulated picture"
                        f" \nCalculated Earth: {np.round(center_point_c['Earth_c'], 4)}"
                        f" \nCalculated Sun: {np.round(center_point_c['Sun_c'], 4)}")
        if img_cv2_ is not None:
            ax[1].imshow(img_cv2_[..., ::-1])
        else:
            ax[1].imshow(np.zeros_like(self.current_imagen))
        ax[1].hlines(self.current_imagen.shape[0] * 0.5, 0, self.current_imagen.shape[1] - 0.5)
        ax[1].vlines(self.current_imagen.shape[1] * 0.5, 0, self.current_imagen.shape[0] - 0.5)
        ax[1].grid()
        fig_.savefig('res_grayscale_as_rgb.png', dpi=300)
        plt.show()


    def add_body_in_picture(self, input_pos_c, body_radii):
        unit_input_pos_c = input_pos_c / np.linalg.norm(input_pos_c)

        a, b, c, d, e, f = self.get_coefficients(unit_input_pos_c, np.linalg.norm(input_pos_c), body_radii)
        print("Conical values from vector: ", unit_input_pos_c, a, b, c, d, e, f)
        conical_value = a * self.px_full ** 2 + b * self.py_full * self.px_full + c * self.py_full ** 2 + d * self.px_full + e * self.py_full + f
        self.current_imagen[conical_value >= 0] = 1

    def get_coefficients(self, e_c, me, body_re):
        if np.linalg.norm(me) * 1e-3  - body_re < 0.0:
            h = me - body_re
        else:
            h = me * 1e-3 - body_re
        da = 0.0

        alpha = np.arcsin((body_re + da) / (body_re + h))
        exc = e_c[0]
        eyc = e_c[1]
        ezc = e_c[2]

        a = exc ** 2 - np.cos(alpha) ** 2
        b = 2 * exc * eyc
        c = eyc ** 2 - np.cos(alpha) ** 2
        d = 2 * exc * ezc * self.focal_length
        e = 2 * eyc * ezc * self.focal_length
        f = self.focal_length ** 2 * (ezc ** 2 - np.cos(alpha) ** 2)
        return a, b, c, d, e, f

    def get_view_contour(self):
        from matplotlib.patches import Rectangle
        half_length = self.focal_length * np.tan(self.half_hor)
        view = Rectangle((-half_length, -half_length), 2 * half_length, 2 * half_length, fc='none', ec='k', lw=2)
        return view

    def get_log_values(self, subsys):
        report = {}
        return report

    def log_value(self):
        self.historical_imagen.append(self.current_imagen)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np
    import matplotlib
    from scipy.spatial.transform import Rotation

    tle_file = "../data/sat000052191.txt"
    ROT_CAM2BODY = Rotation.from_euler('zx', [180, -90], degrees=True).inv().as_matrix()

    cam_test = CamSensor(r_c2b=ROT_CAM2BODY, debug=True, add_filter=True, target_resolution=102)
    cam_test.compute_picture(np.array([0, 0, 0, 1]), np.array([1, 0, 0]), np.array([0, 1, 0]), np.ones(3))

    folder = "example_picture/"
    name_files = [elem for elem in os.listdir(folder) if "jpg" in elem]

    dataset = []
    for earth_image in name_files:
        jd, sat, node, pos_i, vel_i, sun_pos_sc_i, l1, l2 = get_file_info(earth_image, tle_file) # jd, sat, node, pos, vel, sun_pos_from_sc, line1, line2
        altitude_earth = np.linalg.norm(pos_i) - radius_earth
        altitude_sun = np.linalg.norm(sun_pos_sc_i) - radius_sun
        edge_, img_cv2_, p_, r_, center_point_c  = get_vector_v2(folder + earth_image, altitude_earth, altitude_sun,
                                                                 CamSensor.sensor_pixel_v, CamSensor.sensor_pixel_h,
                                                                 CamSensor.focal_length)
        # center_im_list, center_point_c, bw_temp, edge, radius_m, _ = get_earth_pointing(folder + earth_image,
        #                                                                              0.0035, 2.74, altitude,
        #                                                                                 threshold=0.9) # name_file, fl, sw, height, threshold=0.98

        dict_info = {'jd': jd, 'pos': pos_i, 'vel': vel_i, 'sat': sat, 'node': node, 'sun_pos_sc_i': sun_pos_sc_i,
                     'img': img_cv2_, 'earth_pos_c': center_point_c['Earth_c'], 'sun_pos_c': center_point_c['Sun_c'],}
        dataset.append(dict_info)

    # Earth vector
    earth_pos_b_list = [ROT_CAM2BODY @ elem['earth_pos_c'] for elem in dataset]
    earth_pos_lvlh = np.array([0, 0, 1])

    # Sun vector
    sun_pos_b_list = [ROT_CAM2BODY @ elem['sun_pos_c'] for elem in dataset]
    sun_pos_lvlh = [inertial_to_lvlh(d_['sun_pos_sc_i'], d_['pos'], d_['vel']) for d_ in dataset]

    #
    print(earth_pos_b_list[0])
    rot_list = [np.cross(earth_pos_lvlh, earth_pos_b_i) for earth_pos_b_i in earth_pos_b_list]
    ang_list = [np.arccos(earth_pos_lvlh.dot(earth_pos_b_i)) for earth_pos_b_i in earth_pos_b_list]

    yaw_list = [0.0] * len(earth_pos_b_list)
    pitch_list = [-np.arcsin(earth_pos_b_i[0]) for earth_pos_b_i in earth_pos_b_list]
    roll_list = [np.arctan2(earth_pos_b_i[1], earth_pos_b_i[2]) for earth_pos_b_i in earth_pos_b_list]
    q_lvlh2b_m = [Quaternions.quat_from_ypr(0, pitch_, roll_) for pitch_, roll_ in zip(pitch_list, roll_list)]

    cam = [CamSensor(r_c2b=ROT_CAM2BODY, debug=False, add_filter=True, target_resolution=102) for i in range(len(dataset))]

    # simulation with quaternion
    # earth_perimeter = [cam[i].compute_picture(q(), d_['pos'], d_['vel'], d_['sun_pos_sc_i']) for i, (q, d_) in enumerate(zip(q_lvlh2b_m, dataset))]

    # direct simulation of the picture
    earth_perimeter = []
    for i, (q, d_) in enumerate(zip(q_lvlh2b_m, dataset)):
        earth_pos_c_ = d_['earth_pos_c'] * np.linalg.norm(d_['pos'])
        sun_pos_c_ = d_['sun_pos_c'] * np.linalg.norm(d_['sun_pos_sc_i'])
        earth_perimeter.append(cam[i].compute_picture_from_vectors(earth_pos_c_, sun_pos_c_))

    cam_view_image = [cam_.current_imagen for cam_ in cam]


    for i in range(len(dataset)):
        fig_, ax = plt.subplots(1, 2)
        # fig_list.append(fig_)
        fig_.suptitle("Camera frame\nYaw: {:.2f}° - Pitch: {:.2f}° - Roll: {:.2f}°".format(yaw_list[i] * RAD2DEG,
                                                                                       roll_list[i] * RAD2DEG,
                                                                                       pitch_list[i] * RAD2DEG))
        if dataset[i]['img'] is not None:
            ax[0].imshow(dataset[i]['img'][..., ::-1])
        else:
            ax[0].imshow(np.zeros_like(cam_view_image[i]))
        ax[1].imshow(cam_view_image[i], cmap=plt.cm.gray, vmin=0, vmax=1)
        ax[1].hlines(cam_view_image[i].shape[0] * 0.5, 0, cam_view_image[i].shape[1] - 0.5)
        ax[1].vlines(cam_view_image[i].shape[0] * 0.5, 0, cam_view_image[i].shape[1] - 0.5)
        ax[1].scatter(cam_view_image[i].shape[0] * 0.5, cam_view_image[i].shape[1] * 0.5, marker='x', color='red')
        ax[1].grid()
        fig_.savefig(folder + "/results/sim_picture" + name_files[i] + ".png", dpi=300)
        plt.close()
