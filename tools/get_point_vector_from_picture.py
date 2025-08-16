"""
Created by Elias Obreque
Date: 16-09-2023
email: els.obrq@gmail.com
"""
import time
from typing import Any

from PIL import Image, ImageFilter, ImageDraw
from numpy import floating
# from tools.my_scipy import optimize
from scipy import optimize, ndimage
# from tools.my_scipy.optimize import fsolve
from scipy.spatial.transform import Rotation
from scipy.optimize import fsolve
from tools.clustering import cluster, decrease_color
import numpy as np
import os
import matplotlib.pyplot as plt
import cv2
import matplotlib as mpl

mpl.rcParams['font.size'] = 12

re = 6378.137  # km
RAD2DEG = 180 / np.pi
DEG2RAD = 1 / RAD2DEG
au = 149597870.691  # km
d_sun = 1.3927e-9
rs = au
sensor_width_h = 3.68e-3  # m
sensor_width_v = 2.76e-3  # m

# sensor_pixel_h = sensor_width_h / 3280
# sensor_pixel_v = sensor_width_v / 2464
fov_h = 62.2 # deg
fov_v = 48.8 # deg
ROT_CAM2BODY = Rotation.from_euler('zx', [180, -90], degrees=True).inv().as_matrix()
# sensor_width = 2.76 * 1e-3  # m
# focal_length = 0.00304

MAX_NUM_LINES = 3

import numpy as np


def R1(psi):
    """
    Rotation matrix around the X-axis by angle psi (in radians).
    """
    return np.array([
        [1, 0, 0],
        [0, np.cos(psi), np.sin(psi)],
        [0, -np.sin(psi), np.cos(psi)]
    ])

def R2(theta):
    """
    Rotation matrix around the Y-axis by angle theta (in radians).
    """
    return np.array([
        [np.cos(theta), 0, -np.sin(theta)],
        [0, 1, 0],
        [np.sin(theta), 0, np.cos(theta)]
    ])

def R3(phi):
    """
    Rotation matrix around the Z-axis by angle phi (in radians).
    """
    return np.array([
        [np.cos(phi), np.sin(phi), 0],
        [-np.sin(phi), np.cos(phi), 0],
        [0, 0, 1]
    ])


def intersection(line1, line2):
    """Finds the intersection of two lines given in Hesse normal form.

    Returns closest integer pixel locations.
    See https://stackoverflow.com/a/383527/5087436
    """
    rho1, theta1 = line1[0]
    rho2, theta2 = line2[0]
    A = np.array([
        [np.cos(theta1), np.sin(theta1)],
        [np.cos(theta2), np.sin(theta2)]
    ])
    b = np.array([[rho1], [rho2]])

    def func(x):
        return (A @ x - b.T)[0]

    xsol, info, status, msg = fsolve(func, x0=np.zeros(2), full_output=1)
    # x0, y0 = np.linalg.solve(A, b)
    if status == 1:
        x0, y0 = int(np.round(xsol[0])), int(np.round(xsol[1]))
        return [[x0, y0]]
    else:
        return None


def get_lines(img_, center_guess):
    img_cv2 = cv2.cvtColor(np.asarray(img_), cv2.COLOR_RGB2BGR)
    gray = img_.convert('L')
    # Apply edge detection method on the image
    gray = (np.asarray(gray) / 255) ** 0.7 * 255
    gray = np.asarray(gray, dtype=np.uint8)
    kernel = np.ones((3, 3), np.uint8)
    img_dilation = cv2.dilate(np.asarray(gray, dtype=np.uint8), kernel, iterations=1)
    edges = cv2.Canny(gray, 25, 28, apertureSize=3)

    lines = cv2.HoughLines(edges, 2, 0.1, 35)
    count = 0
    max_count = 100
    ths_val = 50
    center_point = None
    lines_list = []
    thr_list = []
    if lines is not None:
        while len(lines) != MAX_NUM_LINES:
            if len(lines) > MAX_NUM_LINES:
                ths_val += 1
            else:
                ths_val -= 1
            lines = cv2.HoughLines(edges, 1, 0.1, ths_val)
            count += 1
            if count > max_count or lines is None:
                print(ths_val, len(lines)) if lines is not None else None

                break

    # Draw the lines
    if lines is not None:
        for i in range(0, len(lines)):
            rho = lines[i][0][0]
            theta = lines[i][0][1]
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * (a)))
            pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * (a)))
            cv2.line(img_cv2, pt1, pt2, (0, 0, 255), 1, cv2.LINE_AA)
    return edges, img_cv2


def rgb_to_gray(R, G, B):
    return int(R * 299 / 1000 + G * 587 / 1000 + B * 114 / 1000)


def get_body(col_, file_name, show=True):
    new_col, lighter_colors, counts = decrease_color(col_, 10)
    color_label = []

    thereisblack = False
    for elem in lighter_colors:
        finding_color = {}
        finding_color['black'] = True if np.mean(elem) < 20 else False
        finding_color['white'] = True if np.mean(elem) > 240 else False
        finding_color['red'] = False
        finding_color['green'] = False
        finding_color['blue'] = False
        if (finding_color['white'] or finding_color['black']) is False:
            finding_color['red'] = True if elem[0] > np.mean(elem) + 30 else False
            finding_color['green'] = True if elem[1] > np.mean(elem) + 30 else False
            finding_color['blue'] = True if elem[2] > np.mean(elem) + 30 else False
        thereisblack = finding_color['black']
        color_label.append(finding_color)

    # Blue filter
    for i, color_l in enumerate(color_label):
        if color_l['blue'] and thereisblack:
            blue_no_black = np.max(new_col, axis=2) - np.min(new_col, axis=2) > 10
            new_col[blue_no_black] = lighter_colors[0]
    #
    col = col_.copy()
    col.putdata([tuple(colors) for colors in new_col.reshape(-1, 3)])

    gray = col.convert('L')
    # gray = np.asarray(col)[:, :, 2]
    lighter_gray = [rgb_to_gray(r, g, b) for r, g, b in zip(lighter_colors[:, 0],
                                                            lighter_colors[:, 1],
                                                            lighter_colors[:, 2])]
    gray = gray.filter(ImageFilter.SMOOTH)
    bw = np.asarray(gray).copy()
    bw = bw / 255

    # if counts[-1] > 4 * counts[-2] and color_label[-1]['black']:
    #     bw = bw ** (1 / 2)

    # hist = np.histogram(bw, bins=256)[1]
    # max_hist = np.max(hist)
    # min_hist = np.min(hist)
    mean_cut = np.mean(bw)
    bw_mean = bw.copy()
    bw_mean[np.where(bw < mean_cut)] = 0.0
    bw_mean[np.where(bw >= mean_cut)] = 1.0

    bw[np.where(bw < 0.98 * max(lighter_gray) / 255)] = 0.0
    bw[np.where(bw >= 0.98 * max(lighter_gray) / 255)] = 1.0
    bw_temp = []

    # labeled_array, num_features = etiquetar_objetos(bw)
    labeled_array, num_features = ndimage.label(bw)
    if num_features > 1:
        k = 0
        for i in range(num_features + 1):
            points = np.where(labeled_array == i)
            centroid = int(sum(points[0]) / len(points[0])), int(sum(points[1]) / len(points[1]))
            if bw[centroid] == 1 and len(points[0]) > 5:  # is object
                temp = np.zeros(bw.shape)
                temp[points] = 1
                bw_temp.append(temp)
                k += 1
                if k == 2:
                    break
    else:
        bw_temp.append(bw)

    # fig_h, ax_h = plt.subplots()
    fig, ax = plt.subplots(1, 7, figsize=(15, 5), sharey=True)
    fig.suptitle(file_name.split("/")[-1][:-4])
    fig.tight_layout()
    ax[0].imshow(col_)
    ax[0].set_title("Original")

    # ax_h.plot(np.arange(0, bw.shape[1]), np.sum(bw, axis=0), label="X")
    # ax_h.plot(np.arange(0, bw.shape[0]), np.sum(bw, axis=1), label="Y")
    # ax_h.set_title("Histogram")
    # ax_h.grid()
    # ax_h.legend()

    ax[1].imshow(new_col/255, vmin=0, vmax=1)
    ax[1].set_title("KMeans")

    ax[2].imshow(np.asarray(gray)/ 255, vmin=0, vmax=1)
    ax[2].set_title("Gray")

    ax[3].imshow(bw_mean, vmin=0, vmax=1)
    ax[3].set_title("Binarize - Mean")

    ax[4].imshow(bw, vmin=0, vmax=1)
    ax[4].set_title("Binarize - Blue Filter")

    if len(bw_temp) > 1:
        for k_, temp_ in enumerate(bw_temp):
            ax[5 + k_].imshow(temp_, vmin=0, vmax=1)
            ax[5 + k_].set_title("Body - {}".format(k_))
    else:
        ax[5].imshow(bw)
        ax[5].set_title("Body")
        ax[6].imshow(bw * 0)
        ax[6].set_title("None")
    name_folder = 'process'
    folder_ = ''.join([el + '/' for el in file_name.split('/')[:-1]])
    if not os.path.exists(folder_ + name_folder):
        os.makedirs(folder_ + name_folder)
    fig.savefig("{}_process.png".format(folder_ + name_folder + '/' + file_name.split('/')[-1][:-4]))
    # fig_h.savefig("{}_hist.png".format(folder_ + name_folder + '/' + file_name.split('/')[-1][:-4]))
    if show:
        plt.show()
    else:
        plt.close(fig=fig)
    return bw_temp


def etiquetar_objetos(imagen):
    # Inicializamos la matriz de etiquetas
    etiquetas = np.zeros_like(imagen, dtype=int)
    etiqueta_actual = 0

    # Recorremos la imagen
    for i in range(imagen.shape[0]):
        for j in range(imagen.shape[1]):
            # Si el pixel actual es parte de un objeto (no es cero)
            # y aún no ha sido etiquetado, le asignamos una nueva etiqueta
            if imagen[i, j] != 0 and etiquetas[i, j] == 0:
                etiqueta_actual += 1
                etiquetas[i, j] = etiqueta_actual

                # Buscamos todos los otros pixeles conectados a este
                # y les asignamos la misma etiqueta
                pixeles_a_chequear = [(i, j)]
                while pixeles_a_chequear:
                    x, y = pixeles_a_chequear.pop()
                    for dx in [-1, 0, 1]:
                        for dy in [-1, 0, 1]:
                            nx, ny = x + dx, y + dy
                            if (0 <= nx < imagen.shape[0] and 0 <= ny < imagen.shape[1] and
                                    imagen[nx, ny] != 0 and etiquetas[nx, ny] == 0):
                                etiquetas[nx, ny] = etiqueta_actual
                                pixeles_a_chequear.append((nx, ny))
    return etiquetas, etiqueta_actual


def metrics_for_contour(cnt, r, a_):
    P_ = cv2.arcLength(cnt, True)
    A_ = max(cv2.contourArea(cnt), 1e-9)
    hull = cv2.convexHull(cnt)
    A_hull = abs(cv2.contourArea(hull))
    circularity = (4 * np.pi * A_hull) / (P_ * P_ + 1e-9) # 1
    arc_cov = (P_ / (2 * np.pi * r)) if r > 1e-6 else 0.0
    return circularity, arc_cov


def get_type_radius(im_list):

    def calc_R(yc, xc, x, y):
        """ calculate the distance of each data points from the center (xc, yc) """
        return np.sqrt((x - xc) ** 2 + (y - yc) ** 2)

    def f_2b(c, data):
        """ calculate the algebraic distance between the 2D points and the mean circle centered at c=(xc, yc) """
        # TODO: revisar referencias de pixeles
        # c = [cv, ch]
        x = data[:, 1]
        y = data[:, 0]
        Ri = calc_R(*c, x, y)
        return Ri - Ri.mean()

    def Df_2b(c, data):
        # c = [cv, ch]
        """ Jacobian of f_2b
        The axis corresponding to derivatives must be coherent with the col_deriv option of leastsq"""
        yc, xc = c
        x = data[:, 1]
        y = data[:, 0]
        df2b_dc = np.empty((len(c), x.size))

        Ri = calc_R(yc, xc, x, y)
        df2b_dc[1] = (xc - x) / Ri  # dR/dxc
        df2b_dc[0] = (yc - y) / Ri  # dR/dyc
        df2b_dc = df2b_dc - df2b_dc.mean(axis=1)[:, np.newaxis]
        return df2b_dc

    point_list = []
    center_2d_index_list = []
    radius_list = []
    edge_array_list = []
    points_c_list = []
    rad_value = []
    metrics = {'circularity': [], 'arc_cov': []}
    for im in im_list:
        # contours, hierarchy = cv2.findContours(np.array(im, dtype=np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        gx, gy = np.gradient(im)[0], np.gradient(im)[1]
        gr = np.hypot(gx, gy)
        gr[np.where(gr < 0.3)] = 0
        points = np.where(gr > 0.3)
        point_list.append(points)
        edge_array = np.array([points[0], points[1]]).T

        center_estimate = np.array(np.mean(edge_array, axis=0), dtype=int)
        #(cx, cy), r = cv2.minEnclosingCircle(edge_array)  # círculo que encierra al contorno

        center_2d_index, ier = optimize.leastsq(f_2b, center_estimate, args=edge_array, Dfun=Df_2b, col_deriv=True)
        center_2d_index_list.append(center_2d_index)
        radius = calc_R(center_2d_index[1], center_2d_index[0], edge_array[:, 0], edge_array[:, 1])
        radius_list.append([np.mean(radius), np.std(radius)])
        edge_array_list.append(edge_array)
        rad_value.append(np.mean(radius))
        #points_c_list.append(points_center)
        c_, a_c_ = metrics_for_contour(edge_array, np.mean(radius), np.sum(im))
        metrics['circularity'].append(c_)
        metrics['arc_cov'].append(a_c_)
    n = len(rad_value)
    order = sorted(range(n), key=lambda i: rad_value[i], reverse=True)

    point_list = [point_list[i] for i in order]
    center_2d_index_list = [center_2d_index_list[i] for i in order]
    radius_list = [radius_list[i] for i in order]
    rad_value = [rad_value[i] for i in order]
    metrics['circularity'] = [metrics['circularity'][i] for i in order]
    metrics['arc_cov'] = [metrics['arc_cov'][i] for i in order]
    return radius_list, point_list, center_2d_index_list, metrics


def get_type_radius_cv2(opencv_binary_imgs):
    """
    Given a list of binary images (each pixel is either 0 or 255),
    detect circular contours and return for each image a tuple:
       (earth_center, earth_radius, sun_center, sun_radius)
    If only one contour is found, classify it based on its position
    (touching bottom edge => Earth; otherwise => Sun).
    """
    results = []
    radius_list, point_list = [], []
    for im_bin in opencv_binary_imgs:
        # -------------------------------------------------
        # Step 1: Make sure the input is a uint8 binary image
        # -------------------------------------------------
        # If the input comes as floats in [0,1] or bool, convert to 0/255 uint8
        if im_bin.dtype != np.uint8:
            im = (im_bin.astype(np.float32) * 255.0).astype(np.uint8)
        else:
            im = im_bin.copy()

        # Ensure binary format: foreground = 255, background = 0
        # (In case some pixels are not exactly 0 or 255)
        _, im_thresh = cv2.threshold(im, 200, 255, cv2.THRESH_BINARY)

        # -------------------------------------------------
        # Step 2: Find external contours
        # -------------------------------------------------
        # RETR_EXTERNAL: only outermost contours
        # CHAIN_APPROX_SIMPLE: compresses horizontal, vertical, diagonal segments

        contours, _ = cv2.findContours(
            im,
            cv2.RETR_TREE,
            cv2.CHAIN_APPROX_NONE
        )
        h, w = im_thresh.shape[:2]
        filtered = []
        for cnt in contours:
            # cnt is an array of shape (N,1,2) with points [[x,y]]
            pts = cnt.reshape(-1, 2)  # shape = (N,2)
            xs = pts[:, 0]
            ys = pts[:, 1]


            # Check if any point of this contour lies on the image border
            on_left = xs == 0
            on_right = xs == w - 1
            on_top = ys == 0
            on_bottom = ys == h - 1

            x_ = xs[~on_left & ~on_right & ~on_top & ~on_bottom]
            y_ = ys[~on_left & ~on_right & ~on_top & ~on_bottom]

            # Otherwise keep it
            cnt = np.array([y_, x_])
            filtered.append(cnt)

        # -------------------------------------------------
        # Step 3: For each contour, compute min enclosing circle
        # -------------------------------------------------

        img_h, img_w = im_thresh.shape[:2]
        areas = [cv2.contourArea(cnt) for cnt in contours]
        idx_big = np.argmax(areas)
        (x_center, y_center), radius = cv2.minEnclosingCircle(contours[idx_big])
        radius_list.append(radius)
        # point_list.append(contours[idx_big][:, 0].T)
        point_list.append(filtered[-1])
    return radius_list, point_list


def calc_conical_curvature(points, fl, pw, ph, shape_, h):
    # https://link.springer.com/article/10.1007/s12567-022-00461-0

    points_center = np.array([points[0], points[1]]).T
    # p_c[x, y, z]
    cx = shape_[1] * 0.5 * pw
    cy = shape_[0] * 0.5 * ph
    p_c = [np.array([(points_center[i])[1] * pw - cx, (points_center[i][0]) * ph - cy, fl]) for i in
           range(len(points_center))]
    hh = np.array(p_c)
    da = 50.0
    alpha = np.arcsin((re + da) / (re + h))
    y = np.cos(alpha) * np.array([np.linalg.norm(p_c_i) for p_c_i in p_c])
    e_c = np.linalg.inv(hh.T @ hh) @ hh.T @ y
    e_c = e_c / np.linalg.norm(e_c)

    exc = e_c[0]
    eyc = e_c[1]
    ezc = e_c[2]

    a = exc ** 2 - np.cos(alpha) ** 2
    b = 2 * exc * eyc
    c = eyc ** 2 - np.cos(alpha) ** 2
    d = 2 * exc * ezc * fl
    e = 2 * eyc * ezc * fl
    f = fl ** 2 * (ezc ** 2 - np.cos(alpha) ** 2)
    semi_major_axis = np.max([np.sqrt(np.abs(-f / a)), np.sqrt(np.abs(-f / c))])
    semi_minor_axis = np.min([np.sqrt(np.abs(-f / a)), np.sqrt(np.abs(-f / c))])

    if e_c[2] < 0.0:
        e_c[2] = np.abs(e_c[2])

    center_pixel = np.zeros(2, dtype=np.int16)
    if e_c[2] >= 0.01:
        vec_pix = (e_c * fl / np.abs(e_c[2]))[:2]
    else:
        vec_pix = e_c[:2]

    center_ref = np.array([vec_pix[0] / pw, vec_pix[1] / ph])

    quadratic = b ** 2  - 4 * a * c
    if quadratic < 0:
        conical_type = "ellipse"
        eccentricity = np.sqrt(1 - semi_minor_axis **2 / semi_major_axis **2 )
    elif quadratic == 0:
        conical_type = "parabolic"
        eccentricity = 1
    else:
        conical_type = "hyperbolic"
        eccentricity = np.sqrt(1 + semi_minor_axis ** 2 / semi_major_axis ** 2)
        center_ref = center_ref / np.min(np.abs(center_ref)) * np.max(shape_)

    center_pixel[0] = center_ref[0] + shape_[1] * 0.5
    center_pixel[1] = center_ref[1] + shape_[0] * 0.5

    center_pixel = center_pixel.astype(np.int32)
    # vector in body frame,
    e_b = ROT_CAM2BODY @ e_c
    e_b[0] *= -1
    # Pitch and roll to rotate from LVLH to Body frame
    pitch = np.arcsin(np.clip(e_b[1], -1.0, 1.0))
    roll = -np.arctan2(e_b[0], e_b[2])
    return points_center, e_c, center_pixel, pitch, roll, semi_major_axis, semi_minor_axis, eccentricity

def get_vector_v2(file_name, height_earth, height_sun, pixel_size_height, pixel_size_width, focal_length):
    edge_ = None
    pitch_, roll_ = np.nan, np.nan
    vectors_c = {'Earth_c': np.zeros(3), 'Sun_c': np.zeros(3)}
    try:
        col_o = Image.open(file_name)
        img_cv2_ = np.asarray(col_o)
        gray = np.asarray(col_o.convert('L'))
        if np.sum(gray) / 255 < 255 * 0.2: # min area sun
            return edge_, img_cv2_, pitch_, roll_, vectors_c
        t1 = time.time()
        # get earth and sun
        bw_bodies = get_body(col_o.copy(), file_name, show=False)
        print(f"Get bodies {time.time() - t1} seconds")

        radius_cv2, point_list_, center_2d_index_list, metrics_list_ = get_type_radius(bw_bodies)
        name_folder = 'vectorization'
        folder_ = ''.join([el + '/' for el in file_name.split('/')[:-1]])
        if not os.path.exists(folder_ + name_folder):
            os.makedirs(folder_ + name_folder)

        results = {"edge_pixels": [], "vector_c": [], "center_pixel": [], "pitch": [], "roll": [],
                   "s_major_a": [], "s_minor_a": [], "eccentricity": []}

        last_tag = []
        for radii, pl, bw_temp, center_aprox_, circ_, arc_cov_ in zip(radius_cv2, point_list_, bw_bodies, center_2d_index_list,
                                                     metrics_list_['circularity'], metrics_list_['arc_cov']):
            if (circ_ < 0.1 and arc_cov_ < 3.5 and radii[0] > 0.1 * max(bw_temp.shape) and radii[0] > 2 * radii[1]) and 'earth' not in last_tag:
                height = height_earth
                last_tag.append("earth")
            elif (circ_ >= 0.1 or arc_cov_ > 2) and 'sun' not in last_tag and radii[0] > 2 * radii[1]:
                height = height_sun
                last_tag.append("sun")
            else:
                continue
            print("Radii: ", radii[0], "circ: ", circ_, "arc_cov: ", arc_cov_, "- tag: ", last_tag[-1])
            body_edge_array, body_vec_c, center_pixel, pitch_, roll_, s_major_a, s_minor_a, eccentricity = calc_conical_curvature(pl,
                                                                                                                                  focal_length,
                                                                                                                                  pixel_size_width,
                                                                                                                                  pixel_size_height,
                                                                                                                                  bw_temp.shape,
                                                                                                                                  height)
            results["edge_pixels"].append(body_edge_array)
            results["vector_c"].append(ROT_CAM2BODY @ body_vec_c)
            results["center_pixel"].append(center_pixel)
            results["pitch"].append(pitch_)
            results["roll"].append(roll_)
            results["s_major_a"].append(s_major_a)
            results["s_minor_a"].append(s_minor_a)
            results["eccentricity"].append(eccentricity)

        # check valid calculation
        assert len(results["edge_pixels"]) > 0, "Bodies not match the conditions"

        col = None
        for i, _ in enumerate(range(len(results["edge_pixels"]))):
            if col is None:
                col = np.asarray(col_o).copy()
                col = col[..., ::-1]
            col[results['edge_pixels'][i][:, 0], results['edge_pixels'][i][:, 1], :] = [0, 255, 0]

        im = Image.fromarray(col)
        draw = ImageDraw.Draw(im)
        center_text = ""
        append_text = ""

        x0, y0 = bw_bodies[0].shape[1] / 2, bw_bodies[0].shape[0] / 2
        for i, (bw_temp, center_2d_index_) in enumerate(zip(bw_bodies, center_2d_index_list)):
            radii = radius_cv2[i]
            if last_tag[i] == "earth":
                # Earth section
                vectors_c['Earth_c'] = results['vector_c'][i]
                body_name = "earth"
                center_text += f"Earth Center vector: {(np.round(results['vector_c'][i], 4))} - {radii[0]}\n"
                append_text += rf"$\psi$: {np.round(results['pitch'][i] * np.rad2deg(1), 3)}, $\theta$: {np.round(results['roll'][i] * np.rad2deg(1), 3)} [deg] @ Body frame" + "\n"
            elif last_tag[i] == "sun":
                vectors_c['Sun_c'] = results['vector_c'][i]
                body_name = "sun"
                center_text += f"Sun Center vector: {(np.round(results['vector_c'][i], 4))} - {radii[0]}\n"
            else:
                continue

            x1, y1 = results['center_pixel'][i][0], results['center_pixel'][i][1]
            if body_name == 'earth':
                draw.line((x0, y0, x1, y1), fill=(255, 0 *255, 0*255), width=2)
            else:
                draw.line((x0, y0, x1, y1), fill=(0 * 255, 0 * 255, 255), width=2)

        img_cv2_ = np.asarray(im)
        time_number = file_name.split('/')[-1][:-4]
        fig_cv2 = plt.figure()
        plt.title(f"{last_tag}\n Timestamp: {time_number if isinstance(time_number, np.number) else 0.0}\n" + append_text + center_text)
        # add arrow
        plt.quiver(bw_bodies[0].shape[1] / 2, bw_bodies[0].shape[0] / 2, 0, -bw_bodies[0].shape[0] / 2, color="green")
        plt.quiver(bw_bodies[0].shape[1] / 2, bw_bodies[0].shape[0] / 2, bw_bodies[0].shape[1] / 2, 0, color="red")
        plt.imshow(img_cv2_[..., ::-1])
        # plt.plot(x1, y1, color="red", marker="X")
        plt.grid()
        plt.tight_layout()
        file_name = os.path.splitext(os.path.basename(file_name))[0]
        fig_cv2.savefig("{}.png".format(folder_ + name_folder + f'/vec_bodies_' + file_name + ".png"), dpi=300)
        plt.close()
        col = None
    except Exception as e:
        print(e)
        edge_, img_cv2_ = [], img_cv2_
    return edge_, img_cv2_, pitch_, roll_, vectors_c

if __name__ == '__main__':
    import pandas as pd
    import cv2
    from mpl_toolkits.mplot3d import Axes3D
    import pickle
    from src.dynamics.dynamics_kinematics import Dynamics
    from tools.get_video_frame import save_frame

    PROJECT_FOLDER = "../data/20230904/"
    VIDEO_DATA = "20230904-video-att9-clip.mp4"
    video_last_frame = "2023/09/04 17:49:04"
    vide_name = VIDEO_DATA.split('.')[0]
    # frame_shape = save_frame(PROJECT_FOLDER, VIDEO_DATA, video_last_frame)
    frame_shape = (90, 160, 3)
    NEW_FOLDER = PROJECT_FOLDER + VIDEO_DATA.split(".")[0] + '/'
    img_type = 'png'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    list_file = [elem for elem in os.listdir(NEW_FOLDER) if img_type in elem]
    num_list = [float(elem[:-4]) for elem in list_file if 'png' in elem]
    datalist = pd.DataFrame({'filename': list_file, 'id': num_list})
    datalist.sort_values(by='id', inplace=True)
    pitch_list = []
    residual_list = []
    roll_list = []
    earth_b_list = []
    k = 0
    dt_ = 1 / 30

    # if pitch_roll.xlsx exist
    if not os.path.exists(NEW_FOLDER + f'pitch_roll_LVLH_{vide_name}.xlsx'):
        video_salida = cv2.VideoWriter(NEW_FOLDER + f"att_process_lvlh_{vide_name}.avi", fourcc, 10.0, (frame_shape[1], frame_shape[0]))
        for filename, ts_i in datalist.values[:]:
            print(k)
            height_sc = 450
            edge_, img_cv2_, pitch_a, roll_a, earth_b = get_vector_v2(NEW_FOLDER + filename, height_sc)
            pitch_list.append(pitch_a * np.rad2deg(1))
            roll_list.append(roll_a * np.rad2deg(1))
            earth_b_list.append(earth_b)
            k += 1
            video_salida.write(img_cv2_)
            # Liberar el objeto VideoWriter
        video_salida.release()
        dict_data = {'time(sec)': np.arange(0, len(pitch_list)) * dt_,
                     'pitch': pitch_list,
                     'roll': roll_list,
                     'e_b_x': np.array(earth_b_list)[:, 0],
                     'e_b_y': np.array(earth_b_list)[:, 1],
                     'e_b_z': np.array(earth_b_list)[:, 2]}
        data = pd.DataFrame(dict_data)
        data.to_excel(NEW_FOLDER + f'pitch_roll_LVLH_{vide_name}.xlsx', index=False)
    else:
        data = pd.read_excel(NEW_FOLDER + f'pitch_roll_LVLH_{vide_name}.xlsx')
        earth_b_list = data[['e_b_x', 'e_b_y', 'e_b_z']].values

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    [ax.quiver(*np.zeros(3), *vec_, alpha=0.3, arrow_length_ratio=0.1) for vec_ in earth_b_list]
    ax.quiver(*np.zeros(3), 1, 0, 0, length=2, arrow_length_ratio=0.1, color='red')
    ax.quiver(*np.zeros(3), 0, 1, 0, length=2, arrow_length_ratio=0.1, color='green')
    ax.quiver(*np.zeros(3), 0, 0, 1, length=2, arrow_length_ratio=0.1, color='blue')
    ax.set_xlim([-2, 2])
    ax.set_ylim([-2, 2])
    ax.set_zlim([-2, 2])
    plt.grid()
    fig.savefig(NEW_FOLDER + "earth_vector_pointing_body.png")

    fig = plt.figure()
    plt.xlabel("Roll [deg]")
    plt.ylabel("Pitch [deg]")
    plt.grid()
    theta = data['roll'] * np.deg2rad(1)
    plt.plot(data['pitch'] * np.sin(theta), data['pitch'] * np.cos(theta), '.-')
    fig.savefig(NEW_FOLDER + "circular_pitch_roll_lvlh.png")

    fig = plt.figure()
    plt.title("Angular rotation (3, 1, 2) - BF @ LVLH")
    plt.ylabel("Angle rotation [deg]")
    plt.xlabel("Relative Time [sec]")
    plt.plot(np.array(data['time(sec)']), data['pitch'], '.', label='Pitch Angle')
    plt.plot(data['time(sec)'], data['roll'], '.', label='Roll Angle')
    plt.grid()
    plt.legend()
    fig.savefig(NEW_FOLDER + "pitch_roll_lvlh.png")

    fig = plt.figure()
    plt.ylabel("Angle rotation [deg]")

    convolve_pitch = np.convolve([elem if elem >= 0 else elem + 360 for elem in data['pitch']], np.ones(5), "valid") / 5

    convolve_roll = np.convolve([elem if elem >= 0 else elem + 360 for elem in data['roll']], np.ones(5), "valid") / 5

    plt.plot(-np.diff(convolve_pitch) / dt_, label='Rate - Pitch Angle')
    plt.plot(-np.diff(convolve_roll) / dt_, label='Rate - Roll Angle')
    plt.grid()
    plt.legend()
    temp = VIDEO_DATA.split(".")[0]
    fig.savefig(NEW_FOLDER + f"process/{temp}_rate_pitch_roll_lvlh.png")
    plt.show()
