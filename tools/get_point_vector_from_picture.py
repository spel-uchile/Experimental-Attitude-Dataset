"""
Created by Elias Obreque
Date: 16-09-2023
email: els.obrq@gmail.com
"""
import time

from PIL import Image, ImageFilter, ImageDraw
# from tools.my_scipy import optimize
from scipy import optimize
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

sensor_pixel_h = sensor_width_h / 3280
sensor_pixel_v = sensor_width_v / 2464
fov_h = 62.2 # deg
fov_v = 48.8 # deg
ROT_CAM2BODY = Rotation.from_euler('zx', [180, -90], degrees=True).inv().as_matrix()
# sensor_width = 2.76 * 1e-3  # m
focal_length = 0.00304

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


def get_body(col, file_name, show=True):
    # fig_h, ax_h = plt.subplots()
    # fig, ax = plt.subplots(1, 6, figsize=(15, 5), sharey=True)
    # fig.suptitle(file_name.split("/")[-1][:-4])
    # fig.tight_layout()
    # ax[0].imshow(col)
    # ax[0].set_title("Original")
    #
    # new_data = []
    # new_col, lighter_colors, counts = decrease_color(col, 10)
    # color_label = []

    # thereisblack = False
    # # for elem in lighter_colors:
    # #     finding_color = {}
    # #     finding_color['black'] = True if np.mean(elem) < 20 else False
    # #     finding_color['white'] = True if np.mean(elem) > 240 else False
    # #     finding_color['red'] = False
    # #     finding_color['green'] = False
    # #     finding_color['blue'] = False
    # #     if (finding_color['white'] or finding_color['black']) is False:
    # #         finding_color['red'] = True if elem[0] > np.mean(elem) + 30 else False
    # #         finding_color['green'] = True if elem[1] > np.mean(elem) + 30 else False
    # #         finding_color['blue'] = True if elem[2] > np.mean(elem) + 30 else False
    # #     thereisblack = finding_color['black']
    # #     color_label.append(finding_color)
    # #
    # # # Blue filter
    # # for i, color_l in enumerate(color_label):
    # #     if color_l['blue'] and thereisblack:
    # #         blue_no_black = np.max(new_col, axis=2) - np.min(new_col, axis=2) > 12
    # #         new_col[blue_no_black] = lighter_colors[0]
    #
    # col.putdata([tuple(colors) for colors in new_col.reshape(-1, 3)])
    gray = col.convert('L')
    # gray = np.asarray(col)[:, :, 2]
    # lighter_gray = [rgb_to_gray(r, g, b) for r, g, b in zip(lighter_colors[:, 0],
    #                                                         lighter_colors[:, 1],
    #                                                         lighter_colors[:, 2])]
    gray = gray.filter(ImageFilter.SMOOTH)
    bw = np.asarray(gray).copy()
    bw = bw / 255
    # ax[1].imshow(bw)
    # ax[1].set_title("Gray")

    bw = bw ** 0.7
    # if counts[-1] > 4 * counts[-2] and color_label[-1]['black']:
    #     bw = bw ** (1 / 2)

    # ax_h.plot(np.arange(0, bw.shape[1]), np.sum(bw, axis=0), label="X")
    # ax_h.plot(np.arange(0, bw.shape[0]), np.sum(bw, axis=1), label="Y")
    # ax_h.set_title("Histogram")
    # ax_h.grid()
    # ax_h.legend()
    # ax[2].imshow(bw)
    # ax[2].set_title("Gray ^ (0.7)")

    # hist = np.histogram(bw, bins=256)[1]
    # max_hist = np.max(hist)
    # min_hist = np.min(hist)
    mean_cut = 0.25
    bw[np.where(bw < mean_cut)] = 0.0
    bw[np.where(bw >= mean_cut)] = 1.0
    bw_temp = []

    # ax[3].imshow(bw)
    # ax[3].set_title("Binarize")
    # bw_earth, bw_sun = separate_body(bw)
    labeled_array, num_features = etiquetar_objetos(bw)
    if num_features > 1:
        k = 0
        for i in range(num_features + 1):
            points = np.where(labeled_array == i)
            centroid = int(sum(points[0]) / len(points[0])), int(sum(points[1]) / len(points[1]))
            if bw[centroid] == 1 and len(points[0]) > 5:  # is object
                temp = np.zeros(bw.shape)
                temp[points] = 1
                bw_temp.append(temp)
                # ax[4 + k].imshow(temp)
                # ax[4 + k].set_title("Body - {}".format(k))
                k += 1
                if k == 2:
                    break
            else:
                pass
                # ax[5].imshow(bw * 0)
                # ax[5].set_title("None")
    else:
        bw_temp.append(bw)
        # ax[4].imshow(bw)
        #ax[4].set_title("Body")
        #ax[5].imshow(bw * 0)
        #ax[5].set_title("None")
    # create folder
    name_folder = 'process'
    folder_ = ''.join([el + '/' for el in file_name.split('/')[:-1]])
    if not os.path.exists(folder_ + name_folder):
        os.makedirs(folder_ + name_folder)
    # fig.savefig("{}_process.png".format(folder_ + name_folder + '/' + file_name.split('/')[-1][:-4]))
    # fig_h.savefig("{}_hist.png".format(folder_ + name_folder + '/' + file_name.split('/')[-1][:-4]))
    # plt.close(fig=fig)
    # plt.close(fig=fig_h)
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
    for im in im_list:
        # contours, hierarchy = cv2.findContours(np.array(im, dtype=np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        gx, gy = np.gradient(im)[0], np.gradient(im)[1]
        gr = np.hypot(gx, gy)
        gr[np.where(gr < 0.3)] = 0
        points = np.where(gr > 0.3)
        point_list.append(points)
        edge_array = np.array([points[0], points[1]]).T
        delta_ = 3
        lines = []
        for k in range(len(points[0])):
            lines.append(edge_array[delta_ + k] - edge_array[k])
            if k >= len(points[0]) - delta_ - 1:
                break
        # mean
        x_mean = np.sum([np.array(elem) * np.arange(0, np.shape(im)[1]) for elem in im]) / np.count_nonzero(im)
        y_mean = np.sum([np.array(elem) * i for i, elem in enumerate(im)]) / np.count_nonzero(im)
        center_estimate = np.array([y_mean, x_mean], dtype=int)
        points_center = edge_array[:-delta_] + np.array(lines) * 0.5

        center_2d_index, ier = optimize.leastsq(f_2b, center_estimate, args=points_center, Dfun=Df_2b, col_deriv=True)
        center_2d_index_list.append(center_2d_index)
        radius = calc_R(center_2d_index[1], center_2d_index[0], points_center[:, 1], points_center[:, 0])
        radius_list.append([np.mean(radius), np.std(radius)])
        edge_array_list.append(edge_array)
        points_c_list.append(points_center)
    return radius_list, point_list


def calc_hyperbola(points, fl, pw, ph, h, shape_, center_is_into_):
    # https://link.springer.com/article/10.1007/s12567-022-00461-0
    edge_array = np.array([points[0], points[1]]).T
    # delta_ = 3
    # lines = []
    # for k in range(len(points[0])):
    #     lines.append(edge_array[delta_ + k] - edge_array[k])
    #     if k >= len(points[0]) - delta_ - 1:
    #         break
    points_center = edge_array# [:-delta_] + np.array(lines) * 0.5
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
    # e_c /= np.linalg.norm(e_c)

    vec_pix = (e_c * fl / e_c[2])[:2]
    center_pixel = np.zeros(2, dtype=np.int16)
    center_pixel[0] = np.int16(vec_pix[0] / pw + shape_[1] * 0.5)
    center_pixel[1] = np.int16(vec_pix[1] / ph + shape_[0] * 0.5)

    m_slope = e_c[1] / e_c[0]

    a_prime = e_c[0] ** 2 + e_c[1] ** 2 - np.cos(alpha) ** 2
    c_prime = -np.cos(alpha) ** 2
    f_prime = -(fl ** 2 * (1 - np.cos(alpha) ** 2) * np.cos(alpha) ** 2) / (e_c[0] ** 2 + e_c[1] ** 2 - np.cos(alpha) ** 2)

    y_sol = m_slope * hh[:, 0]
    idx_inter = np.argmin(np.abs(y_sol - hh[:, 1]))
    x_inter, y_inter = hh[:, 0][idx_inter], hh[:, 1][idx_inter]

    dist_pitch = np.sqrt(x_inter ** 2 + y_inter ** 2)
    ang_center = np.arctan(dist_pitch / fl)

    center_pixel = np.array([x_inter / pw + shape_[1] * 0.5, y_inter / ph + shape_[0] * 0.5])

    angle_orbit = np.pi / 2 - alpha

    # vector in body frame,
    e_b = ROT_CAM2BODY @ e_c
    # pitch = np.arcsin(e_b[1])
    roll = -np.arctan2(e_b[0], e_b[2])

    roll_xy = roll# np.arctan2(x_inter, y_inter)
    pos_x_rot = -np.sin(roll_xy) * y_inter + np.cos(roll_xy) * x_inter
    pos_y_rot = np.sin(roll_xy) * x_inter + np.cos(roll_xy) * y_inter

    pitch = angle_orbit - ang_center * np.sign(pos_y_rot)

    # 3, 1, 2
    # q_Xx_1 = np.cos(pitch) * np.sin(roll)
    # q_Xx_2 = -np.sin(pitch)
    # q_Xx_3 = np.cos(pitch) * np.cos(roll)

    e_b[0] = np.cos(pitch) * np.sin(roll)
    e_b[1] = -np.sin(pitch)
    e_b[2] = np.cos(pitch) * np.cos(roll)
    return edge_array, points_center, e_c, center_pixel, pitch, roll


def least_squares_fit(it, jt):
    """
    Calcula los parámetros a y b para ajustar una línea usando mínimos cuadrados.
    Args:
        it (np.array): Coordenadas de los puntos en el eje i (x).
        jt (np.array): Coordenadas de los puntos en el eje j (y).

    Returns:
        a (float): Pendiente de la línea ajustada.
        b (float): Intersección con el eje j (ordenada al origen).
    """
    # Número total de puntos
    T = len(it)

    # Validar que las longitudes de las listas coincidan
    if T != len(jt):
        raise ValueError("Los vectores 'it' y 'jt' deben tener la misma longitud.")

    # Cálculos intermedios para las sumas
    sum_it = np.sum(it)
    sum_jt = np.sum(jt)
    sum_it_jt = np.sum(it * jt)
    sum_it2 = np.sum(it ** 2)

    # Calcular a y b usando las fórmulas de las ecuaciones 2a y 2b
    denominator = T * sum_it2 - (sum_it) ** 2
    if denominator == 0:
        raise ValueError("El denominador es cero, no se puede calcular la regresión.")

    a = (T * sum_it_jt - sum_it * sum_jt) / denominator
    b = (sum_jt * sum_it2 - sum_it * sum_it_jt) / denominator

    return a, b


def calc_sun_curvature(points, fl, pw, ph, shape_, h):
    edge_array = np.array([points[0], points[1]]).T
    delta_ = 5
    lines = []
    for k in range(len(points[0])):
        lines.append(edge_array[delta_ + k] - edge_array[k])
        if k >= len(points[0]) - delta_ - 1:
            break
    points_center = edge_array[:-delta_] + np.array(lines) * 0.5
    # p_c[x, y, z]
    cx = shape_[0] * 0.5 * pw
    cy = shape_[1] * 0.5 * ph
    p_c = [np.array([(points_center[i])[1] * pw - cx, (points_center[i][0]) * ph - cy, fl]) for i in
           range(len(points_center))]
    hh = np.array(p_c)
    alpha = np.arcsin(rs / (rs + h))
    y = np.cos(alpha) * np.array([np.linalg.norm(p_c_i) for p_c_i in p_c])
    sun_c = np.linalg.inv(hh.T.dot(hh)).dot(hh.T).dot(y)
    sun_c /= np.linalg.norm(sun_c)

    center_pixel = np.zeros(2, dtype=np.int16)
    vec_pix = (sun_c * fl / sun_c[2])[:2]
    center_pixel[0] = np.int16(vec_pix[0] / pw + shape_[0] * 0.5)
    center_pixel[1] = np.int16(vec_pix[1] / ph + shape_[1] * 0.5)
    return edge_array, points_center, sun_c, center_pixel


def get_vector(file_name, height, height_img=None, width_img=None):
    edge_ = None
    pitch_, roll_ = np.nan, np.nan
    img_cv2_ = None
    earth_c = np.array([np.nan, np.nan,np.nan])
    try:
        col = Image.open(file_name)
        if height_img is not None:
          col = col.resize((height_img, width_img))
        img_cv2_ = np.asarray(col)
        dimx, dimy = col.size[0], col.size[1]
        pixel_size_width = sensor_width_h / dimx
        pixel_size_height = sensor_width_v / dimy
        earth_limit_pixels = ((dimx * 0.82) ** 2 + (dimy * 0.81) ** 2) ** 0.5
        t1 = time.time()
        bw_bodies = get_body(col.copy(), file_name, show=False)
        radius_, point_list_ = get_type_radius(bw_bodies)
        print(f"Get bodies {time.time()-t1} seconds")
        print(file_name, radius_)
        for radii, pl, bw_temp in zip(radius_, point_list_, bw_bodies):
            x0, y0 = bw_temp.shape[1] / 2, bw_temp.shape[0] / 2

            if len(pl[0]) > earth_limit_pixels: # np.min([dimx, dimy]):
                if np.max([dimx, dimy]) * 0.1 < radii[0] < np.max([dimx, dimy]) * 0.5 and radii[0] > 100 * radii[1]:
                    print("sun")
                    sun_edge_array, sun_points_center, sun_c, center_pixel = calc_sun_curvature(pl,
                                                                                                focal_length,
                                                                                                pixel_size_width,
                                                                                                pixel_size_height,
                                                                                                [col.size[0], col.size[1]],
                                                                                                height)
                    edge_, img_cv2_ = get_lines(col, center_pixel)
                elif np.max([dimx, dimy]) * 100 >= radii[0] > np.max([dimx, dimy]) and radii[0] > 10 * radii[1]:
                    print("earth")
                    if len(pl[0]) < earth_limit_pixels:
                        continue
                    # Earth: Recalculate with hyperbolic geometry
                    center_is_into = bool(bw_temp[int(bw_temp.shape[0] / 2), int(bw_temp.shape[1] / 2)])
                    t1 = time.time()

                    earth_edge_array, earth_points_center, earth_c, center_pixel, pitch_, roll_ = calc_hyperbola(pl,
                                                                                                                 focal_length,
                                                                                                                 pixel_size_width,
                                                                                                                 pixel_size_height,
                                                                                                                 height,
                                                                                                                 bw_temp.shape,
                                                                                                                 center_is_into)

                    print(f"Full hyperbola calculation ... {time.time() - t1}")
                    col = np.asarray(col).copy()
                    col = col[..., ::-1]
                    col[earth_edge_array[:, 0], earth_edge_array[:, 1], :] = [0, 255, 0]
                    im = Image.fromarray(col)
                    draw = ImageDraw.Draw(im)
                    x1, y1 = x0 + earth_c[0] * 50, y0 + earth_c[1] * 50

                    print(x0, y0, x1, y1)
                    draw.line((x0, y0, x1, y1), fill=(0, 255, 255), width=1)

                    alpha = np.arcsin((re + 50) / (re + height))
                    angle_ = np.pi/2 - alpha
                    proj_plane = focal_length * np.tan(angle_)
                    pix_elipse_w = proj_plane / pixel_size_width
                    pix_elipse_h = proj_plane / pixel_size_height
                    draw.ellipse([(x0 - pix_elipse_w, y0 - pix_elipse_h),
                                  (x0 + pix_elipse_w, y0 + pix_elipse_h)] , fill=None, outline="white")

                    draw.text((0, 80), f"{np.round(angle_ * np.rad2deg(1), 1)} deg - {len(pl[0])}", fill="blue", scale=0.5)
                    draw.text((0, 0), f"R1: {np.round(pitch_ * np.rad2deg(1), 2)} deg", fill="white", scale=0.5)
                    draw.text((0, 10), f"R2: {np.round(roll_ * np.rad2deg(1), 2)} deg", fill="white", scale=0.5)
                    draw.text((0, 70), f"{int(radii[0])} - {int(radii[1])}", fill="red", scale=0.5)

                    edge_, img_cv2_ = None, np.asarray(im)
                    # add arrow
                    name_folder = 'vectorization'
                    folder_ = ''.join([el + '/' for el in file_name.split('/')[:-1]])
                    if not os.path.exists(folder_ + name_folder):
                        os.makedirs(folder_ + name_folder)
                    fig_cv2 = plt.figure()
                    plt.title(f"Timestamp: {file_name.split('/')[-1][:-4]}\n"
                              rf"$\psi$: {np.round(pitch_ * np.rad2deg(1), 3)}, $\theta$: {np.round(roll_ * np.rad2deg(1), 3)} [deg] @ Body frame" + "\n"
                              f"Earth center: vector {(np.round(earth_c, 4))}")
                    plt.quiver(bw_temp.shape[1] / 2, bw_temp.shape[0] / 2, 0, -bw_temp.shape[0] / 3, color="green")
                    plt.quiver(bw_temp.shape[1] / 2, bw_temp.shape[0] / 2, bw_temp.shape[1] / 3, 0, color="red")
                    plt.imshow(img_cv2_[..., ::-1])
                    plt.plot(center_pixel[0], center_pixel[1],
                             color="red", marker="X")
                    plt.grid()
                    plt.tight_layout()
                    fig_cv2.savefig("{}.png".format(folder_ + name_folder + '/vec_' + file_name.split('/')[-1][:-4]))
                    plt.close()
    except Exception as e:
        print(e)
        edge_, img_cv2_ = [], None
    return edge_, img_cv2_, pitch_, roll_, ROT_CAM2BODY @ earth_c


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
            edge_, img_cv2_, pitch_a, roll_a, earth_b = get_vector(NEW_FOLDER + filename, height_sc, frame_shape[1], frame_shape[0])
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
