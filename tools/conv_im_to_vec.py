"""
Created by Elias Obreque
els.obrq@gmail.com
Date: 26-09-2022
"""
# line1 = "1 52191U 22033AL  22147.79526710  .00008056  00000+0  36597-3 0  9992"
# line2 = "2 52191  97.3966 229.5847 0012282  67.1638 293.0894 15.20960720  8536"

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageEnhance, ImageFilter
from os import walk

import cv2
from tools.clustering import cluster, decrease_color

import os
from scipy import optimize, ndimage
from sgp4.api import Satrec
from sgp4.api import WGS84
import random as rng

re = 6378.137  # km
RAD2DEG = 180 / np.pi
DEG2RAD = 1 / RAD2DEG
au = 149597870.691  # km
d_sun = 1.3927e-9
rs = au
sensor_width = 2.74 * 1e-3  # m
flength = 0.00304
rng.seed(12345)


def calc_curvature(im_list, height):
    center_2d_index_list = []
    radius_list = []
    edge_array_list, points_c_list = [], []
    angle = np.linspace(0, 2 * np.pi, 150)
    x = np.cos(angle)
    y = np.sin(angle)
    # Preliminary estimation of radius
    point_list = []
    for im in im_list:
        gx, gy = np.gradient(im)[0], np.gradient(im)[1]
        gr = np.hypot(gx, gy)
        gr[np.where(gr < 0.3)] = 0
        points = np.where(gr > 0.3)
        point_list.append(points)
        edge_array = np.array([points[0], points[1]]).T
        delta_ = 5
        lines = []
        for k in range(len(points[0])):
            lines.append(edge_array[delta_ + k] - edge_array[k])
            if k >= len(points[0]) - delta_ - 1:
                break
        center_estimate = np.mean(edge_array, axis=0)
        points_center = edge_array[:-delta_] + np.array(lines) * 0.5

        center_2d_index, ier = optimize.leastsq(f_2b, center_estimate, args=points_center, Dfun=Df_2b, col_deriv=True)
        points_c = points_center
        center_2d_index_list.append(center_2d_index)
        radius = calc_R(center_2d_index[0], center_2d_index[1], points_center[:, 1], points_center[:, 0])
        radius_list.append(radius)
        # print("Pixel radius:", radius.mean())
        error = np.max(radius - radius.mean())
        # print("ERROR:", error[-1], np.std(radius - radius.mean()))
        # print("Pixel center:", center_2d_index)
        edge_array_list.append(edge_array)
        points_c_list.append(points_c)

    radius_mean = [radius_m_.mean() for radius_m_ in radius_list]
    print(radius_mean)
    edge_array_list, points_c_list = [], []
    vectors_c = {}
    vectors_c['Earth_c'] = np.zeros(3)
    vectors_c['Sun_c'] = np.zeros(3)
    properties = {}
    properties['Earth'] = np.zeros(3)
    properties['Sun'] = np.zeros(3)
    for i in range(len(im_list)):
        print("RADIUS: ", radius_mean[i])
        if radius_mean[i] > 100.0 * np.shape(im_list[i])[0] / 102:
            pixel_size = sensor_width / np.shape(im_list[i])[0]
            # Earth: Recalculate with hyperbolic geometry
            earth_edge_array, earth_points_center, earth_c, properties_h, center_pixel = calc_hyperbola(point_list[i],
                                                                                                        flength,
                                                                                                        pixel_size,
                                                                                                        height,
                                                                                                        np.shape(im_list[i])[0])
            edge_array_list.append(earth_edge_array)
            points_c_list.append(earth_points_center)
            vectors_c['Earth_c'] = earth_c
            print("Earth_c: ", earth_c)
            print("Center Pixel:", center_pixel)
            properties['Earth'] = properties_h
            properties['Earth']['center_px'] = center_pixel
        elif 5.0 * np.shape(im_list[i])[0] / 102 < radius_mean[i] <= 100.0 * np.shape(im_list[i])[0] / 102:
            # Sun: Recalculate with elliptical geometry
            pixel_size = sensor_width / np.shape(im_list[i])[0]
            sun_edge_array, sun_points_center, sun_c, properties_e, center_pixel = calc_sun_curvature(point_list[i],
                                                                                                      flength,
                                                                                                      pixel_size,
                                                                                                      np.shape(im_list[i])[0],
                                                                                                      im_list[i],
                                                                                                      height)
            edge_array_list.append(sun_edge_array)
            points_c_list.append(sun_points_center)
            vectors_c['Sun_c'] = sun_c
            print("Sun_c: ", sun_c)
            print("Center Pixel:", center_pixel)
            properties['Sun'] = properties_e
            properties['Sun']['center_px'] = center_pixel
        else:
            if len(im_list) == 1:
                raise Exception("The radius does not correspond to the Earth or the Sun: {}".format(radius_mean[i]))
    # plt.show()
    return edge_array_list, points_c_list, center_2d_index_list, radius_list, properties, vectors_c


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


def calc_vector_c(pos_im, fl, fov):

    yd = -pos_im[1] + 51  # Origin in the center of imagen @ sensor frame x,y, z
    zd = -pos_im[0] + 51
    # print("Origin in the center of imagen @ sensor frame yd, zd:", yd, zd)
    phi = np.arctan2(zd, yd)
    # print("phi", phi)
    focal_length = fl
    half_hor = np.deg2rad(fov * 0.5)
    resolution = 102
    dx = 2 * focal_length * np.tan(half_hor) / resolution
    yd *= dx
    zd *= dx
    # print(yd, zd, focal_length)
    theta = np.arctan(np.sqrt(yd ** 2 + zd ** 2) / focal_length)
    # print("theta", theta)
    pos_c = np.zeros(3)
    pos_c[0] = np.cos(theta)
    pos_c[1] = np.sin(theta) * np.cos(phi)
    pos_c[2] = np.sin(theta) * np.sin(phi)
    return pos_c


def add_filter(col, threshold, file_path=None):
    fig, ax = plt.subplots(1, 7, figsize=(15, 5), sharey=True, sharex=True)
    fig.suptitle('Test')
    fig.tight_layout()
    ax[0].imshow(col)
    ax[0].set_title("Original")

    new_data = []
    new_col, lighter_colors, counts = decrease_color(col, 10)
    color_label = []
    ax[1].imshow(new_col/255)
    ax[1].set_title("KMeans")
    # print("Lighter:", lighter_colors)
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
        # print(finding_color)
        thereisblack = finding_color['black']
        color_label.append(finding_color)

    # mean_value = np.mean(new_col) * 255
    # for elem in col.getdata():
    #     if elem[2] > mean_value:
    #         new_data.append((255, 255, 255))
    #     else:
    #         new_data.append(elem)

    # Blue filter
    for i, color_l in enumerate(color_label):
        if color_l['blue'] and thereisblack:
            blue_no_black = np.max(new_col, axis=2) - np.min(new_col, axis=2) > 10
            new_col[blue_no_black] = lighter_colors[0]

    col.putdata([tuple(colors) for colors in new_col.reshape(-1, 3)])
    ax[2].imshow(col)
    ax[2].set_title("Blue Filter")
    gray = col.convert('L')
    lighter_gray = [rgb_to_gray(r, g, b) for r, g, b in zip(lighter_colors[:, 0],
                                                            lighter_colors[:, 1],
                                                            lighter_colors[:, 2])]
    bw = np.asarray(gray).copy()
    bw = bw / 255
    if counts[-1] > 4 * counts[-2] and color_label[-1]['black']:
        bw = bw ** (1/2)
    ax[3].imshow(bw)
    ax[3].set_title("Gray")
    # bw = bw ** 2
    bw[np.where(bw < 0.98 * max(lighter_gray) / 255)] = 0.0
    bw[np.where(bw >= 0.98 * max(lighter_gray) / 255)] = 1.0
    bw_temp = []
    ax[4].imshow(bw)
    ax[4].set_title("Binarize")
    # bw_earth, bw_sun = separate_body(bw)
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
                ax[5 + k].imshow(temp)
                ax[5 + k].set_title("Body - {}".format(k))
                k += 1
                if k == 2:
                    break
    else:
        bw_temp.append(bw)
        ax[5].imshow(bw)
        ax[5].set_title("Body")
        ax[6].imshow(bw * 0)
        ax[6].set_title("No Body")
    # plt.show()
    if file_path is not None:
        # path folder from file
        folder_ = os.path.dirname(file_path) + "/results/"
        if not os.path.exists(folder_):
            os.makedirs(folder_)
        file_name = os.path.splitext(os.path.basename(file_path))[0]
        fig.savefig(folder_ + "add_filers" + file_name + ".png", dpi=300)
        plt.close()
    return bw_temp


def rgb_to_gray(R, G, B):
    return int(R * 299 / 1000 + G * 587 / 1000 + B * 114 / 1000)


def separate_body(bw):
    bw_earth = np.zeros(bw.shape)
    bw_sun = np.zeros(bw.shape)
    num_sep = 3
    shape = bw.shape
    width = int(shape[0] / num_sep)
    height = int(shape[1] / num_sep)
    mean_lux = []
    for i in range(num_sep):  # Vertical
        for j in range(num_sep):  # Horizontal
            mean_value = np.mean(bw[i * width: (i + 1) * width, j * height: (j + 1) * height])
            mean_lux.append(mean_value)

    # print(mean_lux)
    return bw_earth, bw_sun


def get_earth_pointing(name_file, fl, sw, height, threshold=0.98):
    # real picture
    col = Image.open(name_file)
    bw_temp = add_filter(col, threshold, file_path=name_file)
    # sun_radius = fl / (au * 1e3) * d_sun
    edge, center_lines, center_im_list, radius_m, properties, vectors_c = calc_curvature(bw_temp, height)
    # edge, center_lines, center_im_list, radius_m, properties, vectors_c = calc_curvature2(name_file, height)
    return center_im_list, vectors_c, bw_temp, edge, radius_m, properties


def calc_hyperbola(points, fl, pw, h, length):
    edge_array = np.array([points[0], points[1]]).T
    delta_ = 2
    lines = []
    for k in range(len(points[0])):
        lines.append(edge_array[delta_ + k] - edge_array[k])
        if k >= len(points[0]) - delta_ - 1:
            break
    points_center = edge_array[:-delta_] + np.array(lines) * 0.5
    # p_c[x, y, z]
    cx = length * 0.5 * pw
    cy = length * 0.5 * pw
    p_c = [np.array([(points_center[i])[1] * pw - cx, (points_center[i][0]) * pw - cy, fl]) for i in
           range(len(points_center))]
    hh = np.array(p_c)
    da = 50.0
    alpha = np.arcsin((re + da) / (re + h))
    y = np.cos(alpha) * np.array([np.linalg.norm(p_c_i) for p_c_i in p_c])
    e_c = np.linalg.inv(hh.T.dot(hh)).dot(hh.T).dot(y)
    e_c /= np.linalg.norm(e_c)
    A = e_c[0] ** 2 + e_c[1] ** 2 - np.cos(alpha) ** 2
    C = - np.cos(alpha) ** 2
    F = - fl ** 2 * (1 - np.cos(alpha) ** 2) * np.cos(alpha) ** 2 / A
    exc = e_c[0]
    eyc = e_c[1]
    ezc = e_c[2]
    center_pixel = np.asarray((e_c * fl / e_c[2])[:2] / pw + length * 0.5, dtype=np.int64)
    a = exc ** 2 - np.cos(alpha) ** 2
    b = 2 * exc * eyc
    c = eyc ** 2 - np.cos(alpha) ** 2
    d = 2 * exc * ezc * fl
    e = 2 * eyc * ezc * fl
    f = fl ** 2 * (ezc ** 2 - np.cos(alpha) ** 2)
    ap = np.sqrt(-F / A)
    bp = np.sqrt(abs(F / C))
    print("CENTER:", (e_c * fl / e_c[2])[:2] / np.linalg.norm((e_c * fl / e_c[2])[:2]),
          cart_to_pol([a, b, c, d, e, f])[:2] / np.linalg.norm(cart_to_pol([a, b, c, d, e, f])[:2]))
    temp_contour = get_contour([a, b, c, d, e, f], 3.04e-3, 2.74e-3 / length, 48.8 * np.deg2rad(1))
    properties = {'a': ap, 'b': bp, 'coeff': [a, b, c, d, e, f, np.array(temp_contour)], 'car': None}
    return edge_array, points_center, e_c, properties, center_pixel


def calc_sun_curvature(points, fl, pw, length, im, h):
    edge_array = np.array([points[0], points[1]]).T
    delta_ = 5
    lines = []
    for k in range(len(points[0])):
        lines.append(edge_array[delta_ + k] - edge_array[k])
        if k >= len(points[0]) - delta_ - 1:
            break
    points_center = edge_array[:-delta_] + np.array(lines) * 0.5
    # p_c[x, y, z]
    cx = length * 0.5 * pw
    cy = length * 0.5 * pw
    p_c = [np.array([(points_center[i])[1] * pw - cx, (points_center[i][0]) * pw - cy, fl]) for i in
           range(len(points_center))]
    hh = np.array(p_c)
    alpha = np.arcsin(rs / (rs + h))
    y = np.cos(alpha) * np.array([np.linalg.norm(p_c_i) for p_c_i in p_c])
    sun_c = np.linalg.inv(hh.T.dot(hh)).dot(hh.T).dot(y)
    sun_c /= np.linalg.norm(sun_c)

    # coeffs = fit_ellipse(np.array(p_c)[:, 0], np.array(p_c)[:, 1])
    # x0, y0, ap, bp, e, phi = cart_to_pol(coeffs)
    # print("x0, y0", x0, y0)
    center_pixel = np.asarray((sun_c * fl / sun_c[2])[:2] / pw + length * 0.5, dtype=np.int64)
    center_lines_sun = None
    A = sun_c[0] ** 2 + sun_c[1] ** 2 - np.cos(alpha) ** 2
    C = - np.cos(alpha) ** 2
    F = - fl ** 2 * (1 - np.cos(alpha) ** 2) * np.cos(alpha) ** 2 / A
    a = np.sqrt(-F / A)
    b = np.sqrt(abs(F / C))
    coeffs_list = get_coefficients(sun_c, fl, h, rs, da=0)

    properties = {'a': a, 'b': b, 'coeff': coeffs_list}
    return edge_array, points_center, sun_c, properties, center_pixel


def get_coefficients(e_c, focal_length, h, r, da=0):
    alpha = np.arcsin((r + da) / (r + h))
    exc = e_c[0]
    eyc = e_c[1]
    ezc = e_c[2]
    a = exc ** 2 - np.cos(alpha) ** 2
    b = 2 * exc * eyc
    c = eyc ** 2 - np.cos(alpha) ** 2
    d = 2 * exc * ezc * focal_length
    e = 2 * eyc * ezc * focal_length
    f = focal_length ** 2 * (ezc ** 2 - np.cos(alpha) ** 2)
    return a, b, c, d, e, f


def get_contour(coeffs, fl, pw, fov):
    temph = 2.74 * 1e-3  # m
    tempv = 2.74 * 1e-3  # m

    contour = []
    a = coeffs[0]
    b = coeffs[1]
    c = coeffs[2]
    d = coeffs[3]
    e = coeffs[4]
    f = coeffs[5]
    den = b ** 2 - a * c
    if den > 0:
        for i in np.arange(-50, 150, 1):
            x_i = -temph + pw / 2 + pw * i
            # z_i_center = z_i - center[1]
            for j in np.arange(-50, 150, 1):
                y_j = -tempv + pw / 2 + pw * j
                f_zero = a * x_i ** 2 + b * y_j * x_i + c * y_j ** 2 + d * x_i + e * y_j + f
                if 1e-8 > f_zero > -1e-8:
                    contour.append([j, i])
    plt.show()
    return contour


def fit_ellipse(x, y):
    """

    Fit the coefficients a,b,c,d,e,f, representing an ellipse described by
    the formula F(x,y) = ax^2 + bxy + cy^2 + dx + ey + f = 0 to the provided
    arrays of data points x=[x1, x2, ..., xn] and y=[y1, y2, ..., yn].

    Based on the algorithm of Halir and Flusser, "Numerically stable direct
    least squares fitting of ellipses'.


    """

    D1 = np.vstack([x ** 2, x * y, y ** 2]).T
    D2 = np.vstack([x, y, np.ones(len(x))]).T
    S1 = D1.T @ D1
    S2 = D1.T @ D2
    S3 = D2.T @ D2
    T = -np.linalg.inv(S3) @ S2.T
    M = S1 + S2 @ T
    C = np.array(((0, 0, 2), (0, -1, 0), (2, 0, 0)), dtype=float)
    M = np.linalg.inv(C) @ M
    eigval, eigvec = np.linalg.eig(M)
    con = 4 * eigvec[0] * eigvec[2] - eigvec[1] ** 2
    ak = eigvec[:, np.nonzero(con > 0)[0]]
    return np.concatenate((ak, T @ ak)).ravel()


def cart_to_pol(coeffs):
    """

    Convert the cartesian conic coefficients, (a, b, c, d, e, f), to the
    ellipse parameters, where F(x, y) = ax^2 + bxy + cy^2 + dx + ey + f = 0.
    The returned parameters are x0, y0, ap, bp, e, phi, where (x0, y0) is the
    ellipse centre; (ap, bp) are the semi-major and semi-minor axes,
    respectively; e is the eccentricity; and phi is the rotation of the semi-
    major axis from the x-axis.

    """

    # We use the formulas from https://mathworld.wolfram.com/Ellipse.html
    # which assumes a cartesian form ax^2 + 2bxy + cy^2 + 2dx + 2fy + g = 0.
    # Therefore, rename and scale b, d and f appropriately.
    # [a, b, d], [b, c, f], [d, f, g]
    a = coeffs[0]
    b = coeffs[1] / 2
    c = coeffs[2]
    d = coeffs[3] / 2
    f = coeffs[4] / 2
    g = coeffs[5]
    x0, y0, ap, bp, e, phi = 0, 0, 0, 0, 0, 0
    den = b ** 2 - a * c
    if den < 0:
        # The location of the ellipse centre.
        x0, y0 = (c * d - b * f) / den, (a * f - b * d) / den
        num = 2 * (a * f ** 2 + c * d ** 2 + g * b ** 2 - 2 * b * d * f - a * c * g)
        fac = np.sqrt((a - c) ** 2 + 4 * b ** 2)
        # The semi-major and semi-minor axis lengths (these are not sorted).
        ap = np.sqrt(num / den / (fac - a - c))
        bp = np.sqrt(num / den / (-fac - a - c))

        # Sort the semi-major and semi-minor axis lengths but keep track of
        # the original relative magnitudes of width and height.
        width_gt_height = True
        if ap < bp:
            width_gt_height = False
            ap, bp = bp, ap

        # The eccentricity.
        r = (bp / ap) ** 2
        if r > 1:
            r = 1 / r
        e = np.sqrt(1 - r)

        # The angle of anticlockwise rotation of the major-axis from x-axis.
        if b == 0:
            phi = 0 if a < c else np.pi / 2
        else:
            phi = np.arctan((2. * b) / (a - c)) / 2
            if a > c:
                phi += np.pi / 2
        if not width_gt_height:
            # Ensure that phi is the angle to rotate to the semi-major axis.
            phi += np.pi / 2
        phi = phi % np.pi
    elif den > 0:
        # The location of the hyperbola centre.
        # [a, b, d], [b, c, f], [d, f, g]
        den *= -1
        # c *= -1
        x0, y0 = -(c * d - b * f) / den, -(a * f - b * d) / den
        matrix_temp = np.array([[a, b, d], [b, c, f], [d, f, g]])
        det = np.linalg.det(matrix_temp)
        d_ = (a + c) ** 2 - 4 * den
        lambda_1 = ((a + c) - np.sqrt(d_)) / 2
        lambda_2 = ((a + c) + np.sqrt(d_)) / 2
        ap = np.sqrt(det / lambda_1 / den)
        bp = np.sqrt(- det / lambda_2 / den)
        width_gt_height = True
        if bp < ap:
            width_gt_height = False
            ap, bp = bp, ap
        # The eccentricity.
        r = (ap / bp) ** 2
        if r > 1:
            r = 1 / r
        e = np.sqrt(1 - r)
        # The angle of anticlockwise rotation of the major-axis from x-axis.
        if b == 0:
            phi = 0 if a < c else np.pi / 2
        else:
            phi = np.arctan((2. * b) / (a - c)) / 2
            if a > c:
                phi += np.pi / 2
        if not width_gt_height:
            # Ensure that phi is the angle to rotate to the semi-major axis.
            phi += np.pi / 2
        phi = phi % np.pi
    return x0, y0, ap, bp, e, phi


def get_ellipse_pts(params, npts=100, tmin=0, tmax=2 * np.pi):
    """
    Return npts points on the ellipse described by the params = x0, y0, ap,
    bp, e, phi for values of the parametric variable t between tmin and tmax.

    """

    x0, y0, ap, bp, e, phi = params
    # A grid of the parametric variable, t.
    t = np.linspace(tmin, tmax, npts)
    x = x0 + ap * np.cos(t) * np.cos(phi) - bp * np.sin(t) * np.sin(phi)
    y = y0 + ap * np.cos(t) * np.sin(phi) + bp * np.sin(t) * np.cos(phi)
    return x, y


def calc_sun_pos_i(jd):
    # all in degree
    n = jd - 2451545.0
    l = (280.459 + 0.98564736 * n) % 360.0
    m = (357.529 + 0.98560023 * n) % 360.0
    m *= DEG2RAD
    lam = (l + 1.915 * np.sin(m) + 0.0200 * np.sin(2 * m)) % 360.0
    lam *= DEG2RAD
    e = 23.439 - 3.56e-7 * n
    e *= DEG2RAD

    r_sun = (1.00014 - 0.01671 * np.cos(m) - 0.000140 * np.cos(2 * m)) * au
    u_v = np.array([np.cos(lam), np.cos(e) * np.sin(lam), np.sin(lam) * np.sin(e)])
    return r_sun * u_v


def jday(year, mon, day, hr, minute, sec):
    jd0 = 367.0 * year - 7.0 * (year + ((mon + 9.0) // 12.0)) * 0.25 // 1.0 + 275.0 * mon // 9.0 + day + 1721013.5
    utc = ((sec / 60.0 + minute) / 60.0 + hr)  # utc in hours#
    return jd0 + utc / 24.


def eci_to_lvlh(earth_pos_i, sc_vel_i):
    lvlh_z = earth_pos_i / np.linalg.norm(earth_pos_i)
    lvlh_y = np.cross(earth_pos_i, sc_vel_i) / np.linalg.norm(np.cross(earth_pos_i, sc_vel_i))
    lvlh_x = np.cross(lvlh_y, lvlh_z)
    matrix_i2lvlh = np.array([lvlh_x, lvlh_y, lvlh_z])
    return matrix_i2lvlh


def search_tle_by_date(date_time, file_tle=None):
    jd = jday(float('20' + date_time[4:6]), float(date_time[2:4]),
              float(date_time[:2]), float(date_time[7:9]) - 1.0,
              float(date_time[9:11]),
              float(date_time[11:]))
    jd_year = jday(float('20' + date_time[4:6]), 1, 1, 0, 0, 0)
    epoch_day = jd - jd_year
    current_epoch_tle = float(date_time[4:6] + str(round(epoch_day, 8)))
    if file_tle is None:
        file_tle = "sat000052191.txt"
    file_tle = open(file_tle, 'r').read()
    epoch_day_tle = [float(line[17:33]) for line in file_tle.split('\n')[:-1] if line[0] == '1']
    idx = np.argmin(np.abs(current_epoch_tle - np.array(epoch_day_tle)))
    line_1 = file_tle.split('\n')[0 + 2 * idx]
    line_2 = file_tle.split('\n')[0 + 2 * idx + 1]
    return line_1, line_2


def get_file_info(name_image, tle_file=None):
    dt = name_image[:13]
    # UTC + 1 ()
    jd = jday(float('20' + dt[4:6]), float(dt[2:4]), float(dt[:2]), float(dt[7:9]) - 1.0, float(dt[9:11]),
              float(dt[11:]))
    line1, line2 = search_tle_by_date(dt, tle_file)
    node = None
    if 'N7' in name_image:
        node = 'N7'
    elif 'N4' in name_image:
        node = 'N4'
    else:
        node = 'NAN'
    sat = None
    if 'S3' in name_image:
        sat = 'S3'
    elif 'PS' in name_image:
        sat = 'PS'
    else:
        sat = 'NAN'
    satellite = Satrec.twoline2rv(line1, line2, WGS84)
    _, pos, vel = satellite.sgp4(int(jd), jd % 1) # km
    sun_pos_i_earth = calc_sun_pos_i(jd)
    sun_pos_from_sc = sun_pos_i_earth - pos
    return jd, sat, node, pos, vel, sun_pos_from_sc, line1, line2


def calc_curvature2(name_file, altitude):
    src = cv2.imread(name_file)
    # Convert image to gray and blur it
    src_gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

    edged = cv2.Canny(src_gray, 10, 50)

    plt.figure()
    plt.imshow(src)
    plt.figure()
    plt.imshow(edged)
    plt.show()
    edge, center_lines, center_im_list, radius_m, properties, vectors_c = 1, 1, 1, 1, 1, 1
    return edge, center_lines, center_im_list, radius_m, properties, vectors_c


if __name__ == '__main__':
    re = 6378.137  # km
    PROJECT_FOLDER = "../data/M-20230824/"
    OBC_DATA = "gyros-S3-240823.xlsx"
    VIDEO_DATA = "20230824-att1-original/frame1.png"  # reference unit time

    # loop through each image in the dataset
    earth_image = PROJECT_FOLDER + VIDEO_DATA
    jd, sat, node, pos_i, vel_i, sun_pos_sc_i, l1, l2 = get_file_info(earth_image)
    height = np.linalg.norm(pos_i) - re
    center_im_list, center_point_c, bw_temp, edge, radius_m = get_earth_pointing(earth_image, 0.003, 48.8, height)

    dict_info = {'jd': jd, 'pos': pos_i, 'vel': vel_i, 'sat': sat, 'node': node,
                 'sun_pos_sc_i': center_point_c['Sun'],
                 'img': bw_temp, 'earth_pos_c': center_point_c['Earth']}
