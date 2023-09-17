"""
Created by Elias Obreque
Date: 16-09-2023
email: els.obrq@gmail.com
"""

from PIL import Image, ImageFilter
from scipy import optimize, ndimage
from tools.clustering import cluster, decrease_color
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

re = 6378.137  # km
RAD2DEG = 180 / np.pi
DEG2RAD = 1 / RAD2DEG
au = 149597870.691  # km
d_sun = 1.3927e-9
rs = au
sensor_width = 2.74 * 1e-3  # m
flength = 0.00304


def get_lines(img_):
    img_cv2 = cv2.cvtColor(np.asarray(img_), cv2.COLOR_RGB2BGR)
    gray = img_.convert('L')
    # Apply edge detection method on the image
    edges = cv2.Canny(np.asarray(gray, dtype=np.uint8), 25, 40, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, 0.01, 55)
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


def get_body(col):
    fig, ax = plt.subplots(1, 7, figsize=(15, 5))
    fig.suptitle('Test')
    fig.tight_layout()
    ax[0].imshow(col)
    ax[0].set_title("Original")

    new_data = []
    new_col, lighter_colors, counts = decrease_color(col, 10)
    color_label = []
    ax[1].imshow(new_col / 255)
    ax[1].set_title("KMeans")
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
        bw = bw ** (1 / 2)
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
    plt.show()
    return bw_temp


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
        radius_list.append([np.mean(radius), np.std(radius)])
        edge_array_list.append(edge_array)
        points_c_list.append(points_c)
    return radius_list, point_list


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
    center_pixel = np.asarray((e_c * fl / e_c[2])[:2] / pw + length * 0.5, dtype=np.uint8)
    return edge_array, points_center, e_c, center_pixel


def calc_sun_curvature(points, fl, pw, length, h):
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

    center_pixel = np.asarray((sun_c * fl / sun_c[2])[:2] / pw + length * 0.5, dtype=np.int)
    return edge_array, points_center, sun_c, center_pixel



def get_vector(file_name, height):
    col = Image.open(file_name)
    bw_bodies = get_body(col)
    radius_, point_list_ = get_type_radius(bw_bodies)
    print("RADIUS: ", radius_)
    for body_, radii, pl in zip(bw_bodies, radius_, point_list_):
        if 10 < radii[0] < 150:
            "sun"
            pixel_size = sensor_width / np.shape(body_)[0]
            sun_edge_array, sun_points_center, sun_c, center_pixel = calc_sun_curvature(pl,
                                                                                        flength,
                                                                                        pixel_size,
                                                                                        np.shape(body_)[0],
                                                                                        height)
            edge_, img_cv2_ = get_lines(col)
        elif radii[0] > 150:
            "earth"
            pixel_size = sensor_width / np.shape(body_)[0]
            # Earth: Recalculate with hyperbolic geometry
            earth_edge_array, earth_points_center, earth_c, center_pixel = calc_hyperbola(pl,
                                                                                          flength,
                                                                                          pixel_size,
                                                                                          height,
                                                                                          np.shape(body_)[0])
    return edge_, img_cv2_


if __name__ == '__main__':
    import pandas as pd

    PROJECT_FOLDER = "../data/M-20230824/20230824-att1-original/"

    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    video_salida = cv2.VideoWriter(PROJECT_FOLDER + "video_test.avi", fourcc, 10.0, (320, 180))

    list_file = [elem for elem in os.listdir(PROJECT_FOLDER) if 'png' in elem]
    num_list = [int(elem.split(".")[0].replace("frame", "")) for elem in list_file if 'png' in elem]
    datalist = pd.DataFrame({'filename': list_file, 'id': num_list})
    datalist.sort_values(by='id', inplace=True)
    height_sc = 480 #  km
    for filename in datalist['filename'].values:
        edge_, img_cv2_ = get_vector(PROJECT_FOLDER + filename, height_sc)
        # plt.figure()
        # plt.imshow(img_cv2_)
        # plt.show()
        video_salida.write(img_cv2_)

        # Liberar el objeto VideoWriter
    video_salida.release()
