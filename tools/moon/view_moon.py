"""
Created by Elias Obreque
Date: 03-06-2025
email: els.obrq@gmail.com
"""
import os
import cv2
from tools.conv_im_to_vec import get_file_info
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams['font.size'] = 12


if __name__ == '__main__':
    folder = "results/"

    name_files = [elem for elem in os.listdir("./") if "jpg" in elem]

    tle_file = "../../data/sat000052191.txt"

    dataset = []
    for earth_image in name_files:
        # jd, sat, node, pos_i, vel_i, sun_pos_sc_i, l1, l2 = get_file_info(earth_image, tle_file)
        #
        # dict_info = {'jd': jd, 'pos': pos_i, 'vel': vel_i, 'sat': sat, 'node': node, 'sun_pos_sc_i': sun_pos_sc_i,
        #              'l1': l1, 'l2': l2}
        # dataset.append(dict_info)

        # print("Log: File name: {} - JD: {} - NODE: {}".format(earth_image, jd, node))

        img = cv2.imread(earth_image, cv2.IMREAD_GRAYSCALE)
        img_gray = np.asarray(img).copy()
        img_gray[img_gray < 2] = 0
        #img_gray[img_gray > 10] = 255

        img_gray = np.array(((img_gray/ img_gray.max()) ** 0.4) * 255, dtype=np.uint8)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        # dilated = cv2.dilate(img_gray, kernel, iterations=2)

        # plt.figure()
        # plt.imshow(img)
        # plt.figure()
        # plt.imshow(img_gray, cmap='Reds', vmin=0, vmax=255)
        fig, ax = plt.subplots(1, 2, figsize=(10, 5), sharex=True, sharey=True)
        ax[0].imshow(img, cmap='gray')#, vmin=0, vmax=255)
        ax[0].set_title("Original")
        ax[1].set_title("Filtered")
        ax[1].imshow(img_gray, cmap='Reds', vmin=0, vmax=255)
        plt.tight_layout()
        #fig.savefig(folder + "filtered_{}.png".format(earth_image.remove(".jpg")), dpi=500)
        plt.show()
        plt.close()