"""
Created by Elias Obreque
els.obrq@gmail.com
Date: 10-11-2022
"""
import matplotlib.pyplot as plt
import scipy.misc
import scipy.cluster
from sklearn.cluster import KMeans
import numpy as np
import scipy
import os
from PIL import Image
import binascii

# Bajo 50, 50, 50 es negro


def cluster(col, n_clusters):
    ar = np.asarray(col).copy()
    shape = ar.shape
    ar = ar.reshape(np.product(shape[:2]), shape[2]).astype(float)

    # print('finding clusters')
    codes, dist = scipy.cluster.vq.kmeans(ar, n_clusters)
    # print('cluster centres:\n', codes)

    vecs, dist = scipy.cluster.vq.vq(ar, codes)  # assign codes
    counts, bins = np.histogram(vecs, len(codes))  # count occurrences
    counts_ = np.sort(counts)
    # for elem in reversed(counts_):
    #     index_max = np.where(elem == counts)[0][0]  # find most frequent
    #     peak = codes[index_max]
    #     colour = binascii.hexlify(bytearray(int(c) for c in peak)).decode('ascii')
    #     print('most frequent is %s (#%s)' % (peak, colour))
    return [codes, np.asarray(counts) * 0.06, bins, vecs]


def decrease_color(im, ncol, plot=False):
    w, h = im.size
    col_array = np.asarray(im.copy(), dtype=np.float32) / 255
    colors = im.getcolors(h * w)
    num_colores = len(colors)
    print(u'Número de colores = ', num_colores)
    R = np.copy(col_array[:, :, 0])
    G = np.copy(col_array[:, :, 1])
    B = np.copy(col_array[:, :, 2])
    XR = R.reshape((-1, 1))
    XG = G.reshape((-1, 1))
    XB = B.reshape((-1, 1))
    X = np.concatenate((XR, XG, XB), axis=1)
    k_means = KMeans(n_clusters=ncol)
    k_means.fit(X)
    centroides = k_means.cluster_centers_
    etiquetas = k_means.labels_
    count, _ = np.histogram(etiquetas, len(centroides))
    m = XR.shape
    for i in range(m[0]):
        XR[i] = centroides[etiquetas[i]][0]
        XG[i] = centroides[etiquetas[i]][1]
        XB[i] = centroides[etiquetas[i]][2]
    XR.shape = R.shape
    XG.shape = G.shape
    XB.shape = B.shape
    XR = XR[:, :, np.newaxis]
    XG = XG[:, :, np.newaxis]
    XB = XB[:, :, np.newaxis]

    Y = np.concatenate((XR, XG, XB), axis=2)
    # Lighter color
    norm_color = [np.linalg.norm(c) for c in centroides]
    order_ = np.sort(norm_color)[::-1]
    lighter_color_idx = [np.argmin(abs(np.array(norm_color) - val)) for idx, val in enumerate(order_)]
    lighter_colors = centroides[lighter_color_idx]
    if plot:
        plt.figure(figsize=(12, 12))
        plt.title("Decrease Color - Array")
        plt.imshow(col_array)
        plt.axis('off')
        plt.figure(figsize=(12, 12))
        plt.title("Decrease Color - RGB")
        plt.imshow(Y)
        plt.axis('off')
        plt.show()
    return Y * 255, lighter_colors * 255, count[lighter_color_idx]


if __name__ == '__main__':
    SC = "S3"
    filename_list = []
    folder = "realPicture/"
    black_count_max = 1
    black_count = 0
    black_color_norm = 23
    white_color_norm = 255
    white_count_max = 1
    white_count = 0
    plot_file = False
    for filename in os.listdir("realPicture"):
        save = False
        if '.jpg' in filename and 'N7' in filename and SC in filename:
            im = Image.open(folder + filename)
            if im.width > 200:
                continue
            colors_list = np.asarray(im.copy()).reshape(im.width * im.height, 3)
            mean_color = np.mean(np.mean(colors_list, axis=1))
            std_color = np.std(np.mean(colors_list, axis=1))
            if white_color_norm * 0.98 > mean_color > black_color_norm:
                print("NORMAL - ", filename, "Mean value:", mean_color)
                filename_list.append(filename)
                save = True
            elif white_count < white_count_max and white_color_norm * 0.98 < mean_color:
                print("WHITE - ", filename)
                filename_list.append(filename)
                white_count += 1
                save = True
            elif black_count < black_count_max and mean_color < black_color_norm:
                print("BLACK - ", filename)
                filename_list.append(filename)
                black_count += 1
                save = True
            if plot_file and save:
                plt.figure()
                plt.imshow(im)

    # plt.show()
    cluster_list = []

    fig_sub, ax = plt.subplots(3, int(np.ceil(len(filename_list) / 3)))
    ax = ax.flatten()
    fig_sub_colorless, ax_colorless = plt.subplots(3, int(np.ceil(len(filename_list) / 3)))
    ax_colorless = ax_colorless.flatten()
    i = 0
    n_clusters = 10
    for filename in filename_list:
        im = Image.open(folder + filename)
        cluster_list.append(cluster(im, n_clusters))
        im_colorless = decrease_color(im, n_clusters)
        ax[i].set_title(str(i + 1))
        ax[i].imshow(im)
        ax_colorless[i].set_title(str(i + 1))
        ax_colorless[i].imshow(im_colorless[0]/ 255)
        i += 1

    fig_clu, ax = plt.subplots(1, 3, figsize=(15, 5))
    i = 1
    ax = ax.flatten()
    markers = ['o', '.', ',', 'x', '+', 'v', '^', '<', '>', 's', 'd']
    for cluster_ in cluster_list[:10]:
        xs = cluster_[0][:, 0]
        ys = cluster_[0][:, 1]
        zs = cluster_[0][:, 2]
        ax[0].plot(range(len(xs)), xs, '--', lw=0.5)
        ax[1].plot(range(len(xs)), ys, '--', lw=0.5)
        ax[2].plot(range(len(xs)), zs, '--', lw=0.5)
        ax[0].scatter(range(len(xs)), xs, label=str(i), marker=markers[i - 1], s=cluster_[1], alpha=0.5)
        ax[1].scatter(range(len(ys)), ys, label=str(i), marker=markers[i - 1], s=cluster_[1], alpha=0.5)
        ax[2].scatter(range(len(zs)), zs, label=str(i), marker=markers[i - 1], s=cluster_[1], alpha=0.5)
        i += 1
    [ax_.set_xlabel('N Cluster') for ax_ in ax]
    ax[0].set_ylabel('RED')
    ax[1].set_ylabel('GREEN')
    ax[2].set_ylabel('BLUE')
    plt.legend()

    fig_rgb, ax = plt.subplots(1, n_clusters, figsize=(15, 5))

    ax = ax.flatten()
    i = 1
    for cluster_ in cluster_list[:10]:
        for j, color_rgb in enumerate(cluster_[0]):
            ax[j].plot(range(len(color_rgb)), color_rgb, '--', color=color_rgb/255, lw=0.5)
            ax[j].legend()
            ax[j].scatter(range(len(color_rgb)), color_rgb, color=color_rgb/255, marker=markers[i - 1],
                          s=cluster_[1][j], alpha=0.5, label=str(i))
        i += 1
    [ax_.set_xlabel('Color RGB') for k, ax_ in enumerate(ax)]
    [ax_.set_ylabel('Cluster ' + str(k)) for k, ax_ in enumerate(ax)]
    plt.show()