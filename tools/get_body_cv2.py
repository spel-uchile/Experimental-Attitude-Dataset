"""
Created by Elias Obreque
Date: 02-06-2025
email: els.obrq@gmail.com
"""
import os

import cv2
import numpy as np
from matplotlib import pyplot as plt


def segmentar_tierra_sol(img_bgr):
    # 1) Convertir a HSV
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    H, S, V = cv2.split(hsv)
    b, g, r = cv2.split(img_bgr)

    total = (b/255 + g/255 + r/255) *255 / 3
    fig, ax_hsv = plt.subplots(2, 8)
    ax_hsv[0, 0].imshow(img_bgr)
    ax_hsv[0, 1].imshow(total)
    ax_hsv[0, 2].imshow(b)
    ax_hsv[0, 3].imshow(g)
    ax_hsv[0, 4].imshow(r)
    ax_hsv[0, 5].imshow(H)
    ax_hsv[0, 6].imshow(S)
    ax_hsv[0, 7].imshow(V)

    # 2) Máscaras elemental básicas
    # 2.1) Espacio puro (negro): V < 10
    mask_img_bgr = cv2.threshold(img_bgr, 200, 255, cv2.THRESH_BINARY)[1]
    mask_total = cv2.threshold(total, np.mean(total) + np.std(total), 255, cv2.THRESH_BINARY)[1]
    mask_b = cv2.threshold(b, np.mean(b) + np.std(b), 255, cv2.THRESH_BINARY)[1]
    mask_g = cv2.threshold(g, np.mean(g) + np.std(g), 255, cv2.THRESH_BINARY)[1]
    mask_r = cv2.threshold(r, np.mean(r) + np.std(r), 255, cv2.THRESH_BINARY)[1]
    mask_H = cv2.threshold(H, 100, 255, cv2.THRESH_BINARY_INV)[1]
    mask_S = cv2.threshold(S, 100, 255, cv2.THRESH_BINARY_INV)[1]
    mask_V = cv2.threshold(V, 100, 255, cv2.THRESH_BINARY_INV)[1]


    ax_hsv[1, 0].imshow(mask_img_bgr)
    ax_hsv[1, 1].imshow(mask_total)
    ax_hsv[1, 2].imshow(mask_b)
    ax_hsv[1, 3].imshow(mask_g)
    ax_hsv[1, 4].imshow(mask_r)
    ax_hsv[1, 5].imshow(mask_H)
    ax_hsv[1, 6].imshow(mask_S)
    ax_hsv[1, 7].imshow(mask_V)
    plt.show()

    # 2.2) Océano (H≈100-140, S>50, V≈ 15-60)
    lower_ocean = np.array([100, 50, 15])
    upper_ocean = np.array([140, 255, 60])
    mask_ocean = cv2.inRange(hsv, lower_ocean, upper_ocean)

    # 2.3) Nubes / superficies muy claras (blanco/leve albedo)
    mask_clouds = cv2.inRange(hsv, np.array([0, 0, 200]), np.array([180, 60, 255]))

    # 2.4) Sol: píxeles extremadamente brillantes (V>240)
    mask_sun = cv2.threshold(V, 240, 255, cv2.THRESH_BINARY)[1]

    # 3) Región inicial candidata a Tierra: no espacio, no sol
    mask_not_space = cv2.bitwise_not(mask_g)
    mask_earth_cand = cv2.bitwise_and(mask_not_space, cv2.bitwise_not(mask_sun))

    # 4) Quitar océano y nubes de la región Tierra
    mask_earth = cv2.bitwise_and(mask_earth_cand, cv2.bitwise_not(mask_ocean))
    mask_earth = cv2.bitwise_and(mask_earth, cv2.bitwise_not(mask_clouds))

    # 5) Limpieza morfológica
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask_earth = cv2.morphologyEx(mask_earth, cv2.MORPH_CLOSE, kernel_close, iterations=2)
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask_earth = cv2.morphologyEx(mask_earth, cv2.MORPH_OPEN, kernel_open, iterations=1)

    # 6) Encontrar el contorno «más grande» y más circular
    conts, _ = cv2.findContours(mask_earth, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mejor_contorno = None
    mejor_valor = 0  # podemos combinar (área * circularidad) o solo circularidad
    for c in conts:
        area = cv2.contourArea(c)
        if area < 500:  # descarta muy pequeños
            continue
        peri = cv2.arcLength(c, True)
        circ = 4 * np.pi * area / (peri * peri + 1e-8)  # métrica de circularidad
        score = area * circ  # priorizamos área grande y buen grado circular
        if score > mejor_valor:
            mejor_valor = score
            mejor_contorno = c

    mask_final_earth = np.zeros(mask_earth.shape, dtype=np.uint8)
    if mejor_contorno is not None:
        cv2.drawContours(mask_final_earth, [mejor_contorno], -1, 255, -1)

    # 7) Hacer lo mismo para el Sol: contorno circular en mask_sun
    mask_sun_morph = cv2.morphologyEx(mask_sun, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8), iterations=1)
    conts_sun, _ = cv2.findContours(mask_sun_morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mejor_c_sun = None
    mejor_valor_sun = 0
    for c in conts_sun:
        area = cv2.contourArea(c)
        if area < 50:
            continue
        peri = cv2.arcLength(c, True)
        circ = 4 * np.pi * area / (peri * peri + 1e-8)
        if circ > 0.7 and area > mejor_valor_sun:
            mejor_valor_sun = area
            mejor_c_sun = c

    mask_final_sun = np.zeros(mask_sun.shape, dtype=np.uint8)
    if mejor_c_sun is not None:
        cv2.drawContours(mask_final_sun, [mejor_c_sun], -1, 255, -1)

    # 8) Opcional: suavizar bordes con un ligero blur
    mask_final_earth = cv2.medianBlur(mask_final_earth, 5)
    mask_final_sun = cv2.medianBlur(mask_final_sun, 5)

    return mask_final_earth, mask_final_sun


if __name__ == '__main__':
    for img_name in os.listdir('./example_picture/'):
        img = cv2.imread('./example_picture/' + img_name)
        mask_tierra, mask_sol = segmentar_tierra_sol(img)
        cv2.imwrite(f"./process/cv2_earth_{img_name}", mask_tierra)
        cv2.imwrite(f"./process/cv2_sun_{img_name}.png", mask_sol)
        print(f"Done {img_name}")


