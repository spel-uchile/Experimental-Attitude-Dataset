
import cv2
import os
import errno


def save_frame(folder, video_file, reduction=False):
    # Crear un objeto de captura de video
    cap = cv2.VideoCapture(folder + video_file)

    try:
        os.mkdir(folder + video_file.split('.')[0])
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

    # Inicializar un contador de frames
    frame_count = 0

    # Bucle para leer los frames del video
    while cap.isOpened():
        # Leer el siguiente frame
        ret, frame = cap.read()

        # Verificar si se ha llegado al final del video
        if not ret:
            break

        # Incrementar el contador de frames
        frame_count += 1

        # resize frame
        width, height = frame.shape[1], frame.shape[0]
        max_dim = min(width, height)
        factor = max_dim / 90  # pixels
        frame = cv2.resize(frame, (int(width / factor), int(height / factor)))
        # cut
        width, height = frame.shape[1], frame.shape[0]
        # frame = frame[int(height / 2 - 50): int(height / 2 + 50), int(width / 2 - 50): int(width / 2 + 50), :]

        # Guardar el frame como una imagen
        frame_file = f"frame{frame_count}.png"
        cv2.imwrite(folder + video_file.split('.')[0] + '/' + frame_file, frame)

    # Liberar los recursos
    cap.release()


if __name__ == '__main__':
    import pandas as pd
    from PIL import Image
    from tools.clustering import cluster, decrease_color
    import numpy as np
    import pickle

    PROJECT_FOLDER = "../data/20230904/"
    VIDEO_DATA = "20230904-video-att9-original.mp4"
    save_frame(PROJECT_FOLDER, VIDEO_DATA, reduction=True)
    NEW_FOLDER = PROJECT_FOLDER + VIDEO_DATA.split(".")[0] + '/'

    img_type = 'png'
    fps = 30
    red_fps = 1
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_salida = cv2.VideoWriter(NEW_FOLDER + "video_reduction.mp4", fourcc, int(fps / red_fps), (180, 90))

    list_file = [elem for elem in os.listdir(NEW_FOLDER) if img_type in elem]
    num_list = [int(elem.split(".")[0].replace("frame", "")) for elem in list_file if img_type in elem]
    datalist = pd.DataFrame({'filename': list_file, 'id': num_list})
    datalist.sort_values(by='id', inplace=True)
    k = 0
    for filename in datalist['filename'].values[:1:red_fps]:
        print(k)
        col = Image.open(NEW_FOLDER + filename)
        # new_col, lighter_colors, counts = decrease_color(col, 5)
        # col.putdata([tuple(colors) for colors in new_col.reshape(-1, 3)])
        gray = col.convert('L')# .resize((80, 45))
        # with open(NEW_FOLDER + "reduction_pickle.p", "wb") as fl:
        #     pickle.dump(col, fl)
        k += 1
        video_salida.write(np.asarray(col.resize((180, 90))))

    video_salida.release()
